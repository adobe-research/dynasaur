import ast
import os
import re
from copy import deepcopy
from glob import glob
from typing import Callable, Dict, List, Optional, Union

import transformers
from transformers.agents import Agent, ReactCodeAgent
from transformers.agents.agents import (
    AgentExecutionError,
    AgentGenerationError,
    AgentParsingError,
    Toolbox,
    parse_code_blob,
)
from transformers.agents.llm_engine import MessageRole
from transformers.agents.prompts import DEFAULT_REACT_CODE_SYSTEM_PROMPT
from transformers.agents.tools import DEFAULT_TOOL_DESCRIPTION_TEMPLATE, Tool

from env import Env
from scripts.llm_engines import AzureOpenAIEngine
from utils import GeneratedTool, add_parent_pointers, parse_generated_tools


# Define a timeout exception
class TimeoutException(Exception):
    pass


def format_prompt_with_tools(toolbox: Toolbox, prompt_template: str) -> str:
    tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in toolbox._tools.values()])
    if tool_descriptions == "":
        tool_descriptions = "None"
    prompt = prompt_template.replace("<<tool_descriptions>>", tool_descriptions)
    return prompt


class AgentWithMetrics(Agent):
    """Just Agent with metric tracking"""

    def reset_metrics(self):
        # Reset LLM engine and tools' metrics
        if hasattr(self.llm_engine, "reset"):
            self.llm_engine.reset()
        for tool in self.toolbox.tools:
            if hasattr(tool, "reset"):
                tool.reset()

        self.metrics = {
            "num_api_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    def update_metrics(self):
        # Delegate OpenAI-related metric tracking to each tool and engine. Make sure all llm_engines are different objects!
        if hasattr(self.llm_engine, "metrics"):
            metrics = self.llm_engine.metrics
            self.metrics["num_api_calls"] += metrics.get("num_calls", 0)
            self.metrics["prompt_tokens"] += metrics.get("prompt_tokens", 0)
            self.metrics["completion_tokens"] += metrics.get("completion_tokens", 0)

        for tool in self.toolbox.tools:
            if hasattr(tool, "metrics"):
                metrics = tool.metrics
                self.metrics["num_api_calls"] += metrics.get("num_calls", 0)
                self.metrics["prompt_tokens"] += metrics.get("prompt_tokens", 0)
                self.metrics["completion_tokens"] += metrics.get("completion_tokens", 0)


class UnrestrictedReactCodeAgent(ReactCodeAgent, AgentWithMetrics):
    def __init__(
        self,
        tools: Union[List[Tool], Toolbox],
        llm_engine: Callable,
        system_prompt: str = DEFAULT_REACT_CODE_SYSTEM_PROMPT,
        tool_description_template: str = DEFAULT_TOOL_DESCRIPTION_TEMPLATE,
        planning_interval: Optional[int] = None,
        env: Env = None,
        **kwargs,
    ):
        super().__init__(
            tools=tools,
            llm_engine=llm_engine,
            system_prompt=system_prompt,
            tool_description_template=tool_description_template,
            planning_interval=planning_interval,
            **kwargs,
        )
        self.system_prompt = self.system_prompt
        self.env = env

    def initialize_for_run(self, task: str, **kwargs):
        self.reset_metrics()
        self.token_count = 0
        self.task = task
        if len(kwargs) > 0:
            self.task += f"\nYou have been provided with these initial arguments: {str(kwargs)}."
        self.state = kwargs.copy()
        self.system_prompt = format_prompt_with_tools(
            self._toolbox,
            self.system_prompt_template,
        )
        self.logs = [{"system_prompt": self.system_prompt, "task": self.task}]
        self.logger.warn("======== New task ========")
        self.logger.log(33, self.task)
        self.logger.debug("System prompt is as follows:")
        self.logger.debug(self.system_prompt)

    def run(self, task: str, *args, **kwargs):
        self.initialize_for_run(task)
        return self.direct_run(task)

    def step(self):
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        The errors are raised here, they are caught and logged in the run() method.
        """
        agent_memory = self.write_inner_memory_from_logs()

        self.prompt = agent_memory.copy()

        self.logger.debug("===== New step =====")

        # Add new step in logs
        current_step_logs = {}
        self.logs.append(current_step_logs)
        current_step_logs["agent_memory"] = agent_memory.copy()

        self.logger.info("===== Calling LLM with these last messages: =====")
        self.logger.info(self.prompt[-2:])

        try:
            llm_output = self.llm_engine(self.prompt, stop_sequences=["<end_action>", "Observation:"])
            self.update_metrics()
        except Exception as e:
            raise AgentGenerationError(f"Error in generating llm output: {e}.")

        self.logger.debug("===== Output message of the LLM: =====")
        self.logger.debug(llm_output)
        current_step_logs["llm_output"] = llm_output

        # Parse
        self.logger.debug("===== Extracting action =====")
        try:
            rationale, raw_code_action = self.extract_action(llm_output=llm_output, split_token="Code:")
        except Exception as e:
            self.logger.debug(f"Error in extracting action, trying to parse the whole output. Error trace: {e}")
            rationale, raw_code_action = llm_output, llm_output

        try:
            code_action = parse_code_blob(raw_code_action)
        except Exception as e:
            error_msg = f"Error in code parsing: {e}. Make sure to provide correct code"
            raise AgentParsingError(error_msg)

        current_step_logs["rationale"] = rationale
        current_step_logs["tool_call"] = {
            "tool_name": "code interpreter",
            "tool_arguments": code_action,
        }

        # Execute
        self.log_code_action(code_action)
        state = self.env.step(code_action)
        if state.error:
            error_msg = f"Code execution failed due to the following error:\n{str(state.error)}"
            if "'dict' object has no attribute 'read'" in str(state.error):
                error_msg += "\nYou get this error because you passed a dict as input for one of the arguments instead of a string."
            raise AgentExecutionError(error_msg)
        else:
            result = state.result
            information = result
            self.logger.warning("Print outputs:")
            self.logger.log(32, information)
            current_step_logs["observation"] = information

        for line in code_action.split("\n"):
            if line[: len("submit_final_answer")] == "submit_final_answer":
                self.logger.warning(">>> Final answer:")
                self.logger.log(32, result)
                current_step_logs["final_answer"] = result

        return current_step_logs


class DynamicActionSpaceAgent(UnrestrictedReactCodeAgent):
    def __init__(self, generated_tool_dir: str, disable_accum: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generated_tool_dir = generated_tool_dir
        self.disable_accum = disable_accum

        # Load generated tools from disk
        generated_tools: list[GeneratedTool] = []
        generated_tool_paths = sorted(glob(os.path.join(self.generated_tool_dir, "*.py")))
        for path in generated_tool_paths:
            code = open(path, "r").read()
            tools = parse_generated_tools(code)
            generated_tools.extend(tools)
            # Load generated tool to env
            self.env.step(code)
        self.generated_toolbox = Toolbox(generated_tools)

        # We need this to undo _num_calls in env when a proposed function encounter logical error
        self.prev_num_calls = {}

        # Make an engine to correct docstring
        self.docstring_corrector = AzureOpenAIEngine(self.llm_engine.model_name)

        # Disable all logging
        state = self.env.step("import transformers")
        assert not state.error
        state = self.env.step("logging = transformers.agents.agents.logging")
        assert not state.error
        state = self.env.step("logging.disable(logging.CRITICAL + 1)")
        assert not state.error

        self._toolbox.remove_tool("final_answer")

    def reset_metrics(self):
        # Reset LLM engine and tools' metrics
        if hasattr(self.llm_engine, "reset"):
            self.llm_engine.reset()
        for tool in self.toolbox.tools:
            if hasattr(tool, "reset"):
                tool.reset()

        self.env.step("_num_calls = {}")

        self.metrics = {
            "num_api_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "collision": {},
            "function_calls": {},
        }

    def initialize_for_run(self, task: str, **kwargs):
        self.reset_metrics()
        self.token_count = 0
        self.task = task
        if len(kwargs) > 0:
            self.task += f"\nYou have been provided with these initial arguments: {str(kwargs)}."
        self.state = kwargs.copy()
        self.system_prompt = transformers.agents.agents.format_prompt_with_tools(
            self._toolbox,
            self.system_prompt_template,
            self.tool_description_template,
        )
        generated_tool_descriptions = self.generated_toolbox.show_tool_descriptions(self.tool_description_template)
        self.system_prompt = self.system_prompt.replace("<<generated_tool_descriptions>>", generated_tool_descriptions)
        self.logs = [{"system_prompt": self.system_prompt, "task": self.task}]
        self.logger.warn("\n" * 5)
        self.logger.warn("======== New task ========")
        # self.logger.log(33, self.task)
        # self.logger.debug("System prompt is as follows:")
        self.logger.warning("[SYSTEM_PROMPT]")
        self.logger.debug(self.system_prompt)
        self.logger.warning("[TASK]")
        self.logger.log(33, self.task)

    def run(self, *args, **kwargs):
        return super().run(*args, **kwargs)

    def write_inner_memory_from_logs(self, summary_mode: Optional[bool] = False) -> List[Dict[str, str]]:
        """
        Reads past llm_outputs, actions, and observations or errors from the logs into a series of messages
        that can be used as input to the LLM.
        """
        prompt_message = {
            "role": MessageRole.SYSTEM,
            "content": self.logs[0]["system_prompt"],
        }
        task_message = {
            "role": MessageRole.USER,
            "content": "Task: " + self.logs[0]["task"],
        }
        if summary_mode:
            memory = [task_message]
        else:
            memory = [prompt_message, task_message]
        for i, step_log in enumerate(self.logs[1:]):
            if "llm_output" in step_log and not summary_mode:
                thought_message = {
                    "role": MessageRole.ASSISTANT,
                    "content": step_log["llm_output"].strip(),
                }
                memory.append(thought_message)

            if "tool_call" in step_log and summary_mode:
                tool_call_message = {
                    "role": MessageRole.ASSISTANT,
                    "content": f"[STEP {i} TOOL CALL]: " + str(step_log["tool_call"]).strip(),
                }
                memory.append(tool_call_message)

            if "error" in step_log or "observation" in step_log:
                if "error" in step_log:
                    message_content = (
                        # f"[OUTPUT OF STEP {i}] Error: "
                        "Observation:\n"
                        + str(step_log["error"])
                        + "\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n"
                    )
                elif "observation" in step_log:
                    # message_content = f"[OUTPUT OF STEP {i}] Observation:\n{step_log['observation']}"
                    message_content = f"Observation:\n{step_log['observation']}"
                tool_response_message = {
                    "role": MessageRole.SYSTEM,
                    "content": message_content,
                }
                memory.append(tool_response_message)

        return memory

    def step(self):
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        The errors are raised here, they are caught and logged in the run() method.
        """
        agent_memory = self.write_inner_memory_from_logs()

        self.prompt = agent_memory.copy()

        self.logger.debug("===== New step =====")

        # Add new step in logs
        current_step_logs = {}
        self.logs.append(current_step_logs)
        current_step_logs["agent_memory"] = agent_memory.copy()

        try:
            llm_output = self.llm_engine(self.prompt, stop_sequences=["<end_action>", "Observation:"])
            self.update_metrics()
        except Exception as e:
            raise AgentGenerationError(f"Error in generating llm output: {e}.")

        self.logger.debug("===== Output message of the LLM: =====")
        self.logger.debug(llm_output)
        current_step_logs["llm_output"] = llm_output

        # Parse
        self.logger.debug("===== Extracting action =====")
        try:
            rationale, raw_code_action = self.extract_action(llm_output=llm_output, split_token="Code:")
        except Exception as e:
            self.logger.debug(f"Error in extracting action, trying to parse the whole output. Error trace: {e}")
            rationale, raw_code_action = llm_output, llm_output
            # error_msg = f"You did not output the correct format!"
            # raise AgentParsingError(error_msg)

        try:
            code_action = parse_code_blob(raw_code_action)
        except Exception as e:
            error_msg = f"Error in code parsing: {e}. Make sure to provide correct code"
            raise AgentParsingError(error_msg)

        current_step_logs["rationale"] = rationale
        current_step_logs["tool_call"] = {
            "tool_name": "code interpreter",
            "tool_arguments": code_action,
        }

        # Execute
        self.log_code_action(code_action)

        code_action = self.prerun(code_action)
        state = self.env.step(code_action)
        if state.error:
            # Execute failed

            # Undo _num_calls counter
            self.env.step(f"_num_calls = {self.prev_num_calls}")

            error_msg = f"Code execution failed due to the following error:\n{str(state.error)}"
            if "'dict' object has no attribute 'read'" in str(state.error):
                error_msg += "\nYou get this error because you passed a dict as input for one of the arguments instead of a string."
            raise AgentExecutionError(error_msg)
        else:
            # Execute succesfully
            result = state.result

            # Extract metrics
            output = self.env.step("_num_calls")
            num_calls = ast.literal_eval(output.result)  # Can't use json.loads here because name must be enclosed in double quotes
            self.prev_num_calls = deepcopy(self.metrics["function_calls"])
            self.metrics["function_calls"].update(num_calls)
            current_step_logs["metrics"] = deepcopy(self.metrics)

            information = result
            self.logger.warning("Print outputs:")
            self.logger.log(32, information)
            current_step_logs["observation"] = information

            # Add generated tools unless it is bash command to install packages
            try:
                if not self.disable_accum:
                    self.save_generated_tools(code_action)
            except Exception as e:
                print(f"Could not save generated tool due to the following error:\n{e}")

        # Parse final answer if any
        for line in code_action.split("\n"):
            if line[: len("submit_final_answer")] == "submit_final_answer":
                self.logger.warning(">>> Final answer:")
                self.logger.log(32, result)
                current_step_logs["final_answer"] = result

        return current_step_logs

    def prerun(self, code_action: str) -> str:
        shell_cmds, code_action = self.remove_shell_commands(code_action)
        code_action = self.correct_docstring(code_action)
        self.check_collision(code_action)
        code_action = self.add_decorators(code_action)

        # Add the shell_commands back
        code_action = shell_cmds + "\n" + code_action
        return code_action

    def remove_shell_commands(self, code_action: str) -> str:
        shell_cmds = []
        no_cmds_code_action = []
        for line in code_action.split("\n"):
            if line.startswith("!"):
                shell_cmds.append(line)
            else:
                no_cmds_code_action.append(line)

        shell_cmds = "\n".join(shell_cmds)
        code_action = "\n".join(no_cmds_code_action)
        return shell_cmds, code_action

    def correct_docstring(self, code_action: str) -> str:
        try:
            tree = ast.parse(code_action)
        except Exception as e:
            print(f"Attempt to correct docstring failed due to the following error: {e}")
            return code_action

        add_parent_pointers(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and isinstance(node.parent, ast.Module):
                if ast.get_docstring(node) is None:
                    func = ast.unparse(node)
                    messages = [
                        {
                            "role": MessageRole.USER,
                            "content": f"Write a one-line docstring for the following Python function:\n```\n{func}\n```",
                        }
                    ]
                    resp = self.docstring_corrector(messages)
                    try:
                        docstring = re.findall(r'"""(.*?)"""', resp, re.DOTALL)[0]
                        node.body.insert(0, ast.Expr(value=ast.Constant(value=docstring)))
                    except Exception as e:
                        print(f"Attempt to correct docstring failed due to the following error: {e}")
                        return code_action

        try:
            corrected_code_action = ast.unparse(tree)
            return corrected_code_action
        except Exception as e:
            print(f"Attempt to correct docstring failed due to the following error: {e}")
            return code_action

    def check_collision(self, code_action: str):
        # Make sure code_action has no syntax errors first
        try:
            tree = ast.parse(code_action)
        except:
            return

        add_parent_pointers(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and isinstance(node.parent, ast.Module):
                name = node.name
                if name in self.generated_toolbox.tools:
                    if name not in self.metrics["collision"]:
                        self.metrics["collision"][name] = 1
                    else:
                        self.metrics["collision"][name] += 1
                    # error_msg = f"Function name '{name}' already exists. Please choose a different name."
                    # raise AgentExecutionError(error_msg)

    def add_decorators(self, code_action: str) -> str:
        # TODO: Need to add decorator to generated functions that were loaded from disk as well
        try:
            tree = ast.parse(code_action)
            add_parent_pointers(tree)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and isinstance(node.parent, ast.Module):
                    decorator = ast.Name(id="track_num_calls", ctx=ast.Load())
                    node.decorator_list.append(decorator)
            updated_code_action = ast.unparse(tree)
            return updated_code_action
        except:
            print("Add decorator failed :( returning original code_action")
            return code_action

    def save_generated_tools(self, code_action: str):
        _, code_action = self.remove_shell_commands(code_action)
        generated_tools = parse_generated_tools(code_action)

        # Save new tool to disk
        # Call ToolRetriever in env to load the new tool from disk
        for tool in generated_tools:
            self.generated_toolbox.add_tool(tool)

            tool_id = len(self.generated_toolbox.tools)
            file_name = f"{tool_id}".zfill(4) + f"_{tool.name}.py"
            file_path = os.path.join(self.generated_tool_dir, file_name)

            content = f"{tool.code}"
            if tool.dependencies:
                content = f"{tool.dependencies}\n\n\n" + content

            with open(file_path, "w") as f:
                f.write(content)

            self.env.step(f'tool_retriever_tool.add_new_tool_from_path("{file_path}")')


class StructuredOutputDynamicActionSpaceAgent(DynamicActionSpaceAgent):
    def step(self):
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        The errors are raised here, they are caught and logged in the run() method.
        """
        agent_memory = self.write_inner_memory_from_logs()

        self.prompt = agent_memory.copy()

        self.logger.debug("===== New step =====")

        # Add new step in logs
        current_step_logs = {}
        self.logs.append(current_step_logs)
        current_step_logs["agent_memory"] = agent_memory.copy()

        try:
            llm_output_dict = self.llm_engine(self.prompt)
            llm_output = f"Thought: {llm_output_dict.thought}\nCode: {llm_output_dict.code}"
            self.update_metrics()
        except Exception as e:
            raise AgentGenerationError(f"Error in generating llm output: {e}.")

        self.logger.debug("===== Output message of the LLM: =====")
        self.logger.debug(llm_output)
        current_step_logs["llm_output"] = llm_output

        # Parse
        rationale, code_action = llm_output_dict.thought, llm_output_dict.code

        current_step_logs["rationale"] = rationale
        current_step_logs["tool_call"] = {
            "tool_name": "code interpreter",
            "tool_arguments": code_action,
        }

        # Execute
        self.log_code_action(code_action)

        code_action = self.prerun(code_action)
        state = self.env.step(code_action)
        if state.error:
            # Execute failed

            # Undo _num_calls counter
            self.env.step(f"_num_calls = {self.prev_num_calls}")

            error_msg = f"Code execution failed due to the following error:\n{str(state.error)}"
            if "'dict' object has no attribute 'read'" in str(state.error):
                error_msg += "\nYou get this error because you passed a dict as input for one of the arguments instead of a string."
            raise AgentExecutionError(error_msg)
        else:
            # Execute succesfully
            result = state.result

            # Extract metrics
            output = self.env.step("_num_calls")
            num_calls = ast.literal_eval(output.result)  # Can't use json.loads here because name must be enclosed in double quotes
            self.prev_num_calls = deepcopy(self.metrics["function_calls"])
            self.metrics["function_calls"].update(num_calls)
            current_step_logs["metrics"] = deepcopy(self.metrics)

            information = result
            self.logger.warning("Print outputs:")
            self.logger.log(32, information)
            current_step_logs["observation"] = information

            # Add generated tools unless it is bash command to install packages
            try:
                if not self.disable_accum:
                    self.save_generated_tools(code_action)
            except Exception as e:
                print(f"Could not save generated tool due to the following error:\n{e}")

        # Parse final answer if any
        for line in code_action.split("\n"):
            if line[: len("submit_final_answer")] == "submit_final_answer":
                self.logger.warning(">>> Final answer:")
                self.logger.log(32, result)
                current_step_logs["final_answer"] = result

        return current_step_logs
