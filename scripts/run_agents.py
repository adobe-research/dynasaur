import copy
import json
import os
import traceback
from datetime import datetime
from queue import Queue
from typing import Any, Callable, Dict, List

import pandas as pd
from datasets import Dataset
from langchain.agents import AgentExecutor
from langchain.tools.base import ToolException
from tqdm import tqdm
from transformers.agents.agents import AgentError


def run_agent(
    example: Dict,
    agent_executor: AgentExecutor,
    agent_name: str,
    agent_call_function: Callable,
    writer_queue: Queue = None,
    **kwargs,
) -> dict:
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    augmented_question = example["augmented_question"]
    try:
        # run executor agent
        response = agent_call_function(agent_executor, augmented_question, **kwargs)

        # check for parsing errors which indicate the LLM failed to follow the ReACT format
        # this could be due to an issue with the tool calling format or ReACT formatting (i.e. Thought, Action, Observation, etc.)
        parsing_error = (
            True
            if any(
                [
                    "Could not parse LLM output" in step
                    for step in response["intermediate_steps"]
                ]
            )
            else False
        )

        # check if iteration limit exceeded
        iteration_limit_exceeded = (
            True
            if "Agent stopped due to iteration limit or time limit." in response["output"]
            else False
        )
        raised_exception = False

    except (ValueError, ToolException) as e:
        print("Error on ", augmented_question, e)
        response = {"output": None, "intermediate_steps": None, "metrics": {}}
        parsing_error = False
        iteration_limit_exceeded = False
        exception = e
        raised_exception = True
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    intermediate_steps = response["intermediate_steps"]
    metrics = response["metrics"]
    annotated_example = {
        "agent_name": agent_name,
        "question": example["question"],
        "augmented_question": augmented_question,
        "prediction": response["output"],
        "intermediate_steps": intermediate_steps,
        "parsing_error": parsing_error,
        "iteration_limit_exceeded": iteration_limit_exceeded,
        "agent_error": str(exception) if raised_exception else None,
        "start_time": start_time,
        "end_time": end_time,
        "level": example["level"],
        "true_answer": example["true_answer"],
        "metrics": metrics,
    }
    temp = copy.deepcopy(example)
    temp.update(annotated_example)
    annotated_example = temp
    if writer_queue:
        writer_queue.put(annotated_example)
    return annotated_example


def serialize_agent_error(obj):
    if isinstance(obj, AgentError):
        return {"error_type": obj.__class__.__name__, "message": obj.message}
    else:
        return str(obj)


def answer_questions(
    dataset: Dataset,
    agent: AgentExecutor,
    agent_name: str,
    agent_call_function: Callable,
    output_folder: str = "output",
) -> List[Dict[str, Any]]:
    """
    Evaluates the agent on a given dataset.

    Args:
        dataset (Dataset): The dataset to test the agent on.
        agent: The agent.
        agent_name (str): The name of the agent model.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the evaluation results for each example in the dataset.
        Each dictionary includes the agent model ID, evaluator model ID, question, ground truth answer, prediction,
        intermediate steps, evaluation score, evaluation feedback, tool call parsing error flag, iteration limit
        exceeded flag, agent error (if any), and example metadata (task).
    """
    os.makedirs(output_folder, exist_ok=True)
    output_path = f"{output_folder}/{agent_name}.jsonl"
    print(f"Loading answers from {output_path}...")
    if os.path.exists(output_path):
        results = pd.read_json(output_path, lines=True).to_dict(orient="records")
        print(f"Found {len(results)} previous results!")
    else:
        print("Found no usable records! 🤔 Starting new")
        results = []

    results_df = pd.DataFrame(results)

    for _, example in tqdm(enumerate(dataset), total=len(dataset)):
        if len(results_df) > 0:
            if example["question"] in results_df["question"].unique():
                continue
        try:
            prompt_use_files = ""
            if example["file_name"]:
                prompt_use_files += f"\n\nTo answer the question above, you will have to use these attached files:"
                prompt_use_files += f"\nAttached file: {example['file_name']}"
            else:
                prompt_use_files += "\n\nYou have been given no local files to access."

            example["augmented_question"] = (
                f"""It is paramount that you complete this task and provide a correct answer.
Give it all you can: I know for a fact that you have access to all the relevant tools to solve it. Failure or 'I cannot answer' will not be tolerated, success will be rewarded.
Here is the task:
"""
                + example["question"]
                + prompt_use_files
            )

            # run agent
            result = run_agent(
                example=example,
                agent_executor=agent,
                agent_name=agent_name,
                agent_call_function=agent_call_function,
            )
        except Exception as e:
            # raise Exception
            error_trace = ("\n\n" + traceback.format_exc()).strip()
            result = example
            result["error_trace"] = error_trace

        # add in example metadata
        result.update(
            {
                "true_answer": example["true_answer"],
                "level": example["level"],
            }
        )
        results.append(result)

        with open(output_path, "w") as f:
            for d in results:
                json.dump(d, f, default=serialize_agent_error)
                f.write("\n")  # add a newline for JSONL format
        # except Exception as e:
        #     print("EXCEPTION!!!!=================\nFIND THE EXCEPTION LOG BELOW:\n", e)
    return results
