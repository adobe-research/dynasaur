# This code is based on Open Interpreter. Original source: https://github.com/OpenInterpreter/open-interpreter

import ast
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Union

from jupyter_client import KernelManager

# turn off colors in "terminal"
# os.environ["ANSI_COLORS_DISABLED"] = "1"


@dataclass
class EnvState:
    """
    Represents the state of an environment in which commands are executed.
    """
    command: List[str] = field(default_factory=list)
    result: Optional[str] = ''
    error: Optional[str] = None
    pwd: Optional[str] = ''
    ls: Optional[str] = ''

    def __str__(self):
        return (f"Result: {self.result}\n"
                f"Error: {self.error}\n"
                f"PWD: {self.pwd}\n"
                f"LS: {self.ls}")


class BaseEnv:
    """
    A base class for environments configurations in action-based systems.

    This class provides foundational attributes and methods for managing environments,
    including timeouts, working directories, and environmental states. It is designed
    to be extended by subclasses that implement specific environments behaviors.
    """

    def __init__(self, original_working_dir: str) -> None:
        """
        Initializes the environments with default settings.

        Sets up the working directory, applying a default timeout and preparing the
        environments state. If the working directory does not exist, it is created.
        """
        self._name: str = self.__class__.__name__
        self.timeout: int = 300
        self.original_working_dir = original_working_dir
        if os.path.isabs(self.original_working_dir):
            self.working_dir = self.original_working_dir
        else:
            self.working_dir = os.path.abspath(os.path.join(__file__, "..", "..", self.original_working_dir))
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

        self.env_state: Union[EnvState, None] = None

    def step(self, code):
        """
        Generator that yields a dictionary in LMC format:
        {"type": "console", "format": "output", "content": "a printed statement"}
        {"type": "console", "format": "active_line", "content": "1"}
        {"type": "image", "format": "base64", "content": "{base64}"}
        """
        return {"type": "console", "format": "output", "content": code}

    def stop(self):
        """
        Halts code execution, but does not terminate state.
        """
        pass

    def terminate(self):
        """
        Terminates state.
        """
        pass

    def list_working_dir(self):
        """
        Lists the contents of the working directory in a detailed format.

        Returns a string representation similar to the output of the 'ls' command in Linux,
        including file/directory names, sizes, and types.

        Returns:
            str: Detailed listings of the working directory's contents, or an error message if the directory does not exist.
        """
        directory = self.working_dir
        # Check if the directory exists
        if not os.path.exists(directory):
            return f"Directory '{directory}' does not exist."

        # List files and directories
        files_and_dirs = os.listdir(directory)

        # Create a list to store the details
        details = []

        for name in files_and_dirs:
            # Get the full path
            full_path = os.path.join(directory, name)

            # Get file or directory size
            size = os.path.getsize(full_path)

            # Check if it's a file or directory
            if os.path.isdir(full_path):
                doc_type = 'Directory'
            else:
                doc_type = 'File'

            details.append(f"{name}\t {size} bytes\t {doc_type}")

        return "\n".join(details)
        
    def step(self, _command) -> EnvState:
        """
        Executes a command within the environments.

        This method is intended to be implemented by subclasses, defining how commands
        are processed and their effects on the environments state.

        Args:
            _command: The command to be executed.

        Raises:
            NotImplementedError: Indicates that the subclass must implement this method.

        Returns:
            EnvState: The state of the environments after executing the command.
        """
        raise NotImplementedError

    def reset(self):
        """
        Resets the environments to its initial state.

        This method is intended to be implemented by subclasses, defining the specific
        actions required to reset the environments.
        """
        working_dir = self.original_working_dir
        if os.path.isabs(working_dir):
            self.working_dir = working_dir
        else:
            self.working_dir = os.path.abspath(os.path.join(__file__, "..", "..", "..", working_dir))
    
    @property
    def name(self):
        """
        The name of the environments.

        Returns:
            str: The name of the environments, typically set to the class name unless overridden in a subclass.
        """
        return self._name

    def __repr__(self):
        """
        Provides a string representation of the environments.

        Returns:
            str: A representation of the environments, including its name.
        """
        return f'{self.name}'

    def __str__(self):
        """
        Returns the string representation of the environments, mirroring `__repr__`.

        Returns:
            str: A string representation of the environments.
        """
        return self.__repr__()


class PythonJupyterEnv(BaseEnv):
    """
    A class representing an environment for executing Python code in a Jupyter environment.

    This class manages the execution of Python code using IPython kernel, providing methods for preprocessing code,
    executing code steps, handling output messages, and terminating the kernel.

    It inherits from BaseEnv, which provides basic environment functionality.
    """    
    file_extension = "py"
    name = "Python"
    aliases = ["py", "API"]

    def __init__(self, working_dir: str = None, extra_env_vars: dict = None):
        """
        Initializes the Python Jupyter environment.

        This method sets up the IPython kernel manager and client, starts the kernel, and configures logging.
        """        
        super().__init__(working_dir)
        self.extra_env_vars = extra_env_vars

        ipkernel_logger = logging.getLogger('IPKernelApp')

        # Create a filter using a lambda function
        warning_filter = lambda record: not any(msg in record.getMessage() for msg in [
            "Parent appears to have exited, shutting down.",
            "Could not destroy zmq context"
        ])
        # Add the filter to the logger
        ipkernel_logger.addFilter(warning_filter)

        # Get the path to the current Python executable
        python_executable = sys.executable
        
        # Ensure only one KernelManager instance is configured and started
        self.km = KernelManager(kernel_name='python3', kernel_cmd=[python_executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'])
        env = os.environ.copy()
        env.update(extra_env_vars)
        self.km.start_kernel(env=env)
        # self.km.start_kernel()
        self.kc = self.km.client()
        self.kc.start_channels()
        while not self.kc.is_alive():
            time.sleep(0.1)
        time.sleep(0.5)
        '''
        ipkernel_logger = logging.getLogger('IPKernelApp')
        # Create a filter using a lambda function
        warning_filter = lambda record: not any(msg in record.getMessage() for msg in [
            "Parent appears to have exited, shutting down.",
            "Could not destroy zmq context"
        ])
        # Add the filter to the logger
        ipkernel_logger.addFilter(warning_filter)

        # Get the path to the current Python executable
        python_executable = sys.executable
        
        # Create a KernelManager instance using the current Python executable
        self.km = KernelManager(kernel_name='python3', kernel_cmd=[python_executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'])
        # self.km.start_kernel()
        # self.kc = self.km.client()
        # self.kc.start_channels()
            
        # self.km = KernelManager(kernel_name="python3")
        self.km.start_kernel()
        self.kc = self.km.client()
        self.kc.start_channels()
        while not self.kc.is_alive():
            time.sleep(0.1)
        time.sleep(0.5)
        '''
        self.listener_thread = None
        self.finish_flag = False

        # DISABLED because sometimes this bypasses sending it up to us for some reason!
        # Give it our same matplotlib backend
        # backend = matplotlib.get_backend()

#         # Use Agg, which bubbles everything up as an image.
#         # Not perfect (I want interactive!) but it works.
#         backend = "Agg"

#         code = f"""
# import matplotlib
# matplotlib.use('{backend}')
#         """.strip()
#         for _ in self.run(code):
#             pass

        # DISABLED because it doesn't work??
        # Disable color outputs in the terminal, which don't look good in OI and aren't useful
        # code = """
        # from IPython.core.getipython import get_ipython
        # get_ipython().colors = 'NoColor'
        # """
        # self.run(code)

    def terminate(self):
        """
        Terminates the IPython kernel and stops its channels.
        """
        self.kc.stop_channels()
        self.km.shutdown_kernel()

    def step(self, code):
        """
        Executes a step of Python code.

        Args:
            code (str): The Python code to execute.

        Yields:
            dict: Output messages generated during execution.
        """        
        # 解析python代码并且将函数体抽取出来存成字典，key是函数名，value是函数体，如果要存的话，就将每个函数存成一个py文件
        # try:
        #     functions = string_to_python(code)  # 
        # except:
        #     # Non blocking
        #     functions = {}

        # if self.computer.save_skills and functions:
        #     skill_library_path = self.computer.skills.path

        #     if not os.path.exists(skill_library_path):
        #         os.makedirs(skill_library_path)

        #     for filename, function_code in functions.items():
        #         with open(f"{skill_library_path}/{filename}.py", "w") as file:
        #             file.write(function_code)

        self.finish_flag = False
        try:
            # try:
            #     preprocessed_code = self.preprocess_code(code)
            # except:
            #     # Any errors produced here are our fault.
            #     # Also, for python, you don't need them! It's just for active_line and stuff. Just looks pretty.
            #     preprocessed_code = code
            preprocessed_code = code
            message_queue = queue.Queue()
            self._execute_code(preprocessed_code, message_queue)
            yield from self._capture_output(message_queue)
        except GeneratorExit:
            raise  # gotta pass this up!
        except:
            content = traceback.format_exc()
            yield {"type": "console", "format": "output", "content": content}

    def _execute_code(self, code, message_queue):
        """
        Executes Python code using the IPython kernel and captures the output messages.

        Args:
            code (str): The Python code to execute.
            message_queue (queue.Queue): The message queue for storing output messages.
        """        
        def iopub_message_listener():
            '''
            The main function of this function is to monitor the messages on the IOPub message channel of the IPython kernel and 
            process them accordingly according to the type of the message. The IOPub message channel is a channel in the Jupyter/IPython 
            system used to broadcast execution results, logs, errors, status updates and other information.            
            '''
            while True:
                # If self.finish_flag = True, and we didn't set it (we do below), we need to stop. That's our "stop"
                if self.finish_flag == True:
                    self.km.interrupt_kernel()
                    return
                try:
                    msg = self.kc.iopub_channel.get_msg(timeout=0.05)
                except queue.Empty:
                    continue

                if (
                    msg["header"]["msg_type"] == "status"
                    and msg["content"]["execution_state"] == "idle"
                ):
                    # Set finish_flag and return when the kernel becomes idle

                    self.finish_flag = True
                    return

                content = msg["content"]

                if msg["msg_type"] == "stream":
                    line, active_line = self.detect_active_line(content["text"])
                    if active_line:
                        message_queue.put(
                            {
                                "type": "console",
                                "format": "active_line",
                                "content": active_line,
                            }
                        )
                    message_queue.put(
                        {"type": "console", "format": "output", "content": line}
                    )
                elif msg["msg_type"] == "error":
                    content = "\n".join(content["traceback"])
                    # Remove color codes
                    ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
                    content = ansi_escape.sub("", content)
                    message_queue.put(
                        {
                            "type": "console",
                            "format": "output",
                            "content": content,
                            "error": True,
                        }
                    )
                elif msg["msg_type"] in ["display_data", "execute_result"]:
                    data = content["data"]
                    if "image/png" in data:
                        message_queue.put(
                            {
                                "type": "image",
                                "format": "base64.png",
                                "content": data["image/png"],
                            }
                        )
                    elif "image/jpeg" in data:
                        message_queue.put(
                            {
                                "type": "image",
                                "format": "base64.jpeg",
                                "content": data["image/jpeg"],
                            }
                        )
                    elif "text/html" in data:
                        message_queue.put(
                            {
                                "type": "code",
                                "format": "html",
                                "content": data["text/html"],
                            }
                        )
                    elif "text/plain" in data:
                        message_queue.put(
                            {
                                "type": "console",
                                "format": "output",
                                "content": data["text/plain"],
                            }
                        )
                    elif "application/javascript" in data:
                        message_queue.put(
                            {
                                "type": "code",
                                "format": "javascript",
                                "content": data["application/javascript"],
                            }
                        )

        self.listener_thread = threading.Thread(target=iopub_message_listener)
        # self.listener_thread.daemon = True
        self.listener_thread.start()

        self.kc.execute(code)

    def detect_active_line(self, line):
        """
        Detects active line markers in the output line.

        Args:
            line (str): The output line from the IPython kernel.

        Returns:
            tuple: The modified line and active line number, if detected.
        """        
        if "##active_line" in line:
            # Split the line by "##active_line" and grab the last element
            last_active_line = line.split("##active_line")[-1]
            # Split the last active line by "##" and grab the first element
            active_line = int(last_active_line.split("##")[0])
            # Remove all ##active_line{number}##\n
            line = re.sub(r"##active_line\d+##\n", "", line)
            return line, active_line
        return line, None

    def _capture_output(self, message_queue):
        """
        Captures output messages from the message queue.

        Args:
            message_queue (queue.Queue): The message queue.

        Yields:
            dict: Output messages.
        """        
        while True:
            if self.listener_thread:
                try:
                    output = message_queue.get(timeout=0.1)
                    yield output
                except queue.Empty:
                    if self.finish_flag:
                        break
            time.sleep(0.1)

    def stop(self):
        """
        Stops the execution of code by setting the finish flag.
        """        
        self.finish_flag = True

    def preprocess_code(self, code):
        """
        Preprocesses the Python code before execution.

        Args:
            code (str): The Python code to preprocess.

        Returns:
            str: The preprocessed code.
        """
        code = code.strip()

        # Add print commands that tell us what the active line is
        # but don't do this if any line starts with ! or %
        if not any(line.strip().startswith(("!", "%")) for line in code.split("\n")):
            code = add_active_line_prints(code)

        # Wrap in a try except (DISABLED)
        # code = wrap_in_try_except(code)

        # Remove any whitespace lines, as this will break indented blocks
        # (are we sure about this? test this)
        code_lines = code.split("\n")
        code_lines = [c for c in code_lines if c.strip() != ""]
        code = "\n".join(code_lines)

        return code
    

def add_active_line_prints(code):
    """
    Adds print statements indicating line numbers to a Python string.

    Args:
        code (str): The Python code.

    Returns:
        str: The code with added print statements.
    """
    # Replace newlines and comments with pass statements, so the line numbers are accurate (ast will remove them otherwise)
    code_lines = code.split("\n")
    in_multiline_string = False
    for i in range(len(code_lines)):
        line = code_lines[i]
        if '"""' in line or "'''" in line:
            in_multiline_string = not in_multiline_string
        if not in_multiline_string and (line.strip().startswith("#") or line == ""):
            whitespace = len(line) - len(line.lstrip(" "))
            code_lines[i] = " " * whitespace + "pass"
    processed_code = "\n".join(code_lines)
    try:
        tree = ast.parse(processed_code)
    except:
        # If you can't parse the processed version, try the unprocessed version before giving up
        tree = ast.parse(code)
    transformer = AddLinePrints()
    new_tree = transformer.visit(tree)
    return ast.unparse(new_tree)


class AddLinePrints(ast.NodeTransformer):
    """
    Transformer to insert print statements indicating the line number
    before every executable line in the AST.
    """

    def insert_print_statement(self, line_number):
        """
        Inserts a print statement for a given line number.

        Args:
            line_number (int): The line number.

        Returns:
            ast.Expr: The print statement AST node.
        """
        return ast.Expr(
            value=ast.Call(
                func=ast.Name(id="print", ctx=ast.Load()),
                args=[ast.Constant(value=f"##active_line{line_number}##")],
                keywords=[],
            )
        )

    def process_body(self, body):
        """
        Processes a block of statements, adding print calls.

        Args:
            body (list): List of AST nodes representing statements.

        Returns:
            list: List of modified AST nodes.
        """
        new_body = []

        # In case it's not iterable:
        if not isinstance(body, list):
            body = [body]

        for sub_node in body:
            if hasattr(sub_node, "lineno"):
                new_body.append(self.insert_print_statement(sub_node.lineno))
            new_body.append(sub_node)

        return new_body

    def visit(self, node):
        """
        Visits and transforms nodes in the AST.

        Args:
            node: The current AST node.

        Returns:
            ast.Node: The modified AST node.
        """
        new_node = super().visit(node)

        # If node has a body, process it
        if hasattr(new_node, "body"):
            new_node.body = self.process_body(new_node.body)

        # If node has an orelse block (like in for, while, if), process it
        if hasattr(new_node, "orelse") and new_node.orelse:
            new_node.orelse = self.process_body(new_node.orelse)

        # Special case for Try nodes as they have multiple blocks
        if isinstance(new_node, ast.Try):
            for handler in new_node.handlers:
                handler.body = self.process_body(handler.body)
            if new_node.finalbody:
                new_node.finalbody = self.process_body(new_node.finalbody)

        return new_node


def wrap_in_try_except(code):
    """
    Wraps Python code in a try-except block to catch exceptions.

    Args:
        code (str): The Python code.

    Returns:
        str: The code wrapped in a try-except block.
    """
    code = "import traceback\n" + code

    # Parse the input code into an AST
    parsed_code = ast.parse(code)

    # Wrap the entire code's AST in a single try-except block
    try_except = ast.Try(
        body=parsed_code.body,
        handlers=[
            ast.ExceptHandler(
                type=ast.Name(id="Exception", ctx=ast.Load()),
                name=None,
                body=[
                    ast.Expr(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="traceback", ctx=ast.Load()),
                                attr="print_exc",
                                ctx=ast.Load(),
                            ),
                            args=[],
                            keywords=[],
                        )
                    ),
                ],
            )
        ],
        orelse=[],
        finalbody=[],
    )

    # Assign the try-except block as the new body
    parsed_code.body = [try_except]

    # Convert the modified AST back to source code
    return ast.unparse(parsed_code)


def string_to_python(code_as_string):
    """
    Parses Python code from a string and extracts function definitions.

    Args:
        code_as_string (str): The Python code as a string.

    Returns:
        dict: A dictionary mapping function names to their code.
    """    
    parsed_code = ast.parse(code_as_string)

    # Initialize containers for different categories
    import_statements = []
    functions = []
    functions_dict = {}

    # Traverse the AST
    for node in ast.walk(parsed_code):
        # Check for import statements
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            for alias in node.names:
                # Handling the alias in import statements
                if alias.asname:
                    import_statements.append(f"import {alias.name} as {alias.asname}")
                else:
                    import_statements.append(f"import {alias.name}")
        # Check for function definitions
        elif isinstance(node, ast.FunctionDef):
            if node.name.startswith("_"):
                # ignore private functions
                continue
            docstring = ast.get_docstring(node)
            body = node.body
            if docstring:
                body = body[1:]

            code_body = ast.unparse(body[0]).replace("\n", "\n    ")

            func_info = {
                "name": node.name,
                "docstring": docstring,
                "body": code_body,
            }
            functions.append(func_info)

    for func in functions:
        # Consolidating import statements and function definition
        function_content = "\n".join(import_statements) + "\n\n"
        function_content += f"def {func['name']}():\n    \"\"\"{func['docstring']}\"\"\"\n    {func['body']}\n"

        # Adding to dictionary
        functions_dict[func["name"]] = function_content

    return functions_dict


class Env(BaseEnv):
    """
    A class representing an environment for executing code in various languages.

    This class manages the execution of code in different languages and provides methods for interacting with
    those languages.

    It inherits from BaseEnv, which provides basic environment functionality.
    """    
    def __init__(self, working_dir: str = "working_dir", extra_env_vars: dict = None):
        """
        Initializes the environment.

        Sets up the supported languages and initializes the active languages dictionary.
        """        
        super().__init__(working_dir)
        # self.languages = [
        #     PythonJupyterEnv,
        # ]
        self.python_jupyter_env = PythonJupyterEnv(self.working_dir, extra_env_vars)
        self._active_languages = {}

    def get_language(self, language):
        """
        Gets the language class based on the provided language name or alias.

        Args:
            language (str): The name or alias of the language.

        Returns:
            class: The language class corresponding to the provided name or alias, or None if not found.
        """        
        # 输入planner的节点类型即可
        for lang in self.languages:
            if language.lower() == lang.name.lower() or (
                hasattr(lang, "aliases") and language.lower() in (alias.lower() for alias in lang.aliases)
            ):
                return lang
        return None

    def step(self, code, language="Python", stream=False, display=False):
        """
        Executes a step of code in the specified language.

        Args:
            language (str): The name or alias of the language to execute the code in.
            code (str): The code to execute.
            stream (bool): Whether to stream the output as it becomes available.
            display (bool): Whether to display the output.

        Returns:
            EnvState: The state after executing the code.
        """        
        # 不用流式的话很简单，就是调一下lang的step就行了
        state = EnvState(command=code)
        # lang = self.get_language(language)(self.working_dir)  # 输入planner的节点类型即可
        lang = self.python_jupyter_env
        for output_line_dic in lang.step(code):
            if output_line_dic['format'] == 'active_line' or output_line_dic['content'] in ['', '\n']:
                continue
            content = output_line_dic['content']
            # if 'Traceback' in content:
            # DN: More reliable condition to check for error
            if "error" in output_line_dic:
                state.error = (state.error or '') + content
            else:
                state.result += content
        # if lang.name == 'Python':
        #     lang.terminate()
        # for output_line_dic in lang.step(code):
        #     if output_line_dic['format'] == 'active_line':
        #         continue
        #     content = output_line_dic['content']
        #     if content != '' and content != '\n':
        #         if 'Traceback' in content:
        #             state.error = (state.error or '') + content
        #         else:
        #             state.result += content
        state.pwd = self.working_dir
        state.ls = subprocess.run(['ls'], cwd=self.working_dir, capture_output=True, text=True).stdout
        return state
        
        # if (
        #     language == "python"
        #     and self.computer.import_computer_api
        #     and "computer" in code
        # ):
        #     if not self.computer._has_imported_computer_api:
        #         self.computer._has_imported_computer_api = True
        #         # Give it access to the computer via Python
        #         self.computer.run(
        #             language="python",
        #             code="import time\nfrom interpreter import interpreter\ncomputer = interpreter.computer",  # We ask it to use time, so
        #             display=self.computer.verbose,
        #         )

        if stream == False:
            # If stream == False, *pull* from _streaming_run.
            output_messages = []
            for chunk in self._streaming_run(language, code, display=display):
                if chunk.get("format") != "active_line":
                    # Should we append this to the last message, or make a new one?
                    if (
                        output_messages != []
                        and output_messages[-1].get("type") == chunk["type"]
                        and output_messages[-1].get("format") == chunk["format"]
                    ):
                        output_messages[-1]["content"] += chunk["content"]
                    else:
                        output_messages.append(chunk)
            return output_messages

        elif stream == True:
            # If stream == True, replace this with _streaming_run.
            return self._streaming_run(language, code, display=display)

    def _streaming_run(self, language, code, display=False):
        """
        Executes code in the specified language and streams the output.

        Args:
            language (str): The name or alias of the language to execute the code in.
            code (str): The code to execute.
            display (bool): Whether to display the output.

        Yields:
            dict: Output chunks generated during execution.
        """        
        if language not in self._active_languages:
            # Get the language. Pass in self.computer *if it takes a single argument*
            # but pass in nothing if not. This makes custom languages easier to add / understand.
            lang_class = self.get_language(language)
            if lang_class.__init__.__code__.co_argcount > 1:
                self._active_languages[language] = lang_class(self.computer)
            else:
                self._active_languages[language] = lang_class()
        try:
            for chunk in self._active_languages[language].run(code):
                # self.format_to_recipient can format some messages as having a certain recipient.
                # Here we add that to the LMC messages:
                if chunk["type"] == "console" and chunk.get("format") == "output":
                    recipient, content = parse_for_recipient(chunk["content"])
                    if recipient:
                        chunk["recipient"] = recipient
                        chunk["content"] = content

                    # Sometimes, we want to hide the traceback to preserve tokens.
                    # (is this a good idea?)
                    if "@@@HIDE_TRACEBACK@@@" in content:
                        chunk["content"] = (
                            "Stopping execution.\n\n"
                            + content.split("@@@HIDE_TRACEBACK@@@")[-1].strip()
                        )

                yield chunk

                # Print it also if display = True
                if (
                    display
                    and chunk.get("format") != "active_line"
                    and chunk.get("content")
                ):
                    print(chunk["content"])

        except GeneratorExit:
            self.stop()

    def stop(self):
        """
        Stops the execution of all active languages.
        """        
        for language in self._active_languages.values():
            language.stop()

    def terminate(self):
        """
        Terminates all active language environments.
        """        
        for language_name in list(self._active_languages.keys()):
            language = self._active_languages[language_name]
            if (
                language
            ):  # Not sure why this is None sometimes. We should look into this
                language.terminate()
            del self._active_languages[language_name]
