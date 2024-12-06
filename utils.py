import ast
import os
from glob import glob
from typing import Union

from transformers.agents import Tool


class GeneratedTool(Tool):
    def __init__(
        self,
        name: str,
        description: str,
        inputs: str,
        output_type: str,
        code: str,
        dependencies: str,
    ):
        super().__init__()
        self.name = name
        self.description = description if description else "No description provided"
        self.inputs = inputs if inputs else ""
        self.output_type = output_type if output_type else ""
        self.code = code
        self.dependencies = dependencies

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Not a real executable tool")


def extract_description(func_def: ast.FunctionDef) -> Union[str, None]:
    docstring = ast.get_docstring(func_def)
    if docstring:
        return docstring.strip().split("\n")[0]


def extract_inputs(func_def: ast.FunctionDef) -> Union[str, None]:
    inputs = ast.unparse(func_def.args) if func_def.args else None
    return inputs if inputs else None


def extract_output_type(func_def: ast.FunctionDef) -> Union[str, None]:
    output_type = ast.unparse(func_def.returns) if func_def.returns else None
    return output_type if output_type else None


def extract_func_info(func_def: ast.FunctionDef) -> dict[str, str]:
    res = {}
    res["name"] = func_def.name
    res["description"] = extract_description(func_def)
    res["inputs"] = extract_inputs(func_def)
    res["output_type"] = extract_output_type(func_def)
    res["code"] = ast.unparse(func_def)
    return res


def add_parent_pointers(node) -> None:
    for child in ast.iter_child_nodes(node):
        child.parent = node
        add_parent_pointers(child)


def parse_generated_tools(code: str) -> list[GeneratedTool]:
    """Parse LLM-generated code. Save new functions if any."""
    tree = ast.parse(code)

    # Do this so later we can extract only the top-level imports and functions
    add_parent_pointers(tree)

    # Traverse the AST to find import statements and function definitions
    import_statements = []
    func_defs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and isinstance(node.parent, ast.Module):
            for alias in node.names:
                import_statements.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and isinstance(node.parent, ast.Module):
            module = node.module if node.module else ""
            for alias in node.names:
                import_statements.append(f"from {module} import {alias.name}")
        # In case there is func def inside a func, we only extract the top-level ones
        elif isinstance(node, ast.FunctionDef) and isinstance(node.parent, ast.Module):
            func_defs.append(node)
    import_statements = "\n".join(import_statements)

    generated_tools = []
    for func_def in func_defs:
        func_info = extract_func_info(func_def)
        generated_tool = GeneratedTool(**func_info, dependencies=import_statements)
        generated_tools.append(generated_tool)

    return generated_tools


def get_action_set(res_path: str) -> set[str]:
    agent_name = os.path.basename(res_path).split(".")[0]
    generated_tool_dir = os.path.join("../generated_tools", agent_name)

    generated_tools = []
    generated_tool_paths = sorted(glob(os.path.join(generated_tool_dir, "*.py")))
    for path in generated_tool_paths:
        code = open(path, "r").read()
        tools = parse_generated_tools(code)
        generated_tools.append(tools[0].name)
    return set(generated_tools)


class FunctionCallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.calls = []

    def visit_Call(self, node):
        # Get the function name being called
        if isinstance(node.func, ast.Name):
            self.calls.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            self.calls.append(node.func.attr)
        # Visit the arguments of the function
        self.generic_visit(node)


def extract_function_calls(code):
    tree = ast.parse(code)
    visitor = FunctionCallVisitor()
    visitor.visit(tree)
    return visitor.calls


def is_sufficient(code: str, action_set: set[str]) -> Union[bool, None]:
    try:
        tree = ast.parse(code)
    except:
        return None

    function_calls = extract_function_calls(code)
    for func_call in function_calls:
        if func_call not in action_set:
            # If agent use any other functions not in action set -> not sufficient
            return False
    return True


def remove_shell_commands(code_action: str) -> tuple[str, str]:
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


def coverage(action_set: set[str], traj: list, is_correct: bool) -> float:
    nom = 0
    denom = 0
    for step in traj:
        if "tool_call" in step:
            _, code = remove_shell_commands(step["tool_call"]["tool_arguments"])
            sufficient = is_sufficient(code, action_set)
            denom += 1
            if sufficient == True:
                nom += 1
    if denom == 0:
        return 0
    return (1 - (nom / denom)) * int(is_correct)
