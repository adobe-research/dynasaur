import os
from functools import wraps
from glob import glob

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from transformers.agents import Toolbox
from transformers.agents.default_tools import Tool

from prompts import ACTION_DESCRIPTION_TEMPLATE
from scripts.tools.mdconvert import MarkdownConverter
from scripts.tools.visual_qa import VisualQAGPT4Tool
from scripts.tools.web_surfer import (
    ArchiveSearchTool,
    DownloadTool,
    FinderTool,
    FindNextTool,
    NavigationalSearchTool,
    PageDownTool,
    PageUpTool,
    SearchInformationTool,
    VisitTool,
)
from utils import parse_generated_tools

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORGANIZATION = os.getenv("OPENAI_ORGANIZATION")

EMBED_MODEL_TYPE = os.getenv("EMBED_MODEL_TYPE")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME")

AZURE_EMBED_MODEL_NAME = os.getenv("AZURE_EMBED_MODEL_NAME")
AZURE_EMBED_API_KEY = os.getenv("AZURE_EMBED_API_KEY")
AZURE_EMBED_ENDPOINT = os.getenv("AZURE_EMBED_ENDPOINT")
AZURE_EMBED_API_VERSION = os.getenv("AZURE_EMBED_API_VERSION")


# Define a decorator to track how many time each generated action is called
_num_calls = {}


def parameterized_track_num_calls(given_name=None):
    def track_num_calls(func):
        func_name = given_name or func.__name__

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if func_name not in _num_calls:
                _num_calls[func_name] = 1
            else:
                _num_calls[func_name] += 1
            return func(*args, **kwargs)

        return wrapped_func

    return track_num_calls


def track_num_calls(func):
    func_name = func.__name__

    @wraps(func)
    def wrapped_func(*args, **kwargs):
        if func_name not in _num_calls:
            _num_calls[func_name] = 1
        else:
            _num_calls[func_name] += 1
        return func(*args, **kwargs)

    return wrapped_func


# transformers's argument validation is annoying so we're just going to disable it
class Tool(Tool):
    def validate_arguments(self, *args, **kwargs):
        pass


class SubmitFinalAnswer(Tool):
    name = "submit_final_answer"
    description = "Submits a final answer to the given problem."
    inputs = "answer: str"
    output_type = "str"

    def forward(self, answer: str) -> str:
        return answer


class ToolRetriever:
    def __init__(self, generated_tool_dir: str):
        self.generated_tool_dir = generated_tool_dir
        self.vectordb_path = f"{self.generated_tool_dir}/vectordb"

        if not os.path.exists(self.vectordb_path):
            os.makedirs(self.vectordb_path)

        # Utilize the Chroma database and employ OpenAI Embeddings for vectorization (default: text-embedding-ada-002)
        if EMBED_MODEL_TYPE == "OpenAI":
            embedding_function = OpenAIEmbeddings(
                openai_api_key=OPENAI_API_KEY,
                openai_organization=OPENAI_ORGANIZATION,
            )
            embed_model_name = "openai"
        elif EMBED_MODEL_TYPE == "OLLAMA":
            embedding_function = OllamaEmbeddings(model=EMBED_MODEL_NAME)
            embed_model_name = "ollama"
        elif EMBED_MODEL_TYPE == "AzureOpenAI":
            embedding_function = AzureOpenAIEmbeddings(
                api_key=AZURE_EMBED_API_KEY,
                azure_endpoint=AZURE_EMBED_ENDPOINT,
                azure_deployment=AZURE_EMBED_MODEL_NAME,
                openai_api_version=AZURE_EMBED_API_VERSION,
            )
            embed_model_name = AZURE_EMBED_MODEL_NAME

        self.vectordb = Chroma(
            collection_name="tool_vectordb",
            embedding_function=embedding_function,
            persist_directory=self.vectordb_path,
        )

        self.generated_tools = {}
        for path in glob(os.path.join(generated_tool_dir, "*.py")):
            code = open(path).read()
            tools = parse_generated_tools(code)
            for tool in tools:
                self.generated_tools[tool.name] = tool

    def retrieve(self, query: str, k: int = 10) -> list[Tool]:
        k = min(len(self.vectordb), k)
        if k == 0:
            return []
        docs_and_scores = self.vectordb.similarity_search_with_score(query, k=k)
        tools = []
        for doc, _ in docs_and_scores:
            name = doc.metadata["name"]
            tools.append(self.generated_tools[name])
        return tools

    def add_new_tool(self, tool: Tool):
        """
        Adds a new tool to the tool manager, including updating the vector database
        and tool repository with the provided information.

        This method processes the given tool information, which includes the task name,
        code, and description. It prints out the task name and description, checks if
        the tool already exists (rewriting it if so), and updates both the vector
        database and the tool dictionary. Finally, it persists the new tool's code and
        description in the repository and ensures the vector database is synchronized
        with the generated tools.

        Args:
            info (dict): A dictionary containing the tool's information, which must
                         include 'task_name', 'code', and 'description'.

        Raises:
            AssertionError: If the vector database's count does not match the length
                            of the generated_tools dictionary after adding the new tool,
                            indicating a synchronization issue.
        """
        program_name = tool.name
        program_description = tool.description
        program_code = tool.code
        program_inputs = tool.inputs
        program_output_type = tool.output_type
        program_dependencies = tool.dependencies

        res = self.vectordb._collection.get(ids=[program_name])
        if res["ids"]:
            # print(f"\033[33mTool {program_name} already exists!\033[0m")
            raise ValueError(f"\033[33mTool {program_name} already exists!\033[0m")
            self.vectordb._collection.delete(ids=[program_name])

        # Store the new task code in the vector database and the tool dictionary
        self.vectordb.add_texts(
            texts=[program_description],
            ids=[program_name],
            metadatas=[
                {
                    "name": program_name,
                }
            ],
        )
        self.generated_tools[tool.name] = tool

        self.vectordb.persist()

    def add_new_tool_from_path(self, path: str):
        code = open(path, "r").read()
        tools = parse_generated_tools(code)
        for tool in tools:
            self.add_new_tool(tool)


class ToolRetrievalTool(Tool):
    name = "get_relevant_tools"
    description = 'This tool retrieves relevant tools that you generated in previous runs. Write what you want to do in the query. If there are no tools in the toolbox, "No tool found" will be returned.'
    inputs = "query: str"
    output_type = "str"

    def __init__(self, generated_tool_dir: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generated_tool_dir = generated_tool_dir
        self.tool_retriever = ToolRetriever(generated_tool_dir)
        self.tool_description_template = ACTION_DESCRIPTION_TEMPLATE

    def forward(self, query: str) -> str:
        relevant_tools: list[Tool] = self.tool_retriever.retrieve(query)
        if relevant_tools:
            relevant_toolbox = Toolbox(relevant_tools)
            return relevant_toolbox.show_tool_descriptions(
                self.tool_description_template
            )
        else:
            return "No tool found"

    def add_new_tool_from_path(self, path: str):
        return self.tool_retriever.add_new_tool_from_path(path)


class TextInspectorTool(Tool):
    name = "inspect_file_as_text"
    description = """Call this tool to read a file as markdown text. This tool handles the following file extensions: [".html", ".htm", ".xlsx", ".pptx", ".wav", ".mp3", ".flac", ".pdf", ".docx"], and all other types of text files. IT DOES NOT HANDLE IMAGES."
    Input descriptions:
        - file_path (str): The path to the file you want to read as text. Must be a '.something' file, like '.pdf'. If it is an image, use the visualizer tool instead! DO NOT USE THIS TOOL FOR A WEBPAGE: use the search tool instead!"""
    inputs = "file_path: str"
    output_type = "str"
    md_converter = MarkdownConverter()

    def forward(self, file_path: str) -> str:
        if file_path[-4:] in [".png", ".jpg"]:
            raise Exception(
                "Cannot use inspect_file_as_text tool with images: use visualizer instead!"
            )
        result = self.md_converter.convert(file_path)
        return result.text_content


def get_user_defined_actions(model_name) -> dict[str, Tool]:
    # Web browsing actions
    informational_web_search = SearchInformationTool()
    navigational_web_search = NavigationalSearchTool()
    visit_page = VisitTool()
    download_file = DownloadTool()
    page_up = PageUpTool()
    page_down = PageDownTool()
    find_on_page_ctrl_f = FinderTool()
    find_next = FindNextTool()
    find_archived_url = ArchiveSearchTool()

    # VQA and text inspection actions
    visualizer = VisualQAGPT4Tool(model_name)
    inspect_file_as_text = TextInspectorTool()

    task_solving_toolbox = {
        "informational_web_search": informational_web_search,
        "navigational_web_search": navigational_web_search,
        "visit_page": visit_page,
        "download_file": download_file,
        "page_up": page_up,
        "page_down": page_down,
        "find_on_page_ctrl_f": find_on_page_ctrl_f,
        "find_next": find_next,
        "find_archived_url": find_archived_url,
        "visualizer": visualizer,
        "inspect_file_as_text": inspect_file_as_text,
    }

    return task_solving_toolbox


def get_required_actions(generated_action_dir) -> dict[str, Tool]:
    submit_final_answer = SubmitFinalAnswer()
    get_relevant_tools = ToolRetrievalTool(generated_action_dir)

    required_actions = {
        "submit_final_answer": submit_final_answer,
        "get_relevant_tools": get_relevant_tools,
    }
    return required_actions


def load_actions(actions: dict[str, Tool]):
    for func_name in actions:
        # Define a helper function that creates the wrapper with the current func_name
        def make_wrapper(name):
            @parameterized_track_num_calls(name)
            def wrapper(*args, **kwargs):
                return actions[name](*args, **kwargs)

            return wrapper

        # Assign the dynamically created function to a variable in the global scope
        globals()[func_name] = make_wrapper(func_name)


if __name__ == "__main__":
    generated_action_dir = os.getenv("GENERATED_ACTION_DIR")
    required_actions = get_required_actions(generated_action_dir)
    load_actions(required_actions)

    model_name = os.getenv("MODEL_NAME")
    user_defined_actions = get_user_defined_actions(model_name)
    load_actions(user_defined_actions)
