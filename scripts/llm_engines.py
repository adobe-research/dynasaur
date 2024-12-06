import os
from time import sleep

import openai
from anthropic import Anthropic, AnthropicBedrock
from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel
from transformers.agents.llm_engine import MessageRole, get_clean_message_list

openai_role_conversions = {MessageRole.TOOL_RESPONSE: MessageRole.USER}


class OpenAIEngine:
    def __init__(self, model_name="gpt-4o"):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    def __call__(self, messages, stop_sequences=[]):
        messages = get_clean_message_list(messages, role_conversions=openai_role_conversions)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stop=stop_sequences,
            temperature=0.5,
        )
        return response.choices[0].message.content


class AnthropicEngine:
    def __init__(self, model_name="claude-3-5-sonnet-20240620", use_bedrock=False):
        self.model_name = model_name
        if use_bedrock:  # Cf this page: https://docs.anthropic.com/en/api/claude-on-amazon-bedrock
            self.model_name = "anthropic.claude-3-5-sonnet-20240620-v1:0"
            self.client = AnthropicBedrock(
                aws_access_key=os.getenv("AWS_BEDROCK_ID"),
                aws_secret_key=os.getenv("AWS_BEDROCK_KEY"),
                aws_region="us-east-1",
            )
        else:
            self.client = Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            )

    def __call__(self, messages, stop_sequences=[]):
        messages = get_clean_message_list(messages, role_conversions=openai_role_conversions)
        index_system_message, system_prompt = None, None
        for index, message in enumerate(messages):
            if message["role"] == MessageRole.SYSTEM:
                index_system_message = index
                system_prompt = message["content"]
        if system_prompt is None:
            raise Exception("No system prompt found!")

        filtered_messages = [message for i, message in enumerate(messages) if i != index_system_message ]
        if len(filtered_messages) == 0:
            print("Error, no user message:", messages)
            assert False

        response = self.client.messages.create(
            model=self.model_name,
            system=system_prompt,
            messages=filtered_messages,
            stop_sequences=stop_sequences,
            temperature=0.5,
            max_tokens=2000,
        )
        full_response_text = ""
        for content_block in response.content:
            if content_block.type == "text":
                full_response_text += content_block.text
        return full_response_text


class AzureOpenAIEngine:
    def __init__(
        self,
        model_name: str = None,
        api_key: str = None,
        azure_endpoint: str = None,
        api_version: str = None,
    ):
        self.model_name = model_name or os.getenv("AZURE_MODEL_NAME")
        self.client = AzureOpenAI(
            api_key=api_key or os.getenv("AZURE_API_KEY"),
            azure_endpoint=azure_endpoint or os.getenv("AZURE_ENDPOINT"),
            api_version=api_version or os.getenv("AZURE_API_VERSION"),
        )
        self.metrics = {
            "num_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    def __call__(self, messages, stop_sequences=[], temperature=0.5, *args, **kwargs):
        messages = get_clean_message_list(messages, role_conversions=openai_role_conversions)

        success = False
        wait_time = 1
        while not success:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    # stop=stop_sequences,
                    temperature=temperature,
                    *args,
                    **kwargs
                )
                success = True
            except openai.InternalServerError:
                sleep(wait_time)
                wait_time += 1

        # Update metrics
        self.metrics["num_calls"] += 1
        self.metrics["prompt_tokens"] += response.usage.prompt_tokens
        self.metrics["completion_tokens"] += response.usage.completion_tokens

        return response.choices[0].message.content

    def reset(self):
        self.metrics = {
            "num_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }


class ThoughtCodeFormat(BaseModel):
    thought: str
    code: str


class ThoughtActionFormat(BaseModel):
    thought: str
    action: str


class StructuredOutputAzureOpenAIEngine(AzureOpenAIEngine):
    def __init__(self, response_format: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_format_str = response_format
        if response_format == "thought_code":
            self.response_format = ThoughtCodeFormat
        elif response_format == "thought_action":
            self.response_format = ThoughtActionFormat

    def __call__(self, messages, temperature=0.5, stop_sequences=None, *args, **kwargs) -> dict:
        messages = get_clean_message_list(messages, role_conversions=openai_role_conversions)

        response = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=messages,
            response_format=self.response_format,
            temperature=temperature,
            *args,
            **kwargs
        )

        # Update metrics
        self.metrics["num_calls"] += 1
        self.metrics["prompt_tokens"] += response.usage.prompt_tokens
        self.metrics["completion_tokens"] += response.usage.completion_tokens

        return response.choices[0].message.parsed
