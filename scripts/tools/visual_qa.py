import base64
import mimetypes
import os
import uuid
from typing import Optional

import requests
from transformers import AutoProcessor, Tool


# transformers's argument validation is annoying so we're just going to disable it
class Tool(Tool):
    def validate_arguments(self, *args, **kwargs):
        pass


idefics_processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")


# Function to encode the image
def encode_image(image_path):
    if image_path.startswith("http"):
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
        request_kwargs = {
            "headers": {"User-Agent": user_agent},
            "stream": True,
        }

        # Send a HTTP request to the URL
        response = requests.get(image_path, **request_kwargs)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")

        extension = mimetypes.guess_extension(content_type)
        if extension is None:
            extension = ".download"

        fname = str(uuid.uuid4()) + extension
        download_path = os.path.abspath(os.path.join("downloads", fname))

        with open(download_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=512):
                fh.write(chunk)

        image_path = download_path

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class VisualQAGPT4Tool(Tool):
    name = "visualizer"
    description = """A tool that can answer questions about attached images.
    Input descriptions:
        - image_path (str): The path to the image on which to answer the question. This should be a local path to downloaded image.
        - question (Optional[str]): The question to answer."""
    inputs = "image_path: str, question: Optional[str]"
    output_type = "str"

    def __init__(self, model_name, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_name = model_name
        self.azure_model_name = model_name
        self.azure_api_key = os.getenv("AZURE_GPT4V_API_KEY")
        self.azure_endpoint = os.getenv("AZURE_GPT4V_ENDPOINT")
        self.azure_api_version = os.getenv("AZURE_GPT4V_API_VERSION")
        self.headers = {
            "Content-Type": "application/json",
            "api-key": self.azure_api_key,
        }

        self.metrics = {
            "num_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    def reset(self):
        self.metrics = {
            "num_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    def forward(self, image_path: str, question: Optional[str] = None) -> str:
        add_note = False
        if not question:
            add_note = True
            question = "Please write a detailed caption for this image."
        if not isinstance(image_path, str):
            raise Exception(
                "You should provide only one string as argument to this tool!"
            )

        base64_image = encode_image(image_path)

        payload = {
            "model": self.azure_model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 500,
        }
        request = f"{self.azure_endpoint}/openai/deployments/{self.azure_model_name}/chat/completions?api-version={self.azure_api_version}"

        response = requests.post(request, headers=self.headers, json=payload)
        try:
            output = response.json()["choices"][0]["message"]["content"]

            # Update metrics
            self.metrics["num_calls"] += 1
            self.metrics["prompt_tokens"] += response.json()["usage"]["prompt_tokens"]
            self.metrics["completion_tokens"] += response.json()["usage"][
                "completion_tokens"
            ]
        except Exception:
            raise Exception(f"Response format unexpected: {response.json()}")

        if add_note:
            output = f"You did not provide a particular question, so here is a detailed caption for the image: {output}"

        return output
