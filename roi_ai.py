import json
from typing import Any

import requests
from llama_index.core.base.llms.types import LLMMetadata, CompletionResponse
from llama_index.core.llms import CustomLLM
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.legacy.core.llms.types import CompletionResponseGen

from config import LLM_SERVER_API, LLAMA_GENERATE_STREAM


class RioAI(CustomLLM):
    context_window: int = 4096
    num_output: int = 2048
    model_name: str = "llama2"
    api_url: str = f'{LLM_SERVER_API}/{LLAMA_GENERATE_STREAM}'

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        data = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 2048,
                "temperature": 0.2,  # Just an example
                "repetition_penalty": 1.2,  # Just an example
                "details": False
            }
        }

        headers = {
            "Content-Type": "application/json"
        }
        try:
            response = requests.post(self.api_url, headers=headers, json=data, verify=False)  # Verify SSL certificate
            response_text = response.json().get("generated_text")
            return CompletionResponse(text=response_text) if response_text else CompletionResponse(
                text="API call failed")
        except requests.exceptions.RequestException as e:
            return CompletionResponse(text=f"API call failed: {e}")
        except (json.JSONDecodeError, ValueError) as e:
            return CompletionResponse(text=f"Invalid API response: {e}")

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        data = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 2048,
                "temperature": 0.2,  # Just an example
                "repetition_penalty": 1.2,  # Just an example
                "details": False
            }
        }

        headers = {
            "Content-Type": "application/json"
        }
        response = requests.post(self.api_url, headers=headers, json=data, verify=False, stream=True)  # Send POST request with streaming enabled
        response_text = ""
        for line in response.iter_lines():
            _, found, data = line.partition(b"data:")
            if found:
                try:
                    message = json.loads(data)
                    # print(f"message: {message}")
                    token = message['token']['text']
                    response_text += token
                except json.JSONDecodeError:
                    print(f"response line could not be json decoded: {line}")
                except KeyError:
                    print(f"KeyError, unexpected response format in line: {line}")
            else:
                continue

            yield CompletionResponse(text=response_text, delta=token)
