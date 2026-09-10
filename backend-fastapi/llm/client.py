import os
import ollama
from dotenv import load_dotenv

load_dotenv()


class UC3MClient:
    """Chat client for the UC3M Ollama gateway (default/fast model)."""

    api_key_env = "OLLAMA_API_KEY"
    url_env = "OLLAMA_URL"
    model_env = "OLLAMA_MODEL"
    default_model = "qwen3:8b"

    def __init__(self):
        self.api_key = os.getenv(self.api_key_env)
        self.base_url = os.getenv(self.url_env, "https://yiyuan.tsc.uc3m.es")
        self.default_model = os.getenv(self.model_env, self.default_model)

        if not self.api_key:
            raise ValueError(f"{self.api_key_env} not found in .env")

        self.client = ollama.Client(
            host=self.base_url,
            headers={"X-API-KEY": self.api_key},
        )

    def stream_chat(
        self,
        user_prompt: str,
        system_prompt: str = "You are a helpful academic research assistant.",
        model: str | None = None,
        temperature: float = 0.0,
    ):
        response = self.client.chat(
            model=model or self.default_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": temperature},
            stream=True,
        )
        for chunk in response:
            yield chunk["message"]["content"]

    def chat(
        self,
        user_prompt: str,
        system_prompt: str = "You are a helpful academic research assistant.",
        model: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        response = self.client.chat(
            model=model or self.default_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": temperature},
        )
        return response["message"]["content"]


class UC3MClient_large(UC3MClient):
    """Same client, pointed at the larger model via the *2 env vars."""

    api_key_env = "OLLAMA_API_KEY2"
    url_env = "OLLAMA_URL2"
    model_env = "OLLAMA_MODEL2"
    default_model = "qwen3:32b"
