import os
import ollama
from dotenv import load_dotenv

load_dotenv()


class UC3MClient:
    def __init__(self):
        self.api_key = os.getenv("OLLAMA_API_KEY")
        self.base_url = os.getenv("OLLAMA_URL", "https://yiyuan.tsc.uc3m.es")
        self.default_model = os.getenv("OLLAMA_MODEL", "qwen3:8b")

        if not self.api_key:
            raise ValueError("OLLAMA_API_KEY not found in .env")

        self.client = ollama.Client(
            host=self.base_url,
            headers={"X-API-KEY": self.api_key},
        )

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