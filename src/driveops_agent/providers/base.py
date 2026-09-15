import os
from typing import Protocol


class Provider(Protocol):
    def complete(self, prompt: str) -> str: ...


class MockProvider:
    def complete(self, prompt: str) -> str:
        return "offline mock response"


class OpenAICompatibleProvider:
    def __init__(self) -> None:
        self.api_key = os.getenv("API_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model = os.getenv("MODEL")
        if not all((self.api_key, self.base_url, self.model)):
            raise RuntimeError("API_KEY, BASE_URL, and MODEL are required")

    def complete(self, prompt: str) -> str:
        raise NotImplementedError("Network provider is intentionally not used by offline V1")
