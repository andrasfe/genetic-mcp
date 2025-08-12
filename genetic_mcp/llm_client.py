"""LLM client for interacting with various language models."""

import logging
import os
from abc import ABC, abstractmethod

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def generate(self, prompt: str, system_prompt: str | None = None,
                      temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embeddings for text."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client."""

    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

    async def generate(self, prompt: str, system_prompt: str | None = None,
                      temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """Generate a response using OpenAI API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise

    async def embed(self, text: str) -> list[float]:
        """Generate embeddings using the configured embedding provider."""
        from .embedding_providers import embed_text
        try:
            return await embed_text(text)
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise


class AnthropicClient(LLMClient):
    """Anthropic API client."""

    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        # Note: Anthropic doesn't have native embeddings, we'll use a fallback
        self._openai_client = None
        self._openai_api_key = None

    def set_embedding_fallback(self, openai_api_key: str) -> None:
        """Set OpenAI as fallback for embeddings."""
        self._openai_api_key = openai_api_key
        self._openai_client = AsyncOpenAI(api_key=openai_api_key)

    async def generate(self, prompt: str, system_prompt: str | None = None,
                      temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """Generate a response using Anthropic API."""
        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.client.messages.create(
                model=self.model,
                messages=messages,
                system=system_prompt or "",
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            raise

    async def embed(self, text: str) -> list[float]:
        """Generate embeddings using the configured embedding provider."""
        from .embedding_providers import embed_text
        try:
            return await embed_text(text)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise


class OpenRouterClient(LLMClient):
    """OpenRouter API client (OpenAI-compatible)."""

    def __init__(self, api_key: str, model: str = "meta-llama/llama-3.2-3b-instruct"):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.model = model
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")  # Will use OpenAI fallback
        self._openai_client = None
        self._openai_api_key = None

    def set_embedding_fallback(self, openai_api_key: str) -> None:
        """Set OpenAI as fallback for embeddings."""
        self._openai_api_key = openai_api_key
        self._openai_client = AsyncOpenAI(api_key=openai_api_key)

    async def generate(self, prompt: str, system_prompt: str | None = None,
                      temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """Generate a response using OpenRouter API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"OpenRouter generation error: {e}")
            raise

    async def embed(self, text: str) -> list[float]:
        """Generate embeddings using the configured embedding provider."""
        from .embedding_providers import embed_text
        try:
            return await embed_text(text)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise


class LLMClientFactory:
    """Factory for creating LLM clients."""

    @staticmethod
    def create(provider: str, api_key: str, model: str | None = None) -> LLMClient:
        """Create an LLM client based on provider."""
        provider = provider.lower()

        if provider == "openai":
            return OpenAIClient(api_key, model or "gpt-4-turbo-preview")
        elif provider == "anthropic":
            return AnthropicClient(api_key, model or "claude-3-opus-20240229")
        elif provider == "openrouter":
            return OpenRouterClient(api_key, model or "meta-llama/llama-3.2-3b-instruct")
        else:
            raise ValueError(f"Unsupported provider: {provider}")


class MultiModelClient:
    """Client that can use multiple LLM providers."""

    def __init__(self):
        self.clients: dict[str, LLMClient] = {}
        self.default_client: str | None = None

    def add_client(self, name: str, client: LLMClient, is_default: bool = False) -> None:
        """Add an LLM client."""
        self.clients[name] = client
        if is_default or not self.default_client:
            self.default_client = name

    async def generate(self, prompt: str, client_name: str | None = None,
                      system_prompt: str | None = None, **kwargs) -> str:
        """Generate using specified or default client."""
        client_name = client_name or self.default_client
        if not client_name or client_name not in self.clients:
            raise ValueError(f"Client '{client_name}' not found")

        return await self.clients[client_name].generate(prompt, system_prompt, **kwargs)

    async def embed(self, text: str, client_name: str | None = None) -> list[float]:
        """Generate embeddings using specified or default client."""
        client_name = client_name or self.default_client
        if not client_name or client_name not in self.clients:
            raise ValueError(f"Client '{client_name}' not found")

        return await self.clients[client_name].embed(text)

    def get_available_models(self) -> list[str]:
        """Get list of available model names."""
        return list(self.clients.keys())
