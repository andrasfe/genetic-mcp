"""Embedding providers for genetic-mcp with multiple backend support."""

import logging
import os
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embeddings for text."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of embeddings produced."""
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embeddings provider."""

    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.dimension = 1536 if "ada" in model else 3072  # ada-002: 1536, text-embedding-3-small: 1536, text-embedding-3-large: 3072

    async def embed(self, text: str) -> list[float]:
        """Generate embeddings using OpenAI API."""
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise

    def get_dimension(self) -> int:
        return int(self.dimension)


class SentenceTransformerProvider(EmbeddingProvider):
    """Local Sentence Transformer embeddings provider."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded Sentence Transformer model: {model_name} (dim={self.dimension})")
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed. Install with: pip install sentence-transformers"
            ) from e

    async def embed(self, text: str) -> list[float]:
        """Generate embeddings using local Sentence Transformer."""
        try:
            # Sentence Transformers is synchronous, so we run it directly
            embedding = self.model.encode(text, convert_to_numpy=True)
            return list(embedding.tolist())
        except Exception as e:
            logger.error(f"Sentence Transformer embedding error: {e}")
            raise

    def get_dimension(self) -> int:
        return int(self.dimension)


class CohereEmbeddingProvider(EmbeddingProvider):
    """Cohere embeddings provider."""

    def __init__(self, api_key: str, model: str = "embed-english-v3.0"):
        try:
            import cohere
            self.client = cohere.AsyncClient(api_key)
            self.model = model
            self.dimension = 1024  # Cohere v3 models use 1024 dimensions
        except ImportError as e:
            raise ImportError("cohere not installed. Install with: pip install cohere") from e

    async def embed(self, text: str) -> list[float]:
        """Generate embeddings using Cohere API."""
        try:
            response = await self.client.embed(
                texts=[text],
                model=self.model,
                input_type="search_document"
            )
            return list(response.embeddings[0])
        except Exception as e:
            logger.error(f"Cohere embedding error: {e}")
            raise

    def get_dimension(self) -> int:
        return int(self.dimension)


class VoyageEmbeddingProvider(EmbeddingProvider):
    """Voyage AI embeddings provider."""

    def __init__(self, api_key: str, model: str = "voyage-2"):
        try:
            import voyageai
            self.client = voyageai.AsyncClient(api_key=api_key)
            self.model = model
            self.dimension = 1024  # voyage-2 uses 1024 dimensions
        except ImportError as e:
            raise ImportError("voyageai not installed. Install with: pip install voyageai") from e

    async def embed(self, text: str) -> list[float]:
        """Generate embeddings using Voyage AI."""
        try:
            result = await self.client.embed(
                [text],
                model=self.model
            )
            return list(result.embeddings[0])
        except Exception as e:
            logger.error(f"Voyage AI embedding error: {e}")
            raise

    def get_dimension(self) -> int:
        return int(self.dimension)


class DummyEmbeddingProvider(EmbeddingProvider):
    """Dummy embedding provider for testing."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        logger.warning(f"Using dummy embeddings (dim={dimension}). Fitness evaluation will be random!")

    async def embed(self, text: str) -> list[float]:
        """Generate random embeddings for testing."""
        # Use text hash as seed for consistent embeddings
        seed = hash(text) % (2**32)
        np.random.seed(seed)
        return list(np.random.randn(self.dimension).tolist())

    def get_dimension(self) -> int:
        return int(self.dimension)


class EmbeddingProviderFactory:
    """Factory for creating embedding providers."""

    @staticmethod
    def create(provider_type: str, **kwargs: object) -> EmbeddingProvider:
        """Create an embedding provider.

        Args:
            provider_type: Type of provider (openai, sentence-transformer, cohere, voyage, dummy)
            **kwargs: Provider-specific arguments

        Returns:
            EmbeddingProvider instance
        """
        provider_type = provider_type.lower()

        if provider_type == "openai":
            api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key required")
            model = kwargs.get("model", "text-embedding-ada-002")
            return OpenAIEmbeddingProvider(api_key, model)

        elif provider_type in ["sentence-transformer", "local", "sentence_transformer"]:
            model = kwargs.get("model", "all-MiniLM-L6-v2")
            return SentenceTransformerProvider(model)

        elif provider_type == "cohere":
            api_key = kwargs.get("api_key") or os.getenv("COHERE_API_KEY")
            if not api_key:
                raise ValueError("Cohere API key required")
            model = kwargs.get("model", "embed-english-v3.0")
            return CohereEmbeddingProvider(api_key, model)

        elif provider_type == "voyage":
            api_key = kwargs.get("api_key") or os.getenv("VOYAGE_API_KEY")
            if not api_key:
                raise ValueError("Voyage API key required")
            model = kwargs.get("model", "voyage-2")
            return VoyageEmbeddingProvider(api_key, model)

        elif provider_type == "dummy":
            dimension = kwargs.get("dimension", 384)
            return DummyEmbeddingProvider(dimension)

        else:
            raise ValueError(f"Unsupported embedding provider: {provider_type}")


# Global embedding provider instance
_embedding_provider: EmbeddingProvider | None = None


def set_embedding_provider(provider: EmbeddingProvider) -> None:
    """Set the global embedding provider."""
    global _embedding_provider
    _embedding_provider = provider
    logger.info(f"Set embedding provider to {type(provider).__name__}")


def get_embedding_provider() -> EmbeddingProvider:
    """Get the current embedding provider."""
    global _embedding_provider

    if _embedding_provider is None:
        # Try to use sentence-transformer by default
        try:
            _embedding_provider = EmbeddingProviderFactory.create("sentence-transformer")
            logger.info("Using default Sentence Transformer embedding provider")
        except ImportError:
            # Fall back to OpenAI if available
            if os.getenv("OPENAI_API_KEY"):
                logger.info("Sentence Transformers not available, falling back to OpenAI")
                _embedding_provider = EmbeddingProviderFactory.create("openai")
            else:
                # No valid embedding provider available - this is a fatal error
                raise RuntimeError(
                    "No embedding provider available. Please either:\n"
                    "1. Install sentence-transformers: pip install sentence-transformers\n"
                    "2. Set OPENAI_API_KEY environment variable\n"
                    "3. Configure an alternative provider (Cohere, Voyage AI)"
                ) from None

    return _embedding_provider


async def embed_text(text: str) -> list[float]:
    """Convenience function to embed text using the current provider."""
    provider = get_embedding_provider()
    return await provider.embed(text)

