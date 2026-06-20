"""Embedding providers for semantic note search.

This module intentionally stays outside of ``agentkernel.memory`` so the default
SQLite/JSONL notebook does not depend on an embedding endpoint. The provider is
only instantiated when ``Config.semantic_search`` is enabled.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from agentkernel.config import Config


class EmbeddingProvider(Protocol):
    """A tiny embedding seam. Anything matching this can back semantic search."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one dense vector per non-empty text input."""
        ...


class EmbeddingError(RuntimeError):
    """Raised when an embedding request cannot be completed."""


@dataclass
class OpenAIEmbeddingProvider:
    """OpenAI-compatible embedding endpoint using only stdlib ``urllib``.

    Works with OpenAI, any OpenAI-compatible local server, or cloud providers
    that expose the ``/embeddings`` route. The API key is read from the
    environment (``OPENAI_API_KEY`` by default) and never persisted.
    """

    model: str = "text-embedding-3-small"
    base_url: str = "https://api.openai.com/v1"
    dimensions: int | None = None
    api_key_env: str = "OPENAI_API_KEY"
    timeout: float = 60.0

    @classmethod
    def from_config(cls, config: Config, *, api_key_env: str = "OPENAI_API_KEY") -> "OpenAIEmbeddingProvider":
        """Convenience constructor fromConfig.

        Infers a sensible endpoint from Config ``provider``/``base_url`` when
        ``embedding_base_url`` is not set:
        - provider "openai" → https://api.openai.com/v1
        - provider "local"  → config.base_url (fallback to ollama-style default)
        - provider "anthropic" or other → requires explicit ``embedding_base_url``.
        """
        base_url = config.embedding_base_url
        if base_url is None:
            if config.provider == "openai":
                base_url = "https://api.openai.com/v1"
            elif config.provider == "local":
                base_url = config.base_url or "http://localhost:11434/v1"
            else:
                raise EmbeddingError(
                    f"Cannot infer embedding endpoint for provider={config.provider!r}. "
                    "Set `embedding_base_url` in agentkernel.toml."
                )
        return cls(
            model=config.embedding_model,
            base_url=base_url,
            dimensions=config.embedding_dimensions,
            api_key_env=api_key_env,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Call the embeddings endpoint and return vectors in input order."""
        texts = [t.strip() for t in texts]
        if not texts:
            return []
        if all(not t for t in texts):
            return [[] for _ in texts]
        key = os.environ.get(self.api_key_env)
        if not key:
            raise EmbeddingError(f"Environment variable {self.api_key_env} is not set")
        payload: dict[str, Any] = {"model": self.model, "input": texts}
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        url = self.base_url.rstrip("/") + "/embeddings"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="ignore")
            raise EmbeddingError(f"Embedding request failed ({exc.code}): {body}") from exc
        except Exception as exc:
            raise EmbeddingError(f"Embedding request failed: {exc}") from exc

        embeddings: list[list[float]] = []
        for item in sorted(result.get("data", []), key=lambda x: x.get("index", 0)):
            embeddings.append(item.get("embedding", []))
        if len(embeddings) != len(texts):
            raise EmbeddingError(
                f"Embedding response length mismatch: expected {len(texts)}, got {len(embeddings)}"
            )
        return embeddings


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors (0.0 if invalid)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# Forward import for type annotations that need a runtime reference to Any.
from typing import Any  # noqa: E402
