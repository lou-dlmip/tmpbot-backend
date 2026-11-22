# Query-time embedders for retrieval.
#
# The chat embeds each question with the same embedder the corpus was
# indexed with (TOOL/embedding.py at ingest). The fake below is a copy of
# TOOL/embedding.py's FakeEmbedder, line for line in its maths, so a
# corpus indexed by the tool with `--embedder fake` is searchable here;
# BACK/tests/test_embedding_parity.py holds the two equal. The OpenAI
# embedder uses httpx (already a backend dependency) instead of the
# tool's requests session, and is never called in tests.
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Embed a question for hybrid search."""

from __future__ import annotations

import hashlib
import logging
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Protocol

import httpx
import yaml

log = logging.getLogger("tmpbot.embedding")

FAKE_MODEL = "fake/hashing-v1"
OPENAI_URL = "https://api.openai.com/v1/embeddings"

_CJK = (
    "぀-ヿ"
    "㐀-䶿"
    "一-鿿"
    "豈-﫿"
    "가-힯"
)
_WORD_RE = re.compile(rf"[a-z0-9]+|[{_CJK}]+")
_CJK_RUN_RE = re.compile(rf"^[{_CJK}]+$")


class EmbeddingError(Exception):
    """Raised when a question cannot be embedded."""


class QueryEmbedder(Protocol):
    """What retrieval needs: the model name and one vector per text."""

    model: str
    dimensions: int

    async def embed_query(self, text: str) -> list[float]:
        """Embed one question.

        Args:
            text: The question.

        Returns:
            list[float]: The vector.
        """
        ...


def load_embedding_config(path: Path) -> dict[str, Any]:
    """Read the ``embeddings:`` block of the routing policy.

    Args:
        path: ``BOTS/POLC/llm_routing.yaml``.

    Returns:
        dict[str, Any]: ``model`` and ``dimensions``.

    Raises:
        EmbeddingError: When the block is missing or malformed.
    """
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise EmbeddingError(f"cannot read {path}: {exc}") from exc
    block = data.get("embeddings")
    if not isinstance(block, dict):
        raise EmbeddingError(f"{path}: no 'embeddings:' block")
    model, dims = block.get("model"), block.get("dimensions")
    if not isinstance(model, str) or "/" not in model:
        raise EmbeddingError("embeddings.model must be 'provider/name'")
    if not isinstance(dims, int) or dims <= 0:
        raise EmbeddingError("embeddings.dimensions must be > 0")
    return {"model": model, "dimensions": dims}


def normalise_text(text: str) -> str:
    """Lower-case and strip accents, as the database's unaccent does.

    Args:
        text: Any text.

    Returns:
        str: The normalised text.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(
        ch for ch in decomposed if not unicodedata.combining(ch)
    )
    return unicodedata.normalize("NFKC", stripped).lower()


def features(text: str) -> Counter:
    """Split text into the features the fake embedder hashes.

    Args:
        text: Any text.

    Returns:
        Counter: Feature to count.
    """
    counts: Counter = Counter()
    for token in _WORD_RE.findall(normalise_text(text)):
        if _CJK_RUN_RE.match(token):
            counts.update(token)
            counts.update(token[i:i + 2] for i in range(len(token) - 1))
        elif len(token) >= 2:
            counts[token] += 1
    return counts


class FakeEmbedder:
    """Deterministic feature-hashing embedder (same maths as TOOL's).

    Args:
        dimensions: Vector length.
        model: Name stored with each vector.
    """

    def __init__(self, dimensions: int = 1536,
                 model: str = FAKE_MODEL) -> None:
        """Create the embedder.

        Args:
            dimensions: Vector length.
            model: Model name.

        Raises:
            ValueError: When dimensions is not positive.
        """
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions
        self.model = model

    def vector(self, text: str) -> list[float]:
        """Embed one text synchronously.

        Args:
            text: The text.

        Returns:
            list[float]: A unit vector.
        """
        vec = [0.0] * self.dimensions
        for feature, count in features(text).items():
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimensions
            sign = 1.0 if digest[8] & 1 else -1.0
            vec[index] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]

    async def embed_query(self, text: str) -> list[float]:
        """Embed one question.

        Args:
            text: The question.

        Returns:
            list[float]: A unit vector.
        """
        return self.vector(text)


class OpenAIQueryEmbedder:
    """OpenAI embeddings for one question at a time.

    Args:
        api_key: ``OPENAI_API_KEY`` of TMPBOT's own organisation.
        model: ``openai/<name>`` from the routing policy.
        dimensions: Expected vector length.
        client: An ``httpx.AsyncClient``; tests pass a mock transport.
        timeout_s: Seconds per request.
    """

    def __init__(self, api_key: str, model: str, dimensions: int,
                 client: Optional[httpx.AsyncClient] = None,
                 timeout_s: float = 10.0) -> None:
        """Create the embedder.

        Args:
            api_key: API key.
            model: Model id.
            dimensions: Vector length.
            client: HTTP client.
            timeout_s: Request timeout.

        Raises:
            EmbeddingError: When the key is missing or the model is not
                an OpenAI model.
        """
        if not api_key:
            raise EmbeddingError("OPENAI_API_KEY is not set")
        provider, _, name = model.partition("/")
        if provider != "openai" or not name:
            raise EmbeddingError(f"not an OpenAI model: {model}")
        self.model = model
        self.dimensions = dimensions
        self._name = name
        self._key = api_key
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    async def embed_query(self, text: str) -> list[float]:
        """Embed one question.

        Args:
            text: The question.

        Returns:
            list[float]: The vector.

        Raises:
            EmbeddingError: On an HTTP error or a malformed response.
        """
        try:
            resp = await self._client.post(
                OPENAI_URL,
                json={"model": self._name, "input": [text],
                      "dimensions": self.dimensions,
                      "encoding_format": "float"},
                headers={"Authorization": f"Bearer {self._key}"},
            )
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"embeddings request failed: {exc}") \
                from exc
        if resp.status_code != 200:
            raise EmbeddingError(f"embeddings status {resp.status_code}")
        try:
            vec = resp.json()["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise EmbeddingError(f"malformed embeddings response: {exc}") \
                from exc
        if len(vec) != self.dimensions:
            raise EmbeddingError(
                f"embedding has {len(vec)} dimensions, "
                f"expected {self.dimensions}"
            )
        return [float(v) for v in vec]


def vector_literal(vec: list[float]) -> str:
    """Format a vector as pgvector text input, for example ``[0.1,0.2]``.

    Args:
        vec: Floats.

    Returns:
        str: The literal; cast it with ``::vector`` in SQL.

    Raises:
        EmbeddingError: When a value is not finite.
    """
    if not all(math.isfinite(v) for v in vec):
        raise EmbeddingError("vector contains NaN or infinity")
    return "[" + ",".join(repr(float(v)) for v in vec) + "]"
