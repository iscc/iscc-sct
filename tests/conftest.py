import pytest
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np


HERE = Path(__file__).parent.absolute()


@pytest.fixture
def text_en():
    return (HERE / "en.txt").read_text(encoding="utf-8")


@pytest.fixture
def text_de():
    return (HERE / "de.txt").read_text(encoding="utf-8")


@pytest.fixture
def mock_model():
    """Mock ONNX model that returns realistic embeddings."""
    mock = MagicMock()

    def mock_run(output_names, input_feed):
        # Return realistic-looking embeddings based on input size
        batch_size = input_feed["input_ids"].shape[0]
        # Model v0 has 384 dims, v1 has 768 dims - default to 384
        embedding_dim = 384
        # Create deterministic but varied embeddings
        embeddings = np.random.RandomState(42).randn(batch_size, embedding_dim).astype(np.float32)
        # Normalize to unit length (realistic for sentence embeddings)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return [embeddings]

    mock.run = mock_run
    return mock


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer that returns realistic token outputs."""
    mock = MagicMock()

    def mock_encode_batch(texts):
        encodings = []
        for text in texts:
            encoding = MagicMock()
            # Create deterministic token IDs based on text length
            num_tokens = min(len(text.split()), 127)
            encoding.ids = list(range(num_tokens))
            encoding.attention_mask = [1] * num_tokens
            encodings.append(encoding)
        return encodings

    mock.encode_batch = mock_encode_batch
    return mock


@pytest.fixture
def mock_embed_tokens():
    """Returns a function that mocks embed_tokens with realistic embeddings."""

    def _mock_embed_tokens(tokens, model_version=0):
        batch_size = tokens["input_ids"].shape[0]
        embedding_dim = 768 if model_version == 1 else 384
        embeddings = np.random.RandomState(42).randn(batch_size, embedding_dim).astype(np.float32)
        # Normalize to unit length
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    return _mock_embed_tokens


@pytest.fixture
def mock_embed_chunks():
    """Returns a function that mocks embed_chunks with realistic embeddings."""

    def _mock_embed_chunks(chunks, model_version=0, **kwargs):
        num_chunks = len(chunks)
        embedding_dim = 768 if model_version == 1 else 384
        embeddings = np.random.RandomState(42).randn(num_chunks, embedding_dim).astype(np.float32)
        # Normalize to unit length
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    return _mock_embed_chunks
