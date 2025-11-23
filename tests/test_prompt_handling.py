"""Test EmbeddingGemma prompt handling."""

import pytest
import iscc_sct.code_semantic_text
from iscc_sct.code_semantic_text import (
    EmbeddingGemmaPrompt,
    tokenize_chunks,
    gen_text_code_semantic,
)
from iscc_sct.options import SctOptions
from unittest.mock import patch, MagicMock


def test_embedding_gemma_prompt_enum():
    # type: () -> None
    """Test that EmbeddingGemmaPrompt enum has the expected values."""
    assert EmbeddingGemmaPrompt.DOCUMENT.value == "title: none | text: "
    assert EmbeddingGemmaPrompt.QUERY.value == "task: search result | query: "
    assert EmbeddingGemmaPrompt.CLUSTERING.value == "task: clustering | query: "
    assert EmbeddingGemmaPrompt.CLASSIFICATION.value == "task: classification | query: "
    assert EmbeddingGemmaPrompt.INSTRUCTION_RETRIEVAL.value == "task: code retrieval | query: "
    assert EmbeddingGemmaPrompt.PAIR_CLASSIFICATION.value == "task: sentence similarity | query: "
    assert EmbeddingGemmaPrompt.STS.value == "task: sentence similarity | query: "
    assert EmbeddingGemmaPrompt.SUMMARIZATION.value == "task: summarization | query: "
    assert EmbeddingGemmaPrompt.NONE.value == ""


def test_tokenize_chunks_applies_prompt_for_model_v1():
    # type: () -> None
    """Test that tokenize_chunks applies prompt for model version 1."""
    test_chunks = ["sample text", "another chunk"]

    # Clear cache to ensure mock takes effect
    iscc_sct.code_semantic_text.tokenizer.cache_clear()

    # Mock the tokenizer to capture what's being tokenized
    with patch("iscc_sct.code_semantic_text.tokenizer") as mock_tokenizer:
        mock_tok_instance = MagicMock()
        mock_tokenizer.return_value = mock_tok_instance

        # Create mock encodings
        mock_encoding1 = MagicMock()
        mock_encoding1.ids = [1, 2, 3]
        mock_encoding1.attention_mask = [1, 1, 1]

        mock_encoding2 = MagicMock()
        mock_encoding2.ids = [4, 5, 6]
        mock_encoding2.attention_mask = [1, 1, 1]

        mock_tok_instance.encode_batch.return_value = [mock_encoding1, mock_encoding2]

        # Test with model version 1 (should apply default DOCUMENT prompt)
        result = tokenize_chunks(test_chunks, model_version=1)

        # Verify the tokenizer was called with prompted text
        expected_chunks = ["title: none | text: sample text", "title: none | text: another chunk"]
        mock_tok_instance.encode_batch.assert_called_once_with(expected_chunks)

        # Verify result structure (no token_type_ids for v1)
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "token_type_ids" not in result


def test_tokenize_chunks_custom_prompt_type():
    # type: () -> None
    """Test that tokenize_chunks uses custom prompt type when specified."""
    test_chunks = ["test query"]

    # Clear cache to ensure mock takes effect
    iscc_sct.code_semantic_text.tokenizer.cache_clear()

    with patch("iscc_sct.code_semantic_text.tokenizer") as mock_tokenizer:
        mock_tok_instance = MagicMock()
        mock_tokenizer.return_value = mock_tok_instance

        mock_encoding = MagicMock()
        mock_encoding.ids = [1, 2, 3]
        mock_encoding.attention_mask = [1, 1, 1]
        mock_tok_instance.encode_batch.return_value = [mock_encoding]

        # Test with QUERY prompt
        tokenize_chunks(test_chunks, model_version=1, prompt_type=EmbeddingGemmaPrompt.QUERY)

        expected_chunks = ["task: search result | query: test query"]
        mock_tok_instance.encode_batch.assert_called_with(expected_chunks)


def test_tokenize_chunks_string_prompt_type():
    # type: () -> None
    """Test that tokenize_chunks converts string prompt types to enum."""
    test_chunks = ["clustering data"]

    # Clear cache to ensure mock takes effect
    iscc_sct.code_semantic_text.tokenizer.cache_clear()

    with patch("iscc_sct.code_semantic_text.tokenizer") as mock_tokenizer:
        mock_tok_instance = MagicMock()
        mock_tokenizer.return_value = mock_tok_instance

        mock_encoding = MagicMock()
        mock_encoding.ids = [1, 2, 3]
        mock_encoding.attention_mask = [1, 1, 1]
        mock_tok_instance.encode_batch.return_value = [mock_encoding]

        # Test with string prompt type (should convert to enum)
        tokenize_chunks(test_chunks, model_version=1, prompt_type="clustering")

        expected_chunks = ["task: clustering | query: clustering data"]
        mock_tok_instance.encode_batch.assert_called_with(expected_chunks)


def test_tokenize_chunks_no_prompt_for_model_v0():
    # type: () -> None
    """Test that tokenize_chunks does not apply prompt for model version 0."""
    test_chunks = ["sample text"]

    # Clear cache to ensure mock takes effect
    iscc_sct.code_semantic_text.tokenizer.cache_clear()

    with patch("iscc_sct.code_semantic_text.tokenizer") as mock_tokenizer:
        mock_tok_instance = MagicMock()
        mock_tokenizer.return_value = mock_tok_instance

        mock_encoding = MagicMock()
        mock_encoding.ids = [1, 2, 3]
        mock_encoding.attention_mask = [1, 1, 1]
        mock_encoding.type_ids = [0, 0, 0]
        mock_tok_instance.encode_batch.return_value = [mock_encoding]

        # Test with model version 0 (should NOT apply prompt)
        result = tokenize_chunks(
            test_chunks, model_version=0, prompt_type=EmbeddingGemmaPrompt.DOCUMENT
        )

        # Verify no prompt was applied
        mock_tok_instance.encode_batch.assert_called_once_with(test_chunks)

        # Verify result structure (includes token_type_ids for v0)
        assert "token_type_ids" in result


def test_tokenize_chunks_none_prompt():
    # type: () -> None
    """Test that NONE prompt type doesn't add any prefix."""
    test_chunks = ["raw text"]

    # Clear cache to ensure mock takes effect
    iscc_sct.code_semantic_text.tokenizer.cache_clear()

    with patch("iscc_sct.code_semantic_text.tokenizer") as mock_tokenizer:
        mock_tok_instance = MagicMock()
        mock_tokenizer.return_value = mock_tok_instance

        mock_encoding = MagicMock()
        mock_encoding.ids = [1, 2, 3]
        mock_encoding.attention_mask = [1, 1, 1]
        mock_tok_instance.encode_batch.return_value = [mock_encoding]

        # Test with NONE prompt (should not add prefix)
        tokenize_chunks(test_chunks, model_version=1, prompt_type=EmbeddingGemmaPrompt.NONE)

        # Should use original chunks without modification
        mock_tok_instance.encode_batch.assert_called_with(test_chunks)


def test_invalid_string_prompt_type():
    # type: () -> None
    """Test that invalid prompt type falls back to DOCUMENT with warning."""
    test_chunks = ["test text"]

    # Clear cache to ensure mock takes effect
    iscc_sct.code_semantic_text.tokenizer.cache_clear()

    with patch("iscc_sct.code_semantic_text.tokenizer") as mock_tokenizer:
        mock_tok_instance = MagicMock()
        mock_tokenizer.return_value = mock_tok_instance

        mock_encoding = MagicMock()
        mock_encoding.ids = [1, 2, 3]
        mock_encoding.attention_mask = [1, 1, 1]
        mock_tok_instance.encode_batch.return_value = [mock_encoding]

        with patch("iscc_sct.code_semantic_text.log") as mock_log:
            # Test with invalid prompt type
            tokenize_chunks(test_chunks, model_version=1, prompt_type="invalid_prompt")

            # Should log warning and use DOCUMENT prompt
            mock_log.warning.assert_called_once()
            expected_chunks = ["title: none | text: test text"]
            mock_tok_instance.encode_batch.assert_called_with(expected_chunks)


def test_prompt_type_in_options():
    # type: () -> None
    """Test that prompt_type can be set via options."""
    opts = SctOptions()

    # Default should be None
    assert opts.prompt_type is None

    # Test setting via override
    opts_with_prompt = opts.override({"prompt_type": "QUERY"})
    assert opts_with_prompt.prompt_type == "QUERY"

    # Test validation converts to uppercase
    opts_lower = opts.override({"prompt_type": "document"})
    assert opts_lower.prompt_type == "DOCUMENT"


def test_prompt_type_validation():
    # type: () -> None
    """Test that invalid prompt types raise validation error."""
    opts = SctOptions()

    with pytest.raises(ValueError, match="Invalid prompt_type"):
        opts.override({"prompt_type": "invalid"})


def test_gen_text_code_semantic_with_prompt_type():
    # type: () -> None
    """Test that gen_text_code_semantic passes prompt_type to embed_chunks."""
    test_text = "This is a test document for embedding."

    with patch("iscc_sct.code_semantic_text.embed_chunks") as mock_embed:
        # Setup mock to return fake embeddings
        import numpy as np

        mock_embeddings = np.random.rand(2, 384).astype(np.float32)
        mock_embed.return_value = mock_embeddings

        # Call with prompt_type option
        result = gen_text_code_semantic(test_text, prompt_type="QUERY", model_version=1)

        # Verify embed_chunks was called with the prompt_type
        assert mock_embed.called
        call_args = mock_embed.call_args
        assert call_args.kwargs.get("prompt_type") == "QUERY"
        assert call_args.kwargs.get("model_version") == 1

        # Verify result has ISCC code
        assert "iscc" in result
        assert result["iscc"].startswith("ISCC:")
