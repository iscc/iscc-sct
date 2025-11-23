"""Integration test demonstrating EmbeddingGemma prompt usage."""

from iscc_sct import create
from iscc_sct.code_semantic_text import EmbeddingGemmaPrompt


def test_prompt_type_with_create():
    # type: () -> None
    """Test that create() function accepts prompt_type parameter for model version 1."""
    test_text = "This is a sample document about machine learning and AI."

    # Test with DOCUMENT prompt (default for documents)
    result_doc = create(test_text, model_version=1, prompt_type="DOCUMENT", simprints=True)
    assert hasattr(result_doc, "iscc")
    assert result_doc.iscc.startswith("ISCC:")

    # Test with QUERY prompt (for search queries)
    result_query = create(test_text, model_version=1, prompt_type="QUERY", simprints=True)
    assert hasattr(result_query, "iscc")
    assert result_query.iscc.startswith("ISCC:")

    # The ISCCs should be different due to different prompts
    assert result_doc.iscc != result_query.iscc

    # The simprints should also differ
    if hasattr(result_doc, "features") and hasattr(result_query, "features"):
        doc_simprints = result_doc.features[0].simprints if result_doc.features else []
        query_simprints = result_query.features[0].simprints if result_query.features else []
        if doc_simprints and query_simprints:
            assert doc_simprints != query_simprints


def test_prompt_type_enum_usage():
    # type: () -> None
    """Test that EmbeddingGemmaPrompt enum can be used directly."""
    test_text = "Code example: def hello(): print('world')"

    # Use enum directly for code retrieval
    result = create(
        test_text, model_version=1, prompt_type=EmbeddingGemmaPrompt.INSTRUCTION_RETRIEVAL
    )
    assert hasattr(result, "iscc")
    assert result.iscc.startswith("ISCC:")


def test_prompt_type_none():
    # type: () -> None
    """Test that NONE prompt type skips prompt prefixing."""
    test_text = "Raw text without any task-specific prompt."

    # Using NONE should not add any prefix
    result = create(test_text, model_version=1, prompt_type="NONE")
    assert hasattr(result, "iscc")
    assert result.iscc.startswith("ISCC:")


def test_different_tasks_different_embeddings():
    # type: () -> None
    """Demonstrate that different task prompts produce different embeddings."""
    test_text = "The quick brown fox jumps over the lazy dog."

    # Generate ISCCs for different tasks
    results = {}
    tasks = ["DOCUMENT", "QUERY", "CLUSTERING", "SUMMARIZATION"]

    for task in tasks:
        results[task] = create(test_text, model_version=1, prompt_type=task)

    # All should have valid ISCCs
    for task, result in results.items():
        assert hasattr(result, "iscc")
        assert result.iscc.startswith("ISCC:")

    # Each task should produce a unique ISCC
    iscc_codes = [result.iscc for result in results.values()]
    assert len(iscc_codes) == len(set(iscc_codes)), "All ISCCs should be unique"


def test_model_version_0_ignores_prompt():
    # type: () -> None
    """Test that model version 0 ignores prompt_type parameter."""
    test_text = "This text is processed with model version 0."

    # Model version 0 should ignore prompt_type
    result_no_prompt = create(test_text, model_version=0)
    result_with_prompt = create(test_text, model_version=0, prompt_type="QUERY")

    # Both should produce the same ISCC since v0 ignores prompts
    assert result_no_prompt.iscc == result_with_prompt.iscc


def test_case_insensitive_prompt_type():
    # type: () -> None
    """Test that prompt types are case-insensitive."""
    test_text = "Testing case sensitivity."

    # These should all work the same
    result_upper = create(test_text, model_version=1, prompt_type="DOCUMENT")
    result_lower = create(test_text, model_version=1, prompt_type="document")
    result_mixed = create(test_text, model_version=1, prompt_type="Document")

    # All should produce the same ISCC
    assert result_upper.iscc == result_lower.iscc
    assert result_lower.iscc == result_mixed.iscc
