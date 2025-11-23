"""Tests for options module validators."""

import pytest
from iscc_sct.options import SctOptions


def test_validate_model_version_invalid():
    # type: () -> None
    """Test that invalid model_version raises ValueError."""
    with pytest.raises(ValueError) as excinfo:
        SctOptions(model_version=999)

    error_msg = str(excinfo.value)
    assert "Invalid model_version 999" in error_msg
    assert "Available versions:" in error_msg


def test_validate_prompt_type_enum_instance():
    # type: () -> None
    """Test that EmbeddingGemmaPrompt enum instances are converted to name string."""
    # Import enum within function to test the validator's import path (lines 119-122)
    from iscc_sct.code_semantic_text import EmbeddingGemmaPrompt as LocalEmbeddingGemmaPrompt

    opts = SctOptions(prompt_type=LocalEmbeddingGemmaPrompt.QUERY)
    assert opts.prompt_type == "QUERY"


def test_validate_prompt_type_object_with_name_attribute():
    # type: () -> None
    """Test that objects with 'name' attribute are handled (lines 153-154)."""

    # Create a simple object with a 'name' attribute
    class FakeEnum:
        def __init__(self, name):
            # type: (str) -> None
            self.name = name

    fake_enum = FakeEnum("DOCUMENT")
    opts = SctOptions(prompt_type=fake_enum)
    assert opts.prompt_type == "DOCUMENT"


def test_validate_prompt_type_invalid_type():
    # type: () -> None
    """Test that invalid prompt_type type raises ValueError (lines 156-158)."""
    with pytest.raises(ValueError) as excinfo:
        SctOptions(prompt_type=12345)  # Invalid: integer instead of string/enum

    error_msg = str(excinfo.value)
    assert "prompt_type must be a string or EmbeddingGemmaPrompt enum" in error_msg
    assert "int" in error_msg


def test_validate_prompt_type_valid_string():
    # type: () -> None
    """Test that valid prompt_type strings are accepted and normalized."""
    # Test case-insensitive string input
    opts1 = SctOptions(prompt_type="document")
    assert opts1.prompt_type == "DOCUMENT"

    opts2 = SctOptions(prompt_type="QUERY")
    assert opts2.prompt_type == "QUERY"

    opts3 = SctOptions(prompt_type="clustering")
    assert opts3.prompt_type == "CLUSTERING"


def test_validate_prompt_type_none():
    # type: () -> None
    """Test that None is accepted for prompt_type."""
    opts = SctOptions(prompt_type=None)
    assert opts.prompt_type is None


def test_validate_prompt_type_invalid_string():
    # type: () -> None
    """Test that invalid prompt_type string raises ValueError."""
    with pytest.raises(ValueError) as excinfo:
        SctOptions(prompt_type="invalid_prompt")

    error_msg = str(excinfo.value)
    assert "Invalid prompt_type 'invalid_prompt'" in error_msg
    assert "Valid options:" in error_msg
