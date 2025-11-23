"""Tests for models_config module."""

import pytest
from iscc_sct.models_config import get_model_config, MODEL_REGISTRY


def test_get_model_config_valid_versions():
    # type: () -> None
    """Test get_model_config with valid version numbers."""
    # Test version 0
    config_v0 = get_model_config(0)
    assert config_v0.version == 0
    assert config_v0.name == "paraphrase-multilingual-minilm-l12-v2"
    assert config_v0.embedding_dim == 384

    # Test version 1
    config_v1 = get_model_config(1)
    assert config_v1.version == 1
    assert config_v1.name == "embeddinggemma-300m"
    assert config_v1.embedding_dim == 768


def test_get_model_config_invalid_version():
    # type: () -> None
    """Test get_model_config raises ValueError for invalid version."""
    with pytest.raises(ValueError) as excinfo:
        get_model_config(999)

    error_msg = str(excinfo.value)
    assert "Model version 999 not found" in error_msg
    assert "Available versions:" in error_msg
    # Should list available versions
    for version in MODEL_REGISTRY.keys():
        assert str(version) in error_msg
