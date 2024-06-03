import pytest

from iscc_sct.models import SctMeta, SctFeature


def test_sct_feature_initialization():
    # Test initialization with all fields None
    feature = SctFeature()
    assert feature.feature is None
    assert feature.offset is None
    assert feature.text is None

    # Test initialization with values
    feature = SctFeature(feature="feature", offset=5, text="example text")
    assert feature.feature == "feature"
    assert feature.offset == 5
    assert feature.text == "example text"


def test_sct_meta_initialization():
    # Test initialization with minimal required fields
    meta = SctMeta(iscc="ISCC1234567890")
    assert meta.iscc == "ISCC1234567890"
    assert meta.characters is None
    assert meta.embedding is None
    assert meta.features is None

    # Test initialization with all fields
    features = [SctFeature(feature="feature1", offset=0, text="text1")]
    meta = SctMeta(iscc="ISCC1234567890", characters=1000, embedding=[0.1, 0.2], features=features)
    assert meta.iscc == "ISCC1234567890"
    assert meta.characters == 1000
    assert meta.embedding == [0.1, 0.2]
    assert meta.features == features


def test_sct_meta_from_dict():
    data = {
        "iscc": "ISCC1234567890",
        "characters": 100,
        "embedding": [0.1, 0.2],
        "features": ["feature1", "feature2"],
        "chunks": ["chunk1", "chunk2"],
    }

    meta = SctMeta.from_dict(data)
    assert meta.iscc == "ISCC1234567890"
    assert meta.characters == 100
    assert meta.embedding == [0.1, 0.2]
    assert len(meta.features) == 2
    assert meta.features[0].feature == "feature1"
    assert meta.features[0].offset is None
    assert meta.features[0].text == "chunk1"
    assert meta.features[1].feature == "feature2"
    assert meta.features[1].offset is None
    assert meta.features[1].text == "chunk2"

    # Test with missing optional fields
    minimal_data = {"iscc": "ISCC1234567890"}
    minimal_meta = SctMeta.from_dict(minimal_data)
    assert minimal_meta.iscc == "ISCC1234567890"
    assert minimal_meta.characters is None
    assert minimal_meta.embedding is None
    assert minimal_meta.features is None


def test_sct_meta_from_dict_with_incomplete_data():
    # Incomplete feature-chunk pairs
    data = {
        "iscc": "ISCC1234567890",
        "features": ["feature1"],
        "chunks": ["chunk1", "chunk2"],  # More chunks than features
    }
    with pytest.raises(IndexError):
        SctMeta.from_dict(data)
