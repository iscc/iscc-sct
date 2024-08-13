import pytest
from pydantic import ValidationError
from iscc_sct.models import Metadata, Feature, FeatureSet


def test_feature_initialization():
    # Test empty initialization
    with pytest.raises(ValidationError):
        Feature()
    feature = Feature(simprint="XZjeSfdyVi0")
    assert feature.simprint == "XZjeSfdyVi0"
    assert feature.offset is None
    assert feature.content is None

    # Test initialization with values
    feature = Feature(simprint="feature", offset=5, content="example text")
    assert feature.simprint == "feature"
    assert feature.offset == 5
    assert feature.content == "example text"


def test_feature_set_initialization():
    fs = FeatureSet()
    assert fs.model_dump(exclude_none=True) == {"maintype": "semantic", "subtype": "text", "version": 0}


def test_sct_meta_initialization():
    # Test initialization with minimal required fields
    meta = Metadata(iscc="ISCC1234567890")
    assert meta.iscc == "ISCC1234567890"
    assert meta.characters is None
    assert meta.features is None

    # Test initialization with all fields
    features = [FeatureSet(simprints=[Feature(simprint="feature1", offset=0, content="text1")], embedding=[0.1, 0.2])]
    meta = Metadata(iscc="ISCC1234567890", characters=1000, features=features)
    assert meta.iscc == "ISCC1234567890"
    assert meta.characters == 1000
    assert meta.features == features
    assert meta.features[0].embedding == [0.1, 0.2]
