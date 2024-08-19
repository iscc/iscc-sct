import pytest
from pydantic import ValidationError
from iscc_sct.models import Metadata, Feature, FeatureSet
import iscc_sct as sct


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
    assert fs.model_dump(exclude_none=True) == {
        "maintype": "semantic",
        "subtype": "text",
        "version": 0,
    }


def test_sct_meta_initialization():
    # Test initialization with minimal required fields
    meta = Metadata(iscc="ISCC1234567890")
    assert meta.iscc == "ISCC1234567890"
    assert meta.characters is None
    assert meta.features is None

    # Test initialization with all fields
    features = [
        FeatureSet(
            simprints=[Feature(simprint="feature1", offset=0, content="text1")],
            embedding=[0.1, 0.2],
        )
    ]
    meta = Metadata(iscc="ISCC1234567890", characters=1000, features=features)
    assert meta.iscc == "ISCC1234567890"
    assert meta.characters == 1000
    assert meta.features == features
    assert meta.features[0].embedding == [0.1, 0.2]


def test_metadata_to_index_format():
    # Test conversion from Object-Format to Index-Format
    features = [
        FeatureSet(
            simprints=[
                Feature(simprint="feature1", offset=0, size=5, content="text1"),
                Feature(simprint="feature2", offset=5, size=5, content="text2"),
            ]
        )
    ]
    meta = Metadata(iscc="ISCC1234567890", features=features)
    index_meta = meta.to_index_format()
    assert isinstance(index_meta.features[0].simprints[0], str)
    assert index_meta.features[0].simprints == ["feature1", "feature2"]
    assert index_meta.features[0].offsets == [0, 5]
    assert index_meta.features[0].sizes == [5, 5]
    assert index_meta.features[0].contents == ["text1", "text2"]

    # Test that Index-Format remains unchanged
    index_meta2 = index_meta.to_index_format()
    assert index_meta2.model_dump() == index_meta.model_dump()


def test_metadata_to_object_format():
    # Test conversion from Index-Format to Object-Format
    features = [
        FeatureSet(
            simprints=["feature1", "feature2"],
            offsets=[0, 5],
            sizes=[5, 5],
            contents=["text1", "text2"],
        )
    ]
    meta = Metadata(iscc="ISCC1234567890", features=features)
    object_meta = meta.to_object_format()
    assert isinstance(object_meta.features[0].simprints[0], Feature)
    assert object_meta.features[0].simprints[0].simprint == "feature1"
    assert object_meta.features[0].simprints[0].offset == 0
    assert object_meta.features[0].simprints[0].size == 5
    assert object_meta.features[0].simprints[0].content == "text1"
    assert object_meta.features[0].offsets is None
    assert object_meta.features[0].sizes is None
    assert object_meta.features[0].contents is None

    # Test that Object-Format remains unchanged
    object_meta2 = object_meta.to_object_format()
    assert object_meta2.model_dump() == object_meta.model_dump()


def test_metadata_to_index_format_with_none_simprints():
    # Test conversion when feature_set.simprints is None
    features = [FeatureSet(simprints=None, embedding=[0.1, 0.2])]
    meta = Metadata(iscc="ISCC1234567890", features=features)
    index_meta = meta.to_index_format()
    assert index_meta.features[0].simprints is None
    assert index_meta.features[0].embedding == [0.1, 0.2]
    assert index_meta.model_dump() == meta.model_dump()


def test_metadata_format_conversion_with_no_features():
    meta = Metadata(iscc="ISCC1234567890")
    index_meta = meta.to_index_format()
    object_meta = meta.to_object_format()
    assert index_meta.model_dump() == meta.model_dump()
    assert object_meta.model_dump() == meta.model_dump()


def test_metadata_get_content(text_en):
    iscc = sct.create(text_en, granular=True)
    assert iscc.get_content() == text_en


def test_metadata_get_content_no_fetures():
    meta = Metadata(iscc="ISCC1234567890")
    assert meta.get_content() is None


def test_metadata_get_content_index_format():
    meta = sct.create("Hello World", granular=True).to_index_format()
    assert meta.get_content() == "Hello World"


def test_metadata_get_content_no_content():
    meta = sct.create("Hello World", granular=True, contents=False)
    assert meta.get_content() is None


def test_metadata_get_overlaps():
    # Test with no features
    meta = Metadata(iscc="ISCC1234567890")
    assert meta.get_overlaps() == []

    # Test with features but no simprints
    meta = Metadata(iscc="ISCC1234567890", features=[FeatureSet()])
    assert meta.get_overlaps() == []

    # Test with non-overlapping chunks
    features = [
        FeatureSet(
            simprints=[
                Feature(simprint="feature1", offset=0, content="Hello"),
                Feature(simprint="feature2", offset=5, content="World"),
            ]
        )
    ]
    meta = Metadata(iscc="ISCC1234567890", features=features)
    assert meta.get_overlaps() == [""]

    # Test with overlapping chunks
    features = [
        FeatureSet(
            simprints=[
                Feature(simprint="feature1", offset=0, content="Hello W"),
                Feature(simprint="feature2", offset=5, content="World"),
            ]
        )
    ]
    meta = Metadata(iscc="ISCC1234567890", features=features)
    assert meta.get_overlaps() == [" W"]

    # Test with multiple overlaps
    features = [
        FeatureSet(
            simprints=[
                Feature(simprint="feature1", offset=0, content="Hello W"),
                Feature(simprint="feature2", offset=5, content="World!"),
                Feature(simprint="feature3", offset=10, content="! How are you?"),
            ]
        )
    ]
    meta = Metadata(iscc="ISCC1234567890", features=features)
    assert meta.get_overlaps() == [" W", "!"]

    # Test with index format
    features = [
        FeatureSet(
            simprints=["feature1", "feature2", "feature3"],
            offsets=[0, 5, 10],
            contents=["Hello W", "World!", "! How are you?"],
        )
    ]
    meta = Metadata(iscc="ISCC1234567890", features=features)
    assert meta.get_overlaps() == [" W", "!"]

    # Test with missing content or offset
    features = [
        FeatureSet(
            simprints=[
                Feature(simprint="feature1", offset=0, content="Hello"),
                Feature(simprint="feature2", content="World"),
                Feature(simprint="feature3", offset=10),
            ]
        )
    ]
    meta = Metadata(iscc="ISCC1234567890", features=features)
    assert meta.get_overlaps() == []
