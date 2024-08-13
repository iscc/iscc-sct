import iscc_sct as sct


def test_create_returns_sct_meta():
    result = sct.create("Hello World")
    assert isinstance(result, sct.Metadata)


def test_create_default():
    result = sct.create("Hello World")
    assert result == sct.Metadata(iscc="ISCC:CAA7GZ4J3DI3XY2R", characters=11)


def test_create_granular():
    result = sct.create("Hello World", granular=True)
    assert result.model_dump(exclude_none=True) == {
        "iscc": "ISCC:CAA7GZ4J3DI3XY2R",
        "characters": 11,
        "features": [
            {
                "maintype": "semantic",
                "subtype": "text",
                "version": 0,
                "simprints": ["82eJ2NG741E"],
                "offsets": [0],
                "sizes": [11],
                "contents": ["Hello World"],
            }
        ],
    }


def test_create_embedding():
    result = sct.create("Hello World", embedding=True)
    assert len(result.features[0].embedding) == 384
