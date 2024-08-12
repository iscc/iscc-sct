import iscc_sct as sct


def test_create_returns_sct_meta():
    result = sct.create("Hello World")
    assert isinstance(result, sct.SctMeta)


def test_create_default():
    result = sct.create("Hello World")
    assert result == sct.SctMeta(iscc="ISCC:CAA7GZ4J3DI3XY2R", characters=11)


def test_create_granular():
    result = sct.create("Hello World", granular=True)
    assert result == sct.SctMeta(
        iscc="ISCC:CAA7GZ4J3DI3XY2R",
        characters=11,
        embedding=None,
        features=[sct.SctFeature(feature="82eJ2NG741E", offset=0, size=11, text="Hello World")],
    )


def test_create_embedding():
    result = sct.create("Hello World", embedding=True)
    assert len(result.embedding) == 384
