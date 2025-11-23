from unittest.mock import patch
import iscc_sct as sct


@patch("iscc_sct.code_semantic_text.embed_chunks")
def test_create_returns_sct_meta(mock_embed, mock_embed_chunks):
    mock_embed.side_effect = mock_embed_chunks
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
                "byte_offsets": False,
                "simprints": [
                    {"content": "Hello World", "offset": 0, "simprint": "82eJ2NG741E", "size": 11}
                ],
            }
        ],
    }


@patch("iscc_sct.code_semantic_text.embed_chunks")
def test_create_embedding(mock_embed, mock_embed_chunks):
    mock_embed.side_effect = mock_embed_chunks
    result = sct.create("Hello World", embedding=True)
    assert len(result.features[0].embedding) == 384
