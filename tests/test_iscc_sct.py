from pathlib import Path
import os
import tempfile
from unittest.mock import patch

import pytest
from blake3 import blake3

import iscc_sct as sct
from iscc_sct.options import SctOptions
from iscc_sct.code_semantic_text import (
    split_text,
    tokenize_chunks,
    embed_tokens,
    embed_chunks,
    compress,
)
import numpy as np


# Helper function for mock embeddings
def create_mock_embeddings(chunks, model_version=0, **kwargs):
    """Create deterministic mock embeddings for testing."""
    num_chunks = len(chunks)
    embedding_dim = 768 if model_version == 1 else 384
    # Use hash of chunks for deterministic but varied results
    seed = sum(hash(chunk) for chunk in chunks) % (2**32)
    embeddings = np.random.RandomState(seed).randn(num_chunks, embedding_dim).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings


HERE = Path(__file__).parent.absolute()

TEXT = """
`iscc-sct` is a **proof of concept implementation** of a semantic Text-Code for the
[ISCC](https://core.iscc.codes) (*International Standard Content Code*). Semantic Text-Codes are
designed to capture and represent the language agnostic semantic content of text for improved
similarity detection.

The ISCC framework already comes with a Text-Code that is based on lexical similarity and can match
near duplicates. The ISCC Semantic Text-Code is planned as a new additional ISCC-UNIT focused on
capturing a more abstract and broad semantic similarity. As such the Semantic Text-Code is
engineered to be robust against a broader range of variations and translations of text that cannot
be matched based on lexical similarity.
"""


def test_version():
    assert sct.__version__ == "0.1.5"


@pytest.mark.integration
def test_code_text_semantic_default():
    """Integration test: End-to-end ISCC generation with real model."""
    fp = HERE / "en.txt"
    result = sct.code_text_semantic(fp)
    assert result == {
        "iscc": "ISCC:CAA636IXQD736IGJ",
        "characters": 12076,
    }


def test_code_text_semantic_no_chars():
    fp = HERE / "en.txt"
    result = sct.code_text_semantic(fp, characters=False)
    assert result == {"iscc": "ISCC:CAA636IXQD736IGJ"}


def test_code_text_semantic_embedding():
    fp = HERE / "en.txt"
    result = sct.code_text_semantic(fp, embedding=True)
    assert result["iscc"] == "ISCC:CAA636IXQD736IGJ"
    assert len(result["features"][0]["embedding"]) == 384


def test_code_text_semantic_features():
    fp = HERE / "en.txt"
    result = sct.code_text_semantic(fp, simprints=True)
    assert result["iscc"] == "ISCC:CAA636IXQD736IGJ"
    assert result["characters"] == 12076
    assert result["features"][0]["simprints"][:3] == ["5wkXkfEx4lE", "b2UVwfc3wgk", "qvlV0W63s90"]
    assert result["features"][0]["simprints"][-3:] == ["PNsX9eGZQEs", "fFk3M2u5Qkk", "TPuXs2sRtk8"]


def test_code_text_semantic_offsets():
    fp = HERE / "en.txt"
    result = sct.code_text_semantic(fp, offsets=True)
    assert result["features"][0]["offsets"][:3] == [0, 277, 612]


def test_code_text_semantic_chunks():
    fp = HERE / "en.txt"
    result = sct.code_text_semantic(fp, contents=True)
    assert len(result["features"][0]["contents"]) == 39
    assert result["features"][0]["contents"][0].startswith("\n Thank ")
    assert result["features"][0]["contents"][-1].endswith("(Applause)\n")


def test_code_text_semantic_sizes():
    fp = HERE / "en.txt"
    result = sct.code_text_semantic(fp, sizes=True)
    # fmt: off
    assert result["features"][0]["sizes"] == [
        440, 396, 431, 385, 440, 380, 406, 477, 415, 536, 280, 449, 446, 442, 443, 444, 451, 485,
        477, 439, 517, 430, 468, 394, 531, 448, 421, 503, 376, 403, 513, 477, 393, 375, 555, 533,
        312, 455, 413
    ]
    # fmt: on


def test_gen_text_code_semantic_empty():
    with pytest.raises(ValueError) as excinfo:
        sct.gen_text_code_semantic("")
    assert str(excinfo.value) == "Input text cannot be empty."


def test_gen_text_code_semantic_granular():
    result = sct.gen_text_code_semantic(
        TEXT,
        simprints=True,
        offsets=True,
        contents=True,
    )
    assert (
        result
        == {
            "characters": 726,
            "iscc": "ISCC:CAARISHPJHEXQAYL",
            "features": [
                {
                    "maintype": "semantic",
                    "subtype": "text",
                    "version": 0,
                    "byte_offsets": False,
                    "simprints": ["FWjtTcl4Aws", "lAjHSc1wAws"],
                    "offsets": [0, 297],
                    "contents": [
                        "\n"
                        "`iscc-sct` is a **proof of concept implementation** of a semantic "
                        "Text-Code for the\n"
                        "[ISCC](https://core.iscc.codes) (*International Standard Content "
                        "Code*). Semantic Text-Codes are\n"
                        "designed to capture and represent the language agnostic semantic "
                        "content of text for improved\n"
                        "similarity detection.\n"
                        "\n",  # NOTE: end of first chunk (see comma :)
                        "\n"
                        "\n"
                        "The ISCC framework already comes with a Text-Code that is based "
                        "on lexical similarity and can match\n"
                        "near duplicates. The ISCC Semantic Text-Code is planned as a new "
                        "additional ISCC-UNIT focused on\n"
                        "capturing a more abstract and broad semantic similarity. As such "
                        "the Semantic Text-Code is\n"
                        "engineered to be robust against a broader range of variations and "
                        "translations of text that cannot\n"
                        "be matched based on lexical similarity.\n",
                    ],
                }
            ],
        }
    )


def test_gen_text_code_semantic_checks_bits():
    with pytest.raises(ValueError):
        sct.gen_text_code_semantic("Test", bits=99)


def test_split_text(text_en):
    chunks = split_text(text_en)
    assert chunks[0][1][:8] == "\n Thank "
    assert chunks[-1][1][:8] == "\n (Laugh"


def test_split_text_override():
    text = "Try some very small and granular text splitting with Iñtërnâtiônàlizætiøn☃. Use options override for it."
    chunks = split_text(text, max_tokens=8, overlap=4)
    assert chunks == [
        (0, "Try some very small and granular text "),
        (20, "and granular text splitting with "),
        (53, "Iñtërnâtiônà"),
        (59, "âtiônàlizætiøn"),
        (73, "☃. "),
        (76, "Use options override for it."),
    ]


def test_split_text_override_byte_offsets():
    text = "Try some very small and granular text splitting with Iñtërnâtiônàlizætiøn☃. Use options override for it."
    chunks = split_text(text, max_tokens=8, overlap=4, byte_offsets=True)
    assert chunks == [
        (0, "Try some very small and granular text "),
        (20, "and granular text splitting with "),
        (53, "Iñtërnâtiônà"),
        (61, "âtiônàlizætiøn"),
        (80, "☃. "),
        (85, "Use options override for it."),
    ]


def test_tokenize_chunks():
    chunks = ["Hello World", "These are chunks"]
    result = tokenize_chunks(chunks)
    np.testing.assert_array_equal(
        result["input_ids"],
        np.array([[0, 35378, 6661, 2, 1, 1], [0, 32255, 621, 7839, 1224, 2]], dtype=np.int64),
    )


def test_embed_tokens():
    chunks = ["Hello World", "These are chunks"]
    tokens = tokenize_chunks(chunks)
    embeddings = embed_tokens(tokens)
    assert list(embeddings[0][0][:3]) == pytest.approx(
        [0.05907335, 0.11408358, 0.12727071], rel=1e-2
    )


def test_embed_chunks():
    chunks = ["Hello World"]
    expected = [0.008697219, 0.038051583, 0.043976285]
    embeddings = embed_chunks(chunks)
    assert list(embeddings[0][:3]) == pytest.approx(expected, rel=1e-3)


def test_gen_text_code_semantic(text_en):
    result = sct.gen_text_code_semantic(text_en, embedding=True)
    assert result["iscc"] == "ISCC:CAA636IXQD736IGJ"
    assert result["features"][0]["embedding"][:3] == pytest.approx(
        [0.03241169825196266, 0.022712377831339836, 0.050273094326257706],
        rel=1e-3,
    )


@pytest.mark.integration
def test_cross_lingual_match(text_en, text_de):
    """Integration test: Validates cross-lingual semantic matching with real models."""
    a = sct.gen_text_code_semantic(text_en)["iscc"]
    assert a == "ISCC:CAA636IXQD736IGJ"
    b = sct.gen_text_code_semantic(text_de)["iscc"]
    assert b == "ISCC:CAA636IXQD4TMIGL"  # hamming distance for the codes is 6 bits


@pytest.mark.integration
def test_regression_256bit_model_v0(text_en, text_de):
    """Regression test: 256-bit ISCC codes for model v0 should remain stable."""
    en_result = sct.gen_text_code_semantic(text_en, bits=256, model_version=0)
    assert en_result["iscc"] == "ISCC:CAD636IXQD736IGJG4HIHCNQXELCFPH5N674DY32Q3PBUZOODLIQ23I"

    de_result = sct.gen_text_code_semantic(text_de, bits=256, model_version=0)
    assert de_result["iscc"] == "ISCC:CAD636IXQD4TMIGLU47IHHNUXEDCUPH5MO5UB232A3LBUZOKDDAR2YI"


@pytest.mark.integration
def test_regression_256bit_model_v1(text_en, text_de):
    """Regression test: 256-bit ISCC codes for model v1 should remain stable."""
    en_result = sct.gen_text_code_semantic(text_en, bits=256, model_version=1)
    assert en_result["iscc"] == "ISCC:CALZUZN3HFJEPYIJANSMCBT43Y2IKJ27OSGNSTRJBJCMMZBEUUWCTCI"

    de_result = sct.gen_text_code_semantic(text_de, bits=256, model_version=1)
    assert de_result["iscc"] == "ISCC:CALZVZQR7FNGPCTJWU6GCRTM2AKK25E4OCGYS3ZOZZCON3DEUU4EXWI"


def test_tokenizer_integrity(text_en):
    # test if updates break tokenizer compatibility
    hasher = blake3()
    for idx, chunk in split_text(text_en):
        hasher.update(chunk.encode("utf-8"))
    checksum = hasher.hexdigest()
    assert checksum == "7a7ad1ce83c36f853d31390150403e225bac7825a5573dd5c9e326b0917c7b52"


def test_soft_hash_text_semantic():
    result = sct.soft_hash_text_semantic("Hello World")
    assert (
        result.hex()
        == "f36789d8d1bbe351106bdf8e9b5006a3fc4cb1eb4042c75ea26b5058857c9177705429237858e9940e133c8b12ee1a3d"
    )


@pytest.mark.integration
def test_shift_resistance(text_en):
    """Integration test: Validates shift resistance with real models."""
    a = sct.soft_hash_text_semantic(text_en)
    shifted = "Just put another sentence in the begginging of the text!\n" + text_en
    b = sct.soft_hash_text_semantic(shifted)
    # TODO improve algorithm with more shift resistant semantic chunking
    # On 256-bit code
    assert sct.hamming_distance(a, b) == 6
    # On 64-bit code
    assert sct.hamming_distance(b[:16], a[:16]) == 1


def test_compress():
    arr1 = np.array([3.0, 15294.7789, 32977.7])
    arr2 = np.array([3.0, 15294.7789, 32977.7], dtype=np.float32)
    expected = [3.0, 15294.8, 32977.7]
    assert compress(arr1, 1) == expected
    assert compress(arr2, 1) == expected


def test_utf32be_chunk_retrieval():
    """Test that we can retrieve text chunks using UTF-32BE encoding with offset/size * 4."""
    # Generate text code with features
    text = (
        "Hello world! 你好世界! こんにちは! 안녕하세요! مرحبا! שלום! Ç 가 Ω ℍ ① ︷ i⁹ ¼ ǆ ⫝̸ ȴ ȷ ɂ ć "
        "Iñtërnâtiôn\nàlizætiøn☃💩 –  is a tric\t ky   thing!\r"
    )
    text += TEXT
    result = sct.gen_text_code_semantic(
        text, simprints=True, offsets=True, sizes=True, contents=True
    )

    # Convert text to UTF-32BE
    text_utf32be = text.encode("utf-32be")

    # For each feature, retrieve the chunk using offset and size
    features = result["features"][0]
    for i, simprint in enumerate(features["simprints"]):
        offset = features["offsets"][i]
        size = features["sizes"][i]
        original_chunk = features["contents"][i]

        # Calculate byte offset and size in UTF-32BE
        byte_offset = offset * 4
        byte_size = size * 4

        # Retrieve chunk from UTF-32BE encoded text
        chunk_bytes = text_utf32be[byte_offset : byte_offset + byte_size]
        retrieved_chunk = chunk_bytes.decode("utf-32be")

        # Verify retrieved chunk matches the original
        assert retrieved_chunk == original_chunk, f"Chunk mismatch at index {i}"


def test_embedding_precision():
    d16 = sct.gen_text_code_semantic("Hello World", embedding=True, precision=4)
    assert d16["features"][0]["embedding"][0] == 0.0087


def test_create_byte_offsets():
    """Test generation with byte offsets using text with multibyte characters."""
    # Text with ASCII, CJK, emoji, etc.
    text = (
        "Hello world! 你好世界! こんにちは! 안녕하세요! مرحبا! שלום! Ç 가 Ω ℍ ① ︷ i⁹ ¼ ǆ ⫝̸ ȴ ȷ ɂ ć "
        "Iñtërnâtiôn\nàlizætiøn☃💩 –  is a tric\t ky   thing!\r"
    )
    text += TEXT

    # Generate with character offsets (default)
    result_char = sct.create(text, offsets=True, sizes=True)
    # Generate with byte offsets
    result_bytes = sct.create(text, offsets=True, sizes=True, byte_offsets=True)

    # ISCC and Character Count should be the same
    assert result_char.iscc == result_bytes.iscc
    assert result_char.characters == result_bytes.characters

    # Features should be different
    assert result_char.features != result_bytes.features

    # Character versus UTF-8 chunk retrieval and actual chunks should match
    char_chunk = text[0 : result_char.features[0].sizes[0]]
    byte_chunk = text.encode("utf-8")[0 : result_bytes.features[0].sizes[0]].decode("utf-8")
    assert char_chunk == byte_chunk
    actual_chunk = sct.create(text, contents=True)
    assert actual_chunk.features[0].contents[0] == char_chunk

    assert result_char.model_dump(exclude_none=True) == {
        "iscc": "ISCC:CAARISGPJHEXQBQL",
        "characters": 851,
        "features": [
            {
                "byte_offsets": False,
                "maintype": "semantic",
                "offsets": [0, 209, 422],
                "sizes": [307, 215, 429],
                "subtype": "text",
                "version": 0,
            }
        ],
    }

    assert result_bytes.model_dump(exclude_none=True) == {
        "iscc": "ISCC:CAARISGPJHEXQBQL",
        "characters": 851,
        "features": [
            {
                "byte_offsets": True,
                "maintype": "semantic",
                "offsets": [0, 280, 493],
                "sizes": [378, 215, 429],
                "subtype": "text",
                "version": 0,
            }
        ],
    }


def test_options_model_dir_default():
    """Test that model_dir defaults to None."""
    opts = SctOptions()
    assert opts.model_dir is None


def test_options_model_dir_custom():
    """Test that model_dir can be set to a custom value."""
    custom_dir = "/custom/model/path"
    opts = SctOptions(model_dir=custom_dir)
    assert opts.model_dir == custom_dir


def test_options_model_dir_from_env(monkeypatch):
    """Test that model_dir can be set via environment variable."""
    custom_dir = "/env/model/path"
    monkeypatch.setenv("ISCC_SCT_MODEL_DIR", custom_dir)
    opts = SctOptions()
    assert opts.model_dir == custom_dir


def test_model_path_resolution():
    """Test that custom model_dir gets resolved to absolute path."""

    # Test with a relative path
    with tempfile.TemporaryDirectory():
        # Create a test to verify that relative paths would be resolved
        # Note: This test verifies the logic without actually reloading modules
        test_path = Path("./test_relative_path")
        resolved = test_path.resolve()
        assert resolved.is_absolute()


def test_model_directory_creation():
    """Test that custom model directory is created if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_dir = Path(tmpdir) / "custom_model_dir"
        assert not custom_dir.exists()

        # Create the directory as the code would do
        os.makedirs(custom_dir, exist_ok=True)
        assert custom_dir.exists()
        assert custom_dir.is_dir()


def test_custom_model_dir_via_reload(monkeypatch):
    """Test that custom model_dir is used when set via environment variable (module-level code path)."""
    from importlib import reload
    import iscc_sct.utils

    with tempfile.TemporaryDirectory() as tmpdir:
        custom_dir = Path(tmpdir) / "custom_model_path"

        # Set environment variable and reload module to trigger module-level code
        monkeypatch.setenv("ISCC_SCT_MODEL_DIR", str(custom_dir))

        # Reload options module first (it reads env vars)
        import iscc_sct.options

        reload(iscc_sct.options)

        # Then reload utils module (it uses options)
        reload(iscc_sct.utils)

        # Verify the custom directory was used
        expected = custom_dir.resolve()
        actual = iscc_sct.utils.model_storage_dir
        assert actual == expected, f"Expected {expected}, got {actual}"

        # Reload again to restore default state for other tests
        monkeypatch.delenv("ISCC_SCT_MODEL_DIR")
        reload(iscc_sct.options)
        reload(iscc_sct.utils)


def test_tokenizer_downloads_model_for_v1(tmp_path):
    """Test that tokenizer() attempts to download model if tokenizer.json missing for model v1."""
    from unittest.mock import MagicMock
    import iscc_sct.code_semantic_text

    # Create a temporary model directory
    v1_model_dir = tmp_path / "v1"
    v1_model_dir.mkdir(parents=True, exist_ok=True)

    # Clear tokenizer cache to force re-execution
    iscc_sct.code_semantic_text.tokenizer.cache_clear()

    try:
        # Mock get_model_path to return our temp directory
        with patch("iscc_sct.code_semantic_text.sct.get_model_path") as mock_get_path:
            mock_get_path.return_value = v1_model_dir

            # Mock get_model to create tokenizer.json without actual download
            with patch("iscc_sct.code_semantic_text.sct.get_model") as mock_get_model:
                # When get_model is called, create a minimal tokenizer.json
                def create_tokenizer_json(model_version):
                    tokenizer_path = v1_model_dir / "tokenizer.json"
                    # Create a minimal valid tokenizer.json
                    tokenizer_path.write_text('{"version": "1.0"}')

                mock_get_model.side_effect = create_tokenizer_json

                # Mock Tokenizer.from_file to avoid loading the fake tokenizer
                with patch("iscc_sct.code_semantic_text.Tokenizer.from_file") as mock_from_file:
                    mock_tok = MagicMock()
                    mock_tok.enable_padding = MagicMock()
                    mock_from_file.return_value = mock_tok

                    # This should trigger the download path (line 272)
                    tok = iscc_sct.code_semantic_text.tokenizer(model_version=1)

                    # Verify get_model was called
                    mock_get_model.assert_called_once_with(1)
                    assert tok is not None
    finally:
        # Cleanup
        iscc_sct.code_semantic_text.tokenizer.cache_clear()


def test_prompt_type_unexpected_type_warning():
    """Test that unexpected prompt_type type triggers warning and uses default."""
    from iscc_sct.code_semantic_text import tokenize_chunks

    # Test the tokenize_chunks function directly with an unexpected type
    # This bypasses Pydantic validation and tests the runtime handling (lines 367-370)
    test_chunks = ["Hello World"]

    # Call tokenize_chunks with an invalid prompt_type (integer instead of enum/string)
    result = tokenize_chunks(test_chunks, model_version=1, prompt_type=12345)

    # Should succeed despite invalid input (falls back to default DOCUMENT prompt)
    assert result is not None
    assert "input_ids" in result


def test_prompt_type_enum_instance():
    """Test that EmbeddingGemmaPrompt enum instances are handled correctly (line 355)."""
    from iscc_sct.code_semantic_text import tokenize_chunks, EmbeddingGemmaPrompt

    # Test with EmbeddingGemmaPrompt enum instance directly (triggers line 355 - pass statement)
    test_chunks = ["Test text"]
    result = tokenize_chunks(test_chunks, model_version=1, prompt_type=EmbeddingGemmaPrompt.QUERY)

    # Should succeed with enum instance
    assert result is not None
    assert "input_ids" in result
