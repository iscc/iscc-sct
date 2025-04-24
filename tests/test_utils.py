import pytest
import iscc_sct as sct
from iscc_sct import utils
from blake3 import blake3


def test_check_integrity(tmp_path):
    # Create a temporary file with known content
    file_path = tmp_path / "testfile.txt"
    content = "This is a test file."
    with open(file_path, "w") as f:
        f.write(content)

    # Generate a correct checksum and then alter it to simulate failure
    hasher = blake3()
    hasher.update(content.encode())
    correct_checksum = hasher.hexdigest()
    assert utils.check_integrity(file_path, correct_checksum) == file_path

    wrong_checksum = correct_checksum + "wrong"  # Deliberately incorrect checksum

    # Test the function with the wrong checksum
    with pytest.raises(RuntimeError) as exc_info:
        utils.check_integrity(file_path, wrong_checksum)

    # Check that the exception message contains expected text
    assert "Failed integrity check" in str(exc_info.value)


def test_hamming_distance_identical():
    a = b"abc"
    b = b"abc"
    assert utils.hamming_distance(a, b) == 0


def test_hamming_distance_different():
    a = b"abc"
    b = b"abd"
    assert utils.hamming_distance(a, b) == 3


def test_hamming_distance_completely_different():
    a = b"\x00"
    b = b"\xff"
    assert utils.hamming_distance(a, b) == 8


def test_hamming_distance_raises_value_error():
    a = b"abc"
    b = b"abcd"
    with pytest.raises(ValueError):
        utils.hamming_distance(a, b)


def test_encode_decode_base32():
    original = b"Hello, World!"
    encoded = utils.encode_base32(original)
    assert isinstance(encoded, str)
    assert encoded == "JBSWY3DPFQQFO33SNRSCC"
    decoded = utils.decode_base32(encoded)
    assert isinstance(decoded, bytes)
    assert decoded == original


def test_encode_decode_base64():
    original = b"Hello, World!"
    encoded = utils.encode_base64(original)
    assert isinstance(encoded, str)
    assert encoded == "SGVsbG8sIFdvcmxkIQ"
    decoded = utils.decode_base64(encoded)
    assert isinstance(decoded, bytes)
    assert decoded == original


def test_encode_decode_edge_cases():
    # Test empty input
    assert utils.encode_base32(b"") == ""
    assert utils.decode_base32("") == b""
    assert utils.encode_base64(b"") == ""
    assert utils.decode_base64("") == b""

    # Test input with padding
    original = b"a"
    assert utils.decode_base32(utils.encode_base32(original)) == original
    assert utils.decode_base64(utils.encode_base64(original)) == original


def test_iscc_distance_different_lengths():
    iscc1 = sct.create("Hello", bits=64).iscc
    iscc2 = sct.create("Hello", bits=96).iscc
    with pytest.raises(ValueError, match="The input ISCCs must have the same length"):
        utils.iscc_distance(iscc1, iscc2)


def test_cosine_similarity_identical():
    a = b"\x00\x00\x00\x00"
    b = b"\x00\x00\x00\x00"
    assert utils.cosine_similarity(a, b) == 100


def test_cosine_similarity_opposite():
    a = b"\x00\x00\x00\x00"
    b = b"\xff\xff\xff\xff"
    assert utils.cosine_similarity(a, b) == -100


def test_cosine_similarity_half_similar():
    a = b"\x00\x00\xff\xff"
    b = b"\x00\x00\x00\x00"
    assert utils.cosine_similarity(a, b) == 0


def test_cosine_similarity_quarter_similar():
    a = b"\x00\xff\xff\xff"
    b = b"\x00\x00\x00\x00"
    assert utils.cosine_similarity(a, b) == -50


def test_cosine_similarity_three_quarter_similar():
    a = b"\x00\x00\x00\xff"
    b = b"\x00\x00\x00\x00"
    assert utils.cosine_similarity(a, b) == 50


def test_cosine_similarity_different_lengths():
    a = b"\x00\x00\x00"
    b = b"\x00\x00\x00\x00"
    with pytest.raises(ValueError, match="The lengths of the two bytes objects must be the same"):
        utils.cosine_similarity(a, b)


def test_granular_similarity():
    from iscc_sct.models import Metadata, FeatureSet, Feature

    # Create two Metadata objects with some matching and non-matching simprints
    metadata_a = Metadata(
        iscc="ISCC:KACYPXW563EDNM",
        features=[
            FeatureSet(
                simprints=[
                    Feature(simprint="AAECAwQFBgc"),  # Will match
                    Feature(simprint="CAkKCwwNDg8"),  # Will not match
                ]
            )
        ],
    )

    metadata_b = Metadata(
        iscc="ISCC:KACYPXW563EDNM",
        features=[
            FeatureSet(
                simprints=[
                    Feature(simprint="AAECAwQFBgc"),  # Will match
                    Feature(simprint="EBESExQVFhc"),  # Will not match
                ]
            )
        ],
    )

    # Test with default threshold
    matches = utils.granular_similarity(metadata_a, metadata_b)
    assert len(matches) == 1
    assert matches[0][0].simprint == "AAECAwQFBgc"
    assert matches[0][1] == 100
    assert matches[0][2].simprint == "AAECAwQFBgc"

    # Test with lower threshold
    matches = utils.granular_similarity(metadata_a, metadata_b, threshold=0)
    assert len(matches) == 2  # All combinations should match

    # Test with higher threshold
    matches = utils.granular_similarity(metadata_a, metadata_b, threshold=101)
    assert len(matches) == 0  # No matches should be found


def test_granular_similarity_no_matches():
    from iscc_sct.models import Metadata, FeatureSet, Feature

    metadata_a = Metadata(
        iscc="ISCC:KACYPXW563EDNM",
        features=[FeatureSet(simprints=[Feature(simprint="AAECAwQFBgc")])],
    )

    metadata_b = Metadata(
        iscc="ISCC:KACYPXW563EDNM",
        features=[FeatureSet(simprints=[Feature(simprint="CAkKCwwNDg8")])],
    )

    matches = utils.granular_similarity(metadata_a, metadata_b)
    assert len(matches) == 0


def test_granular_similarity_multiple_matches():
    from iscc_sct.models import Metadata, FeatureSet, Feature

    metadata_a = Metadata(
        iscc="ISCC:KACYPXW563EDNM",
        features=[
            FeatureSet(
                simprints=[Feature(simprint="AAECAwQFBgc"), Feature(simprint="CAkKCwwNDg8")]
            ),
            FeatureSet(simprints=[Feature(simprint="EBESExQVFhc")]),
        ],
    )

    metadata_b = Metadata(
        iscc="ISCC:KACYPXW563EDNM",
        features=[
            FeatureSet(
                simprints=[Feature(simprint="AAECAwQFBgc"), Feature(simprint="GBkaGxwdHh8")]
            ),
            FeatureSet(simprints=[Feature(simprint="EBESExQVFhc")]),
        ],
    )

    matches = utils.granular_similarity(metadata_a, metadata_b)
    assert len(matches) == 2
    assert {(match[0].simprint, match[2].simprint) for match in matches} == {
        ("AAECAwQFBgc", "AAECAwQFBgc"),
        ("EBESExQVFhc", "EBESExQVFhc"),
    }


def test_char_to_byte_offsets():
    # This text is 21 unicode characters but 30 utf-8 bytes
    text = "Iñtërnâtiônàlizætiøn☃"

    # Test single position conversion
    assert utils.char_to_byte_offsets(text, [0]) == [0]  # First character
    assert utils.char_to_byte_offsets(text, [1]) == [1]  # 'ñ' starts at byte 1
    assert utils.char_to_byte_offsets(text, [3]) == [4]  # 'ë' starts at byte 4

    # Test multiple positions
    positions = [0, 2, 7, 10, 20]  # I, t, â, ô, ☃
    expected = [0, 3, 10, 14, 27]
    assert utils.char_to_byte_offsets(text, positions) == expected

    # Test out of order positions
    shuffled = [10, 2, 20, 0, 7]
    expected_shuffled = [14, 3, 27, 0, 10]
    assert utils.char_to_byte_offsets(text, shuffled) == expected_shuffled

    # Test edge cases
    assert utils.char_to_byte_offsets(text, []) == []  # Empty list
    assert utils.char_to_byte_offsets("", []) == []  # Empty text

    # Test with duplicate positions
    duplicates = [0, 7, 7, 10, 0]
    expected_duplicates = [0, 10, 10, 14, 0]
    assert utils.char_to_byte_offsets(text, duplicates) == expected_duplicates

    # Test with emoji (4-byte UTF-8 characters)
    emoji_text = "Hello 👋 world 🌍!"
    emoji_positions = [0, 6, 8, 14, 16]  # H, 👋, w, 🌍, !
    expected_emoji = [0, 6, 11, 17, 22]
    assert utils.char_to_byte_offsets(emoji_text, emoji_positions) == expected_emoji

    # Test all positions in a string
    simple = "abc"
    all_pos = list(range(len(simple)))
    assert utils.char_to_byte_offsets(simple, all_pos) == all_pos  # ASCII has 1:1 mapping

    # Test with mixed ASCII and non-ASCII
    mixed = "a¢€𐍈z"  # 1, 2, 3, 4-byte characters
    mixed_pos = list(range(len(mixed)))
    assert utils.char_to_byte_offsets(mixed, mixed_pos) == [0, 1, 3, 6, 10]


def test_test_char_to_byte_offsets_against_simple():
    text = "Iñtërnâtiônàlizætiøn☃"
    offsets = list(range(len(text)))
    assert utils.char_to_byte_offsets(text, offsets) == utils.char_to_byte_offsets_simple(
        text, offsets
    )
