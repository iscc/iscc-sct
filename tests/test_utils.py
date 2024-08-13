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
