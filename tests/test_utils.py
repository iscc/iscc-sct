import pytest
from iscc_sct import utils
from blake3 import blake3


def test_check_integrity_failure(tmp_path):
    # Create a temporary file with known content
    file_path = tmp_path / "testfile.txt"
    content = "This is a test file."
    with open(file_path, "w") as f:
        f.write(content)

    # Generate a correct checksum and then alter it to simulate failure
    hasher = blake3()
    hasher.update(content.encode())
    correct_checksum = hasher.hexdigest()
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
    assert utils.hamming_distance(a, b) == 2

def test_hamming_distance_completely_different():
    a = b"\x00"
    b = b"\xff"
    assert utils.hamming_distance(a, b) == 8

def test_hamming_distance_raises_value_error():
    a = b"abc"
    b = b"abcd"
    with pytest.raises(ValueError):
        utils.hamming_distance(a, b)
