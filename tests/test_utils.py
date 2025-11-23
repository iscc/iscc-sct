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


def test_download_file_corrupt_file_redownload(tmp_path, monkeypatch):
    # type: (object, object) -> None
    """Test that corrupt files are re-downloaded (lines 92-97)."""
    from blake3 import blake3

    dest_file = tmp_path / "test_model.bin"
    content = b"correct content"
    hasher = blake3()
    hasher.update(content)
    correct_checksum = hasher.hexdigest()

    # Create a corrupt file first
    corrupt_content = b"corrupt content"
    dest_file.write_bytes(corrupt_content)
    assert dest_file.exists()

    # Mock niquests.get to return correct content
    class MockResponse:
        def __init__(self, content):
            self.content = content
            self.headers = {"content-length": str(len(content))}

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield self.content

    def mock_get(url, stream=True, timeout=None):
        return MockResponse(content)

    monkeypatch.setattr("iscc_sct.utils.niquests.get", mock_get)

    # This should detect corruption and re-download
    result = utils.download_file(
        url="http://example.com/model.bin",
        dest_path=dest_file,
        checksum=correct_checksum,
        timeout=10,
    )

    # File should now have correct content
    assert result == dest_file
    assert dest_file.read_bytes() == content


def test_download_file_lock_timeout(tmp_path, monkeypatch):
    # type: (object, object) -> None
    """Test that lock timeout raises RuntimeError (lines 114-120)."""
    from filelock import FileLock

    dest_file = tmp_path / "test_model.bin"
    lock_path = dest_file.with_suffix(dest_file.suffix + ".lock")

    # Acquire the lock to simulate another process holding it
    lock = FileLock(lock_path, timeout=1)
    lock.acquire()

    try:
        # Attempt download with very short timeout
        with pytest.raises(RuntimeError) as excinfo:
            utils.download_file(
                url="http://example.com/model.bin",
                dest_path=dest_file,
                checksum="fake_checksum",
                timeout=0.1,  # Very short timeout to trigger immediately
            )

        error_msg = str(excinfo.value)
        assert "Timeout waiting for download lock" in error_msg
        assert "ISCC_SCT_DOWNLOAD_TIMEOUT" in error_msg
    finally:
        lock.release()


def test_download_file_temp_cleanup_on_failure(tmp_path, monkeypatch):
    # type: (object, object) -> None
    """Test that temp files are cleaned up on download failure (lines 108-110)."""

    dest_file = tmp_path / "test_model.bin"

    # Mock niquests.get to raise an exception
    def mock_get_fail(url, stream=True, timeout=None):
        raise Exception("Download failed")

    monkeypatch.setattr("iscc_sct.utils.niquests.get", mock_get_fail)

    # Download should fail, but temp file should be cleaned up
    with pytest.raises(Exception, match="Download failed"):
        utils.download_file(
            url="http://example.com/model.bin",
            dest_path=dest_file,
            checksum="fake_checksum",
            timeout=10,
        )

    # Temp file should not exist after cleanup
    import os

    temp_pattern = f".tmp.{os.getpid()}"
    temp_files = list(tmp_path.glob(f"*{temp_pattern}"))
    assert len(temp_files) == 0, f"Temp files not cleaned up: {temp_files}"


def test_download_file_with_progress(tmp_path, monkeypatch):
    # type: (object, object) -> None
    """Test download_file with progress tracking."""
    from blake3 import blake3

    dest_file = tmp_path / "test_model.bin"
    content = b"test model content for progress tracking"

    # Calculate correct checksum
    hasher = blake3()
    hasher.update(content)
    correct_checksum = hasher.hexdigest()

    # Mock niquests.get
    class MockResponse:
        def __init__(self, content):
            self.content = content
            self.headers = {"content-length": str(len(content))}

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            # Split content into chunks to test progress updates
            for i in range(0, len(self.content), chunk_size):
                yield self.content[i : i + chunk_size]

    def mock_get(url, stream=True, timeout=None):
        return MockResponse(content)

    monkeypatch.setattr("iscc_sct.utils.niquests.get", mock_get)

    # Mock progress object
    class MockProgress:
        def __init__(self):
            self.updates = []

        def update(self, task_id, total=None, advance=None, completed=None):
            self.updates.append({"total": total, "advance": advance, "completed": completed})

    progress = MockProgress()
    task_id = "test_task"

    # Download with progress tracking
    result = utils.download_file(
        url="http://example.com/model.bin",
        dest_path=dest_file,
        checksum=correct_checksum,
        timeout=10,
        progress=progress,
        task_id=task_id,
    )

    # Verify file was downloaded
    assert result == dest_file
    assert dest_file.read_bytes() == content

    # Verify progress was updated
    assert len(progress.updates) > 0
    # First update should set total size
    assert progress.updates[0]["total"] == len(content)
    # Subsequent updates should advance progress
    assert any(u["advance"] is not None for u in progress.updates)


def test_download_file_no_progress(tmp_path, monkeypatch):
    # type: (object, object) -> None
    """Test download_file without progress tracking."""
    from blake3 import blake3

    dest_file = tmp_path / "test_model.bin"
    content = b"test model content without progress"

    # Calculate correct checksum
    hasher = blake3()
    hasher.update(content)
    correct_checksum = hasher.hexdigest()

    # Mock niquests.get
    class MockResponse:
        def __init__(self, content):
            self.content = content
            self.headers = {"content-length": str(len(content))}

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield self.content

    def mock_get(url, stream=True, timeout=None):
        return MockResponse(content)

    monkeypatch.setattr("iscc_sct.utils.niquests.get", mock_get)

    # Download without progress tracking
    result = utils.download_file(
        url="http://example.com/model.bin",
        dest_path=dest_file,
        checksum=correct_checksum,
        timeout=10,
    )

    # Verify file was downloaded
    assert result == dest_file
    assert dest_file.read_bytes() == content


def test_download_file_integrity_check_fails_cleanup(tmp_path, monkeypatch):
    # type: (object, object) -> None
    """Test that temp file is cleaned up when integrity check fails."""

    dest_file = tmp_path / "test_model.bin"
    content = b"test content"

    # Mock niquests.get
    class MockResponse:
        def __init__(self, content):
            self.content = content
            self.headers = {"content-length": str(len(content))}

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield self.content

    def mock_get(url, stream=True, timeout=None):
        return MockResponse(content)

    monkeypatch.setattr("iscc_sct.utils.niquests.get", mock_get)

    # Mock check_integrity to fail
    def mock_check_integrity(file_path, checksum):
        raise RuntimeError("Bad checksum")

    monkeypatch.setattr("iscc_sct.utils.check_integrity", mock_check_integrity)

    # Download should fail due to bad checksum
    with pytest.raises(RuntimeError, match="Bad checksum"):
        utils.download_file(
            url="http://example.com/model.bin",
            dest_path=dest_file,
            checksum="fake_checksum",
            timeout=10,
        )

    # Temp file should be cleaned up
    import os

    temp_pattern = f".tmp.{os.getpid()}"
    temp_files = list(tmp_path.glob(f"*{temp_pattern}"))
    assert len(temp_files) == 0, f"Temp file not cleaned up: {temp_files}"
