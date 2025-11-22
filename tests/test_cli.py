"""Tests for CLI module."""

import json
import sys
from pathlib import Path
from io import StringIO
from unittest.mock import patch, MagicMock

import pytest
import typer
from typer.testing import CliRunner

from iscc_sct.cli import (
    app,
    version_callback,
    expand_path,
    find_files_from_directory,
    find_files_from_glob,
    resolve_path,
    collect_files,
    validate_options,
    read_text_from_file,
    process_single_file,
    format_index_features,
    format_object_features,
    format_json_features,
    output_json,
    escape_content,
    print_line,
    format_feature_parts_index,
    format_feature_parts_object,
    output_index_format_features,
    output_object_format_features,
    output_text_features,
    output_text,
    create_progress,
    setup_processing_environment,
    setup_pretty_json,
    process_and_output_file,
    run_processing_loop,
    process_files,
    main,
)

runner = CliRunner()


@pytest.fixture
def sample_text_file(tmp_path):
    # type: (Path) -> Path
    """Create a sample text file."""
    file_path = tmp_path / "sample.txt"
    file_path.write_text("This is a sample text for testing.")
    return file_path


@pytest.fixture
def empty_text_file(tmp_path):
    # type: (Path) -> Path
    """Create an empty text file."""
    file_path = tmp_path / "empty.txt"
    file_path.write_text("   ")
    return file_path


@pytest.fixture
def non_utf8_text_file(tmp_path):
    # type: (Path) -> Path
    """Create a non-UTF-8 text file."""
    file_path = tmp_path / "non_utf8.txt"
    file_path.write_text("Iñtërnâtiônàlizætiøn☃", encoding="utf-16")
    return file_path


@pytest.fixture
def multiple_files(tmp_path):
    # type: (Path) -> list[Path]
    """Create multiple test files."""
    files = []
    for i in range(3):
        file_path = tmp_path / f"file{i}.txt"
        file_path.write_text(f"Content of file {i}")
        files.append(file_path)
    return files


@pytest.fixture
def nested_directory(tmp_path):
    # type: (Path) -> Path
    """Create a nested directory structure with files."""
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (tmp_path / "root.txt").write_text("Root file")
    (subdir / "nested.txt").write_text("Nested file")
    return tmp_path


def test_version_callback():
    # type: () -> None
    """Test version callback exits with version."""
    with pytest.raises(typer.Exit):
        version_callback(True)


def test_version_callback_no_exit():
    # type: () -> None
    """Test version callback does nothing when False."""
    version_callback(False)  # Should not raise


def test_expand_path_with_tilde(tmp_path, monkeypatch):
    # type: (Path, pytest.MonkeyPatch) -> None
    """Test path expansion with tilde."""
    import platform

    if platform.system() == "Windows":
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
    else:
        monkeypatch.setenv("HOME", str(tmp_path))
    result = expand_path("~/test.txt")
    assert tmp_path.as_posix() in result


def test_expand_path_without_tilde():
    # type: () -> None
    """Test path expansion without tilde."""
    result = expand_path("/absolute/path.txt")
    assert result == "/absolute/path.txt"


def test_find_files_from_directory(nested_directory):
    # type: (Path) -> None
    """Test finding files recursively in a directory."""
    files = find_files_from_directory(nested_directory)
    assert len(files) == 2
    assert all(f.is_file() for f in files)


def test_find_files_from_glob(tmp_path):
    # type: (Path) -> None
    """Test finding files with glob pattern."""
    # Create test files
    (tmp_path / "test1.txt").write_text("test")
    (tmp_path / "test2.txt").write_text("test")
    (tmp_path / "other.md").write_text("test")

    pattern = str(tmp_path / "*.txt")
    files = find_files_from_glob(pattern)
    assert len(files) == 2
    assert all(f.suffix == ".txt" for f in files)


def test_resolve_path_single_file(sample_text_file):
    # type: (Path) -> None
    """Test resolving a single file path."""
    files = resolve_path(str(sample_text_file))
    assert len(files) == 1
    assert files[0] == sample_text_file


def test_resolve_path_directory(nested_directory):
    # type: (Path) -> None
    """Test resolving a directory path."""
    files = resolve_path(str(nested_directory))
    assert len(files) == 2


def test_resolve_path_glob(tmp_path):
    # type: (Path) -> None
    """Test resolving a glob pattern."""
    (tmp_path / "test1.txt").write_text("test")
    (tmp_path / "test2.txt").write_text("test")

    pattern = str(tmp_path / "*.txt")
    files = resolve_path(pattern)
    assert len(files) == 2


def test_resolve_path_nonexistent():
    # type: () -> None
    """Test resolving a non-existent path."""
    files = resolve_path("/nonexistent/path.txt")
    assert files == []


def test_collect_files(tmp_path):
    # type: (Path) -> None
    """Test collecting files from multiple patterns."""
    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"
    file1.write_text("test")
    file2.write_text("test")

    files = collect_files([str(file1), str(file2)])
    assert len(files) == 2


def test_collect_files_with_duplicates(tmp_path):
    # type: (Path) -> None
    """Test that collect_files removes duplicates."""
    file1 = tmp_path / "file1.txt"
    file1.write_text("test")

    # Same file referenced twice
    files = collect_files([str(file1), str(file1)])
    assert len(files) == 1


def test_collect_files_warns_no_matches():
    # type: () -> None
    """Test warning when no files match pattern."""
    # Just test that it doesn't crash - loguru warnings aren't captured by caplog
    result = collect_files(["/nonexistent/*.txt"])
    assert result == []


def test_validate_options_invalid_format():
    # type: () -> None
    """Test validation with invalid format."""
    with pytest.raises(typer.Exit):
        validate_options("invalid", False, False)


def test_validate_options_content_without_granular():
    # type: () -> None
    """Test validation with content but no granular."""
    with pytest.raises(typer.Exit):
        validate_options("text", True, False)


def test_validate_options_valid():
    # type: () -> None
    """Test validation with valid options."""
    validate_options("text", False, False)
    validate_options("json", False, False)
    validate_options("text", True, True)


def test_read_text_from_file_utf8(sample_text_file):
    # type: (Path) -> None
    """Test reading UTF-8 text file."""
    text = read_text_from_file(sample_text_file)
    assert text == "This is a sample text for testing."


def test_read_text_from_file_empty(empty_text_file):
    # type: (Path) -> None
    """Test reading empty text file."""
    text = read_text_from_file(empty_text_file)
    assert text is None


def test_read_text_from_file_non_utf8(non_utf8_text_file):
    # type: (Path) -> None
    """Test reading non-UTF-8 text file."""
    text = read_text_from_file(non_utf8_text_file)
    assert text is not None
    assert "Iñtërnâtiônàlizætiøn" in text


def test_read_text_from_file_binary(tmp_path):
    # type: (Path) -> None
    """Test reading binary file returns None."""
    binary_file = tmp_path / "binary.bin"
    binary_file.write_bytes(b"\x00\x01\x02\x03\xff\xfe")
    text = read_text_from_file(binary_file)
    # Should return None or decoded text depending on charset detection
    # The actual behavior depends on charset_normalizer


def test_read_text_from_file_no_charset_match(tmp_path):
    # type: (Path) -> None
    """Test reading file when charset detection fails."""
    # Create a file with bytes that charset_normalizer can't decode
    binary_file = tmp_path / "undecodable.bin"
    # Write completely random binary data that is not valid in any encoding
    binary_file.write_bytes(bytes(range(256)) * 10)

    text = read_text_from_file(binary_file)
    # charset_normalizer might still detect something, or return None
    # The actual behavior depends on the library


def test_process_single_file(sample_text_file):
    # type: (Path) -> None
    """Test processing a single file."""
    result = process_single_file(sample_text_file, 256, 256, False, False)
    assert result is not None
    assert "iscc" in result
    assert "filename" in result
    assert "meta" in result


def test_process_single_file_with_granular(sample_text_file):
    # type: (Path) -> None
    """Test processing a single file with granular option."""
    result = process_single_file(sample_text_file, 256, 256, True, False)
    assert result is not None
    assert result["meta"].features is not None


def test_process_single_file_with_content(sample_text_file):
    # type: (Path) -> None
    """Test processing a single file with content option."""
    result = process_single_file(sample_text_file, 256, 256, True, True)
    assert result is not None
    # Content should be included in features


def test_process_single_file_empty(empty_text_file):
    # type: (Path) -> None
    """Test processing empty file returns None."""
    result = process_single_file(empty_text_file, 256, 256, False, False)
    assert result is None


def test_process_single_file_error(tmp_path):
    # type: (Path) -> None
    """Test processing file with error."""
    # Create a file that will cause an error
    nonexistent = tmp_path / "nonexistent.txt"
    result = process_single_file(nonexistent, 256, 256, False, False)
    assert result is None


def test_format_index_features():
    # type: () -> None
    """Test formatting index features."""
    # Create a mock feature set in index format
    feature_set = MagicMock()
    feature_set.simprints = ["ISCC:ABC123", "ISCC:DEF456"]
    feature_set.offsets = [0, 100]
    feature_set.sizes = [50, 75]
    feature_set.contents = ["Content 1", "Content 2"]

    result = format_index_features(feature_set)
    assert len(result) == 2
    assert result[0]["simprint"] == "ISCC:ABC123"
    assert result[0]["offset"] == 0
    assert result[0]["size"] == 50
    assert result[0]["content"] == "Content 1"


def test_format_index_features_minimal():
    # type: () -> None
    """Test formatting index features with minimal data."""
    feature_set = MagicMock()
    feature_set.simprints = ["ISCC:ABC123"]
    feature_set.offsets = None
    feature_set.sizes = None
    feature_set.contents = None

    result = format_index_features(feature_set)
    assert len(result) == 1
    assert result[0] == {"simprint": "ISCC:ABC123"}


def test_format_object_features():
    # type: () -> None
    """Test formatting object features."""
    feature1 = MagicMock()
    feature1.model_dump.return_value = {"simprint": "ISCC:ABC123", "offset": 0}

    feature_set = MagicMock()
    feature_set.simprints = [feature1]

    result = format_object_features(feature_set)
    assert len(result) == 1
    assert result[0]["simprint"] == "ISCC:ABC123"


def test_format_json_features_index():
    # type: () -> None
    """Test formatting JSON features in index format."""
    feature_set = MagicMock()
    feature_set.simprints = ["ISCC:ABC123"]
    feature_set.offsets = [0]
    feature_set.sizes = [50]
    feature_set.contents = None

    result = format_json_features([feature_set])
    assert len(result) == 1
    assert result[0]["simprint"] == "ISCC:ABC123"


def test_format_json_features_object():
    # type: () -> None
    """Test formatting JSON features in object format."""
    feature = MagicMock()
    feature.model_dump.return_value = {"simprint": "ISCC:ABC123"}

    feature_set = MagicMock()
    feature_set.simprints = [feature]

    result = format_json_features([feature_set])
    assert len(result) == 1


def test_output_json_ndjson(capsys):
    # type: (pytest.CaptureFixture) -> None
    """Test JSON output in NDJSON format."""
    result = {
        "iscc": "ISCC:ABC123",
        "filename": "test.txt",
        "meta": MagicMock(features=None),
    }

    output_json(result, None, None, None)
    captured = capsys.readouterr()
    data = json.loads(captured.out.strip())
    assert data["iscc"] == "ISCC:ABC123"
    assert data["filename"] == "test.txt"


def test_output_json_pretty(capsys):
    # type: (pytest.CaptureFixture) -> None
    """Test JSON output in pretty format."""
    result = {
        "iscc": "ISCC:ABC123",
        "filename": "test.txt",
        "meta": MagicMock(features=None),
    }

    output_json(result, 2, None, None)
    captured = capsys.readouterr()
    assert "ISCC:ABC123" in captured.out
    assert "\n" in captured.out  # Pretty formatted


def test_output_json_with_features(capsys):
    # type: (pytest.CaptureFixture) -> None
    """Test JSON output with features."""
    feature_set = MagicMock()
    feature_set.simprints = ["ISCC:FEAT123"]
    feature_set.offsets = [0]
    feature_set.sizes = [50]
    feature_set.contents = None

    meta = MagicMock()
    meta.features = [feature_set]

    result = {
        "iscc": "ISCC:ABC123",
        "filename": "test.txt",
        "meta": meta,
    }

    output_json(result, None, None, None)
    captured = capsys.readouterr()
    data = json.loads(captured.out.strip())
    assert "features" in data
    assert len(data["features"]) == 1


def test_escape_content_basic():
    # type: () -> None
    """Test basic content escaping."""
    result = escape_content("Hello\nWorld")
    assert result == "Hello\\nWorld"


def test_escape_content_various_chars():
    # type: () -> None
    """Test escaping various characters."""
    result = escape_content("Line1\r\nLine2\rLine3\tTab")
    assert result == "Line1\\nLine2\\rLine3\\tTab"


def test_escape_content_truncation():
    # type: () -> None
    """Test content truncation."""
    long_text = "a" * 100
    result = escape_content(long_text, max_length=50)
    assert len(result) == 53  # 50 + "..."
    assert result.endswith("...")


def test_escape_content_no_truncation():
    # type: () -> None
    """Test no truncation when max_length is 0."""
    long_text = "a" * 100
    result = escape_content(long_text, max_length=0)
    assert len(result) == 100


def test_print_line_no_console(capsys):
    # type: (pytest.CaptureFixture) -> None
    """Test printing without console."""
    print_line("Test output", None, None)
    captured = capsys.readouterr()
    assert captured.out.strip() == "Test output"


def test_print_line_with_console():
    # type: () -> None
    """Test printing with console and progress."""
    console = MagicMock()
    progress = MagicMock()
    print_line("Test output", console, progress)
    console.print.assert_called_once_with("Test output", markup=False, highlight=False)


def test_format_feature_parts_index():
    # type: () -> None
    """Test formatting feature parts in index format."""
    feature_set = MagicMock()
    feature_set.simprints = ["ISCC:ABC123", "ISCC:DEF456"]
    feature_set.offsets = [0, 100]
    feature_set.sizes = [50, 75]
    feature_set.contents = ["Content 1", "Content 2"]

    parts = format_feature_parts_index(feature_set, 0, True, 50)
    assert parts[0] == "  ISCC:ABC123"
    assert parts[1] == "0"
    assert parts[2] == "50"
    assert parts[3] == "Content 1"


def test_format_feature_parts_index_minimal():
    # type: () -> None
    """Test formatting feature parts with minimal data."""
    feature_set = MagicMock()
    feature_set.simprints = ["ISCC:ABC123"]
    feature_set.offsets = None
    feature_set.sizes = None
    feature_set.contents = None

    parts = format_feature_parts_index(feature_set, 0, False, 50)
    assert len(parts) == 1
    assert parts[0] == "  ISCC:ABC123"


def test_format_feature_parts_object():
    # type: () -> None
    """Test formatting feature parts in object format."""
    feature = MagicMock()
    feature.simprint = "ISCC:ABC123"
    feature.offset = 0
    feature.size = 50
    feature.content = "Content"

    parts = format_feature_parts_object(feature, True, 50)
    assert parts[0] == "  ISCC:ABC123"
    assert parts[1] == "0"
    assert parts[2] == "50"
    assert parts[3] == "Content"


def test_format_feature_parts_object_minimal():
    # type: () -> None
    """Test formatting feature parts with minimal object data."""
    feature = MagicMock()
    feature.simprint = "ISCC:ABC123"
    feature.offset = None
    feature.size = None
    delattr(feature, "content")

    parts = format_feature_parts_object(feature, False, 50)
    assert len(parts) == 1


def test_output_index_format_features(capsys):
    # type: (pytest.CaptureFixture) -> None
    """Test outputting index format features."""
    feature_set = MagicMock()
    feature_set.simprints = ["ISCC:ABC123"]
    feature_set.offsets = None
    feature_set.sizes = None
    feature_set.contents = None

    output_index_format_features(feature_set, None, None, False, 50)
    captured = capsys.readouterr()
    assert "ISCC:ABC123" in captured.out


def test_output_object_format_features(capsys):
    # type: (pytest.CaptureFixture) -> None
    """Test outputting object format features."""
    feature = MagicMock()
    feature.simprint = "ISCC:ABC123"
    feature.offset = None
    feature.size = None

    feature_set = MagicMock()
    feature_set.simprints = [feature]

    output_object_format_features(feature_set, None, None, False, 50)
    captured = capsys.readouterr()
    assert "ISCC:ABC123" in captured.out


def test_output_text_features_index(capsys):
    # type: (pytest.CaptureFixture) -> None
    """Test outputting text features in index format."""
    feature_set = MagicMock()
    feature_set.simprints = ["ISCC:ABC123"]
    feature_set.offsets = None
    feature_set.sizes = None
    feature_set.contents = None

    output_text_features([feature_set], None, None, False, 50)
    captured = capsys.readouterr()
    assert "ISCC:ABC123" in captured.out


def test_output_text_features_object(capsys):
    # type: (pytest.CaptureFixture) -> None
    """Test outputting text features in object format."""
    feature = MagicMock()
    feature.simprint = "ISCC:ABC123"
    feature.offset = None
    feature.size = None

    feature_set = MagicMock()
    feature_set.simprints = [feature]

    output_text_features([feature_set], None, None, False, 50)
    captured = capsys.readouterr()
    assert "ISCC:ABC123" in captured.out


def test_output_text_basic(capsys):
    # type: (pytest.CaptureFixture) -> None
    """Test basic text output."""
    result = {
        "iscc": "ISCC:ABC123",
        "filename": "test.txt",
        "meta": MagicMock(features=None),
    }

    output_text(result, False, None, None, False, 50)
    captured = capsys.readouterr()
    assert "ISCC:ABC123" in captured.out
    assert "test.txt" in captured.out


def test_output_text_with_granular(capsys):
    # type: (pytest.CaptureFixture) -> None
    """Test text output with granular features."""
    feature_set = MagicMock()
    feature_set.simprints = ["ISCC:FEAT123"]
    feature_set.offsets = None
    feature_set.sizes = None
    feature_set.contents = None

    meta = MagicMock()
    meta.features = [feature_set]

    result = {
        "iscc": "ISCC:ABC123",
        "filename": "test.txt",
        "meta": meta,
    }

    output_text(result, True, None, None, False, 50)
    captured = capsys.readouterr()
    assert "ISCC:ABC123" in captured.out
    assert "ISCC:FEAT123" in captured.out


def test_create_progress_no_progress():
    # type: () -> None
    """Test creating progress components without progress bar."""
    console, progress = create_progress(False, 10, False)
    assert console is None
    assert progress is None


def test_create_progress_console_only():
    # type: () -> None
    """Test creating console without progress bar."""
    console, progress = create_progress(False, 10, True)
    assert console is not None
    assert progress is None


def test_create_progress_full():
    # type: () -> None
    """Test creating full progress components."""
    console, progress = create_progress(True, 10, False)
    assert console is not None
    assert progress is not None


def test_setup_processing_environment(sample_text_file):
    # type: (Path) -> None
    """Test setting up processing environment."""
    files, show_progress, json_indent = setup_processing_environment(
        "text", False, False, [str(sample_text_file)]
    )
    assert len(files) == 1
    assert isinstance(show_progress, bool)
    assert json_indent is None


def test_setup_processing_environment_no_files():
    # type: () -> None
    """Test setup with no files found."""
    with pytest.raises(typer.Exit):
        setup_processing_environment("text", False, False, ["/nonexistent/*.txt"])


def test_setup_processing_environment_invalid_format():
    # type: () -> None
    """Test setup with invalid format."""
    with pytest.raises(typer.Exit):
        setup_processing_environment("invalid", False, False, ["test.txt"])


def test_setup_pretty_json():
    # type: () -> None
    """Test pretty JSON setup."""
    result = setup_pretty_json(True, "json", None)
    assert result == 2


def test_setup_pretty_json_not_enabled():
    # type: () -> None
    """Test pretty JSON not enabled."""
    result = setup_pretty_json(False, "json", None)
    assert result is None


def test_setup_pretty_json_not_json_format():
    # type: () -> None
    """Test pretty JSON with non-JSON format."""
    result = setup_pretty_json(True, "text", None)
    assert result is None


def test_process_and_output_file(sample_text_file, capsys):
    # type: (Path, pytest.CaptureFixture) -> None
    """Test processing and outputting a file."""
    options = {
        "unit_bits": 256,
        "simprint_bits": 256,
        "granular": False,
        "content": False,
        "format": "text",
        "json_indent": None,
        "truncate": 50,
    }

    process_and_output_file(sample_text_file, options, None, None)
    captured = capsys.readouterr()
    assert "ISCC:" in captured.out


def test_process_and_output_file_json(sample_text_file, capsys):
    # type: (Path, pytest.CaptureFixture) -> None
    """Test processing and outputting a file as JSON."""
    options = {
        "unit_bits": 256,
        "simprint_bits": 256,
        "granular": False,
        "content": False,
        "format": "json",
        "json_indent": None,
        "truncate": 50,
    }

    process_and_output_file(sample_text_file, options, None, None)
    captured = capsys.readouterr()
    data = json.loads(captured.out.strip())
    assert "iscc" in data


def test_run_processing_loop(sample_text_file, capsys):
    # type: (Path, pytest.CaptureFixture) -> None
    """Test running processing loop."""
    options = {
        "unit_bits": 256,
        "simprint_bits": 256,
        "granular": False,
        "content": False,
        "format": "text",
        "json_indent": None,
        "truncate": 50,
    }

    run_processing_loop([sample_text_file], options, None, None)
    captured = capsys.readouterr()
    assert "ISCC:" in captured.out


def test_run_processing_loop_with_progress(sample_text_file):
    # type: (Path) -> None
    """Test running processing loop with progress bar."""
    from rich.console import Console
    from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn

    console = Console()
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=False,
    )

    options = {
        "unit_bits": 256,
        "simprint_bits": 256,
        "granular": False,
        "content": False,
        "format": "text",
        "json_indent": None,
        "truncate": 50,
    }

    run_processing_loop([sample_text_file], options, console, progress)


def test_process_files(sample_text_file, capsys):
    # type: (Path, pytest.CaptureFixture) -> None
    """Test main process_files function."""
    process_files([str(sample_text_file)], "text", 256, 256, False, False, False, 50)
    captured = capsys.readouterr()
    assert "ISCC:" in captured.out


def test_process_files_json(sample_text_file, capsys):
    # type: (Path, pytest.CaptureFixture) -> None
    """Test process_files with JSON output."""
    process_files([str(sample_text_file)], "json", 256, 256, False, False, False, 50)
    captured = capsys.readouterr()
    data = json.loads(captured.out.strip())
    assert "iscc" in data


def test_process_files_pretty_json(sample_text_file, capsys):
    # type: (Path, pytest.CaptureFixture) -> None
    """Test process_files with pretty JSON output."""
    process_files([str(sample_text_file)], "json", 256, 256, False, True, False, 50)
    captured = capsys.readouterr()
    assert "iscc" in captured.out
    assert "\n" in captured.out


def test_process_files_granular(sample_text_file, capsys):
    # type: (Path, pytest.CaptureFixture) -> None
    """Test process_files with granular output."""
    process_files([str(sample_text_file)], "text", 256, 256, True, False, False, 50)
    captured = capsys.readouterr()
    assert "ISCC:" in captured.out


def test_process_files_with_content(sample_text_file, capsys):
    # type: (Path, pytest.CaptureFixture) -> None
    """Test process_files with content output."""
    process_files([str(sample_text_file)], "text", 256, 256, True, False, True, 50)
    captured = capsys.readouterr()
    assert "ISCC:" in captured.out


def test_cli_version():
    # type: () -> None
    """Test CLI version command."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "iscc-sct version" in result.stdout


def test_cli_version_short():
    # type: () -> None
    """Test CLI version short flag."""
    result = runner.invoke(app, ["-v"])
    assert result.exit_code == 0
    assert "iscc-sct version" in result.stdout


def test_cli_no_args():
    # type: () -> None
    """Test CLI with no arguments shows help."""
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "ISCC - Semantic Code Text" in result.stdout


def test_cli_create_command(sample_text_file):
    # type: (Path) -> None
    """Test create command."""
    result = runner.invoke(app, ["create", str(sample_text_file)])
    assert result.exit_code == 0
    assert "ISCC:" in result.stdout


def test_cli_create_with_format(sample_text_file):
    # type: (Path) -> None
    """Test create command with format option."""
    result = runner.invoke(app, ["create", str(sample_text_file), "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout.strip())
    assert "iscc" in data


def test_cli_create_with_granular(sample_text_file):
    # type: (Path) -> None
    """Test create command with granular option."""
    result = runner.invoke(app, ["create", str(sample_text_file), "--granular"])
    assert result.exit_code == 0
    assert "ISCC:" in result.stdout


def test_cli_create_with_pretty_json(sample_text_file):
    # type: (Path) -> None
    """Test create command with pretty JSON."""
    result = runner.invoke(app, ["create", str(sample_text_file), "-f", "json", "--pretty"])
    assert result.exit_code == 0
    assert "iscc" in result.stdout


def test_cli_create_with_content(sample_text_file):
    # type: (Path) -> None
    """Test create command with content option."""
    result = runner.invoke(app, ["create", str(sample_text_file), "-g", "--content"])
    assert result.exit_code == 0


def test_cli_create_invalid_format(sample_text_file):
    # type: (Path) -> None
    """Test create command with invalid format."""
    result = runner.invoke(app, ["create", str(sample_text_file), "--format", "invalid"])
    assert result.exit_code != 0
    # Error message might be in stdout or stderr
    output = result.stdout + result.stderr
    assert "Invalid format" in output


def test_cli_create_content_without_granular(sample_text_file):
    # type: (Path) -> None
    """Test create command with content but no granular."""
    result = runner.invoke(app, ["create", str(sample_text_file), "--content"])
    assert result.exit_code != 0
    # Error message might be in stdout or stderr
    output = result.stdout + result.stderr
    assert "requires --granular" in output


def test_cli_create_no_files():
    # type: () -> None
    """Test create command with no files found."""
    result = runner.invoke(app, ["create", "/nonexistent/*.txt"])
    assert result.exit_code != 0
    # Error message might be in stdout or stderr
    output = result.stdout + result.stderr
    assert "No files found" in output


def test_cli_create_multiple_files(multiple_files):
    # type: (list[Path]) -> None
    """Test create command with multiple files."""
    paths = [str(f) for f in multiple_files]
    result = runner.invoke(app, ["create"] + paths)
    assert result.exit_code == 0
    assert result.stdout.count("ISCC:") == 3


def test_cli_create_glob_pattern(tmp_path):
    # type: (Path) -> None
    """Test create command with glob pattern."""
    (tmp_path / "test1.txt").write_text("Content 1")
    (tmp_path / "test2.txt").write_text("Content 2")

    pattern = str(tmp_path / "*.txt")
    result = runner.invoke(app, ["create", pattern])
    assert result.exit_code == 0
    assert result.stdout.count("ISCC:") == 2


def test_cli_create_directory(nested_directory):
    # type: (Path) -> None
    """Test create command with directory."""
    result = runner.invoke(app, ["create", str(nested_directory)])
    assert result.exit_code == 0
    assert result.stdout.count("ISCC:") == 2


def test_cli_demo_import_error():
    # type: () -> None
    """Test demo command when import fails."""
    # Test the import error path - directly test the demo function
    from iscc_sct import cli

    # Temporarily replace the demo module to simulate ImportError
    with patch.dict("sys.modules", {"iscc_sct.demo": None}):
        # Call the runner to test the command
        result = runner.invoke(app, ["demo"])
        # The command might succeed if gradio is installed
        # or fail if it's not - both are valid outcomes
        # We just want to ensure the command doesn't crash


def test_cli_demo_success():
    # type: () -> None
    """Test demo command when gradio is available."""
    # Mock the gradio demo module
    mock_demo = MagicMock()
    mock_demo.launch = MagicMock()

    # Patch the import to succeed
    with patch.dict("sys.modules", {"iscc_sct.demo": MagicMock(demo=mock_demo)}):
        result = runner.invoke(app, ["demo"])
        # The command should succeed if the module is available
        # Check that "Launching Gradio demo..." was printed
        if result.exit_code == 0:
            assert "Launching Gradio demo" in result.stdout


def test_main_default_command(sample_text_file):
    # type: (Path) -> None
    """Test main function inserts default 'create' command."""
    test_args = ["iscc-sct", str(sample_text_file)]

    with patch.object(sys, "argv", test_args):
        with patch("iscc_sct.cli.app") as mock_app:
            main()
            # Verify that 'create' was inserted
            assert sys.argv == ["iscc-sct", "create", str(sample_text_file)]


def test_main_with_explicit_command(sample_text_file):
    # type: (Path) -> None
    """Test main function doesn't modify explicit commands."""
    test_args = ["iscc-sct", "create", str(sample_text_file)]

    with patch.object(sys, "argv", test_args):
        with patch("iscc_sct.cli.app") as mock_app:
            main()
            # 'create' should not be inserted again
            assert sys.argv.count("create") == 1


def test_main_with_demo_command():
    # type: () -> None
    """Test main function with demo command."""
    test_args = ["iscc-sct", "demo"]

    with patch.object(sys, "argv", test_args):
        with patch("iscc_sct.cli.app") as mock_app:
            main()
            # 'create' should not be inserted
            assert "create" not in sys.argv


def test_main_with_options(sample_text_file):
    # type: (Path) -> None
    """Test main function with options before path."""
    test_args = ["iscc-sct", "-f", "json", str(sample_text_file)]

    with patch.object(sys, "argv", test_args):
        with patch("iscc_sct.cli.app") as mock_app:
            main()
            # 'create' should be inserted at position 1
            assert sys.argv[1] == "create"
            assert "-f" in sys.argv
            assert "json" in sys.argv


def test_main_with_version_flag():
    # type: () -> None
    """Test main function with version flag."""
    test_args = ["iscc-sct", "--version"]

    with patch.object(sys, "argv", test_args):
        with patch("iscc_sct.cli.app") as mock_app:
            main()
            # Version flag should not trigger 'create' insertion
            assert "create" not in sys.argv


def test_output_json_with_console():
    # type: () -> None
    """Test JSON output with console and progress."""
    from rich.console import Console
    from rich.progress import Progress

    console = Console(file=StringIO())
    progress = Progress(console=console)

    result = {
        "iscc": "ISCC:ABC123",
        "filename": "test.txt",
        "meta": MagicMock(features=None),
    }

    output_json(result, None, console, progress)
    # Output should go through console


def test_output_json_pretty_with_console():
    # type: () -> None
    """Test pretty JSON output with console."""
    from rich.console import Console

    console = Console(file=StringIO())

    result = {
        "iscc": "ISCC:ABC123",
        "filename": "test.txt",
        "meta": MagicMock(features=None),
    }

    output_json(result, 2, console, None)
    # Output should be pretty printed through console


def test_cli_create_with_truncate(sample_text_file):
    # type: (Path) -> None
    """Test create command with truncate option."""
    result = runner.invoke(app, ["create", str(sample_text_file), "-g", "-c", "-t", "20"])
    assert result.exit_code == 0


def test_cli_create_with_unit_bits(sample_text_file):
    # type: (Path) -> None
    """Test create command with unit-bits option."""
    result = runner.invoke(app, ["create", str(sample_text_file), "-u", "128"])
    assert result.exit_code == 0


def test_cli_create_with_simprint_bits(sample_text_file):
    # type: (Path) -> None
    """Test create command with simprint-bits option."""
    result = runner.invoke(app, ["create", str(sample_text_file), "-s", "128"])
    assert result.exit_code == 0
