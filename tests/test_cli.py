import sys
import pytest
from io import StringIO
from pathlib import Path
from unittest.mock import patch, MagicMock
from iscc_sct.cli import main, resolve_path
from loguru import logger


@pytest.fixture
def sample_text_file(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("This is a sample text for testing.")
    return file_path


@pytest.fixture
def empty_text_file(tmp_path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text(" ")
    return file_path


@pytest.fixture
def non_utf8_text_file(tmp_path):
    file_path = tmp_path / "non_utf8.txt"
    file_path.write_text("Iñtërnâtiônàlizætiøn☃", encoding="utf-16")
    return file_path


def test_cli_no_args(capsys):
    with patch("sys.argv", ["iscc-sct"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2  # Missing command, shows help
        captured = capsys.readouterr()
        # Should show help message with usage and available commands
        assert "Usage:" in captured.out
        assert "Commands" in captured.out


def test_cli_empty_file(empty_text_file):
    with patch("sys.argv", ["iscc-sct", "create", str(empty_text_file), "-d"]):
        mock_logger = MagicMock()
        with patch("iscc_sct.cli.logger", mock_logger):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
            # Check that warning was called with SKIPPED message
            warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
            assert any("SKIPPED" in str(call) for call in warning_calls)


def test_cli_non_utf8_file(non_utf8_text_file, capsys):
    with patch("sys.argv", ["iscc-sct", "create", str(non_utf8_text_file), "-d"]):
        mock_logger = MagicMock()
        with patch("iscc_sct.cli.logger", mock_logger):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
            captured = capsys.readouterr()
            # Check that debug was called with "Could not decode" message
            debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
            assert any("Could not decode" in str(call) for call in debug_calls)
            assert "ISCC:" in captured.out


def test_cli_generate_sct(sample_text_file, capsys):
    with patch("sys.argv", ["iscc-sct", "create", str(sample_text_file)]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "ISCC:" in captured.out


def test_cli_generate_sct_granular(sample_text_file, capsys):
    with patch("sys.argv", ["iscc-sct", "create", str(sample_text_file), "--granular"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "features" in captured.out


def test_cli_debug_mode(sample_text_file):
    with patch("sys.argv", ["iscc-sct", "create", str(sample_text_file), "--debug"]):
        mock_logger = MagicMock()
        with patch("iscc_sct.cli.logger", mock_logger):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
            # Check that logger.remove() was NOT called (debug mode keeps logger active)
            # and that debug was called
            assert mock_logger.debug.called


def test_cli_help(capsys):
    with patch("sys.argv", ["iscc-sct", "--help"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0  # Explicit help request succeeds
        captured = capsys.readouterr()
        assert "Usage:" in captured.out
        assert "Commands" in captured.out


def test_cli_help_short(capsys):
    with patch("sys.argv", ["iscc-sct", "-h"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0  # Explicit help request succeeds
        captured = capsys.readouterr()
        assert "Usage:" in captured.out
        assert "Commands" in captured.out
        # Verify both -h and --help are shown in the help text
        assert "--help" in captured.out and "-h" in captured.out


def test_cli_version(capsys):
    with patch("sys.argv", ["iscc-sct", "--version"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "iscc-sct version" in captured.out


def test_cli_invalid_format(sample_text_file, capsys):
    with patch("sys.argv", ["iscc-sct", "create", str(sample_text_file), "--format", "xml"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Invalid format" in captured.err


def test_cli_no_files_found(capsys):
    with patch("sys.argv", ["iscc-sct", "create", "nonexistent*.txt"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "No files found" in captured.err


def test_cli_output_to_file(sample_text_file, tmp_path):
    output_file = tmp_path / "output.txt"
    with patch("sys.argv", ["iscc-sct", "create", str(sample_text_file), "-o", str(output_file)]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        assert output_file.exists()
        content = output_file.read_text()
        assert "ISCC:" in content


def test_cli_json_format(sample_text_file, capsys):
    with patch("sys.argv", ["iscc-sct", "create", str(sample_text_file), "--format", "json"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert '"file"' in captured.out
        assert '"iscc"' in captured.out


def test_cli_json_format_granular(sample_text_file, capsys):
    with patch("sys.argv", ["iscc-sct", "create", str(sample_text_file), "--format", "json", "-g"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert '"metadata"' in captured.out


def test_cli_quiet_mode(sample_text_file, capsys):
    with patch("sys.argv", ["iscc-sct", "create", str(sample_text_file), "--quiet"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0


def test_cli_multiple_files_with_pattern(tmp_path, capsys):
    # Create files with a unique prefix
    file1 = tmp_path / "multi1.txt"
    file2 = tmp_path / "multi2.txt"
    file1.write_text("First file content.")
    file2.write_text("Second file content.")

    # Test single file output (no filename)
    with patch("sys.argv", ["iscc-sct", "create", str(file1)]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        # Single file should not have filename
        assert "ISCC:" in captured.out
        assert "multi1.txt" not in captured.out

    # Test multiple files output (checksum-style format: iscc  file)
    with patch("sys.argv", ["iscc-sct", "create", str(file1), str(file2)]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        # Multiple files should have checksum-style format with two spaces
        lines = captured.out.strip().split("\n")
        assert len(lines) == 2
        assert "ISCC:" in lines[0]
        # Check for two-space separator followed by filename (path may be full or relative)
        assert "  " in lines[0] and "multi1.txt" in lines[0]
        assert "ISCC:" in lines[1]
        assert "  " in lines[1] and "multi2.txt" in lines[1]


def test_cli_glob_matches_directories_only(tmp_path, capsys):
    # Create a directory but no files
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    # Try to process with a pattern that matches the directory
    pattern = str(tmp_path / "*")
    with patch("sys.argv", ["iscc-sct", "create", pattern]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "No files found" in captured.err


def test_cli_demo_success(capsys):
    with patch("sys.argv", ["iscc-sct", "demo"]):
        # Mock the gradio demo.launch() to avoid actually starting the server
        with patch("iscc_sct.demo.demo.launch") as mock_launch:
            with pytest.raises(SystemExit) as exc_info:
                main()
            # Should succeed with exit code 0
            assert exc_info.value.code == 0
            captured = capsys.readouterr()
            assert "Launching Gradio demo" in captured.out
            # Verify launch was called with inbrowser=True
            mock_launch.assert_called_once_with(inbrowser=True)


def test_cli_demo_missing_gradio(capsys):
    with patch("sys.argv", ["iscc-sct", "demo"]):
        with patch.dict("sys.modules", {"iscc_sct.demo": None}):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # Should fail because Gradio import will fail
            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "Gradio is not installed" in captured.err


def test_cli_output_to_file_quiet(sample_text_file, tmp_path, capsys):
    output_file = tmp_path / "output.txt"
    with patch(
        "sys.argv", ["iscc-sct", "create", str(sample_text_file), "-o", str(output_file), "--quiet"]
    ):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        assert output_file.exists()
        captured = capsys.readouterr()
        # In quiet mode, no output message should be shown
        assert "Output written" not in captured.out


def test_cli_error_in_file_processing(tmp_path, capsys):
    # Create a file that will cause an error during processing
    error_file = tmp_path / "error.txt"
    error_file.write_text("Test content for error handling.")

    with patch("sys.argv", ["iscc-sct", "create", str(error_file)]):
        # Mock the create function to raise an exception
        with patch("iscc_sct.cli.create", side_effect=RuntimeError("Test error")):
            mock_logger = MagicMock()
            with patch("iscc_sct.cli.logger", mock_logger):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                # Should still exit with 0 because we continue on errors
                assert exc_info.value.code == 0
                # Check that error was logged
                error_calls = [str(call) for call in mock_logger.error.call_args_list]
                assert any("Test error" in str(call) for call in error_calls)


def test_cli_error_in_debug_mode(tmp_path):
    # Create a file
    error_file = tmp_path / "error.txt"
    error_file.write_text("Test content for error handling.")

    with patch("sys.argv", ["iscc-sct", "create", str(error_file), "--debug"]):
        # Mock the create function to raise an exception
        with patch("iscc_sct.cli.create", side_effect=RuntimeError("Test error in debug")):
            mock_logger = MagicMock()
            with patch("iscc_sct.cli.logger", mock_logger):
                # In debug mode, the exception should be re-raised
                with pytest.raises(RuntimeError, match="Test error in debug"):
                    main()


def test_cli_charset_detection_failure(tmp_path):
    # Create a file with invalid UTF-8 bytes
    binary_file = tmp_path / "binary.txt"
    # These bytes are invalid UTF-8 and should trigger charset detection
    binary_file.write_bytes(b"\xff\xfe\x00\x00\x01\x02")

    with patch("sys.argv", ["iscc-sct", "create", str(binary_file), "-d"]):
        # Mock from_bytes().best() to return None for charset detection failure
        with patch("iscc_sct.cli.from_bytes") as mock_from_bytes:
            mock_result = MagicMock()
            mock_result.best.return_value = None
            mock_from_bytes.return_value = mock_result

            mock_logger = MagicMock()
            with patch("iscc_sct.cli.logger", mock_logger):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 0
                # Check that error was logged for failed encoding detection
                error_calls = [str(call) for call in mock_logger.error.call_args_list]
                assert any("failed to detect text encoding" in str(call) for call in error_calls)


def test_resolve_path_single_file(tmp_path):
    """Test resolving a single file path."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("content")

    result = resolve_path(str(test_file))
    assert result == [test_file]


def test_resolve_path_nonexistent_file(tmp_path):
    """Test resolving a nonexistent file returns empty list."""
    result = resolve_path(str(tmp_path / "nonexistent.txt"))
    assert result == []


def test_resolve_path_directory_recursive(tmp_path, monkeypatch):
    """Test resolving a directory finds all files recursively."""
    monkeypatch.chdir(tmp_path)

    # Create nested directory structure
    (tmp_path / "file1.txt").write_text("content1")
    (tmp_path / "file2.txt").write_text("content2")
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "file3.txt").write_text("content3")

    result = resolve_path(str(tmp_path))
    assert len(result) == 3
    assert all(f.suffix == ".txt" for f in result)
    # Results should be sorted
    assert result == sorted(result)


def test_resolve_path_glob_pattern(tmp_path, monkeypatch):
    """Test resolving a glob pattern finds matching files."""
    monkeypatch.chdir(tmp_path)

    # Create test files
    (tmp_path / "test1.txt").write_text("content")
    (tmp_path / "test2.txt").write_text("content")
    (tmp_path / "other.md").write_text("content")

    result = resolve_path("*.txt")
    assert len(result) == 2
    assert all(f.suffix == ".txt" for f in result)


def test_resolve_path_glob_no_matches(tmp_path, monkeypatch):
    """Test glob pattern with no matches returns empty list."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "file.txt").write_text("content")

    result = resolve_path("*.md")
    assert result == []


def test_resolve_path_expanduser(tmp_path, monkeypatch):
    """Test that ~ paths are expanded."""
    # Create a test file in home-like directory
    test_file = tmp_path / "test.txt"
    test_file.write_text("content")

    # Mock Path.expanduser to return our test path
    with patch("pathlib.Path.expanduser") as mock_expand:
        mock_expand.return_value = test_file
        result = resolve_path("~/test.txt")
        assert result == [test_file]


def test_cli_multiple_paths_as_arguments(tmp_path, capsys):
    """Test CLI with multiple path arguments (shell-expanded glob)."""
    # Create test files
    file1 = tmp_path / "test1.py"
    file2 = tmp_path / "test2.py"
    file3 = tmp_path / "other.txt"
    file1.write_text("Python file 1")
    file2.write_text("Python file 2")
    file3.write_text("Text file")

    # Simulate shell-expanded glob by passing multiple file paths
    with patch("sys.argv", ["iscc-sct", "create", str(file1), str(file2), str(file3)]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        assert len(lines) == 3
        # Verify checksum-style format (two-space separator + filename present)
        assert all("ISCC:" in line for line in lines)
        assert "  " in lines[0] and "test1.py" in lines[0]
        assert "  " in lines[1] and "test2.py" in lines[1]
        assert "  " in lines[2] and "other.txt" in lines[2]


def test_resolve_path_recursive_glob(tmp_path, monkeypatch):
    """Test recursive glob pattern (**) finds files in subdirectories."""
    monkeypatch.chdir(tmp_path)

    # Create nested structure
    (tmp_path / "file1.txt").write_text("content")
    subdir = tmp_path / "sub"
    subdir.mkdir()
    (subdir / "file2.txt").write_text("content")
    deep = subdir / "deep"
    deep.mkdir()
    (deep / "file3.txt").write_text("content")

    result = resolve_path("**/*.txt")
    assert len(result) == 3
    assert all(f.suffix == ".txt" for f in result)


def test_cli_progress_bar_with_tty(tmp_path, monkeypatch):
    # type: (object, object) -> None
    """Test progress bar display when output is to TTY."""
    # Create multiple test files
    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"
    file1.write_text("Content for file 1")
    file2.write_text("Content for file 2")

    # Mock isatty to return True (simulating TTY)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)

    # Run with multiple files (should trigger progress bar)
    with patch("sys.argv", ["iscc-sct", "create", str(file1), str(file2)]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0


def test_cli_json_output_to_file(tmp_path):
    # type: (object,) -> None
    """Test JSON output written to file."""
    # Create test file
    input_file = tmp_path / "input.txt"
    input_file.write_text("Test content for JSON output")

    # Output file
    output_file = tmp_path / "output.json"

    # Run with JSON format and output file
    with patch(
        "sys.argv",
        ["iscc-sct", "create", str(input_file), "--format", "json", "--output", str(output_file)],
    ):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

    # Verify output file was created and contains valid JSON
    assert output_file.exists()
    import json

    data = json.loads(output_file.read_text())
    assert isinstance(data, list)
    assert len(data) == 1
    assert "iscc" in data[0]
    assert "file" in data[0]


def test_cli_json_output_to_file_quiet(tmp_path, capsys):
    # type: (object, object) -> None
    """Test JSON output to file with quiet mode."""
    # Create test file
    input_file = tmp_path / "input.txt"
    input_file.write_text("Test content")

    # Output file
    output_file = tmp_path / "output.json"

    # Run with JSON format, output file, and quiet mode
    with patch(
        "sys.argv",
        [
            "iscc-sct",
            "create",
            str(input_file),
            "--format",
            "json",
            "--output",
            str(output_file),
            "--quiet",
        ],
    ):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

    # Verify no output to stdout in quiet mode
    captured = capsys.readouterr()
    assert "Output written to" not in captured.out

    # Verify file was still created
    assert output_file.exists()
