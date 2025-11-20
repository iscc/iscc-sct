import sys
import pytest
from io import StringIO
from unittest.mock import patch, MagicMock
from iscc_sct.cli import main
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


def test_cli_no_args():
    with patch("sys.argv", ["iscc-sct"]):
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            main()
            output = mock_stdout.getvalue()
            assert "Generate Semantic" in output


def test_cli_empty_file(empty_text_file):
    with patch("sys.argv", ["iscc-sct", str(empty_text_file), "-d"]):
        mock_logger = MagicMock()
        with patch("iscc_sct.cli.logger", mock_logger):
            main()
            # Check that warning was called with SKIPPED message
            warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
            assert any("SKIPPED" in str(call) for call in warning_calls)


def test_cli_non_utf8_file(non_utf8_text_file, capsys):
    with patch("sys.argv", ["iscc-sct", str(non_utf8_text_file), "-d"]):
        mock_logger = MagicMock()
        with patch("iscc_sct.cli.logger", mock_logger):
            main()
            captured = capsys.readouterr()
            # Check that debug was called with "Could not decode" message
            debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
            assert any("Could not decode" in str(call) for call in debug_calls)
            assert "ISCC:" in captured.out


def test_cli_generate_sct(sample_text_file, capsys):
    with patch("sys.argv", ["iscc-sct", str(sample_text_file)]):
        main()
        captured = capsys.readouterr()
        assert "ISCC:" in captured.out


def test_cli_generate_sct_granular(sample_text_file, capsys):
    with patch("sys.argv", ["iscc-sct", str(sample_text_file), "--granular"]):
        main()
        captured = capsys.readouterr()
        assert "features" in captured.out


def test_cli_debug_mode(sample_text_file):
    with patch("sys.argv", ["iscc-sct", str(sample_text_file), "--debug"]):
        mock_logger = MagicMock()
        with patch("iscc_sct.cli.logger", mock_logger):
            main()
            # Check that logger.remove() was NOT called (debug mode keeps logger active)
            # and that debug was called
            assert mock_logger.debug.called
