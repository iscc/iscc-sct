import subprocess
import pytest
import shutil

sct = shutil.which("sct")


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
    result = subprocess.run([sct], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Generate Semantic" in result.stdout


def test_cli_empty_file(empty_text_file):
    result = subprocess.run([sct, str(empty_text_file), "-d"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "SKIPPED" in result.stderr


def test_cli_non_utf8_file(non_utf8_text_file):
    result = subprocess.run([sct, str(non_utf8_text_file), "-d"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Could not decode" in result.stderr
    assert "ISCC:" in result.stdout


def test_cli_generate_sct(sample_text_file):
    result = subprocess.run([sct, str(sample_text_file)], capture_output=True, text=True)
    assert result.returncode == 0
    assert "ISCC:" in result.stdout


def test_cli_generate_sct_granular(sample_text_file):
    result = subprocess.run(
        [sct, str(sample_text_file), "--granular"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "features" in result.stdout


def test_cli_debug_mode(sample_text_file):
    result = subprocess.run([sct, str(sample_text_file), "--debug"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "DEBUG" in result.stderr
