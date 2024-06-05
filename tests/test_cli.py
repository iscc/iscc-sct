import subprocess
import pytest
import shutil

sct = shutil.which("sct")


@pytest.fixture
def sample_text_file(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("This is a sample text for testing.")
    return file_path


def test_cli_module():
    result = subprocess.run(["python", "-m", "iscc_sct.cli"], capture_output=True, text=True, shell=True)
    assert result.returncode != 0
    assert "usage:" in result.stderr


def test_cli_no_args():
    result = subprocess.run([sct], capture_output=True, text=True)
    assert result.returncode != 0
    assert "usage:" in result.stderr


def test_cli_generate_sct(sample_text_file):
    result = subprocess.run([sct, str(sample_text_file)], capture_output=True, text=True)
    assert result.returncode == 0
    assert "ISCC:" in result.stdout


def test_cli_generate_sct_granular(sample_text_file):
    result = subprocess.run([sct, str(sample_text_file), "--granular"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "iscc" in result.stdout


def test_cli_debug_mode(sample_text_file):
    result = subprocess.run([sct, str(sample_text_file), "--debug"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "DEBUG" in result.stderr
