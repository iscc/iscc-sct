import pytest


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
