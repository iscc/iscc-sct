import pytest
from pathlib import Path


HERE = Path(__file__).parent.absolute()


@pytest.fixture
def text_en():
    return (HERE / "en.txt").read_text(encoding="utf-8")


@pytest.fixture
def text_de():
    return (HERE / "de.txt").read_text(encoding="utf-8")
