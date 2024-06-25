import doctest
from pathlib import Path

README = Path(__file__).parent.parent / "README.md"


def test_readme_examples():
    doctest.testfile(README.as_posix(), module_relative=False, optionflags=doctest.ELLIPSIS)
