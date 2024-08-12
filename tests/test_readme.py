import doctest
from pathlib import Path

README = Path(__file__).parent.parent / "README.md"


def test_readme_examples():
    failure_count, test_count = doctest.testfile(
        README.as_posix(), module_relative=False, optionflags=doctest.ELLIPSIS, raise_on_error=False
    )
    assert failure_count == 0, f"{failure_count} out of {test_count} doctests failed"
