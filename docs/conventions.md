# Coding Convetions

- Prefer httpx over requests for making http requests!
- Write pragmatic, easily testable, and performant code!
- Prefer short and pure functions where possible!
- Keep the number of function arguments below 4!
- Don´t use nested functions!
- Write concise and to-the-point docstrings for all functions!
- Write type comments style (PEP 484) instead of function annotations (PEP 3107)
- Always add a correct PEP 484 style type comment as the first line after the function definition!
- Use built-in collection types as generic types for annotations (PEP 585)!
- Use the | (pipe) operator for writing union types (PEP 604)!

Example function definition with (PEP 484) type comment and docstring:

```python
def tokenize_chunks(chunks, max_len=None):
    # type: (list[str], int|None) -> dict
    """
    Tokenize text chunks into model-compatible formats.

    :param chunks: Text chunks to tokenize.
    :param max_len: Truncates chunks above max_len characters
    :return: Dictionary of tokenized data including input IDs, attention masks, and type IDs.
    """
```
