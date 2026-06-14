"""Tests for the docs llms-full.txt generation helpers in scripts/gen_llms_full.py."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "scripts" / "gen_llms_full.py"


def _load_module():
    # type: () -> object
    """Load the gen_llms_full script as a module by file path."""
    spec = importlib.util.spec_from_file_location("gen_llms_full", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = _load_module()


def test_clean_content_strips_frontmatter_and_snippets():
    content = (
        "---\nicon: lucide/code\ndescription: x\n---\n"
        "# Title\n\n"
        "Body paragraph.\n\n"
        "*[ISCC]: International Standard Content Code\n"
    )
    result = gen.clean_content(content)
    assert result.startswith("# Title")
    assert "icon:" not in result
    assert "*[ISCC]:" not in result
    assert "Body paragraph." in result


def test_api_page_exports_pointer_not_directives():
    # The API page is mkdocstrings directives; its export must be the pointer, never raw `:::`.
    assert gen.API_PAGE in gen.PAGES
    pointer = gen.page_content(gen.API_PAGE)
    assert ":::" not in pointer
    assert "reference/api/" in pointer
    assert "reference/for-coding-agents/" in pointer


def test_page_content_cleans_normal_page():
    page = "explanation/how-it-works.md"
    src = gen.DOCS_DIR / page
    result = gen.page_content(page)
    assert result == gen.clean_content(src.read_text(encoding="utf-8"))
    assert not result.startswith("---")
