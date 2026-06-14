"""Generate llms-full.txt and per-page .md files for LLM consumption.

Runs after `zensical build` (see the `docs-build` poe task). Copies cleaned Markdown
(frontmatter and abbreviation snippets stripped) into site/ alongside the rendered HTML,
and concatenates every page into site/llms-full.txt.

The API reference page (`reference/api.md`) is all mkdocstrings `:::` directives that only
render via the plugin, so its plain-text export is a pointer to the rendered page and the
For Coding Agents API map rather than the raw directives.
"""

import re
from pathlib import Path

DOCS_DIR = Path(__file__).parent.parent / "docs"
SITE_DIR = Path(__file__).parent.parent / "site"

# Ordered list of doc pages to include (relative to docs/), matching the zensical.toml nav
PAGES = [
    "index.md",
    "tutorials/getting-started.md",
    "howto/compare-texts.md",
    "howto/granular-features.md",
    "howto/configuration.md",
    "howto/command-line.md",
    "explanation/how-it-works.md",
    "reference/api.md",
    "reference/for-coding-agents.md",
]

# Page whose source is mkdocstrings directives; exported as a pointer instead of raw markdown.
API_PAGE = "reference/api.md"
API_POINTER = """# API reference

The API reference is generated from source docstrings and rendered with full signatures at
https://sct.iscc.codes/reference/api/.

For a plain-text API map of every public symbol - its source module and purpose - see the
For Coding Agents page (in this file and at https://sct.iscc.codes/reference/for-coding-agents/).
"""

# Regex to strip YAML frontmatter
FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)

# Regex to strip snippet auto-append directives
SNIPPET_RE = re.compile(r"^\*\[.*?\]:.*$", re.MULTILINE)


def strip_frontmatter(content):
    # type: (str) -> str
    """Remove YAML frontmatter from markdown content."""
    return FRONTMATTER_RE.sub("", content)


def strip_snippets(content):
    # type: (str) -> str
    """Remove abbreviation snippet definitions appended by pymdownx.snippets."""
    return SNIPPET_RE.sub("", content)


def clean_content(content):
    # type: (str) -> str
    """Strip frontmatter, snippets, and normalize whitespace."""
    content = strip_frontmatter(content)
    content = strip_snippets(content)
    return content.strip()


def page_content(page):
    # type: (str) -> str
    """Return the cleaned export markdown for a page, or the pointer for the API page."""
    if page == API_PAGE:
        return API_POINTER.strip()
    return clean_content((DOCS_DIR / page).read_text(encoding="utf-8"))


def main():
    # type: () -> None
    """Generate llms-full.txt and individual .md files from doc sources."""
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    parts = []

    for page in PAGES:
        if not (DOCS_DIR / page).exists():
            print(f"Warning: {page} not found, skipping")
            continue
        content = page_content(page)
        if not content:
            continue
        parts.append(content)

        # Write individual .md file to site directory
        md_path = SITE_DIR / page
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(content + "\n", encoding="utf-8", newline="")

    # Write concatenated llms-full.txt
    output = "\n\n---\n\n".join(parts) + "\n"
    out_path = SITE_DIR / "llms-full.txt"
    out_path.write_text(output, encoding="utf-8", newline="")
    print(f"Generated {out_path} ({len(parts)} pages, {len(output)} bytes)")
    print(f"Generated {len(parts)} individual .md files in {SITE_DIR}")


if __name__ == "__main__":
    main()
