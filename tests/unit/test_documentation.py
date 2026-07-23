"""Structural contracts for the repository documentation."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs"
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


@pytest.mark.unit
def test_every_document_is_substantial_visual_and_discoverable() -> None:
    documents = sorted(DOCS.rglob("*.md"))
    index = (DOCS / "index.md").read_text(encoding="utf-8")

    for document in documents:
        content = document.read_text(encoding="utf-8")
        relative = document.relative_to(DOCS).as_posix()
        minimum_lines = 45 if relative.startswith("adr/") else 60

        assert len(content.splitlines()) >= minimum_lines, f"{relative} is too shallow"
        assert content.count("\n## ") >= 3, f"{relative} needs navigable sections"
        mermaid_blocks = re.findall(
            r"^```mermaid\n.*?^```$",
            content,
            flags=re.MULTILINE | re.DOTALL,
        )
        assert len(mermaid_blocks) == content.count("```mermaid"), (
            f"{relative} has an unbalanced Mermaid fence"
        )
        assert "```mermaid" in content or re.search(
            r"^\|.+\|\n\|[-: |]+\|", content, flags=re.MULTILINE
        ), f"{relative} needs a diagram or comparison table"

        if document.name != "index.md":
            assert f"({relative})" in index, f"{relative} is missing from docs/index.md"


@pytest.mark.unit
def test_relative_documentation_links_resolve() -> None:
    for document in DOCS.rglob("*.md"):
        content = document.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK.findall(content):
            target, _, fragment = raw_target.partition("#")
            if "://" in target or target.startswith("mailto:"):
                continue
            resolved = (document.parent / target).resolve() if target else document.resolve()
            assert resolved.exists(), f"{document}: broken link {raw_target!r}"
            if fragment and resolved.suffix == ".md":
                headings = resolved.read_text(encoding="utf-8")
                anchors = {
                    re.sub(r"\s+", "-", re.sub(r"[^\w\s-]", "", heading.lower()).strip())
                    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*$", headings, re.MULTILINE)
                }
                assert fragment in anchors, f"{document}: broken anchor {raw_target!r}"
