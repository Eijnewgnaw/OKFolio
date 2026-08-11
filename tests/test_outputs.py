from datetime import date
from pathlib import Path

from okfolio.agentwiki.graph import build_graph
from okfolio.agentwiki.indexer import append_log, build_index, write_if_changed


def write_concept(
    path: Path,
    concept_type: str,
    title: str,
    description: str,
    body: str = "正文",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""---
type: {concept_type}
title: {title}
description: {description}
source: source.md
---
{body}
""",
        encoding="utf-8",
    )


def test_index_groups_concepts_by_type_and_is_sorted(tmp_path: Path):
    write_concept(tmp_path / "b.md", "政策建议", "建议 B", "摘要 B")
    write_concept(tmp_path / "a.md", "数据口径", "指标 A", "摘要 A")

    output = build_index(tmp_path)

    assert output.index("# 数据口径") < output.index("# 政策建议")
    assert "[指标 A](concepts/a.md) - 摘要 A" in output
    assert "[建议 B](concepts/b.md) - 摘要 B" in output


def test_write_if_changed_does_not_rewrite_identical_content(tmp_path: Path):
    target = tmp_path / "index.md"

    assert write_if_changed(target, "same\n") is True
    first_mtime = target.stat().st_mtime_ns
    assert write_if_changed(target, "same\n") is False
    assert target.stat().st_mtime_ns == first_mtime


def test_log_only_adds_entries_for_compiled_sources(tmp_path: Path):
    log = tmp_path / "log.md"

    assert append_log(log, (), today=date(2026, 7, 15)) is False
    assert append_log(log, ("b.md", "a.md"), today=date(2026, 7, 15)) is True
    content = log.read_text(encoding="utf-8")
    assert content == (
        "# Knowledge Base Update Log\n\n"
        "## 2026-07-15\n"
        "* **Update**: Compiled `a.md`.\n"
        "* **Update**: Compiled `b.md`.\n"
    )


def test_graph_is_self_contained_and_contains_valid_edge(tmp_path: Path):
    write_concept(
        tmp_path / "a.md",
        "政策建议",
        "A",
        "A 摘要",
        "参见 [B](../concepts/b.md)。",
    )
    write_concept(tmp_path / "b.md", "数据口径", "B", "B 摘要")

    html = build_graph(tmp_path)

    assert "<svg" in html
    assert "https://" not in html
    assert "A" in html and "B" in html
    assert 'data-source="a.md" data-target="b.md"' in html
