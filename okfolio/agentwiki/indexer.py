from __future__ import annotations

from collections import defaultdict
from datetime import date
from pathlib import Path

from .okf import parse_concept_markdown


TYPE_ORDER = {
    "数据口径": 0,
    "分析框架": 1,
    "政策建议": 2,
    "国际比较": 3,
    "术语解释": 4,
}


def build_index(concepts_dir: Path) -> str:
    grouped: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for path in sorted(concepts_dir.glob("*.md"), key=lambda item: item.name):
        concept = parse_concept_markdown(path.name, path.read_text(encoding="utf-8"))
        concept_type = str(concept.frontmatter["type"])
        title = str(concept.frontmatter.get("title", path.stem))
        description = str(concept.frontmatter.get("description", ""))
        grouped[concept_type].append((title, path.name, description))

    lines = ["# Knowledge Base Index"]
    for concept_type in sorted(
        grouped, key=lambda value: (TYPE_ORDER.get(value, len(TYPE_ORDER)), value)
    ):
        lines.extend(("", f"# {concept_type}", ""))
        for title, filename, description in sorted(
            grouped[concept_type], key=lambda item: (item[0], item[1])
        ):
            suffix = f" - {description}" if description else ""
            lines.append(f"* [{title}](concepts/{filename}){suffix}")
    return "\n".join(lines).rstrip() + "\n"


def write_if_changed(path: Path, content: str) -> bool:
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
    return True


def append_log(
    path: Path,
    compiled_sources: tuple[str, ...],
    *,
    today: date | None = None,
) -> bool:
    if not compiled_sources:
        return False
    day = date.today() if today is None else today
    entries = "\n".join(
        f"* **Update**: Compiled `{source}`."
        for source in sorted(set(compiled_sources))
    )
    section = f"## {day.isoformat()}\n{entries}\n"
    header = "# Knowledge Base Update Log\n\n"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if not existing:
        return write_if_changed(path, header + section)
    return write_if_changed(path, header + section + "\n" + existing.removeprefix(header))
