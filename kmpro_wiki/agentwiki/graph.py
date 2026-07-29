from __future__ import annotations

import html
import math
import re
from pathlib import Path

from .okf import parse_concept_markdown


LINK_RE = re.compile(r"\[[^\]]+\]\((?P<target>[^)#?]+\.md)(?:#[^)]*)?\)")


def build_graph(concepts_dir: Path) -> str:
    concepts = {}
    for path in sorted(concepts_dir.glob("*.md"), key=lambda item: item.name):
        concept = parse_concept_markdown(path.name, path.read_text(encoding="utf-8"))
        concepts[path.name] = concept

    names = sorted(concepts)
    width, height, radius = 1000, 700, 260
    center_x, center_y = width / 2, height / 2
    positions = {}
    for index, name in enumerate(names):
        angle = 2 * math.pi * index / max(len(names), 1)
        positions[name] = (
            center_x + radius * math.cos(angle),
            center_y + radius * math.sin(angle),
        )

    edges: list[tuple[str, str]] = []
    for source_name, concept in concepts.items():
        for match in LINK_RE.finditer(concept.body):
            target_name = Path(match.group("target")).name
            if target_name in concepts:
                edges.append((source_name, target_name))

    svg_lines = []
    for source, target in sorted(set(edges)):
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        svg_lines.append(
            f'<line data-source="{html.escape(source)}" data-target="{html.escape(target)}" '
            f'x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" />'
        )

    svg_nodes = []
    for name in names:
        x, y = positions[name]
        title = html.escape(str(concepts[name].frontmatter.get("title", name)))
        svg_nodes.append(
            f'<g data-node="{html.escape(name)}">'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="38" />'
            f'<text x="{x:.1f}" y="{y + 4:.1f}">{title}</text></g>'
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>OKFolio Knowledge Graph</title>
<style>
body {{ margin: 0; font-family: sans-serif; background: #f7f8fa; }}
svg {{ display: block; margin: auto; background: white; }}
line {{ stroke: #94a3b8; stroke-width: 1.5; }}
circle {{ fill: #dbeafe; stroke: #2563eb; stroke-width: 2; }}
text {{ text-anchor: middle; font-size: 12px; fill: #172554; }}
</style></head>
<body><svg viewBox="0 0 {width} {height}" role="img" aria-label="知识图谱">
<g class="edges">{''.join(svg_lines)}</g>
<g class="nodes">{''.join(svg_nodes)}</g>
</svg></body></html>
"""
