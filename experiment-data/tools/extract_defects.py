#!/usr/bin/env python3
"""Read-only defect worklist extractor for a Claim Review run.

Usage:
    python3 extract_defects.py <review-run-dir> > /tmp/defects_worklist.md

Iterates <review-run-dir>/checkpoints/*.json and reports every withheld group
(status=complete with decision != pass) and every failed group.  Groups are
emitted in the run's processing order: the source run's groups.json order, or
the manifest's selected_group_ids order for probe runs.  Output is Markdown
with one "## <group_id>" section per group.

The script never writes into the run directory and never calls a model.  All
checkpoint fields are optional: missing contracts, drafts or coverage produce
"-" placeholders, and a failed group without coverage only reports its error.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable


def _read_json(path: Path) -> dict[str, Any]:
    """Best-effort JSON object read; missing/unparseable files yield {}."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _group_order(run_dir: Path) -> list[tuple[str, str]]:
    """(group_id, title) pairs in the run's processing order.

    The review manifest stores no group catalog, so the order comes from the
    frozen source run's groups.json (same directory family).  Probe runs list
    their exact selection in configuration.selected_group_ids; formal runs
    fall back to the full source group order.
    """
    manifest = _read_json(run_dir / "manifest.json")
    source_dir = run_dir.parent / str(manifest.get("source_run_name") or "")
    groups_payload = _read_json(source_dir / "groups.json")
    raw_groups = groups_payload.get("groups") if isinstance(groups_payload, dict) else None
    ordered: list[tuple[str, str]] = []
    if isinstance(raw_groups, list):
        ordered = [
            (str(item.get("group_id") or ""), str(item.get("title") or ""))
            for item in raw_groups
            if isinstance(item, dict) and item.get("group_id")
        ]
    configuration = manifest.get("configuration")
    selection: list[str] = []
    if isinstance(configuration, dict):
        raw_selection = configuration.get("selected_group_ids")
        if isinstance(raw_selection, list) and raw_selection:
            selection = [str(item) for item in raw_selection]
    if selection:
        by_id = {group_id: title for group_id, title in ordered}
        selected = [(group_id, by_id.get(group_id, "")) for group_id in selection]
        known = {group_id for group_id, _ in selected}
        selected.extend(
            (group_id, title)
            for group_id, title in ordered
            if group_id not in known
        )
        ordered = selected
    return ordered


def _literal(text: Any) -> str:
    """Render a possibly multi-line value as a Markdown literal block."""
    value = "" if text is None else str(text)
    if not value:
        return "-"
    if "\n" not in value:
        return value
    return "|\n" + "".join("    " + line for line in value.splitlines(keepends=True))


def _json_inline(value: Any) -> str:
    if value in (None, {}, []):
        return "-"
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _render_contract(lines: list[str], payload: Any) -> None:
    lines.append("### contract")
    if not isinstance(payload, dict):
        lines.append("-")
        return
    lines.append(f"canonical_question: {_literal(payload.get('canonical_question'))}")
    lines.append("")
    claims = payload.get("claims")
    if not isinstance(claims, list):
        lines.append("claims: -")
        return
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        lines.append(f"- claim_id: {str(claim.get('claim_id') or '-')}")
        lines.append(f"  kind: {str(claim.get('kind') or '-')} | slot: {str(claim.get('slot') or '-')}")
        lines.append(f"  claim: {_literal(claim.get('claim'))}")
        lines.append(f"  evidence_excerpt: {_literal(claim.get('evidence_excerpt'))}")
        lines.append(f"  scope: {_json_inline(claim.get('scope'))}")
    lines.append("")


def _render_draft(lines: list[str], payload: Any) -> None:
    lines.append("### draft")
    if not isinstance(payload, dict):
        lines.append("-")
        return
    lines.append(f"title: {_literal(payload.get('title'))}")
    lines.append(f"description: {_literal(payload.get('description'))}")
    lines.append("body:")
    lines.append("```text")
    lines.append(str(payload.get("body") or ""))
    lines.append("```")


def _render_rows(lines: list[str], rows: Any) -> None:
    lines.append("rows 非 covered:")
    if not isinstance(rows, list) or not rows:
        lines.append("-")
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("status") == "covered":
            continue
        lines.append(f"- claim_id: {str(row.get('claim_id') or '-')} | status: {str(row.get('status') or '-')}")
        lines.append(f"  draft_excerpt: {_literal(row.get('draft_excerpt'))}")
        lines.append(f"  finding: {_literal(row.get('finding'))}")


def _render_excerpt_issues(lines: list[str], section: str, items: Any, *fields: str) -> None:
    lines.append(f"{section}:")
    if not isinstance(items, list) or not items:
        lines.append("-")
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        parts = [
            f"{field}: {_literal(item.get(field))}"
            for field in fields
            if item.get(field) not in (None, "", [], {})
        ]
        lines.append("- " + (parts[0] if parts else "-"))
        for part in parts[1:]:
            lines.append("  " + part)


def _render_attributions(lines: list[str], items: Any) -> None:
    lines.append("sentence_attributions unsupported/uncertain:")
    if not isinstance(items, list) or not items:
        lines.append("-")
        return
    found = False
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("status") not in {"unsupported", "uncertain"}:
            continue
        found = True
        lines.append(f"- sentence_id: {str(item.get('sentence_id') or '-')} | status: {str(item.get('status') or '-')}")
        lines.append(f"  draft_excerpt: {_literal(item.get('draft_excerpt'))}")
        lines.append(f"  finding: {_literal(item.get('finding'))}")
    if not found:
        lines.append("-")


def _render_coverage(lines: list[str], payload: Any) -> None:
    lines.append("### coverage")
    if not isinstance(payload, dict):
        lines.append("-")
        return
    lines.append(f"decision: {str(payload.get('decision') or '-')}")
    _render_rows(lines, payload.get("rows"))
    _render_excerpt_issues(
        lines, "unsupported_claims", payload.get("unsupported_claims"),
        "draft_excerpt", "finding",
    )
    _render_excerpt_issues(
        lines, "scope_violations", payload.get("scope_violations"),
        "claim_ids", "draft_excerpt", "finding",
    )
    _render_attributions(lines, payload.get("sentence_attributions"))


def _render_group(group_id: str, payload: dict[str, Any], title: str) -> list[str]:
    status = str(payload.get("status") or "")
    decision = str(payload.get("decision") or "")
    lines = [f"## {group_id}", ""]
    lines.append(f"- 标题: {title or '-'}")
    lines.append(f"- status: {status}")
    lines.append(f"- decision: {decision or '-'}")
    lines.append(f"- review_reason: {str(payload.get('review_reason') or '-')}")
    lines.append(f"- recompile_attempts: {str(payload.get('recompile_attempts') or '-')}")
    lines.append(f"- draft_origin: {str(payload.get('draft_origin') or '-')}")
    lines.append("")
    if status == "failed":
        lines.append("### error")
        lines.append("```text")
        lines.append(str(payload.get("error") or "-"))
        lines.append("```")
        lines.append("")
    _render_contract(lines, payload.get("contract"))
    _render_draft(lines, payload.get("draft"))
    _render_coverage(lines, payload.get("coverage"))
    return lines


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {Path(argv[0]).name} <review-run-dir>", file=sys.stderr)
        return 2
    run_dir = Path(argv[1]).resolve()
    checkpoints_dir = run_dir / "checkpoints"
    if not run_dir.is_dir():
        print(f"not a directory: {run_dir}", file=sys.stderr)
        return 2
    if not checkpoints_dir.is_dir():
        print(f"no checkpoints directory: {checkpoints_dir}", file=sys.stderr)
        return 2

    order = _group_order(run_dir)
    order_index = {group_id: position for position, (group_id, _) in enumerate(order)}
    title_by_id = dict(order)

    entries: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(checkpoints_dir.glob("*.json")):
        payload = _read_json(path)
        group_id = str(payload.get("group_id") or "")
        if not group_id:
            continue
        status = str(payload.get("status") or "")
        decision = str(payload.get("decision") or "")
        if status == "failed" or (status == "complete" and decision != "pass"):
            entries.append((group_id, payload))

    entries.sort(
        key=lambda item: (order_index.get(item[0], len(order)), item[0])
    )

    withheld = sum(
        1 for _group_id, payload in entries
        if payload.get("status") == "complete"
    )
    failed = len(entries) - withheld

    parts: list[str] = [
        f"# 缺陷工作清单",
        "",
        f"- run: {run_dir.name}",
        f"- withheld: {withheld} | failed: {failed} | total: {len(entries)}",
        "",
    ]
    for group_id, payload in entries:
        parts.extend(_render_group(group_id, payload, title_by_id.get(group_id, "")))
        parts.append("")
    print("\n".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
