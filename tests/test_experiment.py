import json
import hashlib
from pathlib import Path

from scripts.audit_experiment import audit_bundle, render_report
from okfolio.agentwiki.state import CompileFingerprint, Manifest, SourceState


def fingerprint() -> CompileFingerprint:
    return CompileFingerprint(
        source_md5="source",
        asset_sha256="assets",
        discover_prompt_md5="discover",
        compile_prompt_md5="compile",
        preserve_prompt_md5="preserve",
        enrich_prompt_md5="enrich",
        discover_schema_version="1",
        compile_schema_version="1",
        preserve_schema_version="1",
        enrich_schema_version="1",
        model="model",
    )


def write_concept(path: Path, *, source: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        "type: 分析框架\n"
        f"title: {path.stem}\n"
        "description: 摘要。\n"
        f"source: {source}\n"
        "---\n"
        f"{body}\n",
        encoding="utf-8",
    )


def test_audit_reports_input_error_before_model_metrics(tmp_path: Path):
    sources = tmp_path / "sources"
    sources.mkdir(parents=True)
    (sources / "broken.md").write_text(
        "# Broken\n![](images/missing.jpg)", encoding="utf-8"
    )

    metrics = audit_bundle(tmp_path)

    assert metrics["inputs"]["broken.md"]["status"] == "input_error"
    assert "missing.jpg" in metrics["inputs"]["broken.md"]["error"]


def test_audit_counts_links_stage_statuses_graph_and_site(tmp_path: Path):
    sources = tmp_path / "sources"
    images = sources / "images"
    concepts = tmp_path / "wiki" / "concepts"
    outputs = tmp_path / "outputs"
    images.mkdir(parents=True)
    outputs.joinpath("site", "concepts").mkdir(parents=True)
    (sources / "one.md").write_text("# One\n正文。", encoding="utf-8")
    (sources / "two.md").write_text("# Two\n正文。", encoding="utf-8")
    write_concept(
        concepts / "a.md",
        source="one.md",
        body="[同源](../concepts/b.md) [跨源](../concepts/c.md)",
    )
    write_concept(concepts / "b.md", source="one.md", body="正文。")
    write_concept(concepts / "c.md", source="two.md", body="正文。")
    for name in ("a", "b", "c"):
        (outputs / "site" / "concepts" / f"{name}.html").write_text(
            name, encoding="utf-8"
        )
    (outputs / "graph.html").write_text(
        '<g data-node="a.md"></g><g data-node="b.md"></g>'
        '<line data-source="a.md" data-target="b.md" />',
        encoding="utf-8",
    )
    Manifest(
        {
            "one.md": SourceState(
                fingerprint(),
                ("concepts/a.md", "concepts/b.md"),
                "success",
                "success",
                "success",
                "success",
                "complete",
            ),
            "two.md": SourceState(
                fingerprint(),
                ("concepts/c.md",),
                "success",
                "success",
                "success",
                "no_links",
                "complete",
            ),
        }
    ).save(tmp_path / ".state" / "manifest.json")

    metrics = audit_bundle(tmp_path)

    assert metrics["concepts"] == 3
    assert metrics["links"] == {
        "same_source": 1,
        "cross_source": 1,
        "broken": 0,
        "self": 0,
    }
    assert metrics["sources"]["one.md"]["status"] == "complete"
    assert metrics["graph"] == {"nodes": 2, "edges": 1}
    assert metrics["site_pages"] == 3
    assert "跨来源链接 | 1" in render_report(metrics)


def test_audit_parses_llm_usage_and_cache_events(tmp_path: Path):
    event_log = tmp_path / "run.log"
    event_log.write_text(
        "llm.done elapsed_ms=250 thinking=false prompt_chars=3 "
        "content_chars=6 reasoning_chars=0 selected=content finish=stop "
        "prompt_tokens=12 completion_tokens=3 total_tokens=15\n"
        "cache.hit source=a.md stage=discovery\n",
        encoding="utf-8",
    )

    metrics = audit_bundle(tmp_path, event_log=event_log)

    assert metrics["llm"] == {
        "calls": 1,
        "elapsed_ms": 250,
        "prompt_tokens": 12,
        "completion_tokens": 3,
        "total_tokens": 15,
        "cache_hits": 1,
    }


def test_audit_checks_independently_cached_concept_drafts(tmp_path: Path):
    source_name = "report.md"
    sources = tmp_path / "sources"
    images = sources / "images"
    images.mkdir(parents=True)
    images.joinpath("x.jpg").write_bytes(b"image")
    sources.joinpath(source_name).write_text(
        "正文锚点。\n\n![](images/x.jpg)", encoding="utf-8"
    )
    concept_content = (
        "---\n"
        "type: 分析框架\n"
        "title: A\n"
        "description: 摘要。\n"
        f"source: {source_name}\n"
        "---\n"
        "正文锚点。\n\n![](images/x.jpg)\n"
    )
    cache_path = (
        tmp_path
        / ".staging"
        / "sources"
        / hashlib.md5(source_name.encode("utf-8")).hexdigest()
        / "cache.json"
    )
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(
        json.dumps(
            {
                "stages": {
                    "concept:a": {
                        "payload": {
                            "ref": {"concept_id": "a"},
                            "body": "正文锚点。",
                        }
                    },
                    "concept:stale": {
                        "payload": {
                            "ref": {"concept_id": "stale"},
                            "body": "上一轮发现阶段留下的过期概念。",
                        }
                    },
                    "preservation": {
                        "payload": {
                            "placements": [
                                {
                                    "asset_id": "image-001",
                                    "concept_id": "a",
                                    "anchor": "正文锚点。",
                                    "position": "after",
                                }
                            ],
                            "concepts": [
                                {"filename": "a.md", "content": concept_content}
                            ],
                        }
                    },
                    "relation": {
                        "payload": {
                            "concepts": [
                                {"filename": "a.md", "content": concept_content}
                            ]
                        }
                    },
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    metrics = audit_bundle(tmp_path)

    assert metrics["stage_invariants"] == {
        "asset_only": True,
        "link_only": True,
    }
