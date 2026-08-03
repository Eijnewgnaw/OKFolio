from kmpro_wiki.agentwiki.global_cluster import candidate_edges
from kmpro_wiki.agentwiki.versioning import reconcile_refs, scope_compatibility


def ref(ref_id: str, evidence: str, *, scope: dict, article_id: str = "a") -> dict:
    return {
        "ref_id": ref_id,
        "article_id": article_id,
        "type": "数据口径",
        "title": "制造业融资需求指数",
        "description": "定义和测量制造业融资需求指数。",
        "evidence": [evidence],
        "ref_family_hint": "financing-demand-index",
        "semantic_signature": {"key": "financing-demand-index"},
        "scope": scope,
    }


def test_reconcile_keeps_time_variants_active_and_retains_history():
    old = [ref("old-2024", "补贴标准为30%。", scope={"time": "2024年"})]
    new = [ref("new-2025", "补贴标准为50%。", scope={"time": "2025年"})]

    report = reconcile_refs(old, new)

    assert report["history_retained"] is True
    assert report["counts"] == {"temporal_variant": 1}
    assert report["updates"][0]["old_ref_id"] == "old-2024"
    assert report["updates"][0]["new_ref_id"] == "new-2025"


def test_reconcile_marks_same_scope_change_as_revised():
    old = [ref("old", "补贴标准为30%。", scope={"time": "2024年"})]
    new = [ref("new", "补贴标准为50%。", scope={"time": "2024年"})]

    assert reconcile_refs(old, new)["counts"] == {"revised": 1}


def test_reconcile_without_scope_can_still_reuse_identical_evidence():
    old = [ref("old", "同一条证据。", scope={})]
    new = [ref("new", "同一条证据。", scope={})]

    assert reconcile_refs(old, new)["counts"] == {"unchanged": 1}


def test_reconcile_distinguishes_omission_from_retraction_and_addition():
    old = [
        ref("old-a", "指标A。", scope={"time": "2024年"}),
        ref("old-b", "指标B。", scope={"time": "2024年"}),
    ]
    new = [
        {**ref("new-a", "指标A。", scope={"time": "2024年"}), "status": "retracted"},
        {
            **ref("new-c", "指标C。", scope={"time": "2025年"}),
            "ref_family_hint": "new-indicator",
            "semantic_signature": {"key": "new-indicator"},
        },
    ]

    report = reconcile_refs(old, new)

    assert report["counts"] == {
        "added": 1,
        "retracted": 1,
        "not_repeated": 1,
    }


def test_scope_compatibility_and_candidate_signals_use_metadata():
    left = ref("a", "制造业指数。", scope={"time": "2024年"}, article_id="article-a")
    right = ref("b", "融资需求变化。", scope={"time": "2025年"}, article_id="article-b")

    assert scope_compatibility(left, right) == "temporal_variant"
    edges, states = candidate_edges([left, right], minimum_score=0.01)
    assert len(edges) == 1
    assert edges[0].signals["scope_compatibility"] == "temporal_variant"
    assert states == {"a": "candidates", "b": "candidates"}
