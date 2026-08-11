from okfolio.agentwiki.explorer import build_explorer_html
from okfolio.agentwiki.spatial_graph import build_graph_data, build_spatial_graph


def test_graph_renders_full_coverage_metrics_and_projects_related_refs():
    html = build_spatial_graph(
        [{"article_id": "article-01", "title": "来源", "ref_count": 2}],
        [
            {"ref_id": "article-01:r1", "article_id": "article-01", "title": "R1", "evidence": ["证据 1"]},
            {"ref_id": "article-01:r2", "article_id": "article-01", "title": "R2", "evidence": ["证据 2"]},
        ],
        [
            {"id": "c1", "title": "C1", "type": "分析框架", "kind": "topic", "description": "摘要", "body": "正文", "articles": ["article-01"], "ref_ids": ["article-01:r1"]},
            {"id": "c2", "title": "C2", "type": "分析框架", "kind": "topic", "description": "摘要", "body": "正文", "articles": ["article-01"], "ref_ids": ["article-01:r2"]},
        ],
        [
            {
                "decision": "related",
                "left_ref_id": "article-01:r1",
                "right_ref_id": "article-01:r2",
                "reason": "互补关系",
                "relation_type": "supports",
                "direction": "left_to_right",
                "evidence_ref_ids": ["article-01:r1"],
            }
        ],
    )

    assert '"semantic_edges"' in html
    assert '"articles": 1' in html
    assert '"refs": 2' in html
    assert '"concepts": 2' in html
    assert '"relations": 1' in html
    assert '"relation_evidence": 1' in html
    assert '"relation_concepts": 2' in html
    assert '"relation_types": {"supports": 1}' in html
    assert '"relation_type": "supports"' in html
    assert "决策参考知识图谱" in html
    assert "三维关系编织网（推荐）" in html
    assert "三维知识球（全部资产）" in html
    assert "全部概念查阅（清单）" in html
    assert "适应画布" in html
    assert "多来源 Concept 内核" in html
    assert "quadraticCurveTo" in html
    assert "Concept → Article 来源线" in html
    assert "relationArticles" in html
    assert "spatialCurve" in html
    assert "renderCatalog" in html
    assert "跨文档联合概念" in html
    assert "单文档概念" in html
    assert "Concept → ConceptRef → Article" in html
    assert "拖动旋转" in html
    assert "ConceptRef 证据" in html
    assert "关系类型" in html
    assert "证据支撑" in html


def test_explorer_reuses_graph_payload_and_exposes_provenance_views():
    data = build_graph_data(
        [{"article_id": "article-01", "title": "来源", "ref_count": 1}],
        [{"ref_id": "article-01:r1", "article_id": "article-01", "title": "R1", "evidence": ["证据"]}],
        [{"id": "c1", "title": "C1", "type": "分析框架", "description": "摘要", "body": "正文", "articles": ["article-01"], "ref_ids": ["article-01:r1"]}],
        [],
    )
    html = build_explorer_html(data, scope_note="Public showcase")

    assert "OKFolio Knowledge Explorer" in html
    assert "Local context" in html
    assert "Global themes" in html
    assert "All concepts" in html
    assert "ConceptRef evidence" in html
    assert "Source Articles" in html
    assert '"concepts": 1' in html
