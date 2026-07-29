from __future__ import annotations

import json
from pathlib import Path

from scripts.judge_agent_relations import judge_relations


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str, **_: object) -> str:
        self.calls += 1
        count = prompt.split("候选数组：\n", 1)[1].split(
            "\n\nRefCard：", 1
        )[0]
        size = len(json.loads(count))
        return json.dumps(
            {
                "judgements": [
                    {
                        "decision": "related",
                        "reason": "问题与建议存在实质约束关系。",
                    }
                    for _ in range(size)
                ]
            },
            ensure_ascii=False,
        )


def test_judge_relations_excludes_edges_inside_one_final_concept(
    tmp_path: Path,
):
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps({"status": "complete"}),
        encoding="utf-8",
    )
    refs = [
        {
            "ref_id": "r1",
            "article_id": "a1",
            "type": "分析框架",
            "title": "R1",
            "description": "D1",
            "evidence": ["E1"],
        },
        {
            "ref_id": "r2",
            "article_id": "a2",
            "type": "分析框架",
            "title": "R2",
            "description": "D2",
            "evidence": ["E2"],
        },
        {
            "ref_id": "r3",
            "article_id": "a3",
            "type": "分析框架",
            "title": "R3",
            "description": "D3",
            "evidence": ["E3"],
        },
    ]
    (run / "refs.json").write_text(
        json.dumps({"refs": refs}),
        encoding="utf-8",
    )
    (run / "groups.json").write_text(
        json.dumps(
            {
                "groups": [
                    {"group_id": "c1", "ref_ids": ["r1", "r2"]},
                    {"group_id": "c2", "ref_ids": ["r3"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    (run / "candidates.json").write_text(
        json.dumps(
            {
                "edges": [
                    {
                        "edge_id": "e1",
                        "left_ref_id": "r1",
                        "right_ref_id": "r2",
                    },
                    {
                        "edge_id": "e2",
                        "left_ref_id": "r2",
                        "right_ref_id": "r3",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    client = FakeClient()

    result = judge_relations(run, client)

    assert client.calls == 1
    assert result["candidate_edges"] == 2
    assert result["intra_concept_edges"] == 1
    assert result["cross_concept_edges"] == 1
    assert result["related"] == 1
    assert result["status"] == "complete"
