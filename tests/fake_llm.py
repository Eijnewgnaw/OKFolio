from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re


STATE_FILE = Path("/state/.fake_llm_calls")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_error(404)
            return
        self._json({"status": "ok"})

    def do_POST(self) -> None:
        if not self.path.endswith("/chat/completions"):
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        prompt = payload["messages"][0]["content"]
        _increment_calls()
        if "STAGE: discover" in prompt:
            content = _discover(prompt)
        elif "STAGE: compile_joint_concept" in prompt:
            content = _compile_joint(prompt)
        elif "STAGE: compile" in prompt:
            content = _compile(prompt)
        elif "STAGE: preserve" in prompt:
            content = _preserve(prompt)
        elif "STAGE: enrich" in prompt:
            content = '{"status":"no_links","links":[]}'
        elif "STAGE: judge_edges" in prompt:
            content = _judge_edges(prompt)
        else:
            self.send_error(400, "unknown stage")
            return
        self._json({"choices": [{"message": {"content": content}}]})

    def log_message(self, format: str, *args: object) -> None:
        return

    def _json(self, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _discover(prompt: str) -> str:
    source_match = re.search(r"^真实来源文件名：(.+?)\s*$", prompt, re.MULTILINE)
    if source_match is None:
        raise ValueError("source filename missing")
    source = source_match.group(1)
    catalog_text = prompt.split(
        "带稳定 ID 的原文证据目录（顺序与报告一致）：\n", 1
    )[1].split("\n\n源文档标题结构", 1)[0]
    catalog = json.loads(catalog_text)
    evidence_id = next(
        item["evidence_id"]
        for item in catalog
        if item["text"].strip()
    )
    slug = re.sub(r"[^\w\-\u3400-\u9fff]", "-", Path(source).stem).strip("-")
    minimum_match = re.search(r"当前报告至少输出\s*(\d+)\s*个", prompt)
    minimum = int(minimum_match.group(1)) if minimum_match else 1
    count = max(minimum, 5)
    types = ["政策建议", "数据口径", "分析框架", "国际比较", "术语解释"] * count
    return json.dumps(
        {
            "concepts": [
                {
                    "id": f"{slug}-concept-{index}",
                    "type": types[index - 1],
                    "title": "产业基金支持",
                    "description": f"产业基金支持的测试摘要 {index}。",
                    "evidence": [evidence_id],
                    "asset_hints": [],
                }
                for index in range(1, count + 1)
            ]
        },
        ensure_ascii=False,
    )


def _compile(prompt: str) -> str:
    ref_text = prompt.split("## 当前 ConceptRef\n\n", 1)[1].split(
        "\n\n## 可用原文证据", 1
    )[0]
    evidence_text = prompt.split("## 可用原文证据\n\n", 1)[1].split(
        "\n\n## 输出", 1
    )[0]
    ref = json.loads(ref_text)
    evidence = json.loads(evidence_text)
    return json.dumps(
        {
            "title": ref["title"],
            "description": ref["description"],
            "sections": [
                {
                    "heading": "正文",
                    "paragraphs": [
                        re.sub(r"\s+", " ", item).lstrip("# ")
                        for item in evidence
                    ],
                    "bullets": [],
                }
            ],
        },
        ensure_ascii=False,
    )


def _preserve(prompt: str) -> str:
    assets_text = prompt.split("## 源资产清单\n\n", 1)[1].split(
        "\n\n## 当前 Concept 草稿", 1
    )[0]
    concepts_text = prompt.split("## 当前 Concept 草稿\n\n", 1)[1].split(
        "\n\n## 可选锚点目录", 1
    )[0]
    anchors_text = prompt.split("## 可选锚点目录\n\n", 1)[1].split(
        "\n\n## 输出", 1
    )[0]
    assets = json.loads(assets_text)
    concepts = json.loads(concepts_text)
    anchors = json.loads(anchors_text)
    target = concepts[0]
    anchor = next(item for item in anchors if item["concept_id"] == target["concept_id"])
    return json.dumps(
        {
            "placements": [
                {
                    "asset_id": item["asset_id"],
                    "concept_id": target["concept_id"],
                    "anchor_id": anchor["anchor_id"],
                    "position": "after",
                    "reason": "fake placement",
                }
                for item in assets
            ]
        },
        ensure_ascii=False,
    )


def _judge_edges(prompt: str) -> str:
    edges_text = prompt.split("候选边（数组顺序即绑定顺序）：\n", 1)[1].split("\n\nRefCard：", 1)[0]
    edges = json.loads(edges_text)
    return json.dumps(
        {
            "judgements": [
                {
                    "decision": "same",
                    "reason": "fake judge: same reusable unit",
                }
                for edge in edges
            ]
        },
        ensure_ascii=False,
    )


def _compile_joint(prompt: str) -> str:
    cluster_text = prompt.split("Cluster：\n", 1)[1].split("\n\n证据：", 1)[0]
    cluster = json.loads(cluster_text)
    return json.dumps(
        {
            "title": cluster["title"],
            "description": cluster["description"],
            "body": "## 联合结论\n\n该 Concept 由已验证的原文证据联合编译。",
        },
        ensure_ascii=False,
    )


def _increment_calls() -> None:
    current = int(STATE_FILE.read_text()) if STATE_FILE.exists() else 0
    temporary = STATE_FILE.with_suffix(".tmp")
    temporary.write_text(str(current + 1))
    temporary.replace(STATE_FILE)


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
