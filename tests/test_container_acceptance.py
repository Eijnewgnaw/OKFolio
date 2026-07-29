from __future__ import annotations

import os
from pathlib import Path
import subprocess
import uuid

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_CONTAINER_TESTS") != "1",
    reason="set RUN_CONTAINER_TESTS=1 after building okfolio:latest",
)


def test_compose_cold_incremental_and_noop(tmp_path: Path):
    root = Path(__file__).parents[1]
    data = tmp_path / "data"
    sources = data / "sources"
    sources.mkdir(parents=True)
    env = os.environ.copy()
    env.update(
        {
            "TEST_DATA_DIR": str(data),
            "COMPOSE_PROJECT_NAME": f"kmpro-test-{uuid.uuid4().hex[:8]}",
        }
    )
    command = [
        "docker",
        "compose",
        "-f",
        str(root / "docker-compose.yml"),
        "-f",
        str(root / "docker-compose.test.yml"),
    ]

    def run_compiler() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command + ["run", "--rm", "compiler"],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )

    try:
        (sources / "a.md").write_text("# A\n正文 A", encoding="utf-8")
        first = run_compiler()
        assert "compiled=1" in first.stdout
        existing = _tree_bytes(data / "wiki/concepts")
        assert len(existing) == 5
        first_call_count = int((data / ".fake_llm_calls").read_text())
        assert first_call_count > 0

        (sources / "b.md").write_text("# B\n正文 B", encoding="utf-8")
        second = run_compiler()
        assert "compiled=1" in second.stdout
        current = _tree_bytes(data / "wiki/concepts")
        assert all(current[path] == content for path, content in existing.items())
        second_call_count = int((data / ".fake_llm_calls").read_text())
        assert second_call_count == first_call_count * 2

        before = _tree_bytes(data / "wiki") | _tree_bytes(data / "outputs")
        third = run_compiler()
        after = _tree_bytes(data / "wiki") | _tree_bytes(data / "outputs")
        assert "compiled=0" in third.stdout
        assert "No wiki changes" in third.stdout
        assert int((data / ".fake_llm_calls").read_text()) == second_call_count
        assert after == before
    finally:
        subprocess.run(command + ["down", "--remove-orphans"], cwd=root, env=env)


def _tree_bytes(root: Path) -> dict[str, bytes]:
    if not root.exists():
        return {}
    return {
        str(path): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
