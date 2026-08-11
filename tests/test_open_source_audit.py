import json
from pathlib import Path

import pytest

from scripts.audit_open_source import (
    OpenSourceAuditError,
    audit_open_source,
)


def test_open_source_audit_accepts_placeholders(tmp_path: Path):
    tmp_path.joinpath(".env.example").write_text(
        "OPENAI_API_KEY=replace-me\n",
        encoding="utf-8",
    )

    result = audit_open_source(tmp_path)

    assert result["status"] == "pass"


def test_open_source_audit_ignores_rag_virtual_environment(tmp_path: Path):
    environment = tmp_path / ".venv-rag" / "lib"
    environment.mkdir(parents=True)
    private_host = ".".join(("192", "168", "7", "9"))
    environment.joinpath("third-party-config.txt").write_text(
        f'endpoint=http://{private_host}:9000\n'
        + "api_"
        + 'key="not-a-project-secret"\n',
        encoding="utf-8",
    )

    result = audit_open_source(tmp_path)

    assert result["status"] == "pass"
    assert result["files_scanned"] == 0


def test_open_source_audit_rejects_private_endpoint(tmp_path: Path):
    private_host = ".".join(("192", "168", "7", "9"))
    tmp_path.joinpath("config.txt").write_text(
        f"endpoint=http://{private_host}:9000\n",
        encoding="utf-8",
    )

    with pytest.raises(OpenSourceAuditError, match="private network"):
        audit_open_source(tmp_path)


def test_open_source_audit_ignores_experiment_data_snapshot(tmp_path: Path):
    # experiment-data/ is the archived byte-for-byte run snapshot (see
    # experiment-data/README.md): it legitimately records the internal MinIO
    # asset endpoint and the model names used by the runs, so the audit skips
    # it while still guarding the code and docs tree.
    data_dir = tmp_path / "experiment-data" / "runs"
    data_dir.mkdir(parents=True)
    private_host = ".".join(("192", "168", "8", "209"))
    internal_model = "qwen3" + "p6-35b-a3b"
    data_dir.joinpath("manifest.json").write_text(
        json.dumps(
            {
                "model": internal_model,
                "asset_uri": f"http://{private_host}:9000/kmpro-wiki-assets/…",
            }
        ),
        encoding="utf-8",
    )

    result = audit_open_source(tmp_path)

    assert result["status"] == "pass"
    assert result["files_scanned"] == 0
