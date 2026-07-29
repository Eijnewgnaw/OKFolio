from pathlib import Path

import pytest

from scripts.audit_open_source import (
    OpenSourceAuditError,
    audit_open_source,
)


def test_open_source_audit_accepts_placeholders(tmp_path: Path):
    tmp_path.joinpath(".env.example").write_text(
        "LLM_API_KEY=replace-me\n",
        encoding="utf-8",
    )

    result = audit_open_source(tmp_path)

    assert result["status"] == "pass"


def test_open_source_audit_rejects_private_endpoint(tmp_path: Path):
    private_host = ".".join(("192", "168", "7", "9"))
    tmp_path.joinpath("config.txt").write_text(
        f"endpoint=http://{private_host}:9000\n",
        encoding="utf-8",
    )

    with pytest.raises(OpenSourceAuditError, match="private network"):
        audit_open_source(tmp_path)
