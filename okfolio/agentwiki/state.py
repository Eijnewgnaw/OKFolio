from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    serialized = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


@dataclass(frozen=True)
class CompileFingerprint:
    source_md5: str
    asset_sha256: str
    discover_prompt_md5: str
    compile_prompt_md5: str
    preserve_prompt_md5: str
    enrich_prompt_md5: str
    discover_schema_version: str
    compile_schema_version: str
    preserve_schema_version: str
    enrich_schema_version: str
    model: str
    enable_thinking: bool = False
    max_tokens: int = 32768

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CompileFingerprint":
        return cls(**value)

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class SourceState:
    fingerprint: CompileFingerprint | None
    outputs: tuple[str, ...]
    discovery_status: str
    concept_status: str
    preservation_status: str
    relation_status: str
    status: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SourceState":
        raw_fingerprint = value.get("fingerprint")
        return cls(
            fingerprint=(
                CompileFingerprint.from_dict(raw_fingerprint)
                if isinstance(raw_fingerprint, dict)
                else None
            ),
            outputs=tuple(value.get("outputs", ())),
            discovery_status=str(value.get("discovery_status", "failed")),
            concept_status=str(value.get("concept_status", "failed")),
            preservation_status=str(value.get("preservation_status", "failed")),
            relation_status=str(value.get("relation_status", "failed")),
            status=str(value.get("status", "incomplete")),
        )

    @classmethod
    def from_legacy(cls, value: dict[str, Any]) -> "SourceState":
        return cls(
            fingerprint=None,
            outputs=tuple(value.get("outputs", ())),
            discovery_status="failed",
            concept_status="failed",
            preservation_status="failed",
            relation_status="failed",
            status="incomplete",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": (
                None if self.fingerprint is None else self.fingerprint.to_dict()
            ),
            "outputs": list(self.outputs),
            "discovery_status": self.discovery_status,
            "concept_status": self.concept_status,
            "preservation_status": self.preservation_status,
            "relation_status": self.relation_status,
            "status": self.status,
        }


@dataclass
class Manifest:
    sources: dict[str, SourceState] = field(default_factory=dict)

    def needs_compile(
        self, source_name: str, fingerprint: CompileFingerprint
    ) -> bool:
        current = self.sources.get(source_name)
        return (
            current is None
            or current.status != "complete"
            or current.fingerprint != fingerprint
        )

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        if not path.exists():
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        version = payload.get("version", 1)
        if version == 1:
            return cls(
                {
                    name: SourceState.from_legacy(value)
                    for name, value in payload.get("sources", {}).items()
                }
            )
        if version != 2:
            raise ValueError(f"unsupported manifest version: {version}")
        return cls(
            {
                name: SourceState.from_dict(value)
                for name, value in payload.get("sources", {}).items()
            }
        )

    def save(self, path: Path) -> None:
        payload = {
            "version": 2,
            "sources": {
                name: state.to_dict()
                for name, state in sorted(self.sources.items())
            },
        }
        _write_json_atomic(path, payload)


@dataclass
class StageCache:
    path: Path
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "StageCache":
        if not path.exists():
            return cls(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return cls(path)
        stages = payload.get("stages", {})
        if not isinstance(stages, dict):
            return cls(path)
        return cls(path, stages)

    def get(self, stage: str, key: str) -> Any | None:
        item = self.stages.get(stage)
        if not isinstance(item, dict) or item.get("key") != key:
            return None
        return item.get("payload")

    def put(self, stage: str, key: str, payload: Any) -> None:
        self.stages[stage] = {"key": key, "payload": payload}
        _write_json_atomic(self.path, {"version": 1, "stages": self.stages})


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
