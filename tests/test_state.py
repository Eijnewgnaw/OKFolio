import hashlib
import json
from dataclasses import replace
from pathlib import Path

from okfolio.agentwiki.config import Settings
from okfolio.agentwiki.state import (
    CompileFingerprint,
    Manifest,
    SourceState,
    StageCache,
    md5_file,
)


def test_settings_load_paths_and_llm_from_environment():
    settings = Settings.from_env(
        {
            "DATA_DIR": "/tmp/data",
            "PROMPTS_DIR": "/tmp/prompts",
            "OPENAI_BASE_URL": "https://api.example/v1/",
            "OPENAI_API_KEY": "key",
            "OPENAI_MODEL": "model",
            "OPENAI_TIMEOUT_SECONDS": "900",
            "OPENAI_MAX_ATTEMPTS": "2",
            "OPENAI_ENABLE_THINKING": "false",
            "OPENAI_MAX_TOKENS": "32768",
        }
    )

    assert settings.data_dir == Path("/tmp/data")
    assert settings.prompts_dir == Path("/tmp/prompts")
    assert settings.openai_base_url == "https://api.example/v1"
    assert settings.openai_timeout_seconds == 900.0
    assert settings.openai_max_attempts == 2
    assert settings.openai_enable_thinking is False
    assert settings.openai_max_tokens == 32768


def test_settings_accepts_legacy_server_aliases_and_chat_template_flag():
    settings = Settings.from_env(
        {
            "DATA_DIR": "/tmp/data",
            "LLM_API_BASE": "https://compat.example/v1/",
            "LLM_API_KEY": "key",
            "LLM_MODEL": "model",
            "LLM_SEND_CHAT_TEMPLATE_KWARGS": "true",
        }
    )

    assert settings.openai_base_url == "https://compat.example/v1"
    assert settings.openai_api_key == "key"
    assert settings.openai_model == "model"
    assert settings.openai_send_chat_template_kwargs is True


def fingerprint(source_md5: str = "source") -> CompileFingerprint:
    return CompileFingerprint(
        source_md5=source_md5,
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


def state(
    current: CompileFingerprint | None = None,
    *,
    status: str = "complete",
) -> SourceState:
    return SourceState(
        fingerprint=current,
        outputs=("concepts/a.md",),
        discovery_status="success",
        concept_status="success",
        preservation_status="success",
        relation_status="no_links",
        status=status,
    )


def test_md5_file_hashes_content_not_metadata(tmp_path: Path):
    source = tmp_path / "source.md"
    source.write_text("same", encoding="utf-8")

    assert md5_file(source) == hashlib.md5(b"same").hexdigest()


def test_manifest_skips_identical_complete_fingerprint():
    current = fingerprint()
    manifest = Manifest({"a.md": state(current)})

    assert manifest.needs_compile("a.md", current) is False


def test_manifest_recompiles_changed_or_incomplete_or_legacy_state():
    current = fingerprint("new")

    assert Manifest({"a.md": state(fingerprint("old"))}).needs_compile(
        "a.md", current
    )
    assert Manifest({"a.md": state(current, status="incomplete")}).needs_compile(
        "a.md", current
    )
    assert Manifest({"a.md": state(None)}).needs_compile("a.md", current)


def test_manifest_recompiles_changed_generation_configuration():
    current = fingerprint()
    manifest = Manifest({"a.md": state(current)})

    assert manifest.needs_compile(
        "a.md", replace(current, max_tokens=4096)
    )


def test_manifest_round_trips_version_two_via_atomic_json(tmp_path: Path):
    path = tmp_path / ".state" / "manifest.json"
    original = Manifest({"a.md": state(fingerprint())})

    original.save(path)

    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 2
    assert Manifest.load(path) == original
    assert not path.with_suffix(".json.tmp").exists()


def test_manifest_loads_version_one_as_stale_but_keeps_outputs(tmp_path: Path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "sources": {
                    "a.md": {
                        "fingerprint": {
                            "content_md5": "old",
                            "compile_prompt_md5": "old",
                            "enrich_prompt_md5": "old",
                            "model": "old",
                        },
                        "outputs": ["concepts/old.md"],
                        "status": "success",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = Manifest.load(path)

    assert loaded.sources["a.md"].fingerprint is None
    assert loaded.sources["a.md"].outputs == ("concepts/old.md",)
    assert loaded.sources["a.md"].status == "incomplete"


def test_stage_cache_returns_only_matching_validated_key(tmp_path: Path):
    path = tmp_path / "cache.json"
    cache = StageCache.load(path)

    cache.put("discovery", "key-a", {"concepts": ["a"]})

    assert cache.get("discovery", "key-a") == {"concepts": ["a"]}
    assert cache.get("discovery", "key-b") is None
    assert StageCache.load(path).get("discovery", "key-a") == {
        "concepts": ["a"]
    }
