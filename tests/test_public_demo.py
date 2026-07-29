import json
from pathlib import Path

import pytest

from scripts.build_public_demo import PublicDemoError, build_public_demo


def _release(tmp_path: Path) -> Path:
    release = tmp_path / "release"
    site = release / "data" / "outputs" / "site"
    site.joinpath("articles").mkdir(parents=True)
    private_a = ".".join(("192", "168", "20", "8"))
    private_b = ".".join(("10", "2", "3", "4"))
    site.joinpath("index.html").write_text(
        (
            f'<img src="http://{private_a}:9000/bucket/a.jpg">'
            "<p>private-model / private-run / internal-demo</p>"
            "<h1>KMPro Wiki</h1>"
            '<link href="https://cdnjs.cloudflare.com/ajax/libs/'
            'highlight.js/11.8.0/styles/github.min.css">'
            '<script src="https://cdnjs.cloudflare.com/ajax/libs/'
            'highlight.js/11.8.0/highlight.min.js"></script>'
            "<script>hljs.highlightAll();</script>"
        ),
        encoding="utf-8",
    )
    site.joinpath("articles", "a.html").write_text(
        f'{{"evidence":"http://{private_b}:9000/bucket/b.jpg"}}',
        encoding="utf-8",
    )
    release.joinpath("release-manifest.json").write_text(
        json.dumps(
            {
                "version": "internal-demo",
                "source_run": "private-run",
                "model": "private-model",
                "articles": 1,
                "refs": 2,
                "concepts": 1,
                "joint_concepts": 1,
            }
        ),
        encoding="utf-8",
    )
    return release


def test_public_demo_removes_private_infrastructure(tmp_path):
    release = _release(tmp_path)
    output = tmp_path / "public"

    manifest = build_public_demo(release, output)

    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in output.rglob("*")
        if path.is_file() and path.suffix in {".html", ".json"}
    )
    assert ".".join(("192", "168")) not in combined
    assert ".".join(("10", "2", "3", "4")) not in combined
    assert "private-model" not in combined
    assert "private-run" not in combined
    assert "cdnjs.cloudflare.com" not in combined
    assert "KMPro Wiki" not in combined
    assert "OKFolio" in combined
    assert manifest["version"] == "0.1.0-demo"
    assert manifest["model"] == "OpenAI-compatible LLM"
    assert manifest["removed_private_asset_urls"] == 2
    assert "internal-demo" not in combined
    assert manifest["removed_private_metadata_values"] == 3
    assert manifest["localized_external_highlight_assets"] == 2
    assert manifest["renamed_legacy_brand_values"] == 1
    assert (output / "site" / "assets" / "private-asset-removed.svg").is_file()
    assert (output / "MANIFEST.sha256").is_file()


def test_public_demo_preserves_source_release(tmp_path):
    release = _release(tmp_path)
    original = (release / "data" / "outputs" / "site" / "index.html").read_text(
        encoding="utf-8"
    )

    build_public_demo(release, tmp_path / "public")

    assert (
        release / "data" / "outputs" / "site" / "index.html"
    ).read_text(encoding="utf-8") == original


def test_public_demo_refuses_output_inside_release(tmp_path):
    release = _release(tmp_path)

    with pytest.raises(PublicDemoError, match="outside"):
        build_public_demo(release, release / "public")
