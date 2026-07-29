"""Build the static wiki without coupling MkDocs to a runtime path."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_site(
    wiki_dir: Path,
    site_dir: Path,
    *,
    config_file: Path | None = None,
) -> None:
    """Build one wiki tree into one site tree using explicit runtime paths."""
    from mkdocs.commands.build import build
    from mkdocs.config import load_config

    wiki = wiki_dir.resolve()
    site = site_dir.resolve()
    if not wiki.is_dir():
        raise FileNotFoundError(f"wiki directory does not exist: {wiki}")
    config_path = (config_file or PROJECT_ROOT / "mkdocs.yml").resolve()
    config = load_config(
        config_file=str(config_path),
        docs_dir=str(wiki),
        site_dir=str(site),
    )
    build(config)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("DATA_DIR", "/app/runtime/data")),
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=PROJECT_ROOT / "mkdocs.yml",
    )
    args = parser.parse_args()
    build_site(
        args.data_dir / "wiki",
        args.data_dir / "outputs" / "site",
        config_file=args.config_file,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
