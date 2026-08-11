#!/usr/bin/env python3
from okfolio.agentwiki.config import Settings
from okfolio.agentwiki.indexer import build_index, write_if_changed


def main() -> int:
    settings = Settings.from_env()
    wiki = settings.data_dir / "wiki"
    changed = write_if_changed(
        wiki / "index.md", build_index(wiki / "concepts")
    )
    print(f"index_changed={str(changed).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
