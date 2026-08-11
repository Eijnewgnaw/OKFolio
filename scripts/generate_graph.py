#!/usr/bin/env python3
from okfolio.agentwiki.config import Settings
from okfolio.agentwiki.graph import build_graph
from okfolio.agentwiki.indexer import write_if_changed


def main() -> int:
    settings = Settings.from_env()
    changed = write_if_changed(
        settings.data_dir / "outputs" / "graph.html",
        build_graph(settings.data_dir / "wiki" / "concepts"),
    )
    print(f"graph_changed={str(changed).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
