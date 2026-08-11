from pathlib import Path


def test_required_project_layout_exists():
    root = Path(__file__).parents[1]
    required = {
        "Dockerfile",
        "docker-compose.yml",
        "requirements.lock",
        "scripts/__init__.py",
        "okfolio/agentwiki/__init__.py",
        "okfolio/data_processing/__init__.py",
        "okfolio/mcp/__init__.py",
        "prompts/discover.md",
        "prompts/compile.md",
        "prompts/preserve.md",
        "prompts/enrich.md",
    }
    actual = {str(path.relative_to(root)) for path in root.rglob("*")}
    assert not (required - actual)


def test_project_has_one_deployment_surface_and_no_reverse_script_imports():
    root = Path(__file__).parents[1]
    # "data" is the git-ignored master data tree consolidated under the repo
    # root (2026-08-11) and is intentionally present; the remaining names are
    # legacy layout remnants that must not reappear.
    for legacy_directory in ("mcp", "modules", "scripts/okfolio"):
        assert not (root / legacy_directory).exists()
    assert sorted(path.name for path in root.glob("Dockerfile*")) == [
        "Dockerfile"
    ]
    assert sorted(path.name for path in root.glob("docker-compose*.yml")) == [
        "docker-compose.test.yml",
        "docker-compose.yml",
    ]
    for path in (root / "okfolio").rglob("*.py"):
        content = path.read_text(encoding="utf-8")
        assert "from scripts" not in content
        assert "import scripts" not in content


def test_preserve_prompt_assigns_every_source_asset_without_rewriting_it():
    root = Path(__file__).parents[1]
    prompt = (root / "prompts" / "preserve.md").read_text(encoding="utf-8")

    assert "每个 asset_id 必须且只能返回一次" in prompt
    assert "不得生成、改写、删减资产或正文" in prompt
    assert "代码将逐字插入原始资产并校验" in prompt


def test_prompts_separate_discovery_compilation_and_relation_responsibilities():
    root = Path(__file__).parents[1]
    discover_prompt = (root / "prompts" / "discover.md").read_text(encoding="utf-8")
    compile_prompt = (root / "prompts" / "compile.md").read_text(encoding="utf-8")
    enrich_prompt = (root / "prompts" / "enrich.md").read_text(encoding="utf-8")

    assert "不能把多章节报告汇总成一个总概念" in discover_prompt
    assert "不写完整正文" in discover_prompt
    assert "只有指标定义、计算方式或数据来源" in discover_prompt
    assert "数值表现、趋势、原因或风险判断" in discover_prompt
    assert "每次只把一个已确认的 ConceptRef" in compile_prompt
    assert "不得包含 Markdown 图片" in compile_prompt
    assert "不得返回或改写完整文档" in enrich_prompt
    assert '"status":"no_links"' in enrich_prompt
