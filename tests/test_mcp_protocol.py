from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest


pytest.importorskip("mcp.server.fastmcp")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def test_stdio_protocol_lists_complete_capability_surface(tmp_path: Path):
    root = Path(__file__).parents[1]
    environment = os.environ.copy()
    environment.update(
        {
            "DATA_DIR": str(tmp_path / "data"),
            "PROMPTS_DIR": str(root / "prompts"),
            "RELEASES_DIR": str(tmp_path / "releases"),
            "MCP_ENABLE_WRITES": "false",
        }
    )
    current_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        str(root)
        if not current_pythonpath
        else f"{str(root)}{os.pathsep}{current_pythonpath}"
    )

    async def verify() -> None:
        parameters = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "kmpro_wiki.mcp.server",
                "--transport",
                "stdio",
            ],
            env=environment,
        )
        async with stdio_client(parameters) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                names = {tool.name for tool in tools.tools}
                assert {
                    "ingest_markdown",
                    "start_agent_compile",
                    "start_relation_judgement",
                    "publish_release",
                    "build_release_image",
                    "search_concepts",
                    "trace_concept",
                } <= names
                result = await session.call_tool("get_capabilities", {})
                assert result.isError is False
                assert result.structuredContent["writes_enabled"] is False
                prompts = await session.list_prompts()
                assert {item.name for item in prompts.prompts} == {
                    "compile_corpus",
                    "research_topic",
                }

    asyncio.run(verify())
