# OKFolio

OKFolio is an OKF-native knowledge compiler for research reports, policy
briefs, and other structured documents. It turns source material into
traceable ConceptRefs, cross-document Concepts, publishable Bundles, and
clear knowledge graphs for RAG and agent memory.

> **Compile knowledge once. Reuse it with evidence.**

This repository contains the reusable processing capability, prompts, tests,
and a small public showcase. It does not contain private documents, model
weights, credentials, or deployment-specific infrastructure.

## What it produces

```text
PDF / Markdown
       │
       ▼
Document IR → Articles → ConceptRefs
                              │
                              ▼
                 Cross-document Concepts
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
          OKF Bundles      Knowledge Graph    MCP
```

- One Article can yield multiple independently reusable ConceptRefs.
- Refs from different Articles can be compiled into shared Concepts.
- Every Concept can be traced to its Refs, Article, section path, page, and
  evidence block.
- The compiler supports structured extraction, semantic discovery, grouping,
  quality review, safe re-compilation, relation judgement, and deterministic
  publication.
- The output can be consumed as OKF-style Markdown, a static wiki, a graph,
  or an MCP capability for retrieval and agent memory.

## Showcase

Open the included [static showcase](demo/site/index.html) or the
[interactive graph](demo/site/graph.html). The showcase demonstrates the
presentation and provenance experience; it is not a benchmark dataset.

## Repository layout

```text
okfolio/
├── kmpro_wiki/
│   ├── data_processing/       # PDF → Document IR → Article
│   ├── agentwiki/             # Article → Ref → Concept → Bundle / Graph
│   └── mcp/                   # MCP protocol and job orchestration
├── prompts/                   # Stage-specific model contracts
├── scripts/                   # CLI entry points and audits
├── tests/                     # Unit, integration, and release checks
├── docs/                      # Architecture and operational notes
├── demo/                      # Sanitised static showcase
├── Dockerfile
└── docker-compose.yml
```

Runtime data is intentionally kept outside the public source tree. Mount your
own corpus and output directories when running a compilation.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
cp .env.example .env
```

Configure an OpenAI or OpenAI-compatible service in `.env`:

```text
OPENAI_BASE_URL=
OPENAI_API_KEY=<provider-api-key>
OPENAI_MODEL=<provider-model>
```

`OPENAI_BASE_URL` is optional. When it is empty, the standard OpenAI API
endpoint is used. Any provider that implements the OpenAI Chat Completions
format can be selected by setting its compatible base URL. Keep `.env` out of
version control.

For a local development run, use the standard Python entry point or the
provided Compose service. The transport, host, and port are deployment
settings; no endpoint is hard-coded into this showcase README.

```bash
PYTHONPATH=. python3 -m kmpro_wiki.mcp.server --transport stdio
```

## Processing a PDF

Place a PDF in your runtime inbox and configure either a local MinerU binary,
the MinerU HTTP protocol, or an OpenAI-compatible vision model. The pipeline
keeps page-level checkpoints, preserves visual assets, normalises structure,
and activates only documents that pass the structure gate.

```bash
PYTHONPATH=. python3 scripts/process_pdf.py \
  --pdf runtime/data/inbox/report.pdf \
  --mineru-output runtime/data/mineru-output/report \
  --destination runtime/data/processed/report \
  --activate-dir runtime/data/normalized-sources
```

If MinerU has already produced its intermediate output, add `--skip-mineru`
and run the deterministic normalisation stage.

## Compiling a knowledge base

The AgentWiki compiler runs the complete pipeline:

1. inspect Article structure and provenance;
2. discover and refine ConceptRefs;
3. recall and judge cross-document candidates;
4. compile Concepts and review quality;
5. judge relations and build the graph;
6. publish and audit a Bundle and static site.

```bash
docker compose --profile agent run --rm agentwiki --run-id showcase
```

Only an audited, complete run is eligible for publication. Failed or partial
runs remain resumable runtime artefacts and are never presented as a final
Bundle.

## MCP

The MCP server exposes ingestion, compilation, audit, publication, search, and
provenance tracing as tools. Query operations are read-only by default; write
operations require an explicit runtime gate. The server does not ship with
your documents or generated knowledge assets.

See [MCP operations](docs/operations/mcp-server.md) for client configuration
and [the capability release guide](docs/operations/mcp-capability-release.md)
for packaging.

## Validation

```bash
PYTHONPATH=. python3 -m pytest -q
PYTHONPATH=. python3 scripts/audit_open_source.py
docker compose -f docker-compose.yml -f docker-compose.test.yml config
```

The checks cover schema contracts, provenance, asset and source isolation,
MCP safety gates, deterministic publication, and public-release leakage.

## Public boundary

The public repository intentionally excludes:

- source corpora and generated knowledge results;
- API keys, SSH credentials, private endpoints, and deployment paths;
- model weights and internal dependency archives.

Use the included public-demo builder to create an allow-listed showcase from a
separately audited release. Hiding an element in the UI is not a substitute
for removing private content from the published payload.

## License

Apache License 2.0. See [LICENSE](LICENSE), [NOTICE](NOTICE), and
[third-party notices](THIRD_PARTY_NOTICES.md).
