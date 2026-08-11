# OpenAI-compatible RAG generation

The evaluation package exposes one provider-neutral client for HyDE query
expansion and citation-aware answer generation. It uses the OpenAI Chat
Completions wire format and does not depend on the AgentWiki compiler client.

Configure a hosted API or a local compatible runtime through environment
variables. No endpoint or credential has a source-code default:

```bash
export RAG_LLM_BASE_URL="<openai-compatible-base-url>"
export RAG_LLM_MODEL="<model-id>"
export RAG_LLM_API_KEY="<key-if-required>"
```

`RAG_LLM_*` takes precedence over `OPENAI_BASE_URL`, `OPENAI_MODEL`, and
`OPENAI_API_KEY`. An authentication header is omitted when the key is empty.
Optional controls are `RAG_LLM_TIMEOUT_SECONDS`,
`RAG_LLM_MAX_OUTPUT_TOKENS`, `RAG_LLM_TEMPERATURE`, and
`RAG_LLM_INCLUDE_STREAM_USAGE`. `RAG_LLM_RESPONSE_FORMAT` defaults to
`json_schema`; use `json_object` for a compatible provider that implements only
JSON mode, or `prompt_only` only when the endpoint rejects both response-format
options. The same strict local parser runs in every mode.

```python
from kmpro_wiki.evaluation import (
    AnswerContext,
    AnswerGenerationInput,
    OpenAICompatibleRAGClient,
    RAGGenerationConfig,
    parse_structured_answer,
)

client = OpenAICompatibleRAGClient(RAGGenerationConfig.from_env())

hyde = client.generate_hyde("Which policies support green investment?")

request = AnswerGenerationInput(
    question="What is the applicable subsidy rate?",
    contexts=(
        AnswerContext(
            context_id="unit-7",
            source_id="report-a",
            page_numbers=(12,),
            evidence_ids=("report-a:p012:s3",),
            text="The audited retrieved text goes here.",
        ),
    ),
)
result = client.generate_answer(request)
answer = parse_structured_answer(result.text, request)
print(answer.answer)
print(result.timing.ttft_ms, result.timing.total_ms, result.usage.total_tokens)
```

## Frozen answer contract

Answer generation does not return free-form prose. The provider must return one
JSON object with this shape:

```json
{
  "schema": "okfolio.rag-answer.v1",
  "answer": "The supported answer or explicit refusal.",
  "refusal": false,
  "refusal_reason": "",
  "citations": [
    {"citation_id": "cite-1", "context_id": "unit-7", "page": 12}
  ],
  "atomic_claim_candidates": [
    {
      "claim_id": "claim-1",
      "text": "One atomic claim from the answer.",
      "citation_ids": ["cite-1"]
    }
  ]
}
```

The parser fails closed on extra keys, duplicate identifiers, unknown context
IDs, pages that were not supplied with that context, missing claim citations,
or Markdown-wrapped JSON. For each valid `context_id + page`, code maps the
citation to the canonical evidence atoms already attached to that context. It
does not use fuzzy text matching and does not expose canonical atom IDs to the
generator.

For a refusal, `refusal_reason` must be non-empty and both arrays must be empty.
For a non-refusal, `refusal_reason` must be empty, at least one atomic claim is
required, and every claim must cite supplied evidence.

## What is and is not scored automatically

`atomic_claim_candidates` are an auditable decomposition of the generated
answer. They are **not** Gold fact decisions. The built-in path never shows
`required_facts` or `forbidden_facts` to the answer model and never asks the
same model to grade itself.

Without a separate alignment plugin, the run is marked
`provisional_structured`. Deterministic retrieval, refusal, citation validity,
citation precision, and complete-evidence citation diagnostics are still
written. Semantic fact recall/precision, answer accuracy, and Joint Success are
written as `null`, not zero. To publish those final metrics, import a
human-reviewed alignment or configure an independently calibrated judge and
set `prediction_alignment_status` to `human_reviewed` or `independent_judge`.
Using the answer-generating model as its own judge is not an independent
evaluation.

An optional external alignment adapter is declared explicitly:

```json
{
  "prediction_aligner_factory": "review_package:create_prediction_aligner",
  "prediction_alignment_status": "human_reviewed",
  "prediction_aligner_options": {"input": "reviewed-alignments.jsonl"}
}
```

The backend validates that status before contacting the answer provider. A
missing or unknown status fails closed.

The caller remains responsible for using the same context-token budget and
generation configuration in every experiment arm. HyDE output is a retrieval
query expansion only; it must never be treated as source evidence. Streaming
results record first event, first reasoning-or-content generation, first
user-visible token (TTFT), total client latency, and provider-reported token
usage when available.
