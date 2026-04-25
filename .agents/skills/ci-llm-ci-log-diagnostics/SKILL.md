---
name: ci-llm-ci-log-diagnostics
description: Two-phase CI log shrinking for LLM failure diagnosis (vllm-ascend GitHub Actions). Use when wiring or debugging optional LLM diagnostics after test failures.
---

# CI LLM log diagnostics input (two-phase)

Aligned with a multi-skill pattern: **two strategies that can evolve independently**, implemented in `.github/workflows/scripts/ci_log_filter_llm.py` and merged into a single user-prompt section in `ci_log_llm_analyze.py`.

## Skill A — High-signal line selection (Important)

**Goal**: Extract lines the model should read first from the full log; continuity is not guaranteed.

**Rules** (see `select_important_lines`):

- Keep a line if its text (case-insensitive) matches error, warning, failure, traceback, Ascend/CANN/ACL, or related keywords.
- Drop `INFO` by default; keep it only when the line also matches failure-related substrings (e.g. worker, timeout, OOM, ACL) to catch “looks like INFO but is actually a root-cause hint”.

**Output format**: Prefix each line with `L<line>:` for traceability by the model and humans.

## Skill B — Context near failures (Nearby)

**Goal**: Take contiguous windows around key anchors to preserve stack traces and causal chains.

**Anchors** (see `_ANCHOR_RES`): pytest FAILURES / short summary, `FAILED tests/...`, `Traceback`, common exception lines, `ERROR:`, etc.

**Windows**: By default, `VLLM_ASCEND_CI_LLM_CONTEXT_LINES_BEFORE` lines before and `VLLM_ASCEND_CI_LLM_CONTEXT_LINES_AFTER` lines after each anchor; adjacent windows are merged to avoid duplication.

If there are no anchors (rare), fall back to a **tail window** so context is not empty.

## How the phases are combined

`build_llm_log_bundle` produces Markdown with:

1. `### Phase A — high-signal lines`
2. `### Phase B — local context around failure anchors`

Optionally, `ci_log_llm_analyze.py` also attaches the structured summary from `ci_log_summary.py --format llm-json`.

## Operations (GitHub)

- **Master switch**: `VLLM_ASCEND_CI_LLM_ENABLED` in `vllm_ascend/envs.py`. **Default `0`**: no log read, no HTTP; workflows and script entry points stay in place so code can merge before wiring a model. Set to **`1`** and configure key/base to actually call the API.
- **Secrets**: `VLLM_ASCEND_CI_LLM_API_KEY`, `VLLM_ASCEND_CI_LLM_BASE_URL` (OpenAI-compatible base URL, e.g. `https://api.openai.com/v1`).
- **Variables** (non-secret): `VLLM_ASCEND_CI_LLM_MODEL` (optional; default in script is `gpt-4o-mini` if unset).
- If enabled but key/base are missing, the step **skips** and writes a short note to the job summary.

## Extending the backend

Environment variable `VLLM_ASCEND_CI_LLM_BACKEND` (default `openai_compatible`). Add new backends as branches in `.github/workflows/scripts/ci_log_llm_analyze.py`.
