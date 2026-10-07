# Structured Output E2E Tests

This directory contains single-node weekly coverage for vLLM structured
outputs on Ascend.

## Scope

- Offline `LLM.generate` requests.
- OpenAI-compatible chat completion requests.
- JSON Schema, regular expression, choice, and grammar constraints.
- Streaming, concurrent mixed requests, and invalid-request recovery.

The suite uses `vllm-ascend/Qwen3-32B-W4A4` with two-way tensor parallelism.
The OpenAI-compatible service exposes it as `qwen3`. Shared cases,
assertions, and request construction live outside the test modules to keep
new cases data-driven.

## Run

```bash
pytest -sv tests/e2e/weekly/single_node/features/structured_output
```
