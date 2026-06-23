# Framework Integration Baseline

This document defines the current capability baseline for vLLM framework-layer integration on Ascend.

## What this layer focuses on

It focuses on whether runtime issues come from:

- scheduler behavior changes
- worker or model-runner metadata changes
- sampler or logits-path changes
- KV-cache group or backend-selector changes
- drift between upstream code and existing `vllm-ascend` patches

## Current baseline

- `vllm-ascend` already covers part of the common vLLM framework stack.
- Upstream changes do not automatically imply that a new Ascend backend must be added.
- In many cases, the right first step is to realign an existing patch or override.

## Typical evidence

- upstream non-model file diffs
- framework-side `git diff` output
- whether `vllm_ascend` already patches the affected module
- metadata shape or field mismatch
- scheduler or graph-capture related errors
