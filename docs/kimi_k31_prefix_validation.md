# Kimi K3.1 prefix-cache investigation

This is the historical interim-weight investigation. The delivered W4A8,
DCP8 local-KV design and completed acceptance results are recorded in
[kimi_k31_local_kv_dcp8.md](kimi_k31_local_kv_dcp8.md). Earlier failures and
untested combinations below remain evidence for their original configurations.

Tracking: [issue #5](https://github.com/hust17yixuan/vllm-ascend-kimi/issues/5).

## Source provenance

The local release checkout starts at `9126e3af2d4c596c407492169cf1ae15b45a78ff`.
K3.1 requires the adaptations from the handoff snapshot
`1a83cc1732b73f90795015b4b27ddcab31ab7f31`, originally developed in
[PR #4](https://github.com/hust17yixuan/vllm-ascend-kimi/pull/4).
Those changes are prerequisites, not newly discovered prefix-cache fixes.
Both trees pair with upstream vLLM
`ced6857afa0ea7b2e3f0846a62e1394e90f15607`.

## Admission correction

`AscendMambaManager.get_num_blocks_to_allocate` previously inferred locally
computed tokens from newly matched blocks. Running requests can have no newly
matched blocks while already owning computed state. The resulting false
external-load reservation can reject a request even when enough pages are free.
Conversely, rounding a partial hit up to a full Mamba page can hide a real
external extension and omit its reservation.

Use the scheduler's exact `num_local_computed_tokens` when available. Retain
the existing block-count fallback for callers that omit it. The regression
tests cover a real hybrid coordinator, running requests, partial local hits,
and genuine external extensions with and without newly scheduled tokens.

## Runtime scope

The isolated full-weight experiment uses 93 layers and 896 routed experts,
TP8/PP4, MRV2, eager execution, FP8 main KV and indexer KV, block/interleave
768, and `--prefix-match-unit 768`. The latter is needed to avoid the original
8,312,832-token common match alignment. A configured switch alone is not
evidence of a hit: inspect `usage.prompt_tokens_details.cached_tokens` and
the server's local prefix-cache counters.

The current checkpoint uses the existing dynamic MXFP8 activation override.
It is not the pending new W4A8C8 delivery or a native-precision reference.
This isolated topology is DP1/DCP1 and does not validate the original
DP4/DCP8 P/D deployment. Existing D services are preserved.

Judge final answers against an independent oracle; do not require identical
generated token sequences. Preserve unparsed responses for semantic review.
Controlled prefix-off runs are diagnostic baselines only. Prefix must be
enabled for final functional acceptance.

Latest experiment results, limitations, and raw evidence paths are appended
to issue #5. Full GPQA and long-context accuracy must not be inferred from
short smoke cases or CPU unit tests.

## Evidence collected on 2026-10-25

- Changed-file and related CPU regressions: 1262 passed, 1 skipped. Restoring
  the old admission expression in an isolated CPU test process makes four of
  the seven admission tests fail; the corrected implementation passes all seven.
- Ruff 0.14.0 lint and format checks pass for all 27 changed Python files.
- Full-weight boundary and client-concurrency cases: 17/17 final answers
  correct, including four requests submitted in pairs with 1536 cached tokens
  each. This covers DP1/DCP1 only.
- A 4006-token retrieval request correctly reads the requested values after
  reusing 3840 tokens. A different 4002-token arithmetic question fails both
  without prefix caching and with it. Increasing the prefill batch from 768
  to 4096 does not resolve that arithmetic error; its root cause remains open.
- Single-NPU FP8 indexer checks cover contiguous versus page-strided K/scale
  storage at KV lengths 2048/4096. The selected sets agree for all three cases;
  CPU-reference top-k recall is at least 0.9946. These are synthetic inputs.
- Single-NPU KDA checks at lengths 1984/4002 compare one invocation with
  successive 768-token chunks: output and final-state maximum differences
  are zero. This checks chunk continuity, not complete model correctness.

The admission correction is supported by a reproducible regression. The
remaining long-input precision issue must stay open and must not be hidden
by declaring all prefix-cache or GPQA acceptance criteria satisfied.
