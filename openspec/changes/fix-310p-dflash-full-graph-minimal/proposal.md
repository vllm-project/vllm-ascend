## Why

Ascend 310P already supports native `FULL` graph execution without DFlash, but
adding DFlash currently causes the plugin to downgrade, mis-key, or reject
prefill/mixed/speculative graph work.  The earlier experimental repair expanded
into shared graph and operator paths without reaching a maintainable closure, so
the mode must be rebuilt from the verified `1a8feb60` baseline with an exact
`310P + DFlash + FULL` boundary and evidence-driven scope.

## What Changes

- Add a plugin-side activation policy that is true only for Ascend 310P, DFlash,
  configured `FULL`, and runtime `FULL`.
- Add a FULL-only controller and graph-entry identity that distinguish target
  from draft, TP rank, batch descriptor, and execution signature so prefill,
  chunked prefill, mixed, decode, and speculative decode cannot reuse an
  incompatible graph.
- Add persistent per-entry input contracts, deterministic padding, startup
  manifests, and runtime capture/replay evidence for both target and DFlash
  draft components.
- Preserve the upstream dispatcher as authoritative and fail closed when an
  in-range eligible FULL batch unexpectedly falls back instead of silently
  running eager.
- Remove only the capture blockers proven by fresh RED tests on the current
  baseline.  Python/plugin-side fixes are preferred.
- Permit a separately named operator used exclusively by the exact
  `310P + DFlash + FULL` predicate only when a focused RED proves that the
  existing operator or Python path cannot be made capture-safe.  It must not
  change a public schema, replace a shared operator, or be called by Eager,
  Piecewise, FULL_DECODE_ONLY, non-DFlash, or non-310P paths.
- Validate real FULL capture and replay, correctness, acceptance length,
  throughput, memory, and stability on server 1 for 4B TP1/TP2 and 35B W8A8
  TP2/TP4 at concurrency 1 and 10 with GSM8K-256 and random input/output 2048
  workloads, plus AISBench smoke coverage. Validate an independent Git pull on
  server 2 with its available 2B TP1/TP2 and a representative 35B topology.

## Capabilities

### New Capabilities

- `310p-dflash-full-graph`: Defines isolated activation, dispatch, graph
  identity, input contracts, optional dedicated-operator admission, evidence,
  and acceptance requirements for genuine DFlash `FULL` execution on 310P.

### Modified Capabilities

None.

## Impact

The intended implementation is confined to new modules under
`vllm_ascend/_310p/` plus narrow hooks in the 310P model runner, 310P GDN
attention builder, and 310P DFlash proposer. Shared vLLM Ascend graph wrappers,
upstream vLLM 0.24.0, existing operator schemas, and verified Eager, Piecewise,
and FULL_DECODE_ONLY behavior remain frozen. The old experimental FULL worktree
is diagnostic evidence only; no implementation is copied or cherry-picked.
