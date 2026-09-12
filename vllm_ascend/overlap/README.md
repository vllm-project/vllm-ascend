# Overlap: Unified Stream and Event Management

This package centralizes ownership of the NPU streams and cross-stream
events used to overlap communication with computation across the plugin.

## Components

### StreamRegistry (`streams.py`)

A process-wide registry that owns named singleton NPU streams, created
lazily on first access. It consolidates streams that previously lived as
scattered module-level globals:

| Registry name         | Legacy location                                             | Purpose                                      |
| --------------------- | ----------------------------------------------------------- | -------------------------------------------- |
| `current_computation` | `vllm_ascend/utils.py::current_stream`                      | Cached snapshot of the ambient compute stream|
| `global_computation`  | `vllm_ascend/utils.py::global_stream`                       | Sampler compute stream                       |
| `shared_experts`      | `vllm_ascend/utils.py::shared_experts_calculation_stream`   | Shared-expert MLP overlap stream             |
| `cp_chunked_prefill`  | `vllm_ascend/utils.py::cp_chunkedprefill_comm_stream`       | Context-parallel chunked-prefill comm stream |
| `moe_comm`            | `vllm_ascend/ops/fused_moe/moe_utils.py::COMM_STREAM`       | MoE all-to-all dispatch/combine stream       |
| `dsa_overlap`         | `vllm_ascend/attention/dsa_v1.py::dsv4_dsa_overlap_stream`  | DSA quant/matmul part-overlap stream         |

The legacy accessors remain in place and delegate to the registry, so every
existing call site keeps its import path and receives the identical stream
object with unchanged creation timing. `current_computation` is special: it
snapshots the ambient stream (following `torch.npu.set_stream`) instead of
allocating a new one. `attention_calculation_stream` in `utils.py` has no
in-tree consumer and is deliberately not registered; the accessor is kept
for external compatibility.

### EventLedger (`events.py`)

Named event slots for tracking cross-stream dependencies (for example the
boundaries around MoE dispatch, GMM2, and combine that shared-expert
overlap relies on). Events are created lazily on first record, reused
across steps, and graph-capture aware: while an NPU graph is being
captured, `record` and `wait` become no-ops (returning `None`/`False`)
instead of silently baking dependencies into the graph. Per-capture event
pooling is planned once regions are capture-safe by construction.

Today this is API plus unit tests only; the existing bare
`torch.npu.Event` call sites are unchanged and will migrate incrementally.

## Design decisions

- **Module-level singleton.** Workers run one process per NPU and registry
  mutation happens only on first access per name, matching the concurrency
  exposure of the legacy module globals. A lock guards first construction
  only; steady-state lookups are plain dict reads.
- **Lazy creation.** Nothing is allocated at import time or worker startup.
  Streams appear exactly when their first consumer asks, preserving the
  creation timing of the legacy singletons (device init and a warmup
  forward precede auxiliary-stream use).
- **Behavior-zero-change migration.** Each legacy accessor keeps its module
  global as a cache and populates it from the registry on first call, so
  resetting the global (as tests do) re-resolves the same stream.

## Follow-up wiring

The registry and ledger are the foundation for two planned integrations:

1. **MoE communication streaming.** Move dispatch/combine collectives onto
   `moe_comm` with named events (`before_dispatch`, `before_gmm2`,
   `before_combine`) replacing the ad-hoc event objects threaded through
   `FusedMoEEvents`, and overlap them with shared-expert and router
   computation on all comm methods rather than select ones.
2. **Attention/MLP boundary.** Record an event when attention output is
   ready, move the o_proj tensor-parallel all-reduce onto a communication
   stream, and overlap it with the next layer's normalization and MoE gate
   computation.

Both will consume the registry and ledger introduced here; graph-mode rules
(capture-safe region boundaries, consistent stream labeling for the fusion
passes) will be enforced as those regions land.
