# UBatch Overlap (Dual-Batch Compute/Comm Overlap)

This document describes the **ubatch overlap** feature. It covers the
motivation, architecture, control flow, configuration knobs, and limitations.
It targets developers who need to understand, extend, or debug the feature.

## TL;DR

UBatch overlap splits a single scheduler step's token batch into N
micro-batches ("ubatches") and executes them **concurrently** on N dedicated
NPU streams. While one ubatch performs compute (matmul / attention), the other
performs MoE collective communication (all-gather / reduce-scatter / MC2). This
hides the MoE communication latency behind compute for large prefill batches,
which is the dominant cost for sparse-MoE models such as MiniMax-M2 on Ascend
NPUs.

---

## 1. Motivation

For MoE models on Ascend, a single forward pass of a large prefill batch spends a
significant fraction of time inside MoE collective communication
(`tensor_model_parallel_*`, MC2, EP all-gather/reduce-scatter). When the whole
batch runs on one stream, compute and communication are serialized, leaving the
AI cores idle during communication and the communication engines (HCCS) idle
during compute.

UBatch overlap exploits the fact that **the per-ubatch collective
communication and the other ubatch's dense compute touch independent buffers**,
so they can be overlapped on different streams. The net effect is a shorter
critical path at the cost of slightly higher scheduling overhead, which is only
worth paying when the batch is large enough.

---

## 2. Configuration

The feature is controlled by two keys on vLLM's `additional_config` (the same
mechanism used by all other Ascend feature toggles), declared in
`vllm_ascend/ascend_config.py` (`AscendConfig`) and read at runtime via
`get_ascend_config()`:

| Key | Default | Meaning |
|----------|---------|---------|
| `num_ubatches` | `1` | Number of concurrent ubatches. **Only `1` and `2` are supported.** `1` disables the feature; `2` enables the two-stream compute/comm overlap. Values > 2 are rejected at config init because the lock-step handoff is a two-stream ping-pong by design and yields no additional overlap beyond 2. |
| `ubatch_trigger_threshold` | `2048` | Minimum padded token count required to enable ubatch overlap for a given step. |

Set them via the `--additional-config` flag (online) or the `additional_config`
constructor kwarg (offline), e.g.:

```bash
vllm serve <model> --additional-config='{"num_ubatches": 2}'
```

```python
from vllm import LLM
LLM(model=<model>, additional_config={"num_ubatches": 2, "ubatch_trigger_threshold": 4096})
```

The feature only activates for a step when **all** of the following hold:

1. `num_ubatches > 1` (i.e. `maybe_use_ubatch()` is `True`).
2. The padded token count of the step exceeds `ubatch_trigger_threshold`.
3. After splitting into `num_ubatches`, the last ubatch is **not** empty
   (`is_last_ubatch_empty` from upstream vLLM is reused). This avoids pointless
   overlap when the batch is too small to fill all ubatches.
4. In data-parallel mode, **all** DP ranks agree the step should be ubatched.
   This is enforced by an all-reduce of a per-rank flag in
   `NPUModelRunner._sync_batch_across_dp`.
5. When sequence parallelism (SP) is enabled, the padded token count is rounded
   up to a multiple of `tensor_parallel_size * num_ubatches` (so each ubatch
   slice stays a multiple of `tensor_parallel_size` and the per-slice SP
   all_gather/reduce_scatter stays aligned). This applies to **both** DP==1 and
   DP>1; if the enlarged batch no longer satisfies the trigger threshold,
   ubatching is abandoned.
6. **The step contains prefill tokens** (`NPUModelRunner.with_prefill` is `True`,
   i.e. the attention state is not `DecodeOnly` or `SpecDecoding`). Decode and
   speculative-decoding steps always run on the FULL ACL graph and are never
   ubatched.

`UBatchRuntimeManager.log()` periodically reports the fraction of steps and the
fraction of tokens that ran under ubatch overlap (`ubatch_rate` and
`tokens_ubatch_rate`).

---

## 3. Architecture

### 3.1 Components

| File | Component | Role |
|------|-----------|------|
| `vllm_ascend/worker/ubatch_utils.py` | `UBatchRuntimeManager` | Process-global singleton orchestrating streams, events, threads, and forward contexts. |
| `vllm_ascend/worker/ubatch_utils.py` | `split_ascend_common_metadata` / `_make_ascend_common_metadata_with_slice` / `slice_query_start_locs` | Slice `AscendCommonAttentionMetadata` per ubatch, including the tricky "request split across ubatches" case. |
| `vllm_ascend/worker/ubatch_utils.py` | `should_enable_ubatch` / `maybe_use_ubatch` / `get_ubatch_trigger_threshold` / `resolve_num_tokens_across_dp` | Decision helpers. `resolve_num_tokens_across_dp` synthesizes a 1-element per-rank tensor for the DP==1 case so the per-ubatch forward-context loop stays uniform. |
| `vllm_ascend/worker/model_runner_v1.py` | `NPUModelRunner` changes | Hooks ubatch into `_determine_batch_execution_and_padding` (gated to prefill), `_build_attention_metadata`, `execute_model`, and `_model_forward_ubatches`. |
| `vllm_ascend/ops/rotary_embedding.py` | `get_cos_and_sin_slice` / `update_cos_sin` | Per-ubatch slicing of the rotary cos/sin tables driven by the thread-local token slice bound in `UBatchRuntimeManager.exec` (not the shared `curr_batch` cursor, which races under overlap). |
| `vllm_ascend/ops/register_custom_ops.py` | `_maybe_all_gather_and_maybe_unpad_impl` / `_maybe_pad_and_reduce_impl` | Skip DP-unpadding/padding while ubatch is running (mirrors the `enable_sp_by_pass` fast path). |
| `vllm_ascend/ops/fused_moe/fused_moe.py` | `AscendFusedMoE.forward_impl` | Wraps `prepare()` and `finalize()` of `_EXTRA_CTX.moe_comm_method` in `with rt.comm_section():` (a no-op context manager when ubatch is disabled, so the call sites need no `is_ubatch_running` guard). Because `AscendFusedMoE` (and its subclass `AscendSharedFusedMoE`) is the shared MoE base layer used by **all** Ascend MoE models, the overlap is universal — no per-model monkey-patch is required. |

### 3.2 Runtime data flow

```
                    ┌──────────────────────────────────────────────┐
                    │           NPUModelRunner.execute_model        │
                    │  1. decide should_ubatch (threshold + DP vote)│
                    │  2. build per-ubatch attn metadata            │
                    │  3. create one ForwardContext per ubatch       │
                    └───────────────────────┬──────────────────────┘
                                            │
                                            ▼
                ┌───────────────────────────────────────────────┐
                │          _model_forward_ubatches               │
                │   rt.forward_init()  -> creates streams/events │
                │   for each ubid:  rt.add_task_and_get_future(  │
                │           self._model_forward, ubid, <slices>) │
                │   rt.forward_finished() -> wait all events     │
                └───────────────────────┬───────────────────────┘
                                        │  (one task per ubatch, dispatched
                                        │   to a dedicated PersistentThread)
              ┌─────────────────────────┴──────────────────────────┐
              ▼                                                      ▼
    ┌─────────────────────┐                               ┌─────────────────────┐
    │  ubatch 0 (stream 0)│  compute ◄──┐   ┌──► compute  │  ubatch 1 (stream 1)│
    │                     │             │   │             │                     │
    │  MoE prepare()      │ ── yield ──► │   │ ◄── yield ──│  MoE prepare()      │
    │  ▲ switch to comm   │             │   │             │  ▲ switch to comm   │
    │  MoE finalize()     │ ◄── yield ── │   │ ── yield ──►│  MoE finalize()     │
    │  ▼ switch to compute│             │   │             │  ▼ switch to compute│
    └─────────────────────┘             │   │             └─────────────────────┘
                          handoff via compute_done_event /
                          comm_done_event + cpu_event signaling
```

### 3.3 Stream / event model

The manager allocates, per ubatch:

- a compute **stream** (`self.stream[ubid]`),
- a `compute_done_event[ubid]`, and
- a `comm_done_event[ubid]`,
- a CPU-side `threading.Event` (`self.cpu_event[ubid]`) used to gate the worker
  thread until the previous ubatch has produced its result.

Compute/communication serialization *within* a ubatch is enforced by the
`compute_begin`/`compute_end`/`comm_begin`/`comm_end` helpers, which chain the
current ubatch's stream to the previous ubatch's done-event. Cross-ubatch
handoff in the MoE layer is driven by:

- `yield_and_switch_from_compute_to_comm()` — finish compute, wake the next
  ubatch's compute, then wait for this ubatch's comm to become ready.
- `yield_and_switch_from_comm_to_compute()` — the symmetric operation.

The worker threads block on `cpu_event` until `switch_curr_batch()` sets it,
which guarantees the global `curr_batch` cursor advances in lock-step across
all threads. The global `forward_context._forward_context` pointer is updated
on every switch so that custom ops (e.g. rotary, MoE) read the correct
ubatch-local metadata.

### 3.4 Attention metadata splitting

`split_ascend_common_metadata` produces one `AscendCommonAttentionMetadata` per
ubatch slice. It must handle the general case where a **single request is
split across two ubatches**:

- `splits_first_request` — the first request in this slice is a continuation
  of a request that started in the previous slice. The per-request query
  lengths (`query_start_loc`) are shifted so the slice begins at query offset
  0.
- `splits_last_request` — the last request continues into the next slice. Its
  `seq_len` is reduced by the number of tokens that overflowed, and the
  `seq_lens` tensors are cloned because in-place mutation would break
  cudagraph capture.

For chunked prefill, the attention state is remapped: if the original state is
`PrefillNoCache(0)` or `ChunkedPrefill(3)` and the slice is **not** the first
chunk, the state is forced to `ChunkedPrefill(3)` so the backend appends to
existing KV cache rather than treating it as a fresh prefill.

### 3.5 ACL graph dispatch (no ubatch-specific wrapper)

The model is always wrapped with the standard `ACLGraphWrapper`
(`vllm_ascend/compilation/acl_graph.py`); there is **no** ubatch-specific wrapper.
Dispatch between graph-replay and eager happens entirely inside
`ACLGraphWrapper.__call__`, keyed on the forward context's
`cudagraph_runtime_mode`:

- **Decode steps** → `CUDAGraphMode.FULL` → ACL graph capture/replay.
- **Prefill steps** (including ubatch) → `CUDAGraphMode.NONE` → eager
  (`self.runnable(*args)`, `acl_graph.py` NONE-mode passthrough). Eager execution
  is required because ubatch launches real streams/events that cannot be
  captured into a static graph.

Because ubatch is gated to prefill (§2, condition 6), and prefill always
presents `cudagraph_runtime_mode == NONE`, the per-ubatch forward contexts
(which inherit that mode) cause `ACLGraphWrapper` to run the model eagerly during
ubatch — exactly what is needed. No ubatch-specific ACL graph wrapper is
needed.

### 3.6 Model- and op-level integration

The compute/comm handoff is **inlined into the shared MoE base class**
`AscendFusedMoE.forward_impl` (in `vllm_ascend/ops/fused_moe/fused_moe.py`):

- Around `prepare()` and `finalize()` of `_EXTRA_CTX.moe_comm_method`, the
  runtime wraps the call in `with rt.comm_section():`. `comm_section` is the
  single entry point for the lock-step ping-pong: when ubatch is disabled it is
  a no-op context manager (so the MoE layer needs no `is_ubatch_running` guard
  of its own); when enabled it yields the thread to the peer ubatch worker,
  switches from compute to comm on entry and back on exit, and asserts the peer
  reached the same `comm_section` depth (symmetry guard). These are the only
  places the dual-batch path yields, so all overlap is concentrated at MoE
  collective boundaries.

Because `AscendFusedMoE` (and its subclass `AscendSharedFusedMoE`) is the shared
MoE layer used by **every** Ascend MoE model, inlining the handoff there makes
ubatch overlap universal — any model that goes through `AscendFusedMoE` gets the
behavior automatically when ubatch is enabled. **No per-model monkey-patch is
required.**

The eager-vs-graph model dispatch is handled centrally by
`ACLGraphWrapper.__call__` (§3.5), which every model goes through, so no
model-level patching is needed for that either.

Two custom ops are taught to short-circuit while ubatch is running (mirroring
the existing `enable_sp_by_pass()` path), because DP padding/unpadding is
already handled by the model-runner-level `num_tokens_across_dp` slicing:

- `maybe_all_gather_and_maybe_unpad`
- `maybe_pad_and_reduce`

---

## 4. Control flow summary

```
execute_model()
├── _determine_batch_execution_and_padding()
│     ├── should_ubatch = should_enable_ubatch(...) (+ DP all-reduce)
│     ├── should_run_ubatch() consumes the one-shot warmup guard: the FIRST
│     │   step that would otherwise enable ubatch is skipped so the CANN
│     │   caching allocator warms up on a single-forward path first.
│     └── _ubatch_blocked_reason() gate (DeepStack / DSA / PCP)
├── if should_ubatch:
│     ├── create ubatch_slices  (rt.ubatch_slices = ...)
│     └── create per-ubatch ForwardContexts (rt.forward_contexts = [...])
├── _build_attention_metadata()  -> per-ubatch attn metadata
└── _model_forward:
      if should_ubatch:
          _model_forward_ubatches()
            ├── rt.forward_init()                # streams, events, log
            ├── for ubid in range(num_ubatches):
            │     rt.add_task_and_get_future(self._model_forward, ubid, ...)
            ├── gather all futures (torch.cat results)
            └── rt.forward_finished()            # join streams
      else:
          self._model_forward(...)               # original single-batch path
```

---

## 5. Limitations & constraints

- **First PP rank only.** `_model_forward_ubatches` asserts
  `get_pp_group().is_first_rank is True`. The feature is currently not
  supported on PP-intermediate or PP-last ranks.
- **Works for all non-DSA MoE models.** The compute/comm handoff is inlined
  into `AscendFusedMoE.forward_impl`, the shared MoE base layer, so any model
  that routes experts through `AscendFusedMoE` / `AscendSharedFusedMoE` gets
  ubatch overlap automatically. No per-model patch is needed. The exception is
  **DSA (Distributed Sparse Attention) models** (e.g. GLM-5, DeepSeek-V3.2):
  their SFA attention path relies on thread-unsafe global cos/sin buffers and
  DSA-specific operators that have not been validated for concurrent
  multi-stream execution, so `_ubatch_blocked_reason` disables ubatch for them
  (logged at `debug` level). **PCP (prefill context parallelism)** is also
  blocked, because the per-ubatch attention metadata splitter does not yet
  carry PCP slot-mapping state.
- **Sequence-parallel interaction.** When SP is on, the padded token count must
  be rounded up to `tp_size * num_ubatches`; if this enlargement breaks the
  trigger threshold, ubatching is silently disabled for the step.
- **Prefill-only.** Ubatch overlap only activates on steps that contain prefill
  tokens (`with_prefill`). Pure decode and speculative-decoding steps always run
  on the FULL ACL graph. Mixed prefill+decode batches (chunked prefill) are
  eligible, since `with_prefill` is true for any step containing at least one
  prefill request.
- **Eager prefill path.** Prefill (and therefore ubatch) runs eager, so it
  cannot reuse the captured full ACL graph; throughput on small batches that
  *would* hit ubatch is lower than the graph path — hence the 2048-token
  threshold.
- **Global state.** A process-wide `UBatchRuntimeManager` singleton holds
  runtime state. The singleton and its worker-thread pool are now created
  lazily on first use (see *Changes applied*), so merely importing the module
  no longer spawns daemon threads.

### Multimodal model support

Ubatch overlap supports multimodal models (VLM) with the following matrix:

| Model family | M-RoPE | DeepStack | Supported? |
|--------------|--------|-----------|------------|
| qwen2-vl / qwen2.5-vl | Yes | No | Yes |
| internvl / glm4v | No | No | Yes |
| qwen3-vl / qwen3-vl-moe | Yes | Yes | Yes (requires `patch_qwen3_vl_deepstack`) |
| qwen3-omni-moe-thinker | Yes | Yes | Yes (requires `patch_qwen3_vl_deepstack`) |

**Requirements for VLM + ubatch:**

1. **M-RoPE positions** are sliced on dim=1 automatically via
   `slice_model_inputs_for_ubatch`, so `(3, seq_len)` is handled correctly
   without per-model changes.
2. **DeepStack models** must load `patch_qwen3_vl_deepstack` (auto-registered
   via `patch/worker/__init__.py`). It reuses the thread-local `token_slice`
   (same mechanism as rotary embedding) to make the cross-layer buffer
   `_get/_clear_deepstack_input_embeds` read/clear the per-ubatch
   `[start:stop]` slice. Without it, a defensive gate (`_ubatch_blocked_reason`
   on `NPUModelRunner`) disables ubatch and logs the reason at `debug` level.
3. **Vision encoder** runs in `_preprocess` (single-stream, before ubatch
   dispatch) — it is unaffected by ubatch overlap.

---

## 6. Design notes

The following design choices were made up-front for this feature:

1. **Enablement via `additional_config`**. The feature is configured through
   vLLM's `additional_config` mechanism (keys `num_ubatches` and
   `ubatch_trigger_threshold`), declared on `AscendConfig`
   (`vllm_ascend/ascend_config.py`) and read via `get_ascend_config()`, matching
   the convention used by all other Ascend feature toggles.
2. **Lazy singleton**. `get_ubatch_runtime_manager()` constructs the
   `UBatchRuntimeManager` on first use instead of at import time, and the
   worker-thread pool is created lazily on first dispatch. This keeps imports
   cheap and avoids spawning threads when the feature is disabled
   (`num_ubatches == 1`).
3. **Thread robustness**. `UBatchRuntimeManager._PersistentThread.run` wraps the
   per-task `exec()` call in try/except, logs the failure, and propagates the
   exception on the `Future` so callers do not hang on `future.result()` when a
   task raises. Additionally, `exec()` runs `compute_end()`/`switch_curr_batch()`
   in a `finally` block (so the peer worker thread is always woken and the
   cursor advances even when `target()` raises), and `_model_forward_ubatches`
   runs `forward_finished()` in a `finally` block so the runtime is reset
   (`is_ubatch_running=False`, all streams joined) before the exception
   surfaces — otherwise the next scheduler step would hang or mis-overlap on a
   half-initialized runtime.
4. **`item()` usage** (kept). `_make_ascend_common_metadata_with_slice` calls
   `.item()` to compute `max_query_len`/`max_seq_len`. This runs once per
   ubatch (not per token), and operates on the `*_cpu` tensors so there is no
   D2H sync; the values are needed as Python ints to populate the metadata
   dataclass.

---

## 7. Testing

Unit tests live in `tests/ut/worker/test_ubatch_utils.py` and cover the
pure-Python, hardware-free pieces of the feature:

- `slice_query_start_locs` correctness.
- `split_ascend_common_metadata` for whole-request splits and for the
  request-spanning-ubatches case (first/last request continuation), including
  attention-state remapping for chunked prefill.
- `should_enable_ubatch` / `maybe_use_ubatch` / `get_ubatch_trigger_threshold`
  decision logic, including the `is_last_ubatch_empty` short-circuit and the
  `ubatch_trigger_threshold` `additional_config`-key override.

The stream/event/thread orchestration in `UBatchRuntimeManager` and the MoE
handoff points are NPU-dependent and are exercised by the nightly/ST suites
under `tests/e2e/nightly/` on real Ascend hardware.

Run the unit tests with:

```bash
pytest -sv tests/ut/worker/test_ubatch_utils.py
```
