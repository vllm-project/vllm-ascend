# Context Parallel Piecewise Graph

Piecewise graph support for the Ascend context-parallel attention backends (`sfa_cp`, `mla_cp`, `dsa_cp`, `dsa_v41_cp`) keeps the compiled-and-captured ACL graph backbone while running CP attention, its HCCL collectives, and its auxiliary NPU streams as a single eager island between graph replays.

This document covers the design for Prefill Context Parallel (PCP), Decode Context Parallel (DCP), and PCP+DCP stacking. Execution semantics of the two parallelisms themselves are described in [Context Parallel](context_parallel.md); the generic graph model is described in [ACL Graph](ACL_Graph.md).

## Background

### Graph modes on Ascend

vLLM owns graph dispatch. `vllm-ascend` supplies `vllm_ascend.compilation.acl_graph.ACLGraphWrapper` through `NPUPlatform.get_static_graph_wrapper_cls()`, and `BreakableACLGraphWrapper` for the upstream breakable path. Three execution shapes matter here:

| Aspect | `NONE` | `PIECEWISE` / `FULL_AND_PIECEWISE` | `FULL` / `FULL_DECODE_ONLY` |
| --- | --- | --- | --- |
| Backbone | Python eager | FX graph split at `splitting_ops`, one captured ACL graph per compile size | One ACL graph for the whole model |
| Attention | eager | eager: the attention custom op is a split point | captured; kernel arguments patched per step |
| Attention parameter refresh | none | none needed (metadata read directly) | `torch.npu.graph_task_update_*` (`UpdatableGraph`, `GraphParams`) |
| Token extent | as scheduled | padded to the dispatcher's `num_tokens`, identical on all ranks | padded to `num_tokens` |
| Request extent | as scheduled | not padded (`BatchExecutionDescriptor.num_reqs is None`) | padded to `num_reqs` |
| Query length | free | free (`uniform_token_count is None`) | must be uniform (`AttentionCGSupport.UNIFORM_BATCH`) |
| Capture metadata | n/a | dummy `prepare_inputs_to_capture(...)`, `full_cudagraph=False` | `build_for_graph_capture(...)`, decode-only or spec |

Two configuration facts follow from `NPUPlatform._setup_compile_backend` (`vllm_ascend/platform.py`):

- Piecewise mode requires `CompilationMode.VLLM_COMPILE` unless `VLLM_USE_BREAKABLE_CUDAGRAPH` is set, and it extends `splitting_ops` with `vllm::mla_forward`, `vllm::dsa_forward`, `vllm::minimax_m3_fused_sparse_forward`, and `vllm::glm5next_kda_forward_with_output` while disabling `npugraph_ex`, static kernel, and super kernel.
- `mla_forward` and `dsa_forward` are registered as mutating custom ops (`mutates_args=["output"]`) and decorated with `eager_break_during_capture`, so they are both a compile split point and a breakable-capture break point.

### What is Ascend-specific

The mode names come from vLLM, but every mechanism below is the Ascend implementation, and this design changes only Ascend code:

- Capture is ACL graph capture through `torch.npu.graph(...)` into `torch.npu.NPUGraph`, driven by `ACLGraphWrapper` with one shared device graph pool and one cache entry per `BatchDescriptor`. `CUDAGraphMode`, `cudagraph_mode`, `cudagraph_capture_sizes`, and `BreakableCUDAGraph*` are upstream vocabulary reused on NPU; they do not imply CUDA capture.
- Per-step attention parameter refresh is not a CUDA exec-node update. On Ascend it is `torch.npu.graph_task_group_begin/end` plus `torch.npu.graph_task_update_begin/end`, with FIA workspaces and handles held in `GraphParams` and tasks registered through `get_capture_resource` / `register_task`. Piecewise does not use any of it, because attention is eager.
- CP side streams are NPU streams (`cp_decode_comm_stream`, `cp_chunkedprefill_comm_stream`). Capture requires every forked stream to be joined; the dominant capture failure on Ascend is CANN stream-resource exhaustion (error code `207008`, `insufficient_stream_resources`, or the old-HDK `Alloc sq cq fail`), which `ACLGraphWrapper` already translates into guidance.
- Captured pieces may contain HCCL collectives (model-level PCP gather, sequence-parallel exchange, MoE all-to-all, TP all-reduce). At replay these run as captured HCCL tasks with no Python-side group call, so their shapes must match the ones captured on every rank.
- Piecewise disables `npugraph_ex`, static kernel, and super kernel (`NPUPlatform._setup_compile_backend`), so CP gains come from removing host launch overhead and from eager-island overlap, not from graph fusion passes.
- Device-to-host synchronization is expensive on NPU and stalls the async scheduler, so CP metadata keeps lengths on the CPU/Numpy side and the island avoids `.item()` / `.tolist()` on replay-varying device data.

### CP execution in one paragraph

PCP splits prefill queries with DualChunkSwap and, with `DCP == 1`, assigns decode requests round-robin to PCP ranks. DCP shards the KV cache over an interleaved layout and merges partial attention results with softmax statistics. Both reuse the v1 backend structure: metadata builders inherit the v1 builders and add CP fields; implementations inherit the v1 implementations and override only the stages whose communication, cache view, or layout differs (`vllm_ascend/attention/context_parallel/`). All CP-specific communication (Q-head gather, KV gather, `dcp_a2a_fused` merge, fused prolog, O-projection exchange) happens inside the attention implementation, that is, inside `mla_forward` or `dsa_forward`.

### Why this combination is hard

CP and static graphs disagree on the same three quantities: how many tokens a rank runs, how many requests it owns, and which collectives it must issue. Today the CP code handles the FULL axis explicitly:

- `mla_cp.py` / `attention_cp.py` capture FIA tasks and refresh their kernel arguments per step: `register_task` / `get_capture_resource` during capture, then `update_graph_params()` through `torch.npu.graph_task_update_*`.
- `dsa_cp.py` / `dsa_v41_cp.py` carry a `full_graph_mode: bool` that selects padded versus actual extents for restore indices and request metadata.
- `sfa_cp.py` implements `build_for_graph_capture()` for `DecodeOnly` / `SpecDecoding` only, and sizes CP buffers from `cudagraph_capture_sizes`.

Both `PIECEWISE` and `FULL` have `cudagraph_mode != NONE`, so the CP path cannot tell them apart from the existing booleans: a `has_full_cudagraphs()` guard silently selects FULL semantics under piecewise, and a `_EXTRA_CTX.capturing` guard silently selects capture-time semantics. A piecewise-specific contract is missing.

## Scope

In scope:

- Primary delivery: DeepSeek-V4 through `dsa_cp` (phase P1) and DeepSeek-V4.1 through `dsa_v41_cp` (phase P2). CP for the DSA family is PCP-only today (`dsa_v41_cp` rejects `DCP != 1`, and the DCP support matrix covers MLA, GQA, and SFA only), so P1/P2 cover PCP and PCP decode sharding, not DCP.
- Parallel track: `mla_cp` and `sfa_cp` (including the replicated-indexer DCP variant) for the MLA and GLM families, with PCP, DCP, and PCP+DCP. This track is off the DeepSeek critical path, but the shared contract (`CPGraphSpec`, `supports_cp_piecewise()`) is written generically so it does not need a second refactor.
- `PIECEWISE` and `FULL_AND_PIECEWISE`, with the same stage order applied inside every phase (see [Rollout](#rollout)).
- Speculative decoding paths that are already supported with CP (MTP, Eagle3, DSpark).
- `NONE` remains the reference; `FULL` / `FULL_DECODE_ONLY` behavior must not regress.

Out of scope:

- Capturing CP attention itself into the graph. The attention island stays eager (see D1); replacing the island with captured kernels and `graph_task_update` is a separate follow-up.
- Backend/hardware combinations that CP does not support today (see the [Context Parallel Guide](../../user_guide/feature_guide/context_parallel.md)).
- KV transfer, weight switching, and offload connectors beyond the requirement that they keep working unchanged.

## Design

### Design decisions

| ID | Decision | Rationale |
| --- | --- | --- |
| D1 | CP attention is one eager island; its collectives, side streams, and host-side length lists never enter a captured piece | Matches how non-CP piecewise already works (attention is a split op) and how SGLang keeps CP-dependent branches out of graphs |
| D2 | Graph mode is passed to the CP code as an explicit spec, not inferred from booleans | `has_full_cudagraphs()` and `_EXTRA_CTX.capturing` are ambiguous for piecewise and have already caused "works in FULL, wrong in PIECEWISE" bugs |
| D3 | Every rank uses the same piecewise batch descriptor and therefore the same token extent | Captured pieces contain collectives (model-level PCP gather, sequence-parallel exchange, MoE all-to-all); diverging sizes hang |
| D4 | Padded rows are garbage rows: no host branch may depend on them | Piecewise pads to compile sizes that do not correspond to scheduled work |
| D5 | Reuse the existing eager CP implementation; only mode plumbing, buffer sizing, and ordering change | The GLM-5.3 PCP + `torch.compile` fix (#20586) validates this shape of change |

### Contract for the eager island

`mla_forward` / `dsa_forward` are already the boundary. The CP implementations must satisfy:

- **C1 Isolation.** A CP collective, a side-stream fork, or a host-side per-request list is created and consumed inside a single invocation of the attention op. No state that describes a rank-local layout escapes into the compiled backbone.
- **C2 Static outputs.** The op mutates a caller-owned `output` tensor and returns nothing. No tensor allocated inside the island may be retained across steps.
- **C3 Rank-invariant dispatch.** Collective sizes derive from `batch_desc.num_tokens` (identical on all CP ranks) or from the global PCP request view, never from rank-local request counts or rank-local actual token counts.
- **C4 Metadata duality.** Builders must produce replay metadata from the real local batch and capture metadata from dummy batches, in decode *and* prefill/mixed shapes. No capture-time-only branch may change the layout used at replay.
- **C5 Padding tolerance.** Rows past the metadata extent are skipped by attention (slots `-1`, zero-length sequences, `is_padding` rows). No `.item()` / `.tolist()` on data that changes between replay and the following step.
- **C6 Ordering.** A side stream joins the main stream before the island returns; tensors produced by an earlier replay that a side stream reads get `record_stream()`; dependencies between replay and the next eager island are expressed with events, not with a host-side stream synchronize.
- **C7 Capture invariance.** The number and size of captured pieces are identical on every CP rank.
- **C8 Fail fast.** If a backend cannot satisfy C1-C7 for a configuration, initialization raises with a precise message instead of downgrading silently.

### Graph-mode-parametric CP metadata

Replace the `full_graph_mode: bool` parameters with a small value object threaded through the CP builders:

```text
CPGraphSpec:
    mode: NONE | PIECEWISE | FULL          # graph mode that will replay this batch
    padded_num_tokens: int                 # dispatcher token extent, 0 for NONE
    padded_num_reqs: int | None            # None for NONE and PIECEWISE
    capturing: bool                        # capture-time metadata request
```

Rules the CP builders apply:

| Field | `NONE` | `PIECEWISE` | `FULL` |
| --- | --- | --- | --- |
| Restore-index extent (DSA PCP) | actual tokens | actual tokens | `num_tokens_after_padding` |
| Request metadata (DSA PCP) | actual requests | `min(num_tokens, max_num_reqs)` rows, query offsets cover padded decode tokens | `batch_desc.num_reqs` |
| Query-length padding (`actual_seq_lengths_q`) | actual | actual offsets; padded decode rows are covered only for a uniform decode batch (mirrors `_build_graph_common_attn_metadata`) | required |
| CP buffer capacity | `max_num_batched_tokens` | `max(compile_sizes)`, capped by request capacity | `max(cudagraph_capture_sizes)` |
| `update_graph_params()` | unused | unused (attention is eager) | used |

`_EXTRA_CTX.capturing` keeps its current meaning ("we are inside a full-model capture") and must not be reused as a proxy for "graph replay is active". `ModelWithContext.forward` already excludes `PIECEWISE` when setting it; CP code must rely on `CPGraphSpec.mode` instead.

### Cross-rank consistency

Three invariants make CP collectives safe under capture:

1. **Token extent.** `AscendPCPManager.partition_batch()` already pads the local token extent to `batch_desc.num_tokens` whenever `cg_mode != NONE`, so PCP ranks agree for token-major collectives. Piecewise adds no request padding, so this is the only agreement mechanism.
2. **Request-shaped collectives.** Any collective sized by request count (compact KV-gather plans, per-request block-table packing, DSA global metadata) must be derived from the global PCP request view, as `sfa_cp.py` already does with `global_dcp_block_table` / `global_dcp_num_blocks`. New code must not reintroduce a local loop over local rows.
3. **Dispatch key.** `BatchDescriptor` (or `key_fn`) must be a function of the global batch only. A CP rank that pads its local batch differently must still request the same descriptor, otherwise the ranks capture different pieces (C7) and replay mismatched collectives.

Model-level PCP collectives that sit in the compiled backbone (`sp_all_gather` / `tensor_model_parallel_all_gather` in the model forward, the sequence-parallel shard/reduce-scatter ops, and the PCP O-projection exchange when it is not inside the attention op) are allowed only because of invariant 1. Any new collective that violates it must be moved into the attention island, and P0 must either prove the bound for every supported model or move the gather into the island; this is not left as an unspecified follow-up.

A rank with zero scheduled tokens is not an exception. `partition_batch()` gives it the same `num_tokens_after_padding` as every other rank plus a single placeholder request, so it joins token-major collectives at the global extent while contributing one empty row to request-shaped metadata. Code must never branch on `num_local_tokens == 0` or `num_local_reqs == 1` to skip work.

The invariants are only as good as their enforcement, so capture ends with a cross-rank fingerprint: each rank hashes its captured piece sequence (mode, piece count, per-piece `num_tokens`, compile-size key) and the CP group all-reduces the hashes; a mismatch raises and names the differing rank. The check costs one small collective per capture set, is enabled by default in CI and behind `VLLM_ASCEND_CP_GRAPH_CHECK` in production, and converts a first-replay hang into a startup failure.

### Streams and ordering

The CP implementations use two private streams: `cp_decode_comm_stream()` (overlap history packing with current attention) and `cp_chunkedprefill_comm_stream()` (gather the history query while the current chunk is computed). Under piecewise:

- The islands are not captured, so forks are legal; but the fork must be closed before the island returns (`main_stream.wait_stream(comm_stream)`), otherwise the next captured piece can start on the main stream while a collective is still writing.
- In `FULL` mode `ACLGraphWrapper.__call__` inserts `torch.npu.current_stream().synchronize()` before replay. Piecewise does not, and must not add one: replay/step ordering is already guaranteed by the dispatcher. Instead the island records an event after its last write to any tensor that a later replay reads, and the replay side waits on it.
- Tensors received from graph-owned memory and then read on a comm stream must be `record_stream()`-ed, matching the existing pattern in `attention_cp._forward_decode_dcp()`.
- Breakable capture needs the same rule, not a different one. `BreakableCUDAGraphCapture.add_eager()` closes the current segment, runs the island eagerly, and only then restarts capture, so the island never captures its own forks. Because C6 requires the join before the island returns, the next `capture_begin()` observes an idle side stream. The implementation must preserve that ordering (join on the main stream, then return) and a unit test must assert that no comm stream is still active when the island returns, for both wrappers.

### Buffers and memory

Piecewise capture runs roughly once per layer per compile size, with `gc.collect` and `torch.npu.empty_cache` patched out after the first graph (`ACLGraphWrapper`), so per-layer CP buffers must exist before capture:

- CP metadata buffers are allocated in the builder constructor and sized from the piecewise descriptor capacity (`compile_sizes`), not from capture-time observed lengths: `_hidden_restore_idx_buffer`, `_global_rope_buffers`, `dcp_local_seq_lens_buf`, `pcp_indexer_slot_mapping_buf`, `block_table_replicated_view_buf`.
- `dsa_v41_cp.py` currently sizes request buffers from `max(cudagraph_capture_sizes)`. Under piecewise that bound equals `max_num_batched_tokens`, which is far larger than the request count; cap it with the CP request capacity (`2 * max_num_seqs + 1` for PCP splits).
- Captured workspace tensors stay weak-referenced (`weak_ref_workspaces`) so that the graph pool, not Python, owns them.
- The island must not allocate a fresh tensor whose address is captured by an adjacent piece. `torch.empty` inside the island is acceptable only when the result is consumed inside the same island.

### Per-backend design

#### `mla_cp`

- FULL-only machinery: `MLASplitAttentionGraphParams`, `update_graph_params()`, and the `graph_task_group_begin/end` capture path stay as they are. Under `PIECEWISE` they are never entered; the implementation reads `attn_metadata` directly.
- `use_history_current_split_decode()` is a metadata-only decision and is kept unchanged. The split produces two FIA tasks per layer; both, plus `fused_dcp_lse_combine`, stay inside the island so their ordering is preserved.
- Decode: history attention runs on the main stream while current attention is issued on `cp_decode_comm_stream()`; the island joins the stream and returns the combined output.
- Chunked prefill: the history-query gather uses `cp_chunkedprefill_comm_stream()`; gathered sizes follow the padded local token count, which is rank-invariant by invariant 1.
- Spec decode: the history/current split is forced for causal multi-token verification, so the split layout is independent of graph mode. DCP MTP keeps its current layout.

#### `sfa_cp`

- SFA runs through the shared MLA layer boundary (`vllm::mla_forward`), so it inherits the island.
- DCP replicated indexer: index selection uses the replicated LightningIndexer view while SFA KV stays DCP-local. The rank-local remap and the compact KV-gather plan must be computed from the global PCP request view (already implemented for `PCP+DCP`) so all senders pack the same block IDs.
- Fused prolog: local cache rows are written first, then packed, gathered, and written to receiver caches; the gather size follows the padded token extent.
- Piecewise adds one requirement: `build_for_graph_capture()` must accept prefill/mixed dummy batches, or the capture path must skip CP metadata construction for backends whose attention is eager. The preferred fix is to build capture metadata through the ordinary `build()` with `CPGraphSpec(capturing=True)` and drop the `DecodeOnly` / `SpecDecoding` restriction.
- `pcp_indexer_slot_mapping_buf` is sized for `max_num_input_tokens * pcp_size`; keep that bound and validate it against the piecewise padded extent at build time.

#### `dsa_cp` and `dsa_v41_cp`

- `full_graph_mode` becomes `CPGraphSpec.mode == FULL`. The existing `full_graph_mode=False` branches are already the piecewise-correct behavior: restore indices only cover actual tokens, and DSA cache updates run with the actual extent while padded rows are trimmed inside the island.
- `AscendDSAPCPMetadataBuilder.get_cudagraph_support()` stays `UNIFORM_BATCH`; piecewise does not need uniformity, and this value only gates `FULL`.
- PCP query layout: `_build_graph_common_attn_metadata()` expands padded query offsets only when request padding exists (`full_graph_mode`). Under piecewise `num_reqs` is `None`, so the local metadata keeps the real offsets and the island trims padded rows.
- DSA V4.1: the global RoPE buffer stays graph-stable (`_global_rope_buffers`); the SWA/compressor state must not branch on graph mode inside the island. `swa_bounded_replay` is disabled under piecewise at P2 entry because `make_decoder_replay()` only accepts `NONE` and `FULL_DECODE_ONLY`, and 950 products wait for the packaged-cache uncompiled fallback (see [Rollout](#rollout)).
- Both backends pass through `vllm::dsa_forward`, which already wraps attention *and* the O-projection collectives, so OTP HCCL stays in the island.

#### GQA DCP (`attention_cp`)

Not a primary target, but it shares `DCPImplMixin` and `DCPFIAParamProvider`. If it is enabled later, the same contract applies: FIA params are published through `register_task` only for FULL; piecewise calls the operator directly with fresh kernel arguments.

### Speculative decoding

- The target model runs with the configured CP topology; the draft model is replicated with logical `PCP=1` and uses its own graph parameters (`get_draft_graph_params`, `get_draft_graph_prefill_params`).
- Draft graphs are captured separately, so the island contract must hold for both target and draft impl classes, including `_EXTRA_CTX.is_draft_model` / `is_draft_model_prefill` being set before metadata resolution.
- Fixed query length in MTP verification yields uniform decode pieces; varlen verification must still work because piecewise does not require uniformity.
- Dynamic draft length remains out of scope, consistent with the CP guide.

### Capability declaration

Upstream already owns the downgrade path, so the design hooks into it instead of inventing a new interface. `CompilationConfig.resolve_cudagraph_mode_and_sizes()` lowers the requested mode using the attention backend's `get_cudagraph_support()` and `splitting_ops_contain_attention()`, and independently drops piecewise when `piecewise_capture_available` is false (no compiled submodule and no breakable graphs). `vllm_ascend/patch/worker/patch_model_runner.py` computes that flag and forwards the resolved mode to `ModelCudaGraphManager`.

The CP backends therefore declare capability in two places:

- `get_cudagraph_support()` keeps `UNIFORM_BATCH`, which gates `FULL` only. P0 adds a piecewise-specific signal (a new `AttentionCGSupport` value or a CP-side predicate) so a backend can accept `PIECEWISE` while still rejecting `FULL`.
- A per-backend `supports_cp_piecewise(vllm_config)` predicate is the single source of truth for C8, consumed by `AscendPCPManager.validate_config()` (and, for draft models, by the speculator's validation).

Without that predicate, C8 degrades into "run until it hangs", so it is a P0 deliverable rather than an open question.

### Validation and fallback

Initialization validates the requested combination and fails fast (C8):

| Condition | Action |
| --- | --- |
| `prefill_context_parallel_size > 1` and `cudagraph_mode == FULL` | reject (existing message) |
| CP enabled, `cudagraph_mode` has piecewise, `CompilationMode != VLLM_COMPILE`, breakable disabled | reject with the upstream piecewise assertion |
| CP enabled and `supports_cp_piecewise()` is false | reject, suggest `NONE` or `FULL_DECODE_ONLY` |
| `dcp_size > 1` with `PCP` on a backend without a global request view | reject |
| Piecewise requested but the model has neither a compiled submodule nor breakable graphs | upstream `piecewise_capture_available=False` downgrades the mode; log the downgrade instead of failing |

Ascend owns the PCP gates. `AscendPCPManager.validate_config()` replaces the upstream static method and never calls `super()`, so the upstream sparse-MLA whitelist (`vllm/v1/worker/gpu/pcp_manager.py`: "supports CUDA graphs only for DeepSeek-V3.2 and GLM-5.2/5.3", plus the `dcp_comm_backend == "ag_rs"` requirement) does not run on this path. The FULL-only rejection in `vllm_ascend/worker/v2/pcp_manager.py` stays; P0 re-derives the remaining gates for CP piecewise and re-audits them whenever upstream adds one, so the divergence stays intentional and documented rather than accidental.

Runtime hazards keep their existing guidance: capture failures caused by stream-resource exhaustion surface the `ACLGraphWrapper` hint (`reduce cudagraph_capture_sizes`, prefer `FULL_DECODE_ONLY`, or disable graph mode); a collective mismatch shows up as a hang on the first replay step, so the diagnostics section below matters.

### Diagnostics

- `--cudagraph-metrics` with log stats enabled reports padded versus unpadded tokens per mode and makes piecewise replays visible.
- `VLLM_LOGGING_LEVEL=DEBUG` enables the input-address stability assertion in both wrappers; a CP buffer resized after capture fails here instead of producing wrong results.
- A capture-time counter check ("attention ran eagerly N times during capture") belongs in the unit tests rather than in production logging.
- The cross-rank fingerprint prints per-rank piece counts and sizes on mismatch, which separates "wrong descriptor" from "wrong collective size" before any CANN analysis.
- For hangs, capture `CANN` logs on the lowest-rank device first; a mismatch is almost always a rank-invalid collective size rather than a kernel bug.

## Correctness argument

Three properties make piecewise CP equal to eager CP:

1. **Same metadata.** The island receives exactly the metadata that eager execution would build: `CPGraphSpec` only affects capture-time extents and buffer capacities, never the replay-time metadata.
2. **Same arithmetic.** The compiled pieces compute the same operations on the same rows; padded rows contribute no attention, no cache writes (slots `-1`), and are excluded from sampling.
3. **Same collective shapes.** Invariants 1-3 guarantee that every rank replays the same piece and issues the same collective sizes, so the distributed computation matches eager execution step for step.

The remaining risk is not the arithmetic but the ordering (C6) and the buffer lifetime (C2), which is why they are stated as testable contract items.

## Testing

| Layer | Cases |
| --- | --- |
| Unit | `CPGraphSpec` plumbing for each builder (NONE / PIECEWISE / FULL); padded-row trimming in every CP impl; buffer capacity assertions; island registration (`mla_forward` / `dsa_forward` present in `splitting_ops`, `eager_break_during_capture` applied) |
| Numeric equivalence | eager vs `PIECEWISE` logits/attention output for prefill-only, chunked prefill, decode-only, mixed batch, prefix-cache hit, and MTP `K=1,3`; both runs use a fixed seed, fixed expert placement (dynamic EPLB off, static redundant experts) and the same MoE dispatch configuration, since dynamic EPLB changes the numeric path; tolerance limited to kernel nondeterminism |
| Topology | PCP 2/4/8, DCP 2/8, PCP+DCP where supported; uneven DualChunkSwap fragments; DCP interleave sizes 1 and `block_size` |
| Zero-token rank | a rank with no scheduled prefill or decode tokens, per backend: assert its token extent equals the descriptor's, that it joins every collective, and that its placeholder row produces no cache writes and no logits |
| Graph assertions | capture count, cross-rank fingerprint, address stability under DEBUG, "attention ran eagerly at capture" counter, `num_cudagraph_captured` sanity |
| Stress | 200-step random-skew run (request ownership churn on PCP, mixed batches) to catch collective divergence and stream races |
| E2E | A2/A3 and 950 products, Model Runner V2, one config per backend, both `PIECEWISE` and `FULL_AND_PIECEWISE` |
| Regression | `FULL_DECODE_ONLY` and eager baselines unchanged for the same configs |

Tests must cover every source file that hosts the contract points: `vllm_ascend/attention/context_parallel/{mla_cp,sfa_cp,dsa_cp,dsa_v41_cp,attention_cp,common_cp}.py`, `vllm_ascend/worker/dcp_utils.py`, `vllm_ascend/worker/v2/pcp_manager.py`, and `vllm_ascend/platform.py`.

## Performance

Expected gains are the usual piecewise ones: host launch overhead removed from everything except the attention island, which dominates only for short sequences. The CP-specific costs to watch:

- **Island launch share.** The gain is bounded by how much launch time actually leaves the island, and DSA is the worst case: `dsa_forward` keeps the indexer, compressor, sparse MLA, and O-projection HCCL eager. In P0, measure the share of host launches that stay inside the island on a DeepSeek-V4 DSA trace; if it dominates, reconsider the island boundary (for example a dense-MLA sub-island) before P1 starts, instead of shipping a mode that cannot pay off.
- CPU time inside the island: piecewise attention runs in Python, and every extra view, slice, or `.tolist()` is paid per layer per step. `sfa_v1.py` already documents this constraint.
- Capture cost and stream pressure: `compile_sizes × number of pieces` graphs; CP does not need extra capture sizes, so the existing pruning rules apply.
- Side-stream efficiency: overlap between history and current attention must survive the island boundary; measure it rather than assume it.

Benchmarks to report per backend: TTFT and TPOT versus `FULL_DECODE_ONLY` and eager, at 4K/32K/128K inputs, decode-only and mixed workloads, fixed concurrency.

## Rollout

Phasing follows the DeepSeek generations because the code does: `dsa_v41_cp` reuses `dsa_cp.AscendDSAPCPMetadataBuilder` and adds replicated caches, compressor metadata, and the V4.1 replay path, so DeepSeek-V4 is the smaller target and its work is inherited by V4.1.

| Phase | Scope | Exit criteria |
| --- | --- | --- |
| **P0 shared mechanism** (no new model) | `CPGraphSpec` plumbing, `supports_cp_piecewise()` capability declaration, cross-rank fingerprint check, PCP gate re-derivation, island launch-share measurement, contract assertions, validation matrix, unit tests. Validate the contract on an existing piecewise-capable CP path (MLA or SFA PCP, following the GLM-5.3 `torch.compile` fix) so DeepSeek is not the first user of an unproven contract | Contract and unit tests pass on that path; capability predicate rejects the unsupported combinations at startup; fingerprint check fails fast on a deliberately skewed descriptor |
| **P1 DeepSeek-V4** (`dsa_cp`) | PCP with decode sharding and replicated decode; replace `full_graph_mode` with `CPGraphSpec`; graph-shaped metadata for capture and replay; indexer/compressor/O-projection stay in the island | Numeric equivalence for eager versus `PIECEWISE` across all four stages below; zero-token rank and 200-step skew pass; `FULL_DECODE_ONLY` baseline unchanged |
| **P2 DeepSeek-V4.1** (`dsa_v41_cp`) | Replicated-cache PCP, compressor metadata, `swa_bounded_replay` decision, engram fallback verification, request-buffer capacity from compile sizes. A2/A3 first, 950 products as a separate milestone | A2/A3 matrix passes with `swa_bounded_replay` on and off; 950 products only after the packaged-cache fallback path is validated |
| **Parallel track** | `mla_cp` (PCP, DCP, PCP+DCP) and `sfa_cp` (including the replicated-indexer DCP variant and fused prolog) | Same stage matrix as P1, on their own model families |

Every phase applies the same internal order, lowest risk first:

| Stage | Content |
| --- | --- |
| S1 | Decode-only and uniform batches: closest to `FULL` semantics; validates the island, the fingerprint, and token padding |
| S2 | Mixed batches and chunked prefill: metadata duality, zero-token ranks, large padded token extents |
| S3 | Speculative decoding (MTP, DSpark, Eagle3) and prefix caching, including the forced history/current split |
| S4 | `FULL_AND_PIECEWISE` and the breakable wrapper: prove both wrappers produce the same islands |

Two P2 decisions are prerequisites, not follow-ups:

- `make_decoder_replay()` in `models/deepseek_v41/decoder_replay_layers.py` only accepts `NONE` and `FULL_DECODE_ONLY` ("replay prefill requires eager execution"), and the recent `[no piecewise]` trimming work assumes the same. P2 starts by disabling `swa_bounded_replay` under piecewise, then decides whether extending replay to piecewise is worth the added layout work. Shipping without this decision would trade a graph win for a larger replay loss.
- `AscendDeepseekV41LLMForCausalLM.requires_uncompiled_fallback` is set when A5 packaged cache operators are available, which is why the 950 products run in a separate milestone rather than in the A2/A3 matrix.

Each phase keeps `FULL` and eager as the fallback and updates the supported-scenario tables in the [Context Parallel Guide](../../user_guide/feature_guide/context_parallel.md) only after its matrix passes.

### Implementation status

The contract layer is in place and is independent of any model: `CPGraphMode` / `CPGraphSpec` and the `supports_cp_piecewise` capability predicate live in `vllm_ascend/attention/context_parallel/graph_spec.py`; the cross-rank digest and its capture-time hook live in `vllm_ascend/compilation/cp_graph_fingerprint.py` and `vllm_ascend/worker/v2/aclgraph_utils.py`. `AscendPCPManager.validate_config()` now rejects unsupported CP piecewise combinations at startup.

P1 metadata plumbing is in place for `dsa_cp` and `dsa_v41_cp`: the builders resolve a `CPGraphSpec` from the configuration mode plus the batch-level hint, use it for restore-index extents and graph-stable buffers, and keep the legacy `full_graph_mode` keyword consistent for the DSA V4.1 builders underneath. DSv4.1 request-buffer capacity is capped at the request extent for piecewise-only configurations. The S1-S4 matrices for P1/P2 still have to run on hardware before the supported-scenario tables change.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Rank-local collective sizes diverge and hang | Cross-rank fingerprint at the end of capture; invariant checks at metadata build time; stress test with zero-token ranks |
| Padded rows leak into indexer top-k or cache writes | Slot masking and extent-based trimming; compare against eager with padded batches |
| `.item()` / `.tolist()` in a hot path stalls the async scheduler | Review every CP island path for device-to-host syncs; keep length lists CPU-side where possible |
| Per-layer buffers resized after capture | Allocate from capacity at construction; assert in DEBUG |
| Stream-resource exhaustion with CP + piecewise | Reuse the existing capture-size guidance; do not raise capture sizes for CP |
| Breakable mode changes eager/segment boundaries differently | Keep the island identical for both wrappers; test both |
| DeepSeek-V4.1 loses more to a disabled replay path than it gains from piecewise | Decide the replay strategy before P2 starts and report both `swa_bounded_replay` settings against the `FULL_DECODE_ONLY` baseline |

## Open questions

- Should CP prefer the breakable wrapper by default once HCCL capture in that mode is stable, given it captures one graph per size instead of one per layer?
- Can SFA's replicated indexer top-k be shared across ranks? SGLang broadcasts the finalized top-k from rank 0 and wraps the call in a split op so the broadcast is legal during capture. We do not adopt it yet: our indexer already runs inside the eager island, where a plain HCCL broadcast needs no split op, and the open question is whether the saved indexer work outweighs one extra collective per layer.
- If the P0 launch-share measurement shows the DSA island is launch-bound, is a narrower island (dense MLA captured, indexer eager) worth the extra split points?

## References

- Upstream graph model: [CUDA Graphs](https://docs.vllm.ai/en/latest/design/cuda_graphs/)
- Upstream CP deployment: [Context Parallel Deployment](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/)
- `vllm/compilation/piecewise_backend.py`, per-range compiled pieces and symbolic-shape dispatch
- `vllm/compilation/breakable_cudagraph.py`, `BreakableCUDAGraphCapture.add_eager` and `eager_break_during_capture`
- `vllm/v1/worker/gpu/cudagraph_utils.py`, `BatchExecutionDescriptor` and the piecewise capture path
- `vllm/v1/worker/gpu/pcp_manager.py`, PCP partitioning and graph padding
- SGLang `srt/compilation/npu_piecewise_backend.py`, torch_npu piecewise capture
- SGLang `srt/model_executor/runner_backend_utils/{tc_piecewise_cuda_graph,breakable_cuda_graph}/`, split-op registration, in-graph flags, and `eager_on_graph` break points
- SGLang `srt/layers/attention/dsa_backend.py`, disabling metadata-dependent dispatch inside a graph

## Related files

- `vllm_ascend/compilation/acl_graph.py`, `ACLGraphWrapper`, `UpdatableGraph`, graph parameters
- `vllm_ascend/compilation/breakable_aclgraph.py`, breakable wrapper
- `vllm_ascend/compilation/cp_graph_fingerprint.py`, cross-rank capture digest and verification
- `vllm_ascend/platform.py`, `splitting_ops`, compile backend selection, validation
- `vllm_ascend/attention/context_parallel/graph_spec.py`, `CPGraphSpec` and the piecewise capability predicate
- `vllm_ascend/attention/context_parallel/common_cp.py`, shared DCP collectives and merge
- `vllm_ascend/attention/context_parallel/mla_cp.py`, MLA DCP/PCP implementation and full-graph parameter refresh
- `vllm_ascend/attention/context_parallel/sfa_cp.py`, SFA DCP/PCP metadata, replicated indexer, KV gather
- `vllm_ascend/attention/context_parallel/dsa_cp.py`, DSA PCP metadata and global/local layouts
- `vllm_ascend/attention/context_parallel/dsa_v41_cp.py`, DSA V4.1 PCP with SWA and compressor state
- `vllm_ascend/attention/context_parallel/attention_cp.py`, GQA DCP implementation
- `vllm_ascend/worker/v2/pcp_manager.py`, PCP batch partitioning, padding, restore maps
- `vllm_ascend/worker/dcp_utils.py`, DCP slot mapping and block-table helpers
- `vllm_ascend/ops/mla.py`, `vllm_ascend/ops/dsa.py`, attention island boundaries
