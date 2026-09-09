# KVPP Design

## 1. Background

MLA/SFA models can replicate historical KV caches across tensor parallel (TP) ranks, limiting capacity for long contexts and concurrent requests. KVPP distributes persistent caches by layer within each TP group and broadcasts a layer's cache when computation needs it.

Model execution retains its TP/EP/PP configuration. The scheduler keeps the complete logical cache specification and block numbering.

## 2. Design

KVPP combines layer ownership, two reusable scratch buffers, and prefetching one layer ahead:

```text
TP group within one PP stage
    Owner rank: persistent main KV + indexer KV for its layers
         |
         +-- Full-layer broadcast --> Other ranks: scratch buffer
                                          |
                                          +-- Wait --> Compute current layer
                                                       Prefetch next layer
```

### 2.1 Layer Ownership

Each pipeline parallel (PP) stage sorts its local target layers by layer index and assigns contiguous, balanced partitions to owner ranks in its TP group. A layer's main KV, indexer KV, and quantization components form one bundle with a single owner. MTP caches remain independently allocated on each rank that requires them and are excluded from target-layer partitioning.

Each rank allocates persistent storage for its owned layers and reuses two scratch buffers for other layers. Target-layer execution ordinals select alternating scratch buffers, allowing the current layer and the next prefetch to use separate storage.

### 2.2 Physical Layout and Memory Budget

Components within a bundle are contiguous, and each component contains all logical blocks for that layer. Layers do not need one globally contiguous allocation; only each broadcast bundle must be contiguous. The layout follows the actual cache specifications, including main KV, indexer data, and scales.

The physical budget on each rank is:

```text
Physical bytes per block
  = Sum of owned target-bundle bytes per block
  + Local MTP-cache bytes per block
  + 2 * Largest target-bundle bytes per block

Available blocks = floor(Available KV memory / Physical bytes per block)
```

The worker converts the resulting block capacity into a logical memory budget for the planner. Physical allocation uses the final block count, preserving logical cache management while allocating storage according to the KVPP layout.

### 2.3 Full-Layer Broadcast and Prefetch

A forward pass with previously computed tokens in any actual request starts by prefetching the first layer. Each attention layer waits for its broadcast after projections and before its first KV-cache access or write, then starts prefetching the next layer.

Each broadcast stays within the current PP stage's KVPP group. The owner broadcasts the complete bundle once, including all components and all allocated logical blocks. Other ranks receive it into the corresponding scratch buffer. After computation, the owner retains the updated cache for subsequent forward passes.

Prefetching uses a separate transfer stream. A ready event establishes data dependencies before broadcast, and a completion event ensures the device transfer has finished before the wait returns. The next layer's broadcast can overlap with current-layer computation; the overlap depends on communication and compute costs. Forward passes without history, dummy runs, and profiling paths do not broadcast historical caches.

## 3. Support and Constraints

| Area | Scope |
| --- | --- |
| Models and execution | Non-hybrid MLA/SFA models in eager mode; Model Runner V1 and V2 |
| Supported combinations | TP, EP, PP, chunked prefill, prefix caching, asynchronous scheduling, fixed-step MTP |
| Cache layouts | Allocated from actual specifications, including LI-C8 and SFA-C8 |
| Not supported | Graph execution, PCP, DCP, disaggregated prefill/decode, variable-step MTP |

KVPP trades communication for persistent cache capacity. Each rank still needs two scratch buffers sized for the largest target layer, plus its independent MTP caches. Memory savings therefore depend on layer count, TP size, and per-layer cache sizes. Broadcast traffic grows with the allocated block count; payloads are not filtered by request or active block.

## 4. Usage

Enable KVPP through the model launch arguments. The KVPP group size follows the TP size:

```bash
vllm serve <model-path> \
  --tensor-parallel-size 2 \
  --enforce-eager \
  --additional-config '{"enable_kvpp": true}'
```

`enable_kvpp` defaults to `false`. With PP enabled, each stage assigns owners and broadcasts within its own TP group.

## 5. Test Plan

Unit tests cover layout, budgeting, allocation, and scheduling. Device and communication dependencies use existing mocks; cache aliasing is checked with real tensor storage. One combined E2E scenario covers feature integration. The tables describe test expectations, not execution results. This version includes no KVPP nightly tests.

### 5.1 Unit Tests

Paths are relative to the repository root. Each row groups tests for one responsibility.

| ID | Coverage | Scenarios and expected behavior | Test file |
| --- | --- | --- | --- |
| UT-01 | Configuration and support boundaries | Parse booleans and boolean strings; enabled group size equals TP size; reject unsupported combinations during configuration | `tests/ut/test_ascend_config.py` |
| UT-02 | Communication groups | Keep KVPP groups within each PP stage; use single-rank groups when disabled; initialize and destroy independently of MC2 groups | `tests/ut/distributed/test_parallel_state.py` |
| UT-03 | Owners and bundles | Uneven partitions, more ranks than layers, and unordered specifications produce deterministic assignments; components share an owner; MTP is excluded | `tests/ut/core/test_kvpp_cache_placement.py` |
| UT-04 | Component layout | MLA, packed main KV, indexer data/scales, and different scale dtypes have correct byte sizes, offsets, and total lengths | `tests/ut/core/test_kvpp_cache_placement.py` |
| UT-05 | Physical budget | Different ranks, ranks without owned targets, MTP-only stages, and empty stages; correctly floor capacity at complete-block boundaries | `tests/ut/core/test_kvpp_cache_placement.py` |
| UT-06 | Allocation and aliasing | Reuse two scratch buffers by execution ordinal; keep owner/MTP storage independent; verify total storage size and isolation of writes | `tests/ut/worker/test_kvpp_cache.py` |
| UT-07 | Worker budget integration | Preserve complete logical specifications; convert physical capacity into the planner's logical budget; leave the disabled-path budget unchanged | `tests/ut/worker/test_worker_v1.py` |
| UT-08 | V1/V2 allocation entry points | Use the final block count; preserve typed-view dtypes, shapes, offsets, and storage aliases | `tests/ut/worker/test_model_runner_v1.py`, `tests/ut/worker/test_attn_utils_v2.py` |
| UT-09 | Runtime binding | Convert nonzero storage offsets to byte offsets; broadcast complete bundles; bind hooks only to target main attention layers | `tests/ut/worker/test_kvpp.py` |
| UT-10 | Layer prefetch | Skip prefetch without history; prefetch one layer ahead; reset state across forwards; propagate future failures without scheduling another layer | `tests/ut/worker/test_kvpp.py` |
| UT-11 | Broadcast completion | Map owners to global ranks; transfer the entire payload once; complete the future only after device completion; preserve data outside the payload | `tests/ut/distributed/kv_transfer/kv_pool/test_broadcast_transport.py` |
| UT-12 | History detection and lifecycle | V1/V2 use only actual requests and ignore padding; dummy/profile paths skip broadcasts; preserve lifecycle ordering | `tests/ut/worker/test_model_runner_v1.py`, `tests/ut/worker/test_model_runner_v2.py` |
| UT-13 | Attention wait placement | MLA/SFA native and fused paths, with or without an indexer, wait once after projections and before the first cache access | `tests/ut/attention/test_mla_v1.py`, `tests/ut/attention/test_sfa_v1.py` |

### 5.2 End-to-End Test

The E2E test uses Model Runner V1; unit tests cover V2 integration. One pytest test runs KVPP-disabled and KVPP-enabled instances sequentially with otherwise identical settings.

| ID | Configuration | Procedure | Expected behavior |
| --- | --- | --- | --- |
| E2E-01 | Four A3 devices; `vllm-ascend/DeepSeek-V3.2-W8A8-Pruning`; eager mode; TP=2, PP=2, EP; chunked prefill, prefix caching, asynchronous scheduling; one-step MTP; block size 128 and 64 logical blocks | Each instance processes two requests sequentially, sharing a 384-token prefix with different 16-token suffixes; token budget 128; generate 16 tokens per request | The first request requires multiple actual prefill steps with no cache hits; the second reuses at least 384 tokens; MTP caches exist and draft count is positive; KVPP groups stay within PP stages and target hooks exclude MTP; EP/async are enabled; output token IDs and text match with KVPP off and on |

Test location: `tests/e2e/pull_request/four_card/test_kvpp.py::test_kvpp_combined_features`.
