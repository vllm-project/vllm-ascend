# HiSparse foundation on Ascend

## Scope

This is the foundational split of [#19109](https://github.com/vllm-project/vllm-ascend/pull/19109).
It supplies an Ascend backend for the merged vLLM HiSparse API and the minimal
SFA data path. It is a contribution starting point, with new-head NPU and full-model
qualification pending. Historical optimized C8 performance does not qualify this split.

Core owns HMA allocation, Coordinator admission/eviction, prefix-cache state,
request lifetimes, prefetch control flow, mirror completion and release/rollback.
Ascend implements device operations and cache-layout adaptation. Attention uses
the existing CANN sparse-flash-attention and fused RMSNorm/RoPE operations;
this PR adds no Attention core or native C++/AscendC kernel.

```text
Core sparse index group -> logical TopK -> NPU residency/refill plan
                                             |
                          Resident / Hot hit / mapped-Host miss
                                             |
                       packed KV -> existing native SFA operation
```

The data path covers unquantized KV and the native Ascend 950 C8 format with
the RoPE SFA layout and Model Runner V2. Model-weight quantization is independent. Private
Host pools are used per rank because NPU IPC events are not implemented.
The mapped-Host capability is checked at registration; hardware evidence for
the original primitives is from Ascend 950. A3 and the new source pair still
need their own qualification.

The SFA kernel requires TopK divisible by 128. The resolver's int16 LRU format
limits aligned Hot capacity to 32768 rows, and Hot must cover the complete
verification-row union. Context parallelism is not implemented in this split.
Request counts, lengths, layer/group counts and scratch capacities come from
configuration, rather than benchmark-specific constants.

## Configuration

Use the matching merged Core pin in `.github/vllm-main-verified.commit`.
No unmerged Core PR, copied scheduler, experiment profile or source-file hash
allowlist is required. As with other Ascend backends, paired-version ABI changes
still need review and device validation.

Set `VLLM_USE_V2_MODEL_RUNNER=1` and add these options to a compatible model's
normal launch command:

```bash
--block-size 128 \
--kv-cache-dtype bfloat16 \
--attention-config '{"indexer_kv_dtype":"bf16","hisparse_config":{"eager_host_mirror":true}}' \
--kv-transfer-config '{"kv_connector":"HiSparseConnector","kv_role":"kv_both","kv_connector_extra_config":{"host_pool_gib":4}}'
```

The example reserves 4 GiB Host storage per rank. Size it for the model and context
budget; it is not a model capacity claim. Keep Core's eager mirror policy for
this foundation. External P/D transfer is not provided by this PR.

For native A5 C8, replace the two dtype options above with:

```bash
--kv-cache-dtype fp8 \
--attention-config '{"indexer_kv_dtype":"fp8","hisparse_config":{"eager_host_mirror":true}}'
```

The existing SFA quantizer produces each main KV row: 512 E4M3FN NoPE bytes,
64 BF16 RoPE values and four FP32 block scales, totaling 656 bytes. The adapter
copies these components as opaque bytes and calls the existing quantized SFA
Attention operation. Indexer C8 uses 128 E4M3FN keys and one FP32 scale per token.
Main and Indexer dtypes remain independently selectable. C4, TurboQuant,
INT8-format C8 and other payload geometries are not implemented by this bridge.

Core's planner receives the complete key/scale page size and packed-row stride
alignment before choosing the shared-slab capacity. Strided views preserve the
original allocation and layer offset. The existing scatter handles packed bytes;
prefill Resident gathers reinterpret FP8 as bytes before using Core's gather.
HiSparse writes through its own current destination once; the ordinary SFA C8
source-slot store is skipped. No new fused scatter or Attention kernel is added.

## Extension points for contributors

Use existing interfaces, rather than adding unused placeholder hooks:

| Boundary | Current consumer and responsibility | Follow-up work |
| --- | --- | --- |
| `NPUPlatform.get_hisparse_backend_cls` | Core selects the device backend only for HiSparse | Backend capability/version support |
| `HiSparseBackend.create_cache_ops` | Runtime invokes resolver, refill and invalidation operations | Faster resolver/refill, vectorization and balancing |
| `validate_kv_cache_config` / `prepare_kv_cache_views` | Core validates BF16/C8 before allocation and binds views inside rollback scope | Additional payload formats with layout tests |
| Backend `pin_tensor` / `unpin_tensor` / `swap_blocks_batch` | Core owns completion, explicit registration release and transfer lifetime | DMA implementation and supported memory capabilities |
| `AscendHiSparseMetadataBuilder` / `AscendHiSparseImpl` | SFA factories select metadata and the BF16/C8 layout bridge | A smaller layout-conversion cost |
| `AscendHiSparseIndexerBackend.write_cache` | The existing cache backend interface selects HiSparse once; its override handles C8 page strides | Other Indexer payload layouts |

The model's Indexer wrapper reuses the cache layer's `get_attn_backend()` choice,
preserving model-specific backend factories. HiSparse owns its metadata builder
and C8 write override. The ordinary Indexer write path remains unchanged, and
BF16 HiSparse writes reuse it after metadata remaps the slots. Cache binding
creates the key/scale views; it does not attach payload-format routing strings.

`hisparse/cache_ops.py` owns operation validation and persistent per-group workspace;
`ops/triton/hisparse.py` implements the reference NPU path. Followers share their
leader's scratch only on that group's ordered stream. Independent groups and
current request/state mappings remain separate.

Code ownership follows these boundaries:

```text
vllm_ascend/
  attention/sfa_hisparse.py   # Attention dispatch, C8 adaptation and native calls
  hisparse/
    backend.py               # Core backend interface, registration and DMA
    binding.py               # Shared-slab specs and key/scale views
    cache_ops.py             # Residency, refill, invalidation and byte copies
    c8.py                    # C8 packing, views and physical-index helpers
  ops/triton/hisparse.py     # Cache-operation kernels; no Attention arithmetic
```

The common package contains no Attention invocation. The HiSparse Attention
implementation owns query preparation, current length tensors and the call to
the existing native SFA operation. The Triton and native Attention kernels are
unchanged by this organization.

Core's generic `torch.Stream`, `torch.Event` and current-stream interfaces are
used directly. The Ascend backend preserves Core's event placement and does not
replace its Torch bindings. This PR tests its own NPU operations and changing
graph replays rather than duplicating generic Torch stream tests.

The model-owned index-group builder is carried in Core's existing
`MLAModules.index_group_builder` field, preserving the common MLA wrapper ABI.

HiSparse groups use Core's coordinator and shared-slab memory budget. Ascend's
existing private-buffer budget correction and hybrid coordinator are bypassed
for those groups. V2 retains Core's source/group ownership and skips metadata
construction for Resident/Hot groups that have no Attention consumer.

Follow-up contributions should be separate and measurable: other packed formats,
native resolver/refill improvements, Host-writeback optimization, metadata
reuse, additional hardware and P/D transport. PR19109 retains these experiments.
AVSG, lazy-MTP policy, native optimized kernels, private Core adapters and
machine-specific benchmarking tools are excluded here.

Preserve completion-before-unregister, graph allocation owners, invalidation on
overwrite and protection of every selected token before choosing an LRU victim.
A same-shaped batch is not permission to reuse old lengths or request mappings.

## Validation

CPU boundary checks in the normal installed development environment:

```bash
python -m pytest -q \
  tests/ut/distributed/kv_transfer/test_hisparse_foundation_cpu.py
```

These check current Host offsets, failed-unregister ownership, group scratch
isolation, strided indexer aliases, planner alignment, unsupported formats,
and capture-only mirror suppression. C8 checks also
cover opaque component bytes, nonzero slab offsets, stable duplicate/padding
ordering, Indexer key/scale writes and planner alignment before allocation.
The tests directly import the production modules and Core/Ascend cache specs,
using the repository's normal CPU-UT conftest. Install the matching Core and
Ascend test dependencies before running them. CPU checks do not qualify device
execution.

The weight-loading wrapper keeps Ascend main's explicit activation-dtype ABI,
already covered by `tests/ut/ops/test_mla_weight_processing.py`; this foundation
does not add another calling convention or a duplicate test module.

Reuse the existing suites for behavior this PR does not implement:

| Behavior | Existing coverage |
| --- | --- |
| Core planning and HMA budgets | Core `tests/v1/core/test_kv_cache_planning.py` and `test_kv_cache_utils.py` |
| Core Host-mirror ordering, zero-token steps and request residency | Core `tests/v1/kv_connector/unit/test_hisparse_connector.py` |
| Core prefill plan and sparse layout rules | Core `tests/v1/attention/test_sparse_mla_backends.py` and `test_sparse_mla_kv_cache_layout.py` |
| Native SFA preprocessing and operator dispatch | Ascend `tests/ut/attention/test_sfa_v1.py` |

These suites retain their original platform/dependency requirements. In
particular, Core's CUDA-kernel cases do not qualify Ascend kernels. The NPU
modules below keep coverage for Ascend resolver/refill/invalidation, strided
Resident/Host staging and the BF16/C8 data bridge. Preprocessing uses the existing
native SFA operations as its reference; it contains no embedded activation dump
or independent-versus-fused rounding requirement.

Real-NPU cases are included for resolver/refill/overwrite invalidation, independent
FP32 Attention references, selection order/duplicates, native BF16/C8 preprocessing,
C8 key/scale writes, packed-byte scatter and Resident/Host prefill staging, in
eager and changing-graph cases:

```bash
pytest --confcutdir=tests/e2e/pull_request/one_card \
  tests/e2e/pull_request/one_card/test_hisparse_ops.py \
  tests/e2e/pull_request/one_card/test_hisparse_attention.py \
  tests/e2e/pull_request/one_card/test_hisparse_kv_preprocess.py \
  tests/e2e/pull_request/one_card/test_hisparse_prefill_staging.py
```

On Ascend 950 with Torch 2.10.0, torch-npu 2.10.0.post7 and CANN 9.2.1,
the standard CPU module passed all 20 cases. Running the four complete NPU
modules passed 40 of 41 cases, including BF16/C8 Attention and changing graph
replays. The run reused the compiled Ascend extension and vendor package;
it was not an independent native rebuild.

The remaining failure is `test_resident_prefill_plan_shared_layers[2-128]`.
With the paired Core pin unchanged, `res_blocks[:, offsets]` in
`HiSparsePrefillStagingPlan.ensure_gpu_sources` returned incorrect NPU rows
for the tested singleton-column input. An isolated CPU/NPU comparison found
the first difference at that indexing operation; equivalent `index_select`
matched the CPU result. The failing test remains enabled. This device stack
therefore has an unresolved mixed Resident/Host prefill correctness issue.

Fresh full-model correctness, dataset accuracy, eager/Graph lifecycle and
same-workload performance remain prerequisites for marking the foundation
ready. Existing C8 benchmark results are intentionally not presented as this
implementation's performance.
