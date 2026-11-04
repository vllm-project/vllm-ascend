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

This first data path covers unquantized main/indexer KV with the RoPE SFA
layout and Model Runner V2. Model-weight quantization is independent. Private
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

## Extension points for contributors

Use existing interfaces, rather than adding unused placeholder hooks:

| Boundary | Current consumer and responsibility | Follow-up work |
| --- | --- | --- |
| `NPUPlatform.get_hisparse_backend_cls` | Core selects the device backend only for HiSparse | Backend capability/version support |
| `HiSparseBackend.create_cache_ops` | Runtime invokes resolver, refill and invalidation operations | Faster resolver/refill, vectorization and balancing |
| `validate_kv_cache_config` / `prepare_kv_cache_views` | Core validates before allocation and binds views inside rollback scope | Packed/quantized KV formats with layout tests |
| Backend `pin_tensor` / `unpin_tensor` / `swap_blocks_batch` | Core owns completion, explicit registration release and transfer lifetime | DMA implementation and supported memory capabilities |
| `AscendHiSparseMetadataBuilder` / `AscendHiSparseImpl` | SFA factories select metadata and the layout bridge | C8 and a smaller layout-conversion cost |

`ops/hisparse.py` owns operation validation and persistent per-group workspace;
`ops/triton/hisparse.py` implements the reference NPU path. Followers share their
leader's scratch only on that group's ordered stream. Independent groups and
current request/state mappings remain separate.

The stream facade is an isolated PTA compatibility adapter in the two common
HiSparse modules. It preserves event placement and does not replace the actual
global Torch module. Remove it when generic PTA capture/workspace behavior is
equivalent or Core provides the matching factories.

Follow-up contributions should be separate and measurable: packed C8/indexer
formats, native resolver/refill improvements, Host-writeback optimization, metadata
reuse, additional hardware and P/D transport. PR19109 retains these experiments.
AVSG, lazy-MTP policy, native optimized kernels, private Core adapters and
machine-specific benchmarking tools are excluded here.

Preserve completion-before-unregister, graph allocation owners, invalidation on
overwrite and protection of every selected token before choosing an LRU victim.
A same-shaped batch is not permission to reuse old lengths or request mappings.

## Validation

Portable, source-isolated CPU boundary checks:

```bash
python -m pytest -q --noconftest \
  tests/ut/distributed/kv_transfer/test_hisparse_foundation_cpu.py \
  tests/ut/ops/test_mla_weight_hook_cpu.py \
  tests/ut/worker/test_hisparse_capture_mirror_cpu.py
```

These check current Host offsets, failed-unregister ownership, group scratch
isolation, strided indexer aliases, planner alignment, unsupported formats,
Core coordinator forwarding, loader ABI and capture-only mirror suppression.
They do not exercise an installed NPU package or qualify device execution.

Real-NPU cases are included for resolver/refill/overwrite invalidation, independent
FP32 Attention references, selection order/duplicates, native BF16 preprocessing
and Resident/Host prefill staging, in eager and changing-graph cases:

```bash
pytest --confcutdir=tests/e2e/pull_request/one_card \
  tests/e2e/pull_request/one_card/test_hisparse_ops.py \
  tests/e2e/pull_request/one_card/test_hisparse_attention.py \
  tests/e2e/pull_request/one_card/test_hisparse_kv_preprocess.py \
  tests/e2e/pull_request/one_card/test_hisparse_prefill_staging.py \
  tests/e2e/pull_request/one_card/test_hisparse_selection_order.py
```

These cases require an appropriate NPU environment and are not claimed to have
passed on this new split. Fresh full-model correctness, dataset accuracy,
eager/Graph lifecycle and same-workload performance remain prerequisites for
marking the foundation ready. Existing C8 benchmark results are intentionally
not presented as this implementation's performance.
