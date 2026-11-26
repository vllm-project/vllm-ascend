# Qwen4Exp QSA dispatch support kernels

## Description

| Request | Attention path |
| --- | --- |
| Prefill with a cached prefix below 24K at chunk start | FIA for the entire chunk, even if it crosses 24K |
| Prefill with a cached prefix of 24K or more at chunk start | Union GBSA |
| Decode | Standard GBSA, including decode-only graph replay |

- **Packing**: the indexer selects 2048 tokens in groups of four. A compact row therefore contains 512 complete-group IDs followed by three residual-token slots. `pack_qsa_group_indices_npu` writes the complete groups and reconstructs up to three visible residual tokens. `-1` denotes an invalid lane.
- **FIA mask**: `_prepare_qsa_mask_words_kernel` expands group IDs into four-token bit patterns. Its process-local workspace is allocated once before graph replay and reused with its real row stride; the resulting boolean mask uses `False` for visible positions and `True` for masked positions.
- **Standard GBSA preparation**: `_prepare_qsa_gbsa_rows_kernel` removes invalid groups, preserves selection order, appends/replaces the residual group at the 512-entry capacity boundary, replicates sparse indices per KV head, and gathers the request block-table row.
- **Shared Union preparation**: `AscendQSAMetadataBuilder.prepare_step_metadata()` computes per-query visible lengths and prepares request boundaries, row maps and block tables once per step for layers that share the same metadata. The worker invokes it after all cache metadata has been built.
- **Multi-request batching**: Union Prefill requests with contiguous query rows can share one GBSA call. Each request forms its own groups of up to eight query rows. Unused lanes in the row map are set to `-1`.
- **Per-layer Union preparation**: the AscendC operator `QsaUnionInputs` uses the layer's compact indices, shared visible lengths and row map to produce Union indices, block counts and the exact mask for GBSA.

## Tensor ABI

| Tensor | Direction | Shape | Dtype/layout |
| --- | --- | --- | --- |
| compact group indices | input | `[query_rows, groups + 3]` | int32, strided 2-D |
| query positions | input | `[query_rows]` | int64 |
| sequence lengths | input | `[requests]` | int32 |
| token-to-request map | input | `[query_rows]` | int32 |
| block table | input | `[requests, pages]` | int32, arbitrary positive row/column stride |
| FIA mask | output | `[query_rows, mask_kv_length]` | bool; `True` masks a KV position |
| GBSA sparse indices | output | `[kv_heads, query_rows, output_groups]` | int32, `-1` padding |

The Qwen3.8 LightningIndexer ABI fixes `token_topk=2048`, `compress_ratio=4`, and therefore `groups=512`. Shorter synthetic group widths are supported by the reusable FIA workspace and are covered by the wide-to-narrow stride regression.

## Tiling, specialization, and graph constraints

- `GROUPS`, `OUTPUT_GROUPS`, `GROUP_BLOCK`, `KV_HEADS`, `TABLE_BLOCK`, and QSA packing widths affect vector/tile shape and remain `tl.constexpr`. Their serving variant sets are model-static or bounded powers of two.
- KV length, visible length, row count, tensor strides, and request IDs are runtime values and do not create per-request JIT variants.
- `WORD_BLOCK` is a bounded power-of-two mask tile (maximum 4096 words); `TABLE_BLOCK` is the bounded power-of-two capacity of the configured block table.
- The process-local FIA workspace must be sized before graph capture. It is deliberately not reallocated on the serving hot path because replacing captured device pointers would require graph recapture.
- Triton-Ascend 3.6.0.post2 only requires power-of-two `tl.arange` ranges in SIMT mode. The A3 HACC lowering used here accepts and emits the fixed 515-lane QSA packing range; a real-NPU compile/run test and TTIR are retained with validation evidence.

Normal serving constrains each scheduler chunk with `max_num_batched_tokens` (4096 in the validated deployment). The dispatch decision is based on the cached prefix at the start of the chunk, so a chunk that crosses 24K remains wholly on FIA and the following chunk switches to GBSA. This preserves one operator per request/chunk. Direct callers that bypass scheduler chunking are outside this serving contract; FIA additionally tiles mask construction to a 64 MiB workspace budget.

## Validation

Run CPU semantic coverage with:

```bash
pytest -q tests/ut/models/qwen4_exp/test_fia_dispatch.py tests/ut/models/qwen4_exp/test_gbsa_preparation.py
```

Run the A3 native kernels with:

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_qwen4_exp_qsa.py
```

The native test is discovered by the existing `singlecard-ops-a3` nightly matrix. It compiles and executes QSA packing, FIA wide-to-narrow workspace reuse, and GBSA preparation against independent PyTorch expectations, including residual boundaries, invalid holes, ragged requests, and a non-contiguous block table.

The production QSA-to-CANN GBSA integration path is covered by:

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/test_qwen4_exp_gbsa.py
```

This supplemental test passes compact LI-v2 indices through
`qsa_gbsa_attention`, compares the BF16 output with an independent dense FP32
attention reference, and covers eager execution, `torch.compile`, and NPUGraph
replay after updating an input. Triton packing and metadata outputs remain
exactly compared in `test_qwen4_exp_qsa.py`, as required for integer and index
results.

## Origin and differences

These kernels are Ascend-specific adapters for the upstream Qwen4Exp QSA metadata contract. They replace host-side expansion, compaction, and block-table gathering with device-side preparation while keeping the upstream selected-group order and residual-group semantics.
