# Qwen4Exp QSA dispatch support kernels

## Description

| Request | Attention path |
| --- | --- |
| Prefill with a request plan | Union GBSA |
| Decode | Standard GBSA, including decode-only graph replay |

- **Packing**: the indexer selects 2048 tokens in groups of four. A compact row therefore contains 512 complete-group IDs followed by three residual-token slots. `pack_qsa_group_indices_npu` writes the complete groups and reconstructs up to three visible residual tokens. `-1` denotes an invalid lane.
- **Standard GBSA preparation**: `prepare_qsa_gbsa_inputs` launches `_prepare_qsa_gbsa_rows_kernel` to compact valid complete-group IDs in selection order, replicate them per KV head, and gather the request block-table row. Sparse indices and counts describe complete groups only. GBSA metadata generation and attention execution use `residual_block_mode=1` to include the residual tail implicitly.
- **MTP selection reuse**: when `skip_topk=True`, the compact buffer supplies the complete-group IDs and residual-token positions saved at draft step 0. Standard GBSA preparation derives the effective visible length from these saved tail slots. The caller reuses the same compact buffer while query positions and sequence lengths advance.
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
| GBSA sparse indices | output | `[kv_heads, query_rows, groups]` | int32, valid prefix defined by sparse counts |
| GBSA sparse counts | output | `[kv_heads, query_rows]` | int32, complete-group counts only |
| effective visible lengths | output | `[query_rows]` | int32, passed to GBSA as `seqused_kv` |
| cumulative query lengths | output | `[query_rows + 1]` | int64, one query per GBSA row |
| per-query block table | output | `[query_rows, pages]` | same dtype as the input block table, contiguous |

The Qwen3.8 LightningIndexer ABI fixes `token_topk=2048`, `compress_ratio=4`, and therefore `groups=512`.

The first `groups` entries in each compact row are group IDs; the final three entries are token positions, with `-1` marking unused tail slots. Let `current_visible = min(max(position + 1, 0), request_length)`. Tail slots are valid when their token positions are nonnegative and below `request_length`. If any are valid, preparation sets the effective visible length to `max(valid_tail_token + 1)`; otherwise it uses `floor(current_visible / 4) * 4`. Complete-group IDs must be nonnegative and less than `floor(effective_visible / 4)` to count as valid.

Only the first `sparse_count[head, row]` output indices may be consumed. Entries after that prefix are unspecified. Full-group IDs can be noncontiguous. The effective visible length defines the boundary used for group validity and implicit residual selection; sparse counts give the number of selected complete groups.

## Tiling, specialization, and graph constraints

- `GROUPS`, `GROUP_BLOCK`, `KV_HEADS`, `TABLE_BLOCK`, and QSA packing widths affect vector/tile shape and are `tl.constexpr`. Their serving variant sets are model-static or bounded powers of two. The explicit output width equals `GROUPS`.
- Residual-token positions and effective visible lengths are runtime data.
- KV length, visible length, row count, tensor strides, and request IDs are runtime values and do not create per-request JIT variants.
- `TABLE_BLOCK` is the bounded power-of-two capacity of the configured block table.
- Triton-Ascend 3.6.0.post2 only requires power-of-two `tl.arange` ranges in SIMT mode. The A3 HACC lowering used here accepts and emits the fixed 515-lane QSA packing range; a real-NPU compile/run test and TTIR are retained with validation evidence.

PCP full-graph execution uses standard GBSA with its padded metadata.

## Validation

Run CPU semantic coverage with:

```bash
pytest -q tests/ut/models/qwen4_exp/test_attention_dispatch.py tests/ut/models/qwen4_exp/test_gbsa_preparation.py
```

Run the native kernel and integration tests on A3 or A5 with:

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_qwen4_exp_qsa.py
```

The file includes independent PyTorch expectations for packing and GBSA preparation, covering residual boundaries, invalid holes, ragged requests, and a non-contiguous block table. The packing-width and ragged-table tests are explicitly marked `Requires Ascend A3` and skip on A5; the remaining applicable tests exercise the production path on either device. Integer indices, counts, and effective visible lengths require exact equality.

Run the saved-tail MTP regression separately with:

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_qwen4_exp_qsa.py::test_qsa_gbsa_preparation_reuses_saved_mtp_tail
```

The test checks that complete-group indices, sparse counts, and effective visible lengths follow the saved compact selection during MTP reuse.

The production QSA-to-CANN GBSA integration path is covered by:

```bash
pytest -q \
  tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_qwen4_exp_qsa.py::test_qwen4_exp_gbsa_matches_dense_reference \
  tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_qwen4_exp_qsa.py::test_qwen4_exp_gbsa_npugraph_replays_updated_input
```

These tests pass compact LI-v2 indices through
`qsa_gbsa_attention`, compare the BF16 output with an independent dense FP32
attention reference, and cover eager execution, `torch.compile`, and NPUGraph
replay after updating an input. Triton packing and metadata outputs remain
exactly compared in `test_qwen4_exp_qsa.py`, as required for integer and index
results.

## Origin and differences

These kernels are Ascend-specific adapters for the upstream Qwen4Exp QSA metadata contract. They perform packing, compaction, and block-table gathering on the device while preserving selected-group order and residual-token semantics.
