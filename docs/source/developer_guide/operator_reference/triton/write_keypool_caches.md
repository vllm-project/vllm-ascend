# write_keypool_caches

Write GLM KeyPool cache entries.

[Implementation of `write_keypool_caches`](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/glm5_next_cann_keypool.py) · [Caller `_init_cann_ops` in indexer_kpool.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/attention/indexer_kpool.py)

## Parameters and results

`R` is the padded request count, `T` the padded token count, `A` the actual token count, `D` the head dimension, `C` the tail ring capacity, `B` the compressed kernel block size and `W = retained_width or P`. Model values are `D=128`, `P=4` and `topk=2048`. All tensors are on the same NPU; scalar attributes are Python integers.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `tail` | Input/output | Request-owned raw key/gate ring, `[Ntail,C,2D]` | FP32 | ND; padded page stride supported |
| `cache` | Input/output | Completed pooled keys, `[Ncache,B,1,D]` | BF16, FP8 E4M3FN or INT8 | ND; padded block stride supported |
| `key`, `gate` | Input | MTP retained projections, each `[R*W,D]`; both `None` for non-MTP | FP32 or `None` | Contiguous ND |
| `scale_cache` | Optional output | Managed per-pool descales, `[Ncache,B,1]`; absent for BF16 | FP32 for FP8; FP16 for INT8 | ND; padded block stride supported |
| `pooled` | Input | Completed-pool vectors, `[Q,D]`; already quantized for INT8 | BF16 or INT8 | ND; row stride supported |
| `pooled_scales` | Optional input | One FP16 scale per INT8 completed pooled row, `[Q]` | FP16 or `None` | ND, contiguous |
| `metadata` | Input | `pooled_slot_mapping[Q]`, the destination of each compact pooled row | int64 | Contiguous ND |
| `tail_metadata` | Input | `cann_inputs` from `build_tail_inputs` | Object | N/A |
| `state` | Input | Updated CANN state, `[1+2R,P,2D]`; read only for non-MTP | FP32 | Contiguous ND |
| Return value | Output | N/A; returns `None` and mutates `tail`/`cache` and, for quantized modes, `scale_cache` | N/A | N/A |

## Constraints

- Inference only. `P>1`, `D>0`, `L%P==0`, `0<=A<=source<=T`, and input storage must cover every actual token. FP8 cache writes require matching FP32 scale storage and supported Ascend 950 hardware. The FP8 LI path does not support `OffloadingConnector` or `PreemptOffloadConnector`, including through `MultiConnector`.
- The indexer and FP32 tail share padded pages. FP8 reduces allocated cache memory only when the final shared page becomes smaller; a tail-dominated page can have the same physical size for BF16 and FP8 keys.
- BF16 remains the default. INT8 requires Ascend A3, matching managed FP16 scale storage, and a CANN `PoolKeyIndexer` build that accepts an empty completed-pool history without a dummy request.
- Tail blocks are request-local. Use `C=P` for non-MTP and `C=P+num_speculative_tokens` for speculative decoding; `0<retained_width<=C` when MTP is enabled. `page_rows>=C` and `page_rows*2D==tail.stride(0)` preserve shared page padding.
- `tail` rows, `cache` tokens and vector columns must be contiguous. Only tail page, compressed block and pooled-vector row strides may include padding. `cache` has exactly one KV head. Metadata output buffers must not overlap unrelated tensors.
- INT8 `scale_cache` must have shape `cache.shape[:-1]`, and `pooled_scales` must have one value per pooled row. Both must be supplied together; the FP16 scales use the managed cache page stride, so a P/D cache transfer carries keys and scales together. The kernel copies already quantized values and does not compute scales.
- Invalid tail/pooled destinations are skipped. Valid destinations must be unique within their respective write set; duplicate destinations can race. CANN state and projections must be ready before cache writes. Reads of historical tail rows must finish before any overwrite.
- Graph replay requires stable shapes, strides, addresses and scalar attributes. Request lengths, positions, slots and page contents can change in the existing buffers. Metadata is refreshed in capture/replay; cache writes and packing use those refreshed values.

## Origin and integration

Developed for the GLM pooled-indexer path. CANN reads the previous tail before this operation writes the updated tail and completed pools. FP8 keys are quantized here; INT8 keys and FP16 scales are supplied by the backend.

## Validation

The INT8 NPU tests compare complete managed pages byte-for-byte after valid, invalid and page-boundary writes, including untouched padding. They also compare CANN `PoolKeyIndexer` outputs from managed strided pages against contiguous key/scale tensors in eager and graph modes.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_glm5next_int8_kpool_cache.py
```
