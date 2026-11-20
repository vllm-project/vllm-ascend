# build_tail_inputs

Prepare GLM KeyPool tail inputs.

[Implementation of `build_tail_inputs`](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/glm5_next_cann_keypool.py) · [Caller `build` in indexer_kpool.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/attention/indexer_kpool.py)

## Parameters and results

`R` is the padded request count, `T` the padded token count, `A` the actual token count, `D` the head dimension, `C` the tail ring capacity, `B` the compressed kernel block size and `W = retained_width or P`. Model values are `D=128`, `P=4` and `topk=2048`. All tensors are on the same NPU; scalar attributes are Python integers.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `common` | Input | The request attributes/boundaries/lengths above, plus `slot_mapping[A]` and `block_table_tensor[R, M]`; column 0 owns each request's tail block | int32 boundaries/lengths/table; int64 slots | ND; vector and table row strides supported; table columns contiguous |
| `pool` | Attribute | `P` | int | Scalar |
| `capacity` | Attribute | `C`, raw rows retained per request | int | Scalar |
| `page_rows` | Attribute | Physical rows per tail page, including padding | int | Scalar |
| `retained_width` | Attribute | 0 for non-MTP; number of retained query rows per request for MTP | int | Scalar |
| `buffers` | Output storage | `history[R,P]`, and `writes`, `state`, `retained`, each with at least `R*W` elements | int64 | Contiguous ND |
| Return object | Output | `history_indices[R,P]`, `write_indices[R,W]`, `state_indices[R,W]`, `retained_indices[R,W]` and `page_rows`; retained indices are `None` for non-MTP | int64 tensors; int attribute | Views into supplied buffers |

## Constraints

- Queries are contiguous within each request and packed in request order. Normalize a negative query boundary to `A`, clamp boundaries to `A`, then require nondecreasing boundaries beginning at 0 and ending at `A`. Raw sequence lengths must cover the request's query length; actual positions are nonnegative and consistent with those lengths.
- Tail blocks are request-local. Use `C=P` for non-MTP and `C=P+num_speculative_tokens` for speculative decoding; `0<retained_width<=C` when MTP is enabled. `page_rows>=C` and `page_rows*2D==tail.stride(0)` preserve shared page padding.
- Metadata handles padded requests, evicted slots and empty source vectors. A fully empty model forward is handled by the caller before index packing. Padding storage must still have the declared capacity.
- Graph replay requires stable shapes, strides, addresses and scalar attributes. Request lengths, positions, slots and page contents can change in the existing buffers. Metadata is refreshed in capture/replay; cache writes and packing use those refreshed values.
- Pooled metadata tiles adapt from 8 to 64 rows, targeting 32 programs; tail preparation uses 8-row tiles. Request inversion tiles are capped at 128 to bound unified-buffer storage. Runtime counts avoid specializing on every batch size; strides, pool dimensions and power-of-two vector widths still specialize compilation.

## Origin and integration

Developed for the GLM pooled-indexer path. The metadata builder prepares reusable buffers once for the participating attention layers. Completed pools retain their compact output rows even if a cache slot has been evicted.

## Validation

The [INT8 KeyPool integration tests](https://github.com/vllm-project/vllm-ascend/blob/main/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_glm5next_int8_kpool_cache.py) exercise the managed cache and indexer pipeline. Their cache-write assertions apply to the write operation.
