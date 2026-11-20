# build_keypool_metadata

Build metadata for the GLM pooled key indexer.

[Implementation of `build_keypool_metadata`](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/glm5_next_cann_keypool.py) · [Caller `_build_metadata` in indexer_kpool.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/attention/indexer_kpool.py)

## Parameters and results

`R` is the padded request count, `T` the padded token count, `A` the actual token count, `D` the head dimension, `C` the tail ring capacity, `B` the compressed kernel block size and `W = retained_width or P`. Model values are `D=128`, `P=4` and `topk=2048`. All tensors are on the same NPU; scalar attributes are Python integers.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `common` | Input | `num_reqs=R`, `num_actual_tokens=A`; `query_start_loc[R+1]`, `seq_lens[R]`, `positions[source]`, `slot_mapping[source]` | int attributes; int32 boundaries/lengths, int64 positions/slots | ND; input vector strides supported |
| `buffers` | Output storage | Preallocated fields listed below | Object containing NPU tensors | Contiguous ND |
| `tokens` | Attribute | `T`, including graph padding | int | Scalar |
| `source` | Attribute | Number of available source positions/slots | int | Scalar |
| `logical` | Attribute | `L`, raw tokens per logical cache block | int | Scalar |
| `ratio` | Attribute | `P`, raw tokens per completed pool | int | Scalar |
| Return tuple | Output | Views of `positions[T]`, `slot_mapping[T]`, `seq_lens[R]`, `raw_seq_lens[R]`, `query_start_loc[R+1]`, `cum_query_lens[R]`, `start_pos[R]`, `pool_tail[R]`, in that order | Dtypes below | Contiguous ND |

| Buffer field | Description | Data type | Minimum capacity |
| --- | --- | --- | --- |
| `positions`, `slot_mapping` | Absolute positions and compressed slots; missing positions become 0, invalid slots become -1 | int64 | `[T]` each |
| `seq_lens`, `raw_seq_lens` | Complete-pool counts and raw sequence lengths | int64, int32 respectively | `[R]` each |
| `query_start_loc`, `cum_query_lens` | Query boundaries with a leading zero, and exclusive ends without it | int32, int64 respectively | `[R+1]`, `[R]` |
| `start_pos`, `start_offsets`, `pool_tail` | Raw request start, start modulo `P`, and raw sequence length modulo `P` | int32, int32, int64 respectively | `[R]` each |
| `pooled_slot_mapping` | Sentinel at index 0, then compact completed-pool destinations, padded with -1 | int64 | `[1+min(T, T//P+R)]` |

## Constraints

- Queries are contiguous within each request and packed in request order. Normalize a negative query boundary to `A`, clamp boundaries to `A`, then require nondecreasing boundaries beginning at 0 and ending at `A`. Raw sequence lengths must cover the request's query length; actual positions are nonnegative and consistent with those lengths.
- Invalid tail/pooled destinations are skipped. Valid destinations must be unique within their respective write set; duplicate destinations can race. CANN state and projections must be ready before cache writes. Reads of historical tail rows must finish before any overwrite.
- Metadata handles padded requests, evicted slots and empty source vectors. A fully empty model forward is handled by the caller before index packing. Padding storage must still have the declared capacity.
- Graph replay requires stable shapes, strides, addresses and scalar attributes. Request lengths, positions, slots and page contents can change in the existing buffers. Metadata is refreshed in capture/replay; cache writes and packing use those refreshed values.
- Pooled metadata tiles adapt from 8 to 64 rows, targeting 32 programs; tail preparation uses 8-row tiles. Request inversion tiles are capped at 128 to bound unified-buffer storage. Runtime counts avoid specializing on every batch size; strides, pool dimensions and power-of-two vector widths still specialize compilation.

## Origin and integration

Developed for the GLM pooled-indexer path. The metadata builder prepares reusable buffers once for the participating attention layers. Completed pools retain their compact output rows even if a cache slot has been evicted.

## Validation

The [INT8 KeyPool integration tests](https://github.com/vllm-project/vllm-ascend/blob/main/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_glm5next_int8_kpool_cache.py) exercise the managed cache and indexer pipeline. Their cache-write assertions apply to the write operation.
