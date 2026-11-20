# pack_keypool_indices

Pack selected GLM KeyPool token indices.

[Implementation of `pack_keypool_indices`](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/glm5_next_cann_keypool.py) · [Caller `_init_cann_ops` in indexer_kpool.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/attention/indexer_kpool.py)

## Parameters and results

`R` is the padded request count, `T` the padded token count, `A` the actual token count, `D` the head dimension, `C` the tail ring capacity, `B` the compressed kernel block size and `W = retained_width or P`. Model values are `D=128`, `P=4` and `topk=2048`. All tensors are on the same NPU; scalar attributes are Python integers.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `indices` | Input/output | CANN token indices, `[T,I]`; reused in place when `output_buffer=None` | int32 | ND; padded row stride, contiguous columns |
| `metadata` | Input | `positions[T]`, `cum_query_lens[R]` | int64 | Contiguous ND |
| `topk` | Attribute | Maximum selected history tokens, a positive multiple of `P` | int | Scalar |
| `pool_size` | Attribute | `P` | int | Scalar |
| `output_buffer` | Optional output | Existing `[capacity,O]` SFA buffer; defaults to `indices` | int32 or `None` | ND; padded row stride, contiguous columns |
| Return value | Output | Logical token indices, `[T,1,topk+P-1]`; unused entries are -1 | int32 | View into the output buffer |

## Constraints

- Queries are contiguous within each request and packed in request order. Normalize a negative query boundary to `A`, clamp boundaries to `A`, then require nondecreasing boundaries beginning at 0 and ending at `A`. Raw sequence lengths must cover the request's query length; actual positions are nonnegative and consistent with those lengths.
- Packing requires `R>0` for a nonempty launch, `I>=topk`, output capacity at least `T`, and `O>=topk+P-1`. In-place packing also requires `I>=topk+P-1`. Do not use partially overlapping input/output views. Raw positions and cache addresses use int64; final SFA token indices must fit int32.
- Packing clears all output columns for rows beyond the final query end, and clears alignment columns beyond `topk+P-1`. Output rows beyond `T` and unselected cache bytes remain untouched.
- Metadata handles padded requests, evicted slots and empty source vectors. A fully empty model forward is handled by the caller before index packing. Padding storage must still have the declared capacity.
- Graph replay requires stable shapes, strides, addresses and scalar attributes. Request lengths, positions, slots and page contents can change in the existing buffers. Metadata is refreshed in capture/replay; cache writes and packing use those refreshed values.

## Origin and integration

Developed for the GLM pooled-indexer path. The wrapper preserves selected history, appends the visible causal tail and clears padding. For `P=4`, `L=128`, `D=128`, a token at position 65539 completes a pool. Raw slot 135 maps to compressed slot 33. With `topk=2048`, position 65540 has a tail beginning at 65540, placed in column 2048; columns 2049 and 2050 are -1. This is logical token addressing, while compressed writes use physical cache slots.

## Validation

The [INT8 KeyPool integration tests](https://github.com/vllm-project/vllm-ascend/blob/main/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_glm5next_int8_kpool_cache.py) exercise the managed cache and indexer pipeline. Their cache-write assertions apply to the write operation.
