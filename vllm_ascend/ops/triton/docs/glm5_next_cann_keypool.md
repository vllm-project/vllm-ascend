# glm5_next_cann_keypool

## Description

- **Function**: Three Triton kernel types adapt GLM-5.3-Flash metadata, cache writes and token indices to CANN `KeyPool`/`PoolKeyIndexer`. The entry points are in `vllm_ascend/ops/triton/glm5_next_cann_keypool.py`.
- **Formula**: With pool size `P`, logical block size `L`, position `p` and raw slot `s`, a completed pool has `(p + 1) % P == 0` and compressed slot `(s // L) * (L // P) + (s % L) // P`. A negative slot remains invalid. The visible tail starts at `a = ((p + 1) // P) * P` and contains `[a, ..., p]`, appended after `min(a, topk)` history columns.
- **Algorithm flow**:
    1. `_build_keypool_metadata_kernel` has two modes, exposed by `build_keypool_metadata` and `build_tail_inputs`. They build compressed request boundaries/slots and raw-tail read/write addresses once per metadata build. The resulting buffers are reused across layers. DSA-CP also builds rank-local cumulative query ends, visible pool counts and tail lengths in the same launch. Completed pools retain their compact output row even when their cache slot is evicted.
    2. After CANN `KeyPool` has consumed the old tail, `_write_keypool_caches_kernel` writes the raw FP32 tail and completed BF16 pools in one launch. Non-MTP reads updated CANN state; MTP writes retained projected key/gate rows so rejected candidates can be overwritten after logical rollback.
    3. `_pack_indices_kernel` preserves the selected history, appends the causal tail and clears padded rows/alignment columns in one launch. It can write directly into the SFA output buffer.
- **Supported modes**: Eager execution and fixed-shape NPU graph capture/replay.

## Parameters

`R` is the padded request count, `T` the padded token count, `A` the actual token count, `D` the head dimension, `C` the tail ring capacity, `B` the compressed kernel block size and `W = retained_width or P`. Model values are `D=128`, `P=4` and `topk=2048`. All tensors are on the same NPU; scalar attributes are Python integers.

### build_keypool_metadata

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `common` | Input | `num_reqs=R`, `num_actual_tokens=A`; `query_start_loc[R+1]`, `seq_lens[R]`, `positions[source]`, `slot_mapping[source]` | int attributes; int32 boundaries/lengths, int64 positions/slots | ND; input vector strides supported |
| `buffers` | Output storage | Preallocated fields listed below | Object containing NPU tensors | Contiguous ND |
| `tokens` | Attribute | `T`, including graph padding | int | Scalar |
| `source` | Attribute | Number of available source positions/slots | int | Scalar |
| `logical` | Attribute | `L`, raw tokens per logical cache block | int | Scalar |
| `ratio` | Attribute | `P`, raw tokens per completed pool | int | Scalar |
| `query_metadata` | Optional output storage | DSA-CP local `positions`, preallocated `cum_query_lens[R]`, `seq_lens[R]` and `pool_tail[R]`; the latter three receive local query ends and causal prefix lengths | int64 tensors or `None` | Contiguous ND |
| `local_start` | Attribute | First packed global token owned by this rank; its interval has `query_metadata.positions.numel()` tokens | int | Scalar |
| Return tuple | Output | Views of `positions[T]`, `slot_mapping[T]`, `seq_lens[R]`, `raw_seq_lens[R]`, `query_start_loc[R+1]`, `cum_query_lens[R]`, `start_pos[R]`, `pool_tail[R]`, in that order | Dtypes below | Contiguous ND |

| Buffer field | Description | Data type | Minimum capacity |
| --- | --- | --- | --- |
| `positions`, `slot_mapping` | Absolute positions and compressed slots; missing positions become 0, invalid slots become -1 | int64 | `[T]` each |
| `seq_lens`, `raw_seq_lens` | Complete-pool counts and raw sequence lengths | int64, int32 respectively | `[R]` each |
| `query_start_loc`, `cum_query_lens` | Query boundaries with a leading zero, and exclusive ends without it | int32, int64 respectively | `[R+1]`, `[R]` |
| `start_pos`, `start_offsets`, `pool_tail` | Raw request start, start modulo `P`, and raw sequence length modulo `P` | int32, int32, int64 respectively | `[R]` each |
| `pooled_slot_mapping` | Sentinel at index 0, then compact completed-pool destinations, padded with -1 | int64 | `[1+min(T, T//P+R)]` |

### build_tail_inputs

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `common` | Input | The request attributes/boundaries/lengths above, plus `slot_mapping[A]` and `block_table_tensor[R, M]`; column 0 owns each request's tail block | int32 boundaries/lengths/table; int64 slots | ND; vector and table row strides supported; table columns contiguous |
| `pool` | Attribute | `P` | int | Scalar |
| `capacity` | Attribute | `C`, raw rows retained per request | int | Scalar |
| `page_rows` | Attribute | Physical rows per tail page, including padding | int | Scalar |
| `retained_width` | Attribute | 0 for non-MTP; number of retained query rows per request for MTP | int | Scalar |
| `buffers` | Output storage | `history[R,P]`, and `writes`, `state`, `retained`, each with at least `R*W` elements | int64 | Contiguous ND |
| Return object | Output | `history_indices[R,P]`, `write_indices[R,W]`, `state_indices[R,W]`, `retained_indices[R,W]` and `page_rows`; retained indices are `None` for non-MTP | int64 tensors; int attribute | Views into supplied buffers |

### write_keypool_caches

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `tail` | Input/output | Request-owned raw key/gate ring, `[Ntail,C,2D]` | FP32 | ND; padded page stride supported |
| `cache` | Input/output | Completed pooled keys, `[Ncache,B,1,D]` | BF16 | ND; padded block stride supported |
| `key`, `gate` | Input | MTP retained projections, each `[R*W,D]`; both `None` for non-MTP | FP32 or `None` | Contiguous ND |
| `pooled` | Input | CANN completed-pool vectors, `[Q,D]` | BF16 | ND; row stride supported |
| `metadata` | Input | `pooled_slot_mapping[Q]`, the destination of each compact pooled row | int64 | Contiguous ND |
| `tail_metadata` | Input | `cann_inputs` from `build_tail_inputs` | Object | N/A |
| `state` | Input | Updated CANN state, `[1+2R,P,2D]`; read only for non-MTP | FP32 | Contiguous ND |
| Return value | Output | N/A; returns `None` and mutates `tail`/`cache` | N/A | N/A |

### pack_keypool_indices

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `indices` | Input/output | CANN token indices, `[T,I]`; reused in place when `output_buffer=None` | int32 | ND; padded row stride, contiguous columns |
| `metadata` | Input | `positions[T]`, `cum_query_lens[R]` | int64 | Contiguous ND |
| `topk` | Attribute | Maximum selected history tokens, a positive multiple of `P` | int | Scalar |
| `pool_size` | Attribute | `P` | int | Scalar |
| `output_buffer` | Optional output | Existing `[capacity,O]` SFA buffer; defaults to `indices` | int32 or `None` | ND; padded row stride, contiguous columns |
| Return value | Output | Logical token indices, `[T,1,topk+P-1]`; unused entries are -1 | int32 | View into the output buffer |

## Constraints

- Inference only. `P>1`, `D>0`, `L%P==0`, `0<=A<=source<=T`, and input storage must cover every actual token. This adapter documents the BF16 cache path.
- Queries are contiguous within each request and packed in request order. Normalize a negative query boundary to `A`, clamp boundaries to `A`, then require nondecreasing boundaries beginning at 0 and ending at `A`. Raw sequence lengths must cover the request's query length; actual positions are nonnegative and consistent with those lengths.
- DSA-CP partitions the packed global query into contiguous rank intervals. Each local request sees only the prefix through its last local query; requests without local queries have zero visible length. KeyPool/cache writes still use global request metadata, while selection and packing use local query metadata.
- Tail blocks are request-local. Use `C=P` for non-MTP and `C=P+num_speculative_tokens` for speculative decoding; `0<retained_width<=C` when MTP is enabled. `page_rows>=C` and `page_rows*2D==tail.stride(0)` preserve shared page padding.
- `tail` rows, `cache` tokens and vector columns must be contiguous. Only tail page, compressed block and pooled-vector row strides may include padding. `cache` has exactly one KV head. Metadata output buffers must not overlap unrelated tensors.
- Invalid tail/pooled destinations are skipped. Valid destinations must be unique within their respective write set; duplicate destinations can race. CANN state and projections must be ready before cache writes. Reads of historical tail rows must finish before any overwrite.
- Packing requires `R>0` for a nonempty launch, `I>=topk`, output capacity at least `T`, and `O>=topk+P-1`. In-place packing also requires `I>=topk+P-1`. Do not use partially overlapping input/output views. Raw positions and cache addresses use int64; final SFA token indices must fit int32.
- Packing clears all output columns for rows beyond the final query end, and clears alignment columns beyond `topk+P-1`. Output rows beyond `T` and unselected cache bytes remain untouched.
- Metadata handles padded requests, evicted slots and empty source vectors. A fully empty model forward is handled by the caller before index packing. Padding storage must still have the declared capacity.
- Graph replay requires stable shapes, strides, addresses and scalar attributes. Request lengths, positions, slots and page contents can change in the existing buffers. Metadata is refreshed in capture/replay; cache writes and packing use those refreshed values.
- Pooled metadata tiles adapt from 8 to 64 rows, targeting 32 programs; tail preparation uses 8-row tiles. Request inversion tiles are capped at 128 to bound unified-buffer storage. Runtime counts avoid specializing on every batch size; strides, pool dimensions and power-of-two vector widths still specialize compilation.

## Origin and Differences

- **Origin**: Developed for the CANN GLM-5.3-Flash pooled-indexer integration, with the same logical slot and tail contract as the existing GLM KPool path.
- **Differences**: Replaces separate boundary conversions, compact-slot construction, tail/cache scatter and output padding operations with three kernel types. The metadata kernel runs in two modes per metadata build; this is not a claim of only three launches per model step. Key pooling and TopK scoring remain CANN operations. FP32 tail and BF16 completed keys are preserved.

## Example

For `P=4`, `L=128`, `D=128`, a token at position 65539 completes a pool. Raw slot 135 maps to compressed slot 33. With `topk=2048`, position 65540 has a tail beginning at 65540, placed in column 2048; columns 2049 and 2050 are -1. This is logical token addressing, while compressed writes use physical cache slots.
