# glm5_next_kpool_state_compress

## Description

- **Function**: `glm5_next_kpool_state_compress_and_write_cache_triton` compresses completed KeyPool windows into the paged indexer cache, then saves the current keys and gates in the paged compressor state. GLM-5.3-Flash uses FP32 keys, gates, positional bias, and state, with BF16 compressed keys; its pool size is 4 and head dimension is 128.
- **Formula**: For a pool ending at token position `p`, let `K[j, d]` and `G[j, d]` denote the key and gate at position `p - P + 1 + j`. With positional bias `A[j, d]`, compute `W[:, d] = softmax(G[:, d] + A[:, d])` and `C[d] = sum_j(W[j, d] * K[j, d])`. Accumulate in FP32 and cast `C` to the cache dtype on store. State rows contain the concatenation `[K, G]`.
- **Algorithm flow**:
    1. Each program handles one token and a tile of dimensions. Locate its request using cumulative query ends. A valid indexer slot identifies a token completing a pool.
    2. Read window entries in the current query directly from `k` and `gate_score`; read older entries through `state_block_table`. Apply the stable softmax and write the compressed key to `indexer_slot_mapping`.
    3. Launch a second kernel on the same stream to save current keys and gates to `state_slot_mapping`. This ordering ensures compression reads the preceding step's historical state before current state writes.
- **Supported modes**: Eager execution and fixed-shape NPU graph capture/replay on Atlas A2/A3 with Triton-Ascend. See Test Cases for validation scope. Ascend 950: N/A (not validated by this change).

## Parameters

`T` is the number of input rows including graph padding, `R` the request count, `D` the head dimension, `P` the pool size, `Bs` the state block size, and `Bi` the indexer block size. All tensor parameters are on the same NPU.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `state_cache` | Input/Output | Historical and updated keys/gates, shape `[Ns, Bs, 2 * D]` | FP32 | ND; block, token, and dimension strides supported |
| `indexer_cache` | Output | Completed compressed keys, shape `[Ni, Bi, 1, D]`; other slots remain unchanged | BF16 | ND; block, token, and dimension strides supported |
| `k` | Input | Current normalized keys, `[T, D]` | FP32 | ND |
| `gate_score` | Input | Current compression gate scores, `[T, D]` | FP32 | ND |
| `ape` | Input | Positional bias within a pool, `[P, D]` | FP32 | ND |
| `positions` | Input | Absolute token positions, `[T]` | int64 | ND |
| `cum_query_lens` | Input | Cumulative exclusive ends of current queries, `[R]`, without a leading zero | int32 | ND |
| `seq_lens` | Input | Total sequence lengths including the current query, `[R]` | int32 | ND |
| `state_slot_mapping` | Input | Flattened physical state slots, `[T]`; `-1` skips a state write | int64 | ND |
| `state_block_table` | Input | Request logical state page to physical block mapping, `[R, Ms]` | int32 | ND |
| `indexer_slot_mapping` | Input | Flattened physical compressed-cache slots, `[T]`; `-1` skips compression output | int64 | ND |
| `index_kpool` | Attribute | Number of keys per pool, `P`; GLM-5.3-Flash uses 4 | Python int | Scalar |
| Return value | Output | N/A; updates both caches in place | None | N/A |

## Constraints

- Inference only. The model configuration uses `D = 128`, `P = 4`. Tests also cover `P = 8` as a boundary variant. `P` and `D` must be positive; `Bs >= P`, `Ms > 0`, and both caches must provide storage for every valid slot. An empty page table or a state block smaller than the pool raises `ValueError`.
- Queries are packed in request order. Within each request, positions are contiguous and end at `seq_lens[r] - 1`. Cumulative ends are nondecreasing and the last end is at most `T`. An empty token batch or empty request list returns without modifying caches.
- Only pool-completing rows may have valid indexer slots. All historical positions needed to complete their windows must already be present in the state cache and mapped by the page table. Evicted pages outside these windows may contain invalid entries.
- Valid state slots are in `[0, Ns * Bs)` and valid indexer slots in `[0, Ni * Bi)`. Negative and out-of-range slots are skipped. Rows after the last query end do not write either cache, even if their padded slots contain stale positive values. Valid destination slots must be unique within a launch; duplicate writes are not ordered.
- Cache tensors may have noncontiguous storage as described above. The wrapper makes the other tensor inputs contiguous when needed. Avoid layout conversion in a captured hot path by preparing contiguous inputs beforehand.
- Capture/replay requires stable shapes, strides, pool size, request capacity, and tensor addresses. Metadata and tensor contents can change between replays. Compression and state writes must execute in order on the same stream.

## Origin and Differences

- **Origin**: Developed for the GLM-5.3-Flash KeyPool compression and paged state-cache sequence.
- **Differences**: Fuses request lookup, current/history gathering, softmax, and compressed-cache writes in Triton. Uses a separate ordered state-write kernel instead of a chain of host-dispatched indexing operations. Uses the existing scheduler-provided slots and the shared `vllm.utils.math_utils.next_power_of_2` helper.

## Test Cases

The single-operator test uses the model's FP32 state/key/gate/bias and BF16 compressed-cache dtypes with `D = 128`, `P = 4`, and state block sizes 4 and 16. A `P = 8` variant checks a wider window. A CPU reference builds each complete window from logical token history independently of device page lookup.

Cases cover prefill spanning multiple pools, decode/verification updates, rollback, historical tails, reordered physical pages, eviction of unneeded pages, noncontiguous cache storage, negative/out-of-range slots, graph padding with stale positive slots, empty inputs, and invalid state layouts. Eager and graph cases compare the entire backing allocations, including untouched padding. Graph cases update keys, gates, positions, lengths, slots, and page-table contents between replays.

State copies require bit-exact equality (`rtol = 0`, `atol = 0`). BF16 compressed keys use one fixed tolerance across cases (`rtol = 1e-2`, `atol = 1e-2`). Validated on Atlas A3 with PyTorch 2.10.0, torch-npu 2.10.0.post4, and Triton-Ascend 3.2.0. Atlas A2 and Ascend 950 were not exercised in this validation. These are accuracy tests, not throughput measurements.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_glm5next_kpool_triton.py
```
