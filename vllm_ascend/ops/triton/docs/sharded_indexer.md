# DCP sharded indexer candidates and lengths

## Description

- **Function**: Pack local QLI candidates, select exact global Top-K or owner-local winners, and build local causal key lengths.
- **Formula**: For local ID x, global ID is `(x // I * P + r) * I + x % I`. Owner-local output keeps global winners satisfying `(id // I) % P == r`. Local prefix length is `L // (P*I)*I + clamp(L % (P*I) - r*I, 0, I)`.
- **Algorithm flow**: Mask invalid candidates; optionally sort by score descending and ID ascending; map IDs and pack score bits; AllGather one tile; merge globally or search the owner's exact winning prefix; write every output slot including -1 padding. Local lengths derive from global causal prefixes on device.
- **Supported modes**: NPU eager and NPUGraph. A3/A5 share the implementation. Native FP32 and BF16 comparator inputs retain their original precision.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| indices, scores | Input | Native local QLI candidates `[T,1,K_local]`, positive strides | INT32; FP32/BF16 | ND |
| world_size, rank, interleave | Attribute | DCP group geometry and token stripe width | int | Scalar |
| sort_candidates, fuse_sort | Attribute | Establish comparator order; permit workload-based fused preparation | bool | Scalar |
| packed | Input/Output | Exact global ID and FP32 score-bit pairs `[P*T,K_local,2]` after gather | INT32 | Contiguous ND |
| rows, topk | Attribute | Query count and global winner count | int | Scalar |
| local_rank | Optional attribute | Owner-local output; None selects global coordinates | int/None | Scalar |
| presorted, owner_sharded | Attribute | Explicit sorted-order and per-source owner contracts | bool | Scalar |
| out | Optional output | Destination `[T,1,topk]`; may be a contiguous row slice | INT32 | ND |
| seq_lens, query_ends | Input | Global sequence lengths and cumulative query endpoints | INT32 | ND |
| key_lengths, visible_lengths | Output | Per-request local length and per-query local causal prefix | INT32 | ND |
| query_start_loc | Input | Cumulative query bounds including leading zero for native candidate normalization | INT32 | ND |

## Constraints

- The current supported candidate input contract requires `0 <= local_idx < 2**20` for valid token IDs, for both BF16 and FP32 scores. `local_idx` is the token index within a rank before local-to-global mapping, not its position in the K-candidate array. Negative IDs remain invalid-candidate sentinels. Callers must guarantee this bound; the wrapper does not perform a runtime range validation. This local-ID requirement is separate from the backend's global model-length bound below.
- DCP/stripe sizes are positive; all ranks use identical collective tile boundaries. The owning backend requires complete replicated indexer heads and `max_model_len <= 2**24` for exact FP32 ID comparison.
- K2048 has fused preparation/merge. Other supported K values retain native stable sorting. Fused preparation uses one inline stable radix algorithm for BF16 and FP32, with 16 and 32 score-key bits respectively. The public wrapper rejects other score dtypes before allocation or launch; the kernel receives `SCORE_BITS` as a compile-time parameter. BF16 values widen exactly to FP32 for transport; FP32 scores retain all their bits.
- Owner-prefix merge requires `presorted=True`, `owner_sharded=True`, `K_local=topk=2048`, and a valid local rank. Other calls use full selection.
- Non-power-of-two rank tiles clamp padded peer addresses before masked loads; this prevents A3 GM faults without changing any valid peer or selection result.
- Invalid/nonfinite candidates do not enter the valid output prefix. Empty requests and graph padding produce zero lengths or -1 IDs. Normalization writes only the native QLI-owned output buffers in place.
- ROWS is runtime and does not specialize transport/merge kernels. K, topology, strides and sorting tiles are fixed compilation/layout parameters. Length kernels keep row/request counts dynamic; power-of-two tiles and binary-search step counts select compilation buckets. Input values update on replay. Black-box compilation counts and hardware/model validation belong in the PR evidence.

## Origin and Differences

- **Origin**: DCP candidate exchange developed for token-sharded QLI.
- **Differences**: Exchanges exact candidates rather than token-history KV. Owner-local output removes a later remap; occupancy-based preparation and grouped merge bound work without chip-model dispatch.

## Fused candidate sorting

The radix key orders scores descending and local IDs ascending. The local-to-global ID map is strictly increasing within a rank, so mapping IDs after sorting preserves the tie order. Signed zeros share one sorting key while the original score payload is retained. Invalid candidates receive a negative-infinity score and are written with ID -1.

Each least-significant-digit pass extracts ID bits first, followed by the encoded score bits. A native FP32 value sort orders `digit * BLOCK + position`; the position tag makes the pass stable. With K2048, 11 position bits leave 13 digit bits within FP32's 24-bit significand. The resulting permutation gathers both scores and IDs before the next pass.

Local IDs use a fixed 20-bit field under the input contract above. The logical key is `[descending score key | ascending local ID]`: 36 bits for BF16 and 52 bits for FP32. With K2048, the shared compile-time radix loop therefore takes three passes for BF16 or four for FP32. No per-row maximum-ID reduction, ID-width selection or large-ID comparison fallback is needed. `SCORE_BITS` is supplied by the wrapper, so dtype selection adds no per-row branch.

The wrapper keeps grid selection and routing outside the kernel. Fusion requires K2048, at least 75% as many rows as vector cores, and enough downstream merge levels to amortize local sorting. Other workloads keep separate sorting and packing. Runtime `num_rows` does not specialize the fused kernel; score width, strides, topology and block size do. This change adds no new warmup entry or routing policy.

## Test Cases

Actual K2048 deployment shapes include T48/DCP16 and T128/DCP8. Shared tests also cover generic K, partial tiles, non-power-of-two DCP, ties, near-ULP scores, invalid owners, strided input, exact score bits, the local-ID limit `2**20 - 1`, large mapped global IDs and changing-input graph replay.

```bash
pytest -q tests/ut/attention/test_sharded_indexer.py
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_dcp_candidates.py
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_candidate_radix.py
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_dcp_pre_post.py -k local_lengths
```
