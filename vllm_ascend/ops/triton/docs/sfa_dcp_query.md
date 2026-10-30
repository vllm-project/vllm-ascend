# SFA DCP query layout

## Description

- **Function**: Packs the nope/rope query fragments into head-major storage before DCP AllGather. After waiting for communication, the caller either retains views or unpacks directly into contiguous fragments, according to `keep_view`.
- **Formula**: `packed[h, t] = concat(q_nope[t, h], q_rope[t, h])`; split the gathered result using the original nope/rope widths retained in the gather context.
- **Algorithm flow**: Prepare one query row per grid-stride iteration, with vector loads/stores across its features; let the attention caller issue AllGather and wait for its dependency. With `keep_view=True`, return permuted/split views without an unpack launch; otherwise unpack one token/head row per iteration. Neither kernel performs communication or a reduction.
- **Supported modes**: NPU eager execution and ACLGraph capture/replay. The API is hardware-generation independent; execution has been verified on Atlas A3.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `qn` | Input | Nope query, `[T, local_heads, N]` | BF16, FP16 or FP32 | ND, strided |
| `qr` | Input | Rope query, `[T, local_heads, P]` | Same as `qn` | ND, strided |
| prepared query | Output | AllGather input, `[local_heads, T, N + P]` | Same as `qn` | ND, contiguous |
| `gathered` | Input | Rank-ordered AllGather output, `[gathered_heads, T, N + P]` | Same as `qn` | ND, contiguous |
| `split_sizes` | Attribute | Original `(N, P)` for unpacking | Integer pair | N/A |
| unpacked queries | Output | `[T, gathered_heads, N]` and `[T, gathered_heads, P]` | Same as `qn` | ND, contiguous |

## Constraints

- These are NPU-only internal kernel entry points. The attention backend owns input compatibility and communication ordering; the kernel wrappers do not implement a CPU execution mode or repeat those checks.
- Local and gathered head counts are derived from the tensors, not fixed to 8/64 or a particular DCP size.
- Fragment widths come from the input tensors, without fixed512/64 dimensions or a T2–12 whitelist. Each fragment must have positive power-of-two width for its full `tl.arange` vector. Empty and non-power-of-two layouts retain the existing query path.
- The context retains both fragment widths: their sum alone cannot recover the split. `keep_view` selects the consumer's storage layout rather than using token count as a proxy.
- Native head gathering and token-scattered DSA-CP are different layouts. Physical PCP composition retains its TP-head gather path.
- Payload movement is bit-preserving, including BF16 NaN/Inf encodings. The caller must wait for asynchronous communication before unpacking.

## Origin and Differences

- **Origin**: The native DCP query gather in `AscendSFADCPImpl`.
- **Differences**: Fuses query concatenation with layout preparation and final fragment copies. Runtime token/head/work counts and ordinary strides are not exact-value constexpr keys; only vector widths are static kernel configuration.

## Test Cases

`tests/e2e/nightly/single_node/ops/multicard_ops_a3/test_sfa_dcp.py` covers rank-ordered gathering, multiple local/gathered head counts, input column strides1/2 and changed-input graph replay, using bit-exact comparisons. Group sizes2/4/8 have distributed NPU execution evidence.

```bash
torchrun --standalone --nproc-per-node=4 -m pytest --noconftest -sv \
  tests/e2e/nightly/single_node/ops/multicard_ops_a3/test_sfa_dcp.py -k query
```
