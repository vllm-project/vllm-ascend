# SFA DCP query layout

## Description

- **Function**: Packs the nope/rope query fragments into head-major storage before DCP AllGather, then unpacks the gathered buffer directly into the contiguous fragments consumed by SFA. This avoids materializing a token-major concatenation and copying its two slices afterwards.
- **Formula**: `packed[h, t] = concat(q_nope[t, h], q_rope[t, h])`; after gathering rank-ordered head shards, `out_nope[t, h] = gathered[h, t, :512]` and `out_rope[t, h] = gathered[h, t, 512:]`.
- **Algorithm flow**: Prepare one query row per grid-stride iteration; let the attention caller issue AllGather and wait for its dependency; unpack one token/head row per iteration. Neither kernel performs communication.
- **Supported modes**: NPU eager execution and ACLGraph capture/replay. The API is hardware-generation independent; execution has been verified on Atlas A3.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `qn` | Input | Nope query, `[T, local_heads, 512]` | BF16 | ND, strided |
| `qr` | Input | Rope query, `[T, local_heads, 64]` | BF16 | ND, strided |
| prepared query | Output | AllGather input, `[local_heads, T, 576]` | BF16 | ND, contiguous |
| `gathered` | Input | Rank-ordered AllGather output, `[gathered_heads, T, 576]` | BF16 | ND, contiguous |
| unpacked queries | Output | `[T, gathered_heads, 512]` and `[T, gathered_heads, 64]` | BF16 | ND, contiguous |

## Constraints

- These are NPU-only internal kernel entry points. The attention backend owns input compatibility and communication ordering; the kernel wrappers do not implement a CPU execution mode or repeat those checks.
- Local and gathered head counts are derived from the tensors, not fixed to 8/64 or a particular DCP size.
- Current backend routing selects this small-decode layout for BF16, feature widths512/64 and `2 <= T <= 12`. This is a routing policy, not a theoretical token limit of the kernels. Other shapes use the existing query path.
- Native head gathering and token-scattered DSA-CP are different layouts. Physical PCP composition retains its TP-head gather path.
- Payload movement is bit-preserving, including BF16 NaN/Inf encodings. The caller must wait for asynchronous communication before unpacking.

## Origin and Differences

- **Origin**: The native DCP query gather in `AscendSFADCPImpl`.
- **Differences**: Fuses query concatenation with layout preparation and final fragment copies. Runtime token/head/work counts and ordinary strides are not exact-value constexpr keys; only vector widths are static kernel configuration.

## Test Cases

`tests/ut/attention/a2/test_sfa_dcp_generalized_port.py` covers rank-ordered gathering, multiple local/gathered head counts, input column strides1/2 and changed-input graph replay, using bit-exact comparisons. Group sizes2/4/8 have distributed NPU execution evidence.

```bash
torchrun --standalone --nproc-per-node=4 -m pytest --noconftest -sv \
  tests/ut/attention/a2/test_sfa_dcp_generalized_port.py -k query
```
