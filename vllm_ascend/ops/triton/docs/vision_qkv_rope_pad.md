# Vision QKV extraction, RoPE, and padding

## Description

- **Function**: Prepare packed Qwen vision Q/K/V for the eager fused infer attention score (FIA) call.
- **Formula**: For each token and head, split Q and K into two halves. The outputs are
  `x0 * cos - x1 * sin` and `x1 * cos + x0 * sin`. Copy V without rotation and
  write zeros in columns 72 through 127 of all three outputs.
- **Algorithm flow**: Read Q, K, and V planes from the packed projection. Apply
  half-split RoPE to Q and K in the tile and store each result in its final padded
  TND layout. Each Triton program owns a contiguous run of token tiles.
- **Supported modes**: Eager Ascend NPU execution. Validated on Ascend 910B3.
  The wrapper uses the device's reported vector-core count. Other devices have
  no accuracy or performance results here.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `qkv_proj` | Input | Packed projection with eight heads per plane | BF16 | Contiguous `[T, 1728]` |
| `cos_half`, `sin_half` | Input | Half-width rotary tables for the same tokens | BF16 | Contiguous `[T, 36]` |
| Q, K, V | Output | Padded FIA inputs, with zero tails | BF16 | Contiguous `[T, 8, 128]` each |

## Constraints

All inputs are on the current NPU and remain unchanged. The default allocation
uses disjoint views of one output buffer. The model adapter supplies valid
CPU sequence boundaries to FIA separately. Unsupported layouts, dtypes, or
encoder ACL graph capture use the generic vision attention path.

The default tile contains four tokens. The eight-token cap bounds local-buffer
usage for the supported store modes based on 910B3 compiler validation. The
grid uses `get_vectorcore_num()` and defaults to the smaller of the tile count
and the device's available vector cores.

## Origin and Differences

- **Origin**: The generic Qwen path extracts Q/K/V, invokes
  `AscendApplyRotaryEmb` (`npu_rotary_mul`), and applies
  [the existing padding helper](../../mm_encoder_attention.py) before FIA.
- **Differences**: [The standalone Triton RoPE](../rope.py) takes unpacked Q and K
  and writes unpadded outputs. Using it here still requires separate Q/K views,
  materialization of noncontiguous inputs, and padding of Q, K, and V. The fused
  producer consumes the packed projection directly and writes FIA-ready buffers.
  Both paths use the same FIA call and output unpadding.

Token count and tile scheduling values are runtime arguments excluded from
value specialization. BF16 and the layout are fixed. Tile size and store mode
are compile-time choices, so changing them can create variants. Buffer
alignment and backend options can also change the cache key. Grid size alone
does not require a new variant. Dedicated startup warmup is N/A.
In a fresh process on 910B3, the default configuration loaded one specialization
at the first launch. The in-process cache remained at one entry for token counts
1, 4, 41, 257, 1024, 4096, and 8160, including a repeat of 41.

## Test Cases

The nightly test compares Q/K/V against a PyTorch reference and checks padded
columns exactly. It also compares FIA outputs with matching sequence boundaries
and covers the launch options and vector-core counts.

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_vision_qkv_rope_pad.py
```

### Performance comparison on 910B3

All paths use the same packed BF16 projection and rotary tables. The CANN
baseline uses the existing Qwen extraction, `npu_rotary_mul`, and
`_maybe_pad_qkv`. The standalone Triton baseline substitutes
`rope_forward_triton` in that same sequence. The attention measurement
includes sequence metadata, the same FIA settings, unpadding, and output
layout conversion. It excludes the QKV and output linear projections.

Median wall time per call in microseconds, measured with vLLM 0.30.0,
PyTorch 2.10.0, torch-npu 2.10.0.post4, and Triton 3.2.0 on one 910B3.
Each shape and path was warmed up before timing. The three paths ran in
shuffled order for 25 rounds, each with 12 calls followed by synchronization.

| Tokens (sequences) | CANN prep | Triton RoPE prep | Fused prep | CANN attention | Triton RoPE attention | Fused attention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 41 (1) | 385.6 | 523.2 | 312.5 | 681.0 | 817.9 | 641.5 |
| 257 (1) | 436.1 | 590.2 | 349.3 | 781.8 | 918.2 | 716.5 |
| 1024 (1) | 426.6 | 566.4 | 336.3 | 782.0 | 913.9 | 716.9 |
| 4096 (3) | 431.1 | 574.5 | 334.0 | 785.9 | 924.4 | 718.1 |
| 8160 (4) | 430.1 | 734.3 | 347.1 | 1042.6 | 1439.8 | 948.4 |

Compared with the CANN path, fusion reduced preprocessing time by 19.0% to
22.5% and attention time by 5.8% to 9.0%. Compared with standalone Triton
RoPE, the reductions were 40.3% to 52.7% and 21.6% to 34.1% respectively.
Padded Q/K/V and attention outputs were bitwise equal across all three paths
for these inputs.

In the earlier model A/B on Qwen3.5-35B-A3B with Ascend 910B3 TP2, each
arm completed 765 RealWorldQA requests (batch size 64, maximum 512 output
tokens, temperature 0.01). Request throughput went from 1.2319 to 1.2547
requests/s (+1.85%) and mean request latency from 50.92 to 49.80 s
(-2.20%). Accuracy was 72.29% and 74.12%, respectively. This was one run
per arm using an earlier revision. The accuracy difference does not establish
an accuracy gain, and those measurements do not validate current-revision
model performance or accuracy.
