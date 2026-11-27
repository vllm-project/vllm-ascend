# Vision QKV extraction, RoPE, and padding

## Description

- **Function**: Prepare packed Qwen vision Q/K/V for the eager fused infer attention score (FIA) call.
- **Formula**: For each token and head, split Q and K into two halves. The outputs are
  `x0 * cos - x1 * sin` and `x1 * cos + x0 * sin`. Copy V without rotation and
  write zeros in columns 72 through 127 of all three outputs.
- **Algorithm flow**: Read Q, K, and V planes from the packed projection. Apply
  half-split RoPE to Q and K in the tile and store each result in its final padded
  TND layout. Each Triton program owns a contiguous run of token tiles.
- **Supported modes**: The model adapter enables this eager path on Ascend A2
  when Triton and the input contract are available. Operator accuracy and
  performance were measured on Ascend 910B3. Other device families use the
  generic attention path.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `qkv_proj` | Input | Packed projection with eight heads per plane | BF16 | Contiguous `[T, 1728]` |
| `cos_half`, `sin_half` | Input | Half-width rotary tables for the same tokens | BF16 | Contiguous `[T, 36]` |
| Q, K, V | Output | Padded FIA inputs, with zero tails | BF16 | Contiguous `[T, 8, 128]` each |

## Constraints

All inputs are on the current NPU and remain unchanged. The outputs are
disjoint views of one buffer. The model adapter reads CPU sequence boundaries
for FIA in each attention block. Unsupported layouts, dtypes, compiled vision
blocks, and encoder ACL graph capture use the generic vision attention path.

The target model configuration is Qwen3.5-35B-A3B with vision tensor
parallelism of two. Its 16 vision heads of dimension 72 give eight heads per
rank. The shared Qwen vision attention class also serves Qwen2.5-VL models,
whose standard 80-dimensional vision heads use the generic path.

Each tile contains four tokens and uses aligned block stores. The grid is
capped by the tile count and the device's reported vector-core count.

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
value specialization. BF16, layout, and the four-token tile are fixed. Buffer
alignment and backend options can still change the cache key. Grid size alone
does not require a new variant. Dedicated startup warmup is N/A.
In a fresh process on 910B3, the in-process cache contained one specialization
after the first launch and stayed at one entry for token counts 1, 4, 41, 196,
257, 1024, 4096, and 8160, including a repeat of 41. This direct operator test
does not measure compilation during model startup or serving.

## Test Cases

The nightly test checks Q/K and FIA against a PyTorch FP32 reference cast to
BF16 (`atol=rtol=0.01`), and checks V and padding exactly. It covers single
and packed sequences, including 196 tokens from a 224x224 image.
The standalone operator test is validated on Ascend A2 and A3. It is skipped
on A5 and 310P, which have not been validated. The production model adapter
continues to enable the fused path on A2 only.

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
| 41 (1) | 394.6 | 526.7 | 257.5 | 706.3 | 836.7 | 570.8 |
| 1024 (1) | 408.7 | 553.7 | 275.6 | 759.1 | 901.0 | 617.5 |
| 8160 (4) | 404.1 | 725.5 | 277.7 | 1051.1 | 1434.6 | 942.2 |

Across all five shapes, fusion reduced preprocessing time versus CANN by 31.3% to
34.7% and attention time by 10.4% to 19.2%. Compared with standalone Triton
RoPE, the reductions were 50.2% to 61.7% and 31.5% to 34.3% respectively.
Padded Q/K/V and attention outputs were bitwise equal across all three paths
for these inputs.
