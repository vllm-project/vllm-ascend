# lora_linear

## Description

- **Function**: Computes the dense single-adapter LoRA update for eligible linear layers with a split-K Triton shrink kernel followed by a fused Triton expand kernel. It supports single-slice projections and packed merged projections such as QKV and QKVZ. The serving integration selects this path only for one LoRA slot on Atlas A2/A3; unsupported shapes retain the packed-matmul or Punica fallback.
- **Formula**:
    - `workspace[s, :, :] = partial_s(x @ A_packed^T)` for `s in [0, K_SPLIT)`
    - `shrink = sum_s(workspace[s]) * adapter_mask * scale`
    - `y = y + shrink @ B_packed`
    - For merged projections, `A_packed = [A0; A1; ...]` and `B_packed` is block diagonal, so each output slice receives only its corresponding LoRA update.
- **Algorithm flow**:
  1. `_lora_shrink_splitk_kernel` partitions the hidden K dimension into `K_SPLIT=4` ranges. Each program processes one K range and one token tile, accumulates `x @ A_packed^T` in FP32, and stores its partial result in a reusable FP32 workspace.
  2. `_lora_expand_kernel` reduces the four workspace slices, applies the per-token adapter mask and LoRA scale, multiplies by B, and adds the result to y in one kernel. It supports both `[output, rank]` and pre-transposed `[rank, output]` B layouts.
  3. For two to four packed slices with per-slice rank at least 32, `_lora_expand_sliced_kernel` maps each output tile to its owning slice and reads only the corresponding nonzero block of block-diagonal B.
  4. The serving wrapper caches workspace by packed rank. Shapes outside the Triton eligibility rules fall back to packed `torch.matmul`; multi-adapter, fully-sharded, embedding, fused-MoE, and other general cases retain Punica SGMV/BGMV.
- **Supported modes**: Atlas A2 and Atlas A3 in ACL graph capture. Ascend 950 is not enabled by the current LoRA routing.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `x` | Input | Flattened linear input `[token_count, hidden_size]` | bf16 | ND, contiguous |
| `A_packed` | Input | Single-slice or rank-concatenated LoRA A weights `[total_rank, hidden_size]` | bf16 | ND, contiguous |
| `workspace` | Input / Output | Split-K partial results `[K_SPLIT, WORKSPACE_TOKENS, total_rank]` | fp32 | ND, contiguous |
| `B` / `B_packed` | Input | Single-slice B in `[output_size, rank]` layout, its `[rank, output_size]` transposed copy, or merged block-diagonal B `[total_rank, output_size]` | bf16 | ND, contiguous |
| `y` | Input / Output | Base linear output `[token_count, output_size]`; LoRA delta is added in place | bf16 | ND, contiguous |
| `adapter_mask` | Input | Per-token mask of length at least `WORKSPACE_TOKENS`; 1 enables LoRA and 0 preserves the base output | bf16 | ND, contiguous |
| `token_count` | Input (attribute) | Number of flattened tokens processed by this launch | int32 | scalar |
| `hidden_size` | Input (attribute) | K dimension shared by x and A | int32 | scalar |
| `output_size` | Input (attribute) | Total output width of y and B | int32 | scalar |
| `scale` | Input (attribute) | LoRA scaling factor applied after split-K reduction | fp32 | scalar |
| `RANK` / `TOTAL_RANK` | Compile-time attribute | A row count; for packed layers, per-slice rank multiplied by slice count | int32 | scalar |
| `SLICE_RANK` | Compile-time attribute | Rank of one packed output slice for `_lora_expand_sliced_kernel` | int32 | scalar |
| `WORKSPACE_TOKENS` | Compile-time attribute | Workspace token capacity; currently 128 | int32 | scalar |
| `BLOCK_M` | Compile-time attribute | Token tile size; 16 for token count at most 8, otherwise 32 | int32 | scalar |
| `BLOCK_K` | Compile-time attribute | Hidden-dimension tile size for shrink; currently 256 | int32 | scalar |
| `BLOCK_N` | Compile-time attribute | Output tile size; 1024 for token count at most 8 and total rank at most 48, otherwise 512 | int32 | scalar |
| `K_SPLIT` | Compile-time attribute | Number of hidden-dimension partitions; currently 4 | int32 | scalar |
| `B_TRANSPOSED` | Compile-time attribute | Selects B layout: true for `[output, rank]`, false for `[rank, output]` | bool | scalar |
| `TILE_END_0..2` | Compile-time attribute | Cumulative output-tile boundaries used to map a tile to one of up to four slices | int32 | scalar |
| `OUTPUT_START_1..3` | Compile-time attribute | Cumulative element offsets of packed output slices | int32 | scalar |

## Constraints

- The serving fast path requires `max_loras == 1`, `fully_sharded_loras == false`, and device type Atlas A2 or A3.
- x, y, A, and B must be contiguous BF16 tensors on the same NPU.
- Supported per-slice ranks are 8, 16, 32, and 64.
- `1 <= token_count <= 128`; the workspace and adapter-mask buffers must have capacity for 128 tokens.
- Single-slice execution supports B in `[output_size, rank]` form with `B_TRANSPOSED=True` or a pre-transposed `[rank, output_size]` copy with `B_TRANSPOSED=False`.
- A merged projection must provide both packed A and packed B. Its output-slice sizes must sum to `output_size`.
- Sliced expand supports two to four slices and requires `SLICE_RANK >= 32`. Smaller ranks use the generic expand kernel with block-diagonal packed B.
- The production Triton route requires ACL graph capture. Eager/non-captured execution falls back to packed matmul.
- Large projections fall back to packed matmul when `output_size >= 8192` and token count exceeds 32 for rank 8/16 or 64 for rank 32/64, or when `output_size >= 5120`, rank is at least 16, and token count exceeds 64.
- The kernel performs residual addition. The serving route uses it for the in-place LoRA update semantics of dense linear layers.
- FP32 is used for shrink accumulation and split reduction; the final output is stored as BF16.

## Origin and Differences

- **Origin**: Developed specifically for the vllm-ascend dense LoRA linear path. Its mathematical reference is the standard vLLM LoRA computation `(x @ A^T) @ B^T * scale` followed by an in-place addition to the base-layer output.
- **Differences**:
    - NPU adaptation for performance: split-K exposes parallelism for small-token, large-hidden shrink shapes; expand fuses partial reduction, adapter masking, scaling, B projection, and residual writeback; FP32 workspace is reused across graph replays.
    - Modified for vllm-ascend single-slot logic: merged A weights are concatenated and B weights are block diagonal. Sliced expand skips the zero off-diagonal blocks, while a fixed-address token mask preserves mixed base/LoRA batch semantics.
    - The implementation is shape-aware rather than unconditional: large projections use packed matmul, and general multi-adapter or fully-sharded cases continue to use Punica.

## Test Cases

The single-operator accuracy test uses TP-local Qwen3.5-27B projection shapes:

- single-slice O projection: hidden size 1536 and output size 5120;
- three-slice QKV projection: hidden size 5120 and output slices `(1024, 512, 512)`;
- four-slice QKVZ projection: hidden size 5120 and output slices `(1536, 512, 512, 1536)`.

It tests every supported per-slice rank in `{8, 16, 32, 64}` at token counts `{1, 31, 128}`. This covers the single-slice transposed-B path, generic block-diagonal expand for lower ranks, sliced expand for ranks 32/64, the maximum token boundary, mixed base/LoRA masking, scale application, FP32 split reduction, and residual writeback.

The arithmetic accuracy criterion is relative L2 error below 0.5% against a PyTorch FP32 reference. Tokens masked as base-only must remain bit-exact (`rtol=0`, `atol=0`) relative to their original y values.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_lora_linear.py
```
