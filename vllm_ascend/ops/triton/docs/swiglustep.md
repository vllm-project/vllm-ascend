# swiglustep

## Description

- **Location**: `vllm_ascend/ops/triton/activation/swiglustep.py` — `_swiglustep_kernel`, 1:1 wrapper `swiglustep_forward_triton`
- **Function**: Fused `SwigluStepAndMul` activation for the MoE expert MLP of Step-3.7-style models: splits the last dimension of the input into gate/up halves and computes `silu(gate).clamp(max=limit) * up.clamp(-limit, limit)` in a single kernel. It replaces vllm's `SwigluStepAndMul.forward_native` torch chain on NPU. Entry: `AscendSwigluStepAndMul.swiglustep_forward` (`vllm_ascend/ops/activation.py`), called from `vllm_ascend/ops/fused_moe/moe_mlp.py` (expert layers, `limit=7.0`) and `vllm_ascend/lora/quant_moe.py`; falls back to the vllm native implementation when Triton is unavailable.
- **Formula** (per row, computed in fp32):
    - `gate = x[..., :N]`, `up = x[..., N:]` for a row-major input `x[..., 2N]`
    - `s = silu(gate) = gate * sigmoid(gate)`
    - `out = min(s, limit) * clamp(up, -limit, limit)`
    - Order matters: silu BEFORE clamp on the gate half (SwigluStep), as opposed to clamp-then-silu (SwigluOAI).
- **Algorithm flow** (processed row by row, independently):
  1. Grid `(num_vectorcore,)`: rows are evenly split as `block_size = ceil(M / num_vectorcore)`; each program iterates over the rows of its range (program count == vector-core count, which minimizes host launch overhead compared with a 2D `BLOCK_M x BLOCK_N` grid that spawns `O(M * N / tile)` programs on large MoE shapes).
  2. Per row: load the full `2N` row, split gate/up halves via `extract_slice`, compute `silu` + double-sided clamps + multiply in fp32, and store the `N`-element output row in the input dtype.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950. Used by the MoE expert-MLP activation path of `vllm_ascend/ops/fused_moe/moe_mlp.py` and the quantized-MoE LoRA path; works in both eager and graph-capture modes.

## Parameters

> [!NOTE]
> All parameters are required.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `x` | Input | Gate/up concatenation, row-major `[..., 2N]` (arbitrary leading dims, flattened to `M` rows internally) | fp16 / bf16 | ND |
| `limit` | Input | Clamp bound (default `7.0`; Step-3.7 expert layers use `7.0`) | fp32 | scalar |
| `out_2d` | Output | Activated output `[..., N]`, same dtype as `x` | fp16 / bf16 | ND |

## Constraints

- The last dimension must be even (`2N`).
- `N` must be a multiple of 16 for fp16/bf16 (32-byte UB alignment of the per-row store on the NPU vector core; asserted up front). Real MoE shapes (Step-3.7 `N=1280`) satisfy this.
- Non-contiguous inputs are made contiguous by the wrapper; the output is contiguous with the input's dtype, while all arithmetic runs in fp32.
- `TOTAL_COLS`, `HALF_COLS`, `LIMIT`, `NUM_CORES` are compile-time `constexpr`; `M` is a runtime value so dynamic token counts do not trigger recompilation. The kernel is launched with `multibuffer=True`.
- Only for inference (MoE expert MLP forward) on NPU.

## Origin and Differences

- **Origin**: Math identical to vllm's `SwigluStepAndMul.forward_native` (see source-file header); the Triton kernel was developed from scratch in vllm-ascend (#11467) following the launch pattern of `swiglu_quant.py` / `rope.py`.
- **Differences**:
    - NPU adaptation for performance: fuses `silu + clamp + mul` into a single vector-core launch; persistent 1D grid (one program per vector core, rows looped per core) instead of a 2D tiled grid, which reduces program spawn and host launch overhead on the large-`M` shapes typical of MoE expert batches;
    - Modified for a specific vllm-ascend logic or different input parameters: exposes the `limit` clamp bound as an argument (Step-3.7 expert layers use `7.0`), enforces the 32-byte UB alignment constraint on `N`, and splits gate/up in-register via `extract_slice` instead of two global loads.

## Test Cases

The test compares `swiglustep_forward_triton` with an independent PyTorch baseline across fp16/bf16, 2D and 3D shapes (`M = 1/128/4000`, `N = 1280/2048`), and `limit = 1.0/7.0`, with unified elementwise tolerances (`rtol = atol = 5e-3` for fp16, `2e-2` for bf16). It additionally covers clamp-extreme inputs against exact golden values, NaN/Inf propagation, and non-contiguous input.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_swiglustep.py
```
