# Chunk KDA Prefill Checkpoint (`chunk_kda` / `chunk_gated_delta_rule_fwd_h_kda`)

## Description

- **Function**: Compute chunkwise KDA prefill output and final state on Ascend NPUs while optionally snapshotting chunk-start recurrent checkpoints (`STORE_CHECKPOINT=True`).
- **Formula**: For each sequence `i` and head `h`, iterate over 64-token chunks `t = 0 .. NT - 1`:

  ```text
  if STORE_CHECKPOINT and checkpoint_offset[i] > 0 and t * BT == checkpoint_offset[i]:
      ckpt = H.to(checkpoint_state.dtype); checkpoint_state[i, h] = ckpt; H = ckpt.to(float32)
  v_new = v_t - w_t @ H.T; H = H * exp2(gk_last) + k_t.T @ v_new_decayed
  ```

- **Algorithm flow**:
  1. `chunk_kda` / `chunk_kda_with_fused_gate` expands optional gate fusion (`raw_beta` / `safe_gate`), runs intra-chunk WY/decay stages, and calls `chunk_gated_delta_rule_fwd_h_kda`.
  2. `chunk_gated_delta_rule_fwd_h_kda` validates NPU placement, index/checkpoint dtypes, shapes, and contiguity, then runs the recurrence and stores boundary snapshots when `t * BT == checkpoint_offset[i]`.
  3. `chunk_gla_fwd_o_gk` produces `o` (updating `v` in place).
- **Supported modes**: Atlas A2 and Atlas A3 inference (eager and piecewise NPU graphs with `@eager_break_during_capture` outermost). 950PR&950DT: N/A.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `q`, `k`, `v` | Input (`v` updated in place) | Queries/keys `[B, T, Hg, K]` and values `[B, T, H, V]` | FP16 / BF16 / FP32 | ND |
| `g` / `raw_g`, `beta` / `raw_beta` | Input | Log decay (`g`/`raw_g` `[B, T, H, K]`) and update gate (`[B, T, H]`) | FP16 / BF16 / FP32 | ND |
| `A_log`, `g_bias` | Input | Per-head log decay scale `[H]` (or `[1, 1, H, 1]`) and optional bias | FP32 / act dtype | ND |
| `scale`, `safe_gate`, `lower_bound` | Attribute | Query scale (`K**-0.5` if `None`) and gate activation options | float / bool / `None` | Scalar |
| `initial_state`, `output_final_state` | Input / Attribute | Optional initial state `[N, H, V, K]` and final-state flag | FP16 / BF16 / FP32, bool | Contiguous ND |
| `use_qk_l2norm_in_kernel`, `prebuilt_meta` | Attribute | In-kernel Q/K L2 norm flag and optional chunk metadata | bool, Object / `None` | Scalar / N/A |
| `cu_seqlens`, `checkpoint_offsets` | Input | Cumulative lengths `[N + 1]` and per-seq checkpoint offsets `[N]` | INT32 / INT64 | Contiguous 1D ND |
| `checkpoint_state` | Output | Recurrent checkpoint buffer `[N, H, V, K]` on NPU, or `None` | FP16 / BF16 / FP32 | Contiguous 4D ND |
| `o`, `final_state` | Output | Attention output `[B, T, H, V]` and final state `[N, H, V, K]` or `None` | Same as `v`, FP32 | ND / Contiguous ND |

## Constraints

- `k`, `w`, `u`, `cu_seqlens`, `checkpoint_state`, and `checkpoint_offsets` must reside on the same NPU device (`device.type == "npu"`).
- `cu_seqlens` and `checkpoint_offsets` require `int32`/`int64` and are normalized to contiguous; `checkpoint_state` and `checkpoint_offsets` must be provided together with contiguous `checkpoint_state` of shape `(N, H, V, K)` (`fp16`/`bf16`/`fp32`) and `checkpoint_offsets` of shape `(N,)`.
- `K <= 256` (`BT = 64`); positive `checkpoint_offsets[i]` must align to `64 * k` before sequence end; serving requires `--additional-config '{"kda_internal_prefill_checkpoints": true}'`.

## Origin and Differences

- **Origin**: FLA KDA chunkwise prefill kernels (`vllm/models/kimi_k3/nvidia/ops/third_party/kda.py`) integrated with `MambaSpec` internal prefill checkpoints.
- **Differences**: Added `STORE_CHECKPOINT`, `checkpoint_state`/`checkpoint_offsets`, `int64` offsets, dtype rounding at export boundaries, NPU input guards, and `raw_beta`/`safe_gate`/`lower_bound=None` support in `chunk_kda_with_fused_gate`.

## Test Cases

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_chunk_kda_npu.py
TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest -sv tests/ut/ops/test_kda_checkpoint.py
```
