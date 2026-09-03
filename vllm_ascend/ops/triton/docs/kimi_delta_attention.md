# Kimi Delta Attention (KDA)

## Description

- **Function**: Implements gated delta-rule linear attention with chunked prefill and fused recurrent paths.
- **Formula**: Optionally L2-normalize `q_t` and `k_t`. In mathematical `[K,V]` layout, compute `S_tilde = Diag(exp(g_t)) * S_(t-1)`, `delta_t = beta_t * (v_t - S_tilde^T * k_t)`, `S_t = S_tilde + k_t * delta_t^T`, and `o_t = scale * S_t^T * q_t`. `beta_t` can be headwise or value-wise. The kernels physically store the state transposed in `[V,K]` order.
- **Algorithm flow** (processed sequence by sequence): optionally normalize Q/K; the prefill path takes chunk-local cumulative gates, forms and solves a causal lower-triangular transform, derives chunk update factors, propagates fp32 state across 64-token chunks, and emits outputs; the recurrent path updates an optionally indexed state token by token and can write it in place.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950. The direct Triton APIs support chunked prefill, recurrent decode, packed variable-length sequences, and indexed in-place state updates in eager execution. Graph capture is not directly validated by the operator tests; the current `AscendKimiK3DeltaAttention` implementation routes its attention calls through AscendC operators.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `q`, `k` | Input | Query/key `[B,T,H,K]` | fp16 / bf16 / fp32 | BTHD |
| `v` | Input | Value `[B,T,HV,V]` | fp16 / bf16 / fp32 | BTHD |
| `g` | Input | Per-token key-dimension gate `[B,T,H,K]` | fp16 / bf16 / fp32 | BTHD |
| `beta` | Input | Headwise `[B,T,H]` or value-wise `[B,T,HV,V]` update rate | fp16 / bf16 / fp32 | BTH / BTHD |
| `scale` | Attribute | Query-key scale; defaults to `K**-0.5` | fp32 | scalar |
| `initial_state` | Input | Initial state `[state_slots, HV, V, K]` in physical layout | fp32 | ND |
| `cu_seqlens` | Input | Optional packed sequence boundaries | int32 / int64 | ND |
| `ssm_state_indices` | Input | Optional sequence/token-to-state-slot mapping | int64 | ND |
| `inplace_final_state` | Attribute | Reuse `initial_state` for final state | bool | scalar |
| `use_qk_l2norm_in_kernel` | Attribute | Fuse Q/K L2 normalization | bool | scalar |
| `o` | Output | Attention output `[B,T,HV,V]` | same as `v` | BTHD |
| `final_state` | Output | Per-sequence final state for chunk mode, per-token state for non-in-place recurrent mode, or the updated `initial_state` buffer in in-place mode | fp32 | ND |

## Constraints

- Q and K share head count and key dimension; V may use a different head count/value dimension according to Kimi K3 grouping. Inputs are made contiguous by the public wrapper.
- The recurrent Triton path currently requires one key tile (`NK == 1`). Packed variable-length input requires `B == 1`.
- Prefill uses chunk size 64 and supports tail chunks; state indices must reference allocated state slots. In-place state updates require exclusive ownership of each written slot.

## Origin and Differences

- **Origin**: Ported from the MIT-licensed flash-linear-attention KDA/gated-delta-rule implementation used by upstream vLLM.
- **Differences**:
    - NPU adaptation for performance: uses Ascend-oriented tiling, fp32 intermediate and state accumulation, optional fused Q/K normalization, and separate chunked and recurrent kernels.
    - Modified for vllm-ascend logic: uses a physical `[V,K]` state layout, vector gates, packed-sequence metadata, and indexed in-place recurrent states.

## Test Cases

`test_chunk_kda_npu.py` compares output and final state with a naive fp32
recurrent reference for 32 or 64 heads, dimension 128, fp16/bf16 inputs,
single and packed sequences from 15 to 8192 tokens, and tail chunks. The
recurrent suite covers single-token decode and sequences up to 64 tokens,
32/64 heads, fp16/bf16 inputs, non-in-place per-token state, and indexed
in-place state buffers with 1, 4, or 16 slots. Both suites require a relative
RMSE below 0.005 for output and state.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_chunk_kda_npu.py
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_fused_recurrent_kda_npu.py
```
