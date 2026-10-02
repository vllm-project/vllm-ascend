# split_qkv_rmsnorm_rope

## Description

- **Function**: Fuses QKV splitting, per-head Q/K RMSNorm, optional affine bias, and NeoX-style rotary position embedding into one Triton operator. V is copied without arithmetic. The public custom operator is `torch.ops.vllm.qkv_rmsnorm_rope`.
- **Formula**: Let `T` be the token count, `D` the head dimension, `Hq` and `Hkv` the Q and KV head counts, `Q = Hq * D`, `K = Hkv * D`, and `R` the rotary dimension. Split

  $$
  [q_0, k_0, v] = \operatorname{split}(\operatorname{input}, [Q, K, K]).
  $$

  For each token and head, compute in fp32

  $$
  \operatorname{norm}(x) = x \cdot \operatorname{rsqrt}\left(\operatorname{mean}(x^2) + \epsilon\right),
  $$

  followed by `zq = norm(q0) * q_weight + q_bias` and `zk = norm(k0) * k_weight + k_bias`, with the bias terms omitted when disabled. For cache row `positions[t]`, let `c` and `s` be the first and second `R / 2` elements. NeoX half rotation produces

  $$
  y_{0:R/2} = z_{0:R/2}c - z_{R/2:R}s,
  \qquad
  y_{R/2:R} = z_{R/2:R}c + z_{0:R/2}s.
  $$

  Dimensions `R:D` are preserved for partial RoPE.
- **Algorithm flow** (processed row by row, independently):
  1. Derive `Hq`, `Hkv`, the rotary mode, and UB-aware token tile sizes in the wrapper.
  2. Launch one program per available vector core. Each `program_id(0)` owns a contiguous interval of `ceil(T / num_vector_cores)` token rows.
  3. Load a Q+K token tile, reshape it to `[tile, Hq + Hkv, D]`, compute per-head fp32 RMSNorm, apply shared per-head weights and optional biases, and round the affine result to bf16.
  4. Gather the selected cos/sin cache rows, rotate Q and K, and store them with token-tail masks. A separate UB-tiled loop copies V unchanged.
- **Supported modes**: Atlas A2 and Atlas A3 for this kernel, in eager and graph-fused inference. On Ascend 950, `DeviceOperator` dispatches the same public semantics to the separate SIMT implementation instead of this kernel.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `input` | Input | Fused QKV projection laid out as Q followed by K and V | bf16 | Contiguous 2-D, `[T, Q + 2 * K]` |
| `cos_sin_cache` | Input | Position cache; each row contains `R / 2` cos values followed by `R / 2` sin values | bf16 | Contiguous 2-D, `[max_position, R]` |
| `positions` | Input | Cache-row index for every token | int64 | Contiguous 1-D, `[T]` |
| `q_weight` | Input | RMSNorm weight shared by all Q heads | bf16 | Contiguous 1-D, `[D]` |
| `k_weight` | Input | RMSNorm weight shared by all K heads | bf16 | Contiguous 1-D, `[D]` |
| `q_hidden_size` | Input (attribute) | Local flattened Q width `Q` | int32 | Scalar |
| `kv_hidden_size` | Input (attribute) | Local flattened K and V width `K` | int32 | Scalar |
| `head_dim` | Input (attribute) | Per-head feature size `D` | int32 | Scalar |
| `eps` | Input (attribute) | Positive RMSNorm epsilon | fp32 | Scalar |
| `q_bias` | Optional input | Bias shared by all normalized Q heads | bf16 | Contiguous 1-D, `[D]` |
| `k_bias` | Optional input | Bias shared by all normalized K heads | bf16 | Contiguous 1-D, `[D]` |
| `q_output` | Output | Normalized and rotated Q | bf16 | Contiguous 2-D, `[T, Q]` |
| `k_output` | Output | Normalized and rotated K | bf16 | Contiguous 2-D, `[T, K]` |
| `v_output` | Output | V copied from the fused input | bf16 | Contiguous 2-D, `[T, K]` |

## Constraints

- `input.shape[1]` must equal `q_hidden_size + 2 * kv_hidden_size`; `q_hidden_size` and `kv_hidden_size` must both be divisible by `head_dim`.
- This implementation is a bf16 kernel: the tested and production fusion path uses bf16, and the Q/K affine intermediate is explicitly converted to bf16 before RoPE.
- `R = cos_sin_cache.shape[-1]` must be positive, even, and no greater than `head_dim`. The rotation uses NeoX half splitting, not interleaved GPT-J pairing.
- Every `positions[t]` must satisfy `0 <= positions[t] < cos_sin_cache.shape[0]`.
- `q_weight` and `k_weight` must each contain exactly `head_dim` elements. `q_bias` and `k_bias` must either both be present or both be absent.
- All tensors must use the contiguous layouts listed above because the kernel does not consume tensor strides.
- The registered custom operator provides a fake implementation for Dynamo/AOT tracing. The normal A2/A3 kernel and the Ascend 950 SIMT kernel are separate implementations and should be tested on their respective hardware routes.

## Origin and Differences

- **Origin**: Developed in vLLM-Ascend PR #4711 as a native NPU fusion of QKV split, Q/K RMSNorm, and RoPE. A direct nightly accuracy test was added in PR #5267; indexed cos/sin-cache access and partial-RoPE support were added in later optimizations.
- **Differences**:
    - NPU adaptation for performance: performs Q/K normalization, affine transform, rotary embedding, and V extraction in one vector-core launch with UB-aware row tiling;
    - Modified for a specific vllm-ascend logic or different input parameters: registered as `torch.ops.vllm.qkv_rmsnorm_rope` and used by the Ascend QK-Norm/RoPE graph-fusion pass and direct model patches. The A2/A3 kernel consumes a compact `[cos_base | sin_base]` cache selected by `positions`.

## Test Cases

The test runs through `DeviceOperator.split_qkv_rmsnorm_rope` on the A2/A3 route and compares Q, K, and V with a PyTorch fp32 reference. It covers `T={1,16,1024,10240}`, `(Hq,Hkv)={(12,1),(64,4)}`, `D=128`, full and partial RoPE with `R={128,64}`, bias enabled and disabled, bf16 input, and a maximum position of 262144. The tolerance is `rtol=5e-3, atol=5e-2`.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_split_qkv_rmsnorm_rope.py
```
