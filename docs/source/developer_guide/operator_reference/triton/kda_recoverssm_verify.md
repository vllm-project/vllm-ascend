# kda_recoverssm_verify

Verify speculative KDA tokens against a paged checkpoint.

[Implementation of `kda_recoverssm_verify`](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/kda/recoverssm.py) · [Caller `_forward_core` in kda.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/models/glm5next/kda.py)

## Description

- **Function**: Verify a speculative KDA window using one persistent recurrent
  checkpoint per cache page. Record each token's correction and raw key/gate,
  then reconstruct only the accepted recurrent state after sampling. Commit also
  compacts the convolution history prepared by the caller. This supports GLM
  speculative decoding without keeping a full recurrent checkpoint per draft.
  Kimi K3/K3.1 and GLM use request-indexed temporary records.
- **Formula**: For one head, let `S` have shape `[V, K]`, and normalize nonzero
  query/key vectors to unit L2 norm; zero vectors remain zero. Scale the normalized
  query by `K**-0.5`. With `a = exp(A_log)` and `x = raw_g + dt_bias`:

  ```text
  g = lower_bound * sigmoid(a * x)       if lower_bound is not None
      -a * softplus(x)                  otherwise
  decay = exp(g)
  decayed_state = S * decay[None, :]
  beta = sigmoid(raw_beta)
  correction = beta * (v - decayed_state @ normalized_k)
  next_state = decayed_state + correction[:, None] * normalized_k[None, :]
  output = next_state @ normalized_scaled_q
  ```

  Verification records `correction` and the unnormalized `k/raw_g`. Commit
  reconstructs `next_state` through the accepted prefix from those records and
  the original checkpoint. Softplus uses its linear branch above 20.
- **Algorithm flow** (independent requests and heads, with value-dimension tiles):
  1. `_kda_recoverssm_verify_kernel` loads a request's checkpoint, advances a
     register-held recurrence, writes outputs and compact records, and leaves the
     persistent checkpoint unchanged. Null-page rows produce zero output and do
     not write records.
  2. `_prepare_commit_plan_kernel` maps speculative rows to sampler rows, clamps
     accepted lengths to the query length and window capacity, and prepares final
     and optional boundary destinations. Null pages and zero accepted lengths
     become no-op plan entries.
  3. `_compact_conv_state_kernel` copies the accepted convolution history to its
     destination. For history length `W` and accepted length `c > 0`, the source
     interval is `[c - 1, c - 1 + W)` and the destination interval is `[0, W)`.
  4. `_commit_kda_state_kernel` traverses the accepted records backwards,
     accumulates decay/products and correction outer products, and writes the
     reconstructed recurrent checkpoint. `_commit_kda_state_grouped_kernel`
     performs the same operation for two adjacent heads per program on its A5
     fast path. All aligned contexts use two disjoint on-device request filters
     to separate commits with and without a boundary snapshot. Non-crossing
     requests use two matrix buffers; crossing requests retain the existing
     six-buffer recovery path. A5 uses grouped heads for FP32 `H32/H64`,
     `K=V=128`; other layouts use the ordinary kernel with UB-bounded value tiles.
- **Supported modes**: Atlas A3 and Ascend 950DT (A5), inference only, eager and
  NPU graph execution. Atlas A2 and 950PR are not validated by these tests.
  Operator validation is separate from model/worker integration. Serving
  configuration and restrictions are documented in the model-specific
  RecoverSSM feature guide; low-level align arguments alone do not establish
  model support.

## Parameters

Notation: `B` speculative requests, `T` flattened tokens, `H` local heads,
`K/V` key/value widths, `P` cache pages, `Q` speculative-window capacity,
`L` local KDA layers, `C` convolution channels, `W` convolution history length,
`R` temporary record slots (`R >= B`).
An optional argument's default is part of its description.

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `q`, `k`, `raw_g` | Input | Query, raw key, raw gate, each `[1, T, H, K]` | BF16 / FP16 | ND; token-strided views allowed |
| `v` | Input | Values `[1, T, H, V]` | Same activation dtype | ND |
| `raw_beta` | Input | Update gate `[1, T, H]`; sigmoid is applied internally | Activation dtype / FP32 | ND |
| `A_log` | Input | Per-head log decay scale `[H]` | FP32 in validated callers | Contiguous ND |
| `dt_bias` | Input | Gate bias with `H * K` elements | FP32 in validated callers | Contiguous ND |
| `lower_bound` | Attribute | Gate lower bound; `None` selects negative softplus instead of bounded sigmoid | Python float / `None` | Scalar / N/A |
| `checkpoint_state` | Input | Persistent recurrent state `[P, H, V, K]`, read-only during verification | FP32 / BF16 | Paged ND, explicit strides |
| `correction_cache` | Output | Per-token correction records `[R, H, Q, V]` indexed by request | FP32 | ND, explicit strides |
| `kg_cache` | Output | Raw key and gate, `[R, H, Q, 2 * K]` indexed by request | Same dtype as `k` | ND, explicit strides |
| `query_start_loc` | Input | Nondecreasing token offsets `[B + 1]` | INT32 / INT64 | Strided 1D |
| `state_indices` | Input | Source page per speculative row `[B]`; values at or below `NULL_BLOCK_ID` select null rows | INT32 | Strided 1D |
| `spec_query_len` | Attribute | `Q`, normally `num_speculative_tokens + 1` | Python int | Positive scalar |
| `out` | Output | Output `[1, T, H, V]`; default `None` allocates an output with `v`'s shape and dtype; returns the supplied buffer when present | Same dtype as `v` in validated callers | ND |

## Constraints

- All tensor arguments to a call must be on the same NPU. Shapes and floating
  dtypes must match the parameter tables. The tensor batch dimension for
  activations is 1: requests are represented by flattened token offsets.
- `K`, `V` and `Q` are positive. Every query span is in `[0, Q]`, offsets begin at
  zero and end at `T`, and live page IDs and sampler-row mappings are in range.
  Concurrent live rows must not write overlapping cache pages.
- The last two strides of `q/k/raw_g` are `(K, 1)` and those of `v/out` are
  `(V, 1)`; the head stride of `raw_beta` is 1. Token strides may differ.
  `A_log` and `dt_bias` are contiguous. Cache pages may include page/head padding;
  supplied strides must describe allocated, nonoverlapping storage.
- The caller prepares convolution state before commit. Commit writes only the
  accepted checkpoint destinations and the first `W` convolution positions;
  verification writes only `out` and the valid correction/key/gate record spans.
  Input/output buffers must not alias these write regions unexpectedly.

## Origin and validation

Adapted from the vLLM KDA RecoverSSM recurrence. The [numerical tests](https://github.com/vllm-project/vllm-ascend/blob/main/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_kda_recoverssm_npu.py) compare verify output and replay records against independent references. See [joint verification and commit results](kda_recoverssm_commit.md#test-cases) for tolerances, device evidence and regression cases.
