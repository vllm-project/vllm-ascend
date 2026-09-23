# ChunkKdaFwd in vLLM Ascend

This directory vendors the AscendC KDA forward operator from
[flash-linear-attention-npu](https://github.com/flashserve/flash-linear-attention-npu/tree/ad8a7908e6ee57496500a02995b5416b99f66e32),
revision `ad8a7908e6ee57496500a02995b5416b99f66e32`.
Migration author: Lizheng <12232325@mail.sustech.edu.cn>.

## Kimi K3 integration

Kimi K3 prefill uses `vllm_ascend.ops.kda.run_chunk_kda`, which calls
`torch.ops._C_ascend.chunk_kda_fwd`. The operator and all three V2 stages are
compiled into this repository's custom operator package; installing `fla_npu`
or loading an external operator library is unnecessary.

The model passes raw gates, FP32 preprocessed beta, FP32 initial state in
`[sequence, head, V, K]` order, and CPU sequence/chunk metadata. BF16 inputs with
`K=V=128`, chunk size 64 and strictly increasing sequence offsets use V2 with
Q/K L2 normalization inside Prepare. FP16, `K=V=64`, and empty logical sequences
use the fused entry with Q/K normalization performed by the existing caller.
Both paths return the attention output and a separate FP32 final state for the
model's existing cache update. Decode continues to use `recurrent_kda`.

No new environment variable is required. Rebuild the custom operators and the
PyTorch extension together after switching branches. This is an operator
migration; it does not by itself validate a one-million-token model deployment.

## Public PyTorch interface

```python
outputs = torch.ops._C_ascend.chunk_kda_fwd(
    q, k, v, g, beta, scale, chunk_size,
    layout="BSND",
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    chunk_indices=None,
    safe_gate=False,
    lower_bound=None,
    use_gate_in_kernel=False,
    A_log=None,
    dt_bias=None,
    disable_recompute=False,
    return_intermediate_states=False,
    state_v_first=False,
    epsilon=1e-6,
    use_qk_l2norm_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    use_exp2=True,
)
```

The existing 12-result contract is preserved:

```text
(attn_out, final_state, gk, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state)
```

- `q/k/v` must be FP16 or BF16 with matching dtype. `K/V` must both be 64 or both
  be 128; mixed dimensions and `V=256` are rejected before launch.
- Input layouts are `BSND`, `BNSD`, `TND` and `NTD`. Attention output is always
  sequence-major; intermediate token tensors are head-major. Public `h` is
  sequence-major. `state_v_first` controls the last two state dimensions.
- `g/beta` accept FP32 or BF16. Raw gate activation requires FP32 `A_log[H_v]`
  and accepts optional FP32 `dt_bias[H_v*K]`. Safe raw gates require
  `lower_bound` in `[-5, 0)`; `None` selects `-5`.
- `cu_seqlens` and flattened `(sequence, chunk)` pairs are host integer lists.
  When omitted, chunk indices are generated in canonical sequence order.
  There may be at most 1024 logical sequences; rank-4 packed input requires
  `B=1`. Device metadata must not be copied to the CPU in the inference hot path.
- `final_state` is returned only when requested. `gk` is returned for
  preactivated gates or `disable_recompute=True`. `Aqk/Akk` are always returned;
  `w/u/qg/kg/v_new` require `disable_recompute=True`. `h` additionally supports
  `return_intermediate_states=True`. The last result aliases the input state.
- `epsilon` must be positive and finite in FP32. `allow_neg_eigval=True` requires
  `use_beta_sigmoid_in_kernel=True`; beta then becomes `2 * sigmoid(beta)`.
- Non-default normalization/beta/exponent switches require BF16, `K=V=128`,
  chunk size 64 and strictly increasing offsets. Unsupported combinations fail
  explicitly. The model retains its existing beta preprocessing, so it does
  not request beta sigmoid twice.

V2 invokes `ChunkKdaFwdPrepare -> ChunkFwdH -> ChunkKdaFwdFinalize` through one
executor. For direct calls with all default switches, the upstream small-workload
policy retains the fused entry below `H_v * total_chunks = 4096`; requesting
in-kernel normalization selects V2 regardless of workload size. Optional
backward-only V2 output slots are passed as null by this inference binding.

The [upstream API snapshot](docs/upstream_api.md) describes the ACLNN interfaces.
The PyTorch entry above uses the repository's `_C_ascend` namespace and does not
expose upstream training wrappers or direct diagnostic launchers.

## Source layout and build

| Component | Repository path |
| --- | --- |
| Fused entry and V2 executor | `csrc/attention/chunk_kda_fwd` |
| Prepare | `csrc/attention/chunk_kda_fwd_prepare` |
| State propagation | `csrc/moe/chunk_fwd_h` |
| Finalize | `csrc/attention/chunk_kda_fwd_finalize` |
| Fused-path dependencies | `csrc/attention/kda_gate_cumsum`, `csrc/moe/chunk_gated_delta_rule_fwd_h` |
| Shared kernel utilities | `csrc/moe/common` |

The CMake dependency list includes dependent operator definitions as well as
sources, so a filtered `--ops=chunk_kda_fwd` build also packages the V2 stage
configurations. The regular A2/A3/Ascend 950PR & 950DT custom operator build selects this operator.
Relative includes are adapted to this repository and its installed kernel layout.

The migration retains this checkout's Ascend 950PR/950DT Prepare final-key alias/tail handling,
the existing GDN state kernel's 310P/tail fixes, and CANN version compatibility.
It imports the upstream gate-cumsum scalar dependency fixes and Ascend 950PR/950DT Finalize
event synchronization along with the V2 host implementation. Shared kernel
utilities already present in the repository are reused, with the upstream MMAD
unit-flag event initialization fix applied.

On the matching Linux/CANN/PyTorch-NPU development image, build from the repository
root using its normal source installation workflow, for example on A3:

```shell
git submodule update --init --recursive
SOC_VERSION=ascend910_9391 COMPILE_CUSTOM_KERNELS=1 python -m pip install -v -e . --no-build-isolation
```

Use the matching vLLM revision and CANN dependencies required by this checkout.
Restart workers after installation so the extension and custom package are
loaded from the same build.

## Validation

Run the focused CPU dispatch tests and, on Ascend hardware, the numerical
regressions:

```shell
pytest -q tests/ut/ops/test_kda.py tests/ut/ops/test_kimi_kda.py tests/ut/models/test_glm5next_kda_contracts.py
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/test_chunk_kda_aclnn.py
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/test_chunk_kda_fwd_v2_npu.py
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/test_kimi_k3_chunk_kda_tail_npu.py
```

The numerical tests compare attention outputs and final states with reference
implementations, including packed tails, both state layouts, V2 normalization
and beta switches, fallback paths and invalid arguments. Run these tests on each
target SoC before claiming device accuracy or performance. Local CPU checks do
not validate AscendC compilation, device synchronization, or NPU throughput.

## Attribution

Upstream file-level copyright and license notices are retained. Original
flash-linear-attention-npu code is covered by the accompanying
[BSD 3-Clause license](LICENSES/BSD-3-Clause.txt); files derived from CANN keep
the [CANN Open Software License Agreement Version 2.0](LICENSES/CANN-Open-Software-License-Agreement-Version-2.0.txt).
These notices also apply to the migrated Prepare, Finalize, ChunkFwdH and shared
utility sources according to their file headers. Migration authorship does not
replace the original authors' notices.
