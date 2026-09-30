# Kimi K3 A5 MegaMoE

## Scope

This design routes eligible Kimi K3 routed experts through CANN MegaMoE on
Ascend A5. It applies to the vLLM 0.27-compatible release line and W4A8 MXFP
checkpoints. The existing MoE implementation remains the fallback for other
hardware, quantization formats, and unsupported runtime configurations.

MegaMoE replaces only the routed-expert core:

```text
full hidden states
  |-- gate -> top-k ids and weights
  |-- latent down projection -> [tokens, 3584]
       |-- MegaMoE: dispatch -> GMM1 -> SiTU -> GMM2 -> combine
       `-- routed RMSNorm -> latent up projection
  `-- shared experts -----------------------------------------> sum
```

The Kimi-specific gate, latent projections, routed output transform, and
shared experts stay in the existing `AscendKimiMoE` implementation.

## Selection

The `FUSED_MC2`/MegaMoE path is selected only when all of these conditions
hold:

- the device is A5, the process is not a PD decode node, and expert
  parallelism has more than one rank;
- `additional_config.enable_fused_mc2` is `1`;
- the instantiated routed-expert quantization is `W4A8MXFP`;
- the activation is SiLU/SwiGLU or SiTU;
- the MXFP group size is 32 when the metadata exposes it;
- dynamic EPLB, redundant experts, mixed placement, and MoE LoRA are disabled;
- the batch fits the process-wide symmetric-buffer capacity.

If a condition is not met, communication selection uses the existing MC2,
AllGather, or AllToAll path. Selection is made from the instantiated layer's
quantization type where possible, rather than from a model-name check.

## Kimi K3 Operator Contract

Kimi K3 routes in latent space. For the released shape, the MegaMoE arguments
are:

| Argument | Value |
| --- | --- |
| `num_experts` | 896 |
| `num_topk` | 16 |
| `hidden` | `routed_expert_hidden_size`, 3584 |
| `intermediate_hidden` | full gate/up projection width, 6144 |
| dispatch mode | MXFP, 4 |
| activation dtype | FP8 E4M3 |
| weight dtype | packed FP4 E2M1 |

The weight tensors passed by the W4A8 MXFP method use the checkpoint-oriented
stacked layout expected by MegaMoE:

```text
w1       [local_experts, 2 * intermediate, hidden / 2]
w2       [local_experts, hidden, intermediate / 2]
w1_scale [local_experts, 2 * intermediate, hidden / 64, 2]
w2_scale [local_experts, hidden, intermediate / 64, 2]
```

The existing grouped-matmul representation is restored with transposed views;
no expert-weight copy is introduced in the forward path. Unsigned-byte MXFP
scale storage is reinterpreted as E8M0 without a copy.

Kimi K3 uses SiTU, which is not equivalent to SwiGLU. The backend maps the
vLLM activation configuration to this CANN call contract:

```python
mega_moe(
    ...,
    activation="situglu",
    activation_params={
        "beta": activation_situ_beta,
        "linear_beta": activation_situ_linear_beta,
    },
)
```

For the released Kimi K3 configuration, `linear_beta` is 25 and `beta` is 4.
`activation_clamp` is not used for SiTU. If the installed ops-transformer
package does not expose `activation_params`, execution fails explicitly
instead of silently substituting SwiGLU.

## Buffer And Collective Rules

MegaMoE is a collective operation. Every rank must choose the same backend,
use the same expert configuration, and acquire a compatible symmetric buffer.
The buffer is cached on the dedicated MegaMoE process group and shared by all
compatible MoE layers in the process.

The per-rank capacity is:

```text
min(mega_moe_max_tokens / expert_parallel_size,
    execution_tokens_per_rank)
```

The execution limit comes from the scheduler budget in standalone and prefill
deployments. Decode nodes never select MegaMoE and retain the existing
MC2/AllToAll path. A request larger than the buffer capacity falls back during
communication selection; a direct contract violation raises an error before
entering CANN.

## Configuration

Enable the path with the regular additional configuration interface:

```shell
vllm serve <KIMI_K3_W4A8_MXFP_PATH> \
  --enable-expert-parallel \
  --additional-config '{"enable_fused_mc2":1,"mega_moe_max_tokens":65536}'
```

All ranks must use the same configuration and a CANN/ops-transformer build that
exports `mega_moe`, `get_symm_buffer_for_mega_moe`, and the SiTU
`activation_params` argument.

## Validation

CPU unit tests cover communication selection, the SiTU argument mapping,
SwiGLU compatibility, the packed weight/scale layout, and the K3 buffer width.
Release qualification on A5 must additionally cover:

1. a two-rank operator smoke test comparing MegaMoE with the decomposed path;
2. eager prefill and decode with uneven per-rank token counts;
3. a full Kimi K3 service request on the intended multi-node EP topology;
4. accuracy comparison with `enable_fused_mc2=0`;
5. profiler confirmation that routed layers execute MegaMoE and do not execute
   separate dispatch, SiTU, GMM2, and combine operators;
6. peak HBM and throughput measurements for the selected
   `mega_moe_max_tokens` value.
