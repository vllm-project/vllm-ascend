# DeepSeek V4.1 A2 prefill MegaMoE

This experimental, disabled-by-default integration targets a two-server 1P1D
deployment with 16 A2 devices on each server. Only the P worker's main-model
prefill uses CANN MegaMoE. D-side decode, DSpark and graph settings retain their
existing execution paths. Hardware precision and performance qualification is
required before deployment.

## Requirements

- Start from vLLM Ascend `v0.32.0rc1` with its matching vLLM `v0.32.0` stack.
- Install a compatible `cann_ops_transformer` package and communication extension
  on every P rank. It must include the `ascend910b` A2 W4A8 MegaMoE kernel.
- Use DeepSeek-V4.1-Flash, registered as `DeepseekV41ForCausalLM`, with H=5120,
  I=2304, 384 routed experts, top-k=6 and dynamic per-token INT8 / per-channel
  INT4 (W4A8) routed experts. BF16 activations are required.
- P topology is DP2/TP1/PCP8 with expert parallel enabled: EP16, 24 experts per
  rank. PP and DCP must be 1. Use model runner V2 and `--enforce-eager`.
- The P worker must have `kv_role="kv_producer"`. LoRA, EPLB, redundant experts,
  mixed placement and microbatching are rejected.
- Keep `weight_nz_mode` at `1` (default) or `2`; W4A8 MegaMoE requires NZ
  weights. Expert placement must be linear, with no custom expert map.

I=2304 already meets the A2 operator's 256 alignment. No I640 alignment patch
or intermediate-width padding is needed. Check the actual W4A8 checkpoint's
quantization description; other routed expert schemes are rejected on the main
model. Existing DSpark configuration may remain enabled, but draft execution
does not select MegaMoE.

## Enable on P only

Retain the network, model path, KV transfer, Engram and scheduler options from
your working [DeepSeek V4.1 deployment](../../tutorials/models/DeepSeek-V4.1-Flash.md).
Apply these options to **both P DP processes**, using the existing PCP-aware
launcher to assign eight devices to each process:

```shell
export VLLM_USE_V2_MODEL_RUNNER=1

# Merge these options into the existing P launch command, retaining its
# --data-parallel-rank, address, RPC, model and KV connector settings.
vllm serve /path/to/DeepSeek-V4.1-Flash-w4a8 \
  --dtype bfloat16 \
  --quantization ascend \
  --data-parallel-size 2 \
  --tensor-parallel-size 1 \
  --prefill-context-parallel-size 8 \
  --enable-expert-parallel \
  --enforce-eager \
  --swa-bounded-replay \
  --additional-config '{"enable_fused_mc2":2,"a2_mega_moe_prefill":{"enabled":true,"scope":"prefill"}}' \
  --kv-transfer-config '{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_producer"}'
```

The example shows the feature arguments; it is not a complete two-process
launcher. Preserve the working connector's endpoint and topology configuration.
Merge additional-config keys rather than replacing unrelated existing settings.
Keep FlashComm disabled for the bounded-replay layout. Do not add this opt-in to D.

| Additional configuration | Default | Meaning |
| --- | --- | --- |
| `enable_fused_mc2` | `0` | Set to `2` to request the CANN MegaMoE package. `1` selects the legacy fused operator. |
| `a2_mega_moe_prefill.enabled` | `false` | Enable the supported A2 P-only backend. Requires `enable_fused_mc2=2`. |
| `a2_mega_moe_prefill.scope` | `"prefill"` | Only `prefill` is supported; `all` and `prefill_mixed` are rejected. |

Missing CANN dependencies or unsupported worker settings fail validation.
The existing fused-MC2 configuration disables shared-expert multistream overlap.
Shared-expert computation and routed scaling otherwise retain existing semantics.

## Selection and resource ownership

Each eager forward votes explicit DSA request-phase counts over the EP CPU
group before selecting the collective backend. A decode, mixed or unknown phase
on any participating rank vetoes MegaMoE for all ranks. Empty replicas may join
prefill peers with no active rows. A one-token prefix-cache tail is prefill;
query length is not used to infer the phase. All-empty real batches fall back.
This adds a small Gloo collective per eligible forward; include its cost when
measuring P-side latency and throughput.

Explicit eager memory profiling also initializes and measures MegaMoE resources.
Allocation alone does not prove real requests used MegaMoE. The temporary
validation diagnostics below show the selected backend outside profiling.
Capture never selects this path. The P worker remains eager, including SWA
bounded replay. Bounded replay's compact decoder rows retain their own active
mask and are not treated as autoregressive decode.

MRv2 may enter a draft with the target's forward configuration. Routed draft
layers explicitly select their ordinary backend from their own model config.
Main-model W4A8 uses the existing expert TensorList format for both MegaMoE and
ordinary GMM fallback. No complete second execution-weight copy is retained;
loading and NZ conversion may still have a transient memory peak.

The sender reservation uses the scheduler budget conservatively, without dividing
by PCP8 or applying a token-count selection cutoff. For per-rank sender capacity
B, the receive-route bound is `B * 16 * min(6, 24) = 96B`. All ranks allocate
identical resources. The A2 path uses this bound instead of the general
`mega_moe_max_tokens` hint. Large scheduler budgets can substantially increase
communication memory and workspace; measure available KV capacity and peak HBM.
Inputs larger than the allocated sender capacity fail rather than truncating.
Ordinary fallback retains its original 512-token-per-rank MC2 selection limit.

## Temporary validation diagnostics

With `a2_mega_moe_prefill.enabled=true`, this validation build logs every forward
at INFO level by default. No extra configuration or DEBUG logging is needed.
Collect the output of **both P DP processes**, including all 16 EP workers:

```shell
rg 'A2_MEGA_MOE' p-dp0.log p-dp1.log
```

- `batch`: local and PCP-global request counts, token counts, appended padding
  rows, `is_dummy`, `is_cpp_profile`, CPU prefill flags and scheduled lengths.
  Appended padding excludes padding already present inside the local batch.
- `eligibility`: draft identity, runner, package/hardware support, fused-MC2
  setting, EP/LoRA state, DP/TP/PCP, token count, graph and profile flags.
- `vote_enter`: local `(prefill, veto)` and attention metadata counts. Shared
  metadata objects are grouped by identity; `aliases` counts their layer names.
  Reasons distinguish empty rows, decode/mixed requests, missing/invalid counts
  and inconsistent request counts.
- `vote_exit`: EP-wide MAX of the two vote fields and the resulting selection.
  A local veto of 1 identifies a rank rejecting fusion; a global veto of 1
  rejects fusion for every rank. A global prefill of 0 also rejects fusion.
- `gate_skip`: graph execution rejects fusion; `profile_forced` selects fusion
  only for resource profiling, without a phase vote.
- `decision`: selected communication method and fallback reason/capacity.
- `op_enter`: actual entry into `cann_mega_moe` or ordinary
  `npu_moe_distribute_dispatch_v2`, including the input shape and quantization
  mode. Ordinary MC2 also logs its communication algorithm. This line precedes
  the operator call and does not establish successful kernel completion.

Start with `profile=False` eligibility records from a failing request, then
compare the batch flags and local votes across ranks. For example, local
`(1, 0)` with global `[1, 1]` means another rank vetoed the request; find its
`vote_enter` with local `(0, 1)` and inspect its batch/metadata reason.
`op_enter ... x_shape=(512, 5120)` confirms the actual MC2 input has 512 rows,
even if only some rows are active.

These diagnostics leave routing and existing collectives unchanged. They read
CPU metadata and tensor shapes/dtypes without copying device tensor contents.
They log neither prompts nor request IDs. Per-forward and per-layer output can
be large and affect timing; this commit is intended for fault reproduction,
not performance measurement.

## Validation and rollback

CPU policy and extracted production-function checks can run without vLLM/NPU:

```shell
python -m pytest --noconftest -q tests/ut/ops/test_a2_mega_moe_prefill.py
```

On the 16-device P server, source the matching CANN environment, set the working
HCCL interfaces, and run in a fresh process per shape:

```shell
torchrun --nproc-per-node=16 tests/e2e/nightly/single_node/ops/test_a2_deepseek_prefill.py --tokens 32
```

Repeat with tokens 1, 8, 512 and representative larger inputs. The standalone
test compares the production W4A8 TensorList/MegaMoE endpoint with independent
all-gather routing and ordinary stacked GMM, including skew and padding. It
checks exact expert counts and BF16 mixed tolerance (`atol=rtol=2^-6`, at least
99% matching elements, maximum absolute error at most 1.0), rejecting NaNs.
It
does not qualify the complete model, PCP layout, PD transfer or performance.

Before merging, validate P DP2/TP1/PCP8 with different lengths on the two DP
replicas, idle ranks, prefix-cache tails, chunk boundaries, encoder/decoder
bounded replay, and realistic expert skew. Compare output/logits and task
accuracy against the unchanged deployment. Run end-to-end PD requests and
regress the existing D decode/DSpark/graph path. Record TTFT, P prefill time,
throughput, KV capacity and peak HBM with identical workloads.

To roll back, restore P's original additional-config, set
`a2_mega_moe_prefill.enabled=false`, and restart all P workers. Changing routing
without restarting does not restore the original weight-loading configuration.
