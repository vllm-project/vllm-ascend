# Qwen GDN RecoverSSM on Ascend

This experimental implementation targets `Qwen/Qwen3.6-35B-A3B`, whose model
architecture is `Qwen3_5MoeForConditionalGeneration`. It uses the existing
`--use-replayssm` switch with Model Runner V2 and speculative decoding,
including MTP and DSpark. The default remains off.

## Initial scope

- BF16 activations and convolution state; BF16 or FP32 checkpoint state,
  as returned by the base GDN dtype calculator. With BF16 convolution state,
  `--mamba-ssm-cache-dtype auto` preserves BF16 checkpoints.
- TP must divide the configured key and value head counts; PP, PCP and DCP
  must all be 1.
- Mamba cache modes `none` and `align`; `align` supports prefix caching and
  requires a Mamba block size covering one speculative window. No stochastic
  rounding. 310P is not supported.
- PD separation uses `MooncakeConnectorV2` or `MooncakePullConnector` with
  matching weights, TP size, speculative configuration and RecoverSSM settings;
  PD currently requires cache mode `none` and prefix caching disabled.
- Linear verification windows only. Full attention and the draft model
  retain their existing implementations.

Example on a compatible NPU host (adjust model path and TP to the hardware):

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve Qwen/Qwen3.6-35B-A3B \
    --tensor-parallel-size 4 \
    --dtype bfloat16 --mamba-ssm-cache-dtype float32 \
    --mamba-cache-mode none --no-enable-prefix-caching \
    --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
    --use-replayssm --enforce-eager
```

Start validation in eager mode. Remove `--enforce-eager` only after comparing
graph execution with eager on the same NPU/CANN/triton-ascend stack.
For colocated execution without KV transfer, replace
`--mamba-cache-mode none --no-enable-prefix-caching` with
`--mamba-cache-mode align --enable-prefix-caching` to enable prefix caching.

## State contract

For a window of `L = num_speculative_tokens + 1`, each GDN layer owns three
in-page cache views, matching upstream [#61309](https://github.com/vllm-project/vllm/pull/61309):

- Convolution history, with the base shape and dtype.
- Checkpoint `[B,HV,V,K]`, preserving the base GDN checkpoint dtype.
- FP32 replay `[B,HV,L,V+K+1]`, packing correction `[0:V]`, normalized key
  `[V:V+K]`, and decay `[V+K]`.

`B` is the cache-block count; heads are TP-local. Each value head `h` records
the effective key from key head `h // (HV/HK)`. Keys are repeated per value
head, so fewer views do not imply fewer replay bytes under GQA.
Block and inner strides may contain padding. The context keeps references to
actual views and pointer/stride descriptors, and supports both convolution
axis orientations. Replay stays inside the cache page; the later out-of-page
redesign in #62625 is not used.

The selector accepts Qwen3.5 Dense/MoE and Qwen3-Next model types. Both
Qwen3.5 CausalLM and ConditionalGeneration entry points have matching model-
and layer-level descriptors. Unknown derivative model types are not enabled
merely because they expose GDN geometry fields. This descriptor coverage is
not a claim of NPU validation for every model checkpoint.
Model class entrypoints delegate to the shared model-side
`gdn_mamba_state_shape_from_config` and `gdn_mamba_state_dtype_from_config`
helpers. Both use `_uses_gdn_recoverssm`, also used to initialize the layer's
RecoverSSM flag. The predicate requires a supported model, the feature switch,
and a positive speculative token count. Without speculative tokens the helpers
return base descriptors; configuration validation still rejects an explicit
RecoverSSM request without speculative decoding.

Verification loads the checkpoint and evaluates `S = exp(g)*S + delta*k^T`,
where `delta = beta*(v - exp(g)*S_previous*k)`. It writes outputs and records,
but never writes the checkpoint. Q/K normalization and beta/g computation
are shared with the ordinary GDN path. Records store the effective key after
the existing normalization's dtype conversion. Replay records store the FP32
`exp(g)` value multiplied into the state;
commit loads this decay directly without evaluating `exp` again.

After rejection sampling, the existing v2 `RecoverSSMState` passes sampler
counts to the metadata commit method. Commit has three stages: the shared
`_prepare_commit_plan_kernel` resolves accepted lengths and destination pages;
the shared `_compact_conv_state_kernel` copies final and optional boundary
convolution histories; the GDN state kernel composes accepted records in
reverse order, accumulating decay and correction before merging the checkpoint.
This changes floating-point evaluation order from the former forward GDN commit;
device numerical validation must cover the selected kernel. Records are always
read from the source page. In `align`
mode, state is saved at the crossed block boundary and at the final destination;
in `none` mode the destination remains the source page. Convolution history is
shifted by `count - 1`, so next-step acceptance offsets are neutral (`1`). A count
of zero leaves both checkpoint and convolution unchanged. Plan buffers and
per-layer convolution descriptors are allocated once when the context is created.
KDA and GDN retain separate ordinary and grouped state kernels. GDN selects
two value heads per program on A5 for `K=V=128` and an even value-head count,
otherwise the ordinary kernel. Packed records carry each head's own effective
key, including when a grouped pair crosses a key-head boundary. Existing
Ascend tile and UB-budget policies are retained. FP32 accumulation is stored
back using the checkpoint dtype; BF16 checkpoint comparisons must account
for final-store rounding. Hardware numerical and performance checks remain
required before merging.

The upstream `use_kda_recoverssm` field is a legacy name for the shared
single-checkpoint lifecycle switch. Enabling it here does not select a KDA
kernel: the GDN metadata builder overrides context construction explicitly.
The Ascend GLM metadata adapter supplies request partitioning and graph staging;
the GDN subclass supplies its own prefill descriptors and commit context.

## Prefill/decode separation

Use the same model weights, TP size, BF16 activation/conv dtype, checkpoint
dtype, speculative configuration (including the draft model and window),
and `--use-replayssm` setting on P and D. Both sides require Model Runner V2,
`--mamba-cache-mode none`, and `--no-enable-prefix-caching`. A producer without
speculative decoding or RecoverSSM has a different state layout and is not
supported by this contract. PP, PCP and DCP remain 1.

The existing Mooncake stateful-model flow transfers the prompt state before
the final prompt token; D recomputes that token using the received conv history
and checkpoint in its configured dtype. RecoverSSM allocates one state block per request, without
per-draft checkpoint blocks. The existing scheduler preserves that block using
the Mamba group's `num_speculative_blocks` rather than the speculative window.
In the current connector registry, `MooncakeConnectorV2` and
`MooncakePullConnector` both select the pull-based implementation. Its existing
all-view protocol transfers all three views and requires matching layouts.
The packed replay records are scratch and are overwritten before use;
they remain in the transfer payload for compatibility with this protocol.
Transfer addresses use physical page strides, which can differ between P and D
because of allocator padding. The legacy `MooncakeConnectorV1` is outside this
supported configuration.

For a same-host TP4 + TP4 example, set these variables in both terminals:

```bash
MODEL=Qwen/Qwen3.6-35B-A3B
CONNECTOR=MooncakePullConnector  # Or MooncakeConnectorV2 on both sides.
COMMON_ARGS=(
    --tensor-parallel-size 4
    --dtype bfloat16 --mamba-ssm-cache-dtype float32
    --mamba-cache-mode none --no-enable-prefix-caching
    --speculative-config '{"method":"mtp","num_speculative_tokens":4}'
    --use-replayssm --enforce-eager
)
```

Start P in one terminal:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve "$MODEL" "${COMMON_ARGS[@]}" --port 8100 \
    --kv-transfer-config "{\"kv_connector\":\"$CONNECTOR\",\"kv_role\":\"kv_producer\",\"kv_port\":30000}"
```

Start D in the other terminal:

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve "$MODEL" "${COMMON_ARGS[@]}" --port 8200 \
    --kv-transfer-config "{\"kv_connector\":\"$CONNECTOR\",\"kv_role\":\"kv_consumer\",\"kv_port\":31000}"
```

For DSpark, replace `--speculative-config` with the same working DSpark draft
configuration on both sides. There is no MTP-only restriction in PD validation.
Follow the [Mooncake PD guide](../../tutorials/features/pd_disaggregation_mooncake_multi_node.md)
for transport setup and request routing. Use the same connector name on P/D
for a consistent deployment configuration. Validate eager execution before
enabling graph execution on the deployment stack.

## Validation

```bash
python -m pytest tests/ut/models/test_qwen_gdn_recoverssm_config.py -v
python -m pytest tests/ut/models/test_glm5next_recoverssm_config.py -v
python -m pytest tests/ut/kv_offload/mooncake_v2/test_pull_worker.py -v
python -m pytest tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_npu.py -v
python -m pytest tests/e2e/pull_request/four_card/test_qwen_gdn_recoverssm.py -v
```

The kernel tests cover GQA, ragged windows, all acceptance lengths, zero-length
padding, padded block strides, two layers, remapped sampler rows, both conv
layouts, and consecutive rounds. Align tests cover crossing/exact boundaries,
different and in-place final pages, zero/full acceptance, mixed per-layer conv
strides, sampler mappings and graph replay. The model test compares greedy
generation with RecoverSSM off/on using the same MTP settings, then eager/graph
execution, in both `none` and `align` modes with repeated prompts.
The four-card PD test runs TP2 producer and consumer instances for each
connector, compares token IDs with local greedy generation, and repeats
concurrent requests with recycled state pages. Its MTP case exercises the
state-transfer integration; DSpark configurations still need deployment-specific
model/drafter validation.
NPU compilation, numerical tolerances, end-to-end accuracy, actual allocated
memory and throughput must be measured on the deployment stack before enabling
this path in production. Logical record-size savings alone do not establish
allocator savings or a speedup.
