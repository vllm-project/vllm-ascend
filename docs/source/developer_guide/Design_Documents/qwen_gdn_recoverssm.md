# Qwen GDN RecoverSSM on Ascend

This experimental implementation targets `Qwen/Qwen3.6-35B-A3B`, whose model
architecture is `Qwen3_5MoeForConditionalGeneration`. It uses the existing
`--use-replayssm` switch with MTP and Model Runner V2. The default remains off.

## Initial scope

- BF16 activations, FP32 recurrent state; 16 key heads and 32 value heads,
  both with dimension 128, and convolution width 4.
- TP must divide 16; PP, PCP and DCP must all be 1.
- Mamba cache mode `none`, prefix caching disabled, no KV transfer or stochastic
  rounding. 310P is not supported.
- Linear MTP verification windows only. Full attention and the draft model
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

## State contract

For a window of `L = num_speculative_tokens + 1`, each GDN layer owns five
paged views: conv history, FP32 checkpoint `[B,HV,V,K]`, FP32 corrections
`[B,HV,L,V]`, FP32 effective keys `[B,HK,L,K]`, and FP32 log-decays `[B,HV,L]`.
The key head for value head `h` is `h // (HV/HK)`. Heads are TP-local.
Block strides may contain allocator padding; inner record/state dimensions
are dense. The context keeps references to the actual page views and device
pointer descriptors. It supports both convolution axis orientations.

Verification loads the checkpoint and evaluates `S = exp(g)*S + delta*k^T`,
where `delta = beta*(v - exp(g)*S_previous*k)`. It writes outputs and records,
but never writes the checkpoint. Q/K normalization and beta/g computation
are shared with the ordinary GDN path. Records store the effective key after
the existing normalization's dtype conversion.

After rejection sampling, the existing v2 `RecoverSSMState` passes sampler
counts to the metadata commit method. The GDN context reconstructs only the
accepted prefix in verification order, then shifts convolution history by
`count - 1`. Next-step convolution acceptance offsets are therefore neutral
(`1`). A count of zero leaves both checkpoint and convolution unchanged.

The upstream `use_kda_recoverssm` field is a legacy name for the shared
single-checkpoint lifecycle switch. Enabling it here does not select a KDA
kernel: the GDN metadata builder overrides context construction explicitly.
The Ascend GLM metadata adapter supplies request partitioning and graph staging;
the GDN subclass supplies its own prefill descriptors and commit context.

## Validation

```bash
python -m pytest tests/ut/models/test_qwen_gdn_recoverssm_config.py -v
python -m pytest tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_recoverssm_npu.py -v
python -m pytest tests/e2e/pull_request/four_card/test_qwen_gdn_recoverssm.py -v
```

The kernel tests cover GQA, ragged windows, all acceptance lengths, zero-length
padding, padded block strides, two layers, remapped sampler rows, both conv
layouts, and consecutive rounds. The model test compares greedy generation
with RecoverSSM off/on using the same MTP settings, then eager/graph execution.
NPU compilation, numerical tolerances, end-to-end accuracy, actual allocated
memory and throughput must be measured on the deployment stack before enabling
this path in production. Logical record-size savings alone do not establish
allocator savings or a speedup.
