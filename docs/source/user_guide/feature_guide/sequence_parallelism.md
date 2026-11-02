# Sequence Parallelism

## Overview

Sequence Parallelism (SP) shards the token dimension across tensor-parallel
ranks around the communication boundaries of transformer layers.

On vLLM Ascend, SP covers two paths:

- **SP MoE** (MoE checkpoints): the attention `o_proj` ends with a TP
  all-reduce, so its inputs are replicated on every TP rank. Feeding those
  replicated tokens directly into the experts duplicates compute and
  communication under expert parallelism. SP MoE keeps the expert inputs
  sharded by sequence and restores the expected layout at the MoE output
  boundary instead.
- **Dense SP** (all dense checkpoints): the decoder stack keeps hidden
  states and the residual on the rank-local token shard between the layer
  projections. Elementwise work (norms, activations, residual adds) and
  between-layer activations shrink to the shard; communication volume is
  unchanged (the row-parallel all-reduce becomes a reduce-scatter).

## Principle

SP MoE shards the input along the token dimension in each
Transformer layer. Different TP ranks therefore process different tokens,
avoiding duplicate expert computation for the same tokens.

The main data flow of an MoE layer is:

```text
Sequence-parallel input sharding
  -> TP all-gather: collect tokens from all ranks
  -> attention
  -> TP reduce scatter
  -> RMS Norm
  -> Router
  -> all-to-all
  -> Moe
```

Dense SP applies the same layout through the generic layer primitives every
dense family shares, so no per-model wiring is needed:

```text
embedding lookup
  -> TP reduce-scatter           (entry: hidden states become token-sharded)
  -> per decoder layer:
       RMSNorm (rank-local shard)
       -> column projection      (qkv / gate_up / GDN in-projections:
                                  TP all-gather before the GEMM)
       -> attention / GDN core  (full tokens, as without SP)
       -> row projection         (o_proj / down_proj / out_proj:
                                  TP reduce-scatter instead of all-reduce)
       -> RMSNorm + residual     (rank-local shard)
  -> final norm (rank-local shard)
  -> TP all-gather                (exit: full token rows before logits)
```

Numerics are equivalent to the all-reduce layout; gains come from elementwise
work and between-layer activations shrinking to the rank-local shard. The
magnitude is hardware-dependent (measurable on high-HCCS-bandwidth hardware
such as 950PR/A5; may be negligible on 800I A3). Decode-heavy workloads may
regress — the switch is opt-in per deployment.

Different DP ranks may have different numbers of valid tokens. Therefore, the
buffer after all-gather cannot be treated as a contiguous sequence of valid
tokens; it must be unpadded and zero-padded according to each rank's local
token size. This keeps tokens sequence-sharded during expert computation and
reduces duplicate computation and unnecessary communication.

## How to use

Steps to follow to enable SP currently:

- `tensor_parallel_size > 1`.
- MoE checkpoints: `enable_expert_parallel` is set.
- `--additional-config '{"enable_flashcomm1": true}'` set `flashcomm1`

> [!NOTE]
> **Difference from upstream.** Upstream vLLM enables MoE sequence parallelism only when `data_parallel_size > 1`, together with a supported all2all backend, expert parallelism, and `tensor_parallel_size > 1`. On vLLM Ascend, `data_parallel_size > 1` is not part of the enablement condition. Ascend FlashComm also supports the TP/EP topology with `data_parallel_size = 1`, so SP MoE can be enabled when DP is 1 as long as the conditions above are met. `data_parallel_size > 1` remains supported.

### Dense sequence parallelism (all dense families)

With the same FlashComm switch, any dense checkpoint following the standard
decoder skeleton is covered — Llama, Qwen2/Qwen3 dense, GLM dense, the dense
Qwen3.5 family (including its GDN linear-attention layers), and others —
without per-model switches. Requirements:

- `tensor_parallel_size > 1` and `pipeline_parallel_size == 1`.
- `--additional-config '{"enable_flashcomm1": true}'` (or the
  `VLLM_ASCEND_ENABLE_FLASHCOMM1` env var).

```bash
vllm serve <dense-model> \
  --tensor-parallel-size 2 \
  --additional-config '{"enable_flashcomm1": true}'
```

The runner pads every step's executed token count up to the TP multiple
(V2 rounds the batch token counts in `_prepare_inputs_impl`, V1 keeps its
existing `_pad_for_sequence_parallelism`), so the padding rows ride along
exactly like graph-bucket padding rows (PAD_SLOT_ID cache slots,
`is_padding` mask) and are dropped before logits.

MTP speculative decoding (`qwen3_5_mtp`) is supported together with dense SP
under the V2 model runner: the drafter shares the target's embedding, and the
patched MTP forward gathers the embedding's token shard back to full rows
before combining them with the target's hidden states, so the draft head
itself runs the stock full-token layout. Other speculative methods (EAGLE,
DSpark, ...) and the V1 runner keep the SP-off layout.

Multimodal-capable checkpoints of the dense Qwen3.5 family
(`Qwen3_5ForConditionalGeneration`, a separate `text_config` in the HF config)
also run dense SP under the V2 runner: the multimodal merge happens
full-token side in the runner's `inputs_embeds` preparation, and the wrapper
restores the full-token text embeddings around the merge before sharding the
merged embeds to the rank-local token shard, so the vision feature scatter
never touches the sequence-parallel region. Other multimodal families keep
the SP-off layout (their model classes do not declare the wiring).

Unsupported topologies keep the stock all-reduce layout: MoE checkpoints
(upstream SP MoE owns their layout), MLA checkpoints, multimodal and
speculative-decoding deployments, PCP/DCP topologies, pipeline parallelism,
LoRA, and the finegrained TP groups. A checkpoint whose residual-stream
projection is not recognized fails loudly (shape check in the row wrapper)
rather than computing on a wrong token layout; report the projection name to
extend the coverage list.

### Matmul reduce-scatter fusion

On supported Ascend devices, enable `fuse_gemm_comms` to fuse a
sequence-parallel, bias-free row-parallel projection with its token-dimension
reduce-scatter. The fused implementation supports BF16/FP16 with TP size 2,
4, or 8; unmatched graphs continue to use the unfused path.

```bash
--compilation-config '{"pass_config":{"fuse_gemm_comms":true}}'
```

### FlashComm switch (Ascend only)

vLLM Ascend enables SP MoE through the FlashComm switch. The switch is still
required; SP MoE is not enabled from the parallel configuration alone.

To enable SP MoE, set `additional_config.enable_flashcomm1`:

```bash
# On vLLM Ascend, data-parallel-size may be 1.
# Upstream requires data-parallel-size > 1 for the same SP path.
vllm serve <moe-model> \
  --data-parallel-size 1 \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --additional-config '{"enable_flashcomm1": true}'
```
