# Kimi BF16 alignment reference (draft)

This draft isolates reference execution paths for comparing Kimi K3.1 BF16
rollout logits with an independent FSDPTurbo actor forward. It does not load
quantized weights. The switches below are disabled unless explicitly set to `1`.
They must reach every relevant rollout worker before model initialization.

| Switch | Reference behavior |
| --- | --- |
| `KIMI_ALIGNMENT_RECURRENT_KDA` | One recurrent KDA call per token and document, with FP32 state and BF16 operands. The paired training path computes from its own inputs. |
| `KIMI_ALIGNMENT_SPLIT_GATED_NORM` | Separate RMSNorm and sigmoid gate to match training rounding boundaries. |
| `KIMI_ALIGNMENT_MOE` | BF16 gate output/sigmoid boundaries and explicit routing normalization. |
| `KIMI_ALIGNMENT_FULL_O_PROJ` | Gather full-K inputs/weights, project once, then return the expected token shard. |
| `KIMI_ALIGNMENT_DENSE_MLP` | Use the training SiTU activation and full-K sequence projection. |
| `KIMI_ALIGNMENT_NATIVE_ATTN_RES` | Preserve native residual state and use the training residual mixture/normalization. |
| `KIMI_ALIGNMENT_DSA_NORM` | Use the training DSA normalization epsilon. |
| `KIMI_ALIGNMENT_DSA_RAW_CHANNELS` | Keep raw NoPE channels for the supported native SFA preprocessing path. |
| `KIMI_ALIGNMENT_ALLTOALL` | Select the plain all-to-all MoE communication path for the experiment. |
| `KIMI_ALIGNMENT_DETERMINISTIC_LEVEL=3` | Enable `torch_npu.npu.set_deterministic_level(3)` in the rollout worker. Leave unset for native performance baselines. |

The reference recurrence is deliberately slow. Supported layouts are checked at
the helper boundaries; unsupported dimensions, speculative paths, cache layouts
or preprocessing modes can raise instead of silently claiming equivalence.
Default and non-Kimi deployments retain their original paths. This is a
diagnostic recipe, not a throughput improvement.

The `KIMI_ALIGNMENT_AUDIT_DIR` tracing hooks are for a dedicated diagnostic
process. They can retain inputs/activations, transfer tensors to the host and
synchronize execution; do not enable them in a shared production server.
Multimodal and expert tensor traces are retained for investigation, and are not
evidence of invariance by themselves.

This branch was ported from an older experimental layout onto current main.
The prior four-layer short runs observed zero independent logprob differences,
but that result does not validate this migration. No new NPU model run was
started for this draft. NPU forward/backward, BI, PD, real weight updates and
synchronization, full-depth Kimi, model quality and throughput remain pending.
