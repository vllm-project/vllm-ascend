# GDN checkpoint/replay for speculative decoding

With `enable_gdn_recover`, speculative Gated DeltaNet verification keeps one
FP32 checkpoint and small per-token replay records. After rejection sampling,
the runner commits the processed input prefix and its matching convolution
history. This removes the GDN cache manager's per-draft state reservations.
The option defaults to `false`.

## Supported configuration

The initial implementation supports dense Qwen3.5-family GDN targets with:

- An A2/A3 NPU and Model Runner V1, with eager target and drafter execution.
- Native MTP with 1–5 speculative tokens, a BF16 model/conv cache and FP32 SSM.
- TP, PP, DP, PCP and DCP sizes all equal to one.
- Prefix caching disabled and `mamba_cache_mode=none`.
- No KV transfer or sleep mode.

Unsupported combinations fail at configuration validation. The primitives
support graph capture separately; the integrated engine path requires eager
execution. Multi-device operation and other Mamba cache modes need additional
ownership and cache-copy integration.

```bash
VLLM_USE_V2_MODEL_RUNNER=0 vllm serve /path/to/dense-qwen35-model \
  --dtype bfloat16 --language-model-only --enforce-eager \
  --mamba-ssm-cache-dtype float32 --no-enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"enforce_eager":true}' \
  --additional-config '{"enable_gdn_recover":true}'
```

## Memory and latency

For 48 value heads, K=V=128 and three speculative tokens, four full SSM
snapshots require 12MiB per layer/request. A checkpoint and four records
require approximately 3.188MiB, before convolution history and allocator
padding. The integrated layout includes checkpoint, record and saved conv
fields in its reported cache specification. It sets GDN's speculative block
reservation to zero and expands page padding if the payload requires it.

This lowers state memory charged per request; a fixed physical KV pool can
then hold more requests. It does not necessarily lower the process's total
allocated HBM. Attention KV, page grouping, context length and other buffers
also limit concurrency.

Commit adds replay work. Compatible physical layouts share one SSM/conv commit
launch across layers; other layouts use the per-layer primitives. Single
request latency may increase, while workloads limited by state capacity may
benefit from admitting more requests. Measure throughput and latency under
the deployment's own load and SLO before enabling it.

## State ownership

Each attention metadata builder snapshots its state slots, packed query
offsets and sampler row mapping. The runner owns pending commits and commits
each verified layer once. The processed count includes the mandatory input
and accepted drafts; it excludes the newly sampled bonus token.

Speculative convolution always starts from committed history. Both SSM and
conv writes finish before completion is published and before the next forward
can reuse state blocks. Cache reinitialization clears physical descriptors.
The batched kernel uses 64-bit offsets and does not copy state pools.

See the [kernel contract](../../developer_guide/Design_Documents/gdn_recover_primitives.md)
for the recurrence, storage invariants and independent oracle tests. The
algorithm follows upstream's [GDN RecoverSSM proposal](https://github.com/vllm-project/vllm/pull/58863);
this feature supplies the Ascend kernels and framework integration.
