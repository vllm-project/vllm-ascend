# FlashState Frontier Checkpoints

## Feature Introduction

Hybrid models mix attention layers with layers that carry a recurrent or
linear-attention state (GDN, KDA, and other Mamba-style kernels). Their prefix
cache cannot be reused on full-attention KV alone: continuing from a prefix
also needs the state checkpoint that belongs to it.

FlashState frontier checkpoints cache that state at the exact end of a finished
request and resume the next request that extends the same prefix from it. The
recurrent layers then recompute only the tokens after the checkpoint instead of
the whole prefix, which shortens time to first token on long shared prefixes.

This guide covers the Ascend port. Upstream vLLM carries the same mechanism in
core; vLLM Ascend installs it as a plugin, so no model code changes.

### Usage Scenarios

- Long shared system prompts or documents reused across requests.
- Agent and RAG workloads where many requests share a prefix that crosses
  recurrent layers.
- Multi-turn sessions that repeatedly extend one conversation.

### Constraints and Limitations

| Category | Constraint |
| --- | --- |
| Model | Models with aligned recurrent state (`mamba_cache_mode="align"`). |
| Prefix caching | `enable_prefix_caching` must be on. |
| Retention | `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` must be `0`. |
| Model runner | v2 model runner (`VLLM_USE_V2_MODEL_RUNNER=1`). |
| Parallelism | Context parallelism is not supported. |
| Speculative decoding | Not supported together with this feature. |
| KV cache events | `enable_kv_cache_events` must be off; the checkpoint also owns block keys and cannot publish a consistent event stream yet. |
| Scheduler | Long-prefill splitting must keep the threshold at or above the block size. |

Turning the feature on with an unsupported combination fails at startup with a
`ValueError` naming the offending option rather than degrading silently.

## How It Works

### Frontier Units

A finished request publishes one *unit* per state checkpoint: the state blocks
that hold the recurrent state at the request's last token, plus the page blocks
of the full-attention groups along the same path. Units form a tree by prefix,
and an eviction removes a subtree, so a unit is only reused while its whole
path is still cached.

Keys follow the prefix cache's own hashing, so a hit is only reported for
requests that share the exact tokens. A frontier key names the frontier's
length; a separate *locator* key chains a fixed token window and lets an
external store find the frontiers under a shared prefix without knowing their
lengths.

### Reuse Path

1. A new request computes its block-aligned prefix hit as usual.
2. The checkpoint manager extends that hit to the longest frontier whose path
   is entirely cached.
3. The request resumes from the checkpoint and recomputes only the remaining
   tokens.
4. If the frontier is evicted between the lookup and the allocation, the
   request falls back to the block-aligned hit, so a stale lookup costs
   recomputation and never correctness.

### Eviction

Units are ranked by GreedyDual-Size: a unit's gain is the prefill compute it
saves, its cost is the memory it holds, and `L` rises as units are evicted so
that a unit evicted once is ranked lower the next time it is cached.

Two optional refinements:

- `prefix_cache_leaf_last` evicts inner units before leaves, so a subtree stays
  reachable as long as its leaves are worth more than their shared path.
  `prefix_cache_leaf_idle_seconds` demotes a leaf that has gone unused.
- `prefix_cache_eviction: "gds"` with `prefill_cost_coefficients` weights the
  gain by a prefill cost model instead of raw token count. Without it the
  policy is plain LRU on the same units.

Requests at the front of the waiting queue have their hits pinned briefly so
the next scheduling step can still find them; the pin set is a hint and is
refreshed at a bounded rate.

## Configuration

Enable the feature with `additional_config`:

```bash
vllm serve <model> \
  --additional-config '{"flashstate": {"enable_frontier": true}}'
```

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `enable_frontier` | bool | `false` | Turn frontier checkpoints on. |
| `prefix_cache_eviction` | str | `"lru"` | `"lru"` or `"gds"`. |
| `prefill_cost_coefficients` | list[float] | none | Required by `"gds"`; the prefill cost model's coefficients. |
| `prefix_cache_leaf_last` | bool | `false` | Evict inner units before leaves. |
| `prefix_cache_leaf_idle_seconds` | float | none | With `leaf_last`, demote a leaf unused for this long. |

The feature is inert unless `enable_frontier` is set: importing the plugin
alone installs hooks that return immediately.

## Verification

Check that the feature came up:

```bash
grep "flashstate: frontier checkpoints enabled" <server log>
```

Then compare a shared-prefix workload with and without the switch, holding the
prompt, output length, and concurrency fixed:

```bash
vllm bench serve --backend openai-chat --model <model> --served-model-name <name> \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len 16384 --prefix-repetition-suffix-len 1024 \
  --prefix-repetition-output-len 256 --prefix-repetition-num-prefixes 4 \
  --num-prompts 200 --request-rate 4 --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el
```

Time to first token is the metric that moves; decode throughput should stay
flat. A run where TTFT is unchanged usually means the workload never shares a
prefix across recurrent layers -- check that the prefix length exceeds
`mamba_block_size` and that prefix caching actually hits.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Startup `ValueError` about `mamba_cache_mode` | Set `mamba_cache_mode="align"`. |
| Startup `ValueError` about the retention interval | Set `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0`. |
| Startup `ValueError` about speculative decoding | Turn the feature off; the two are not supported together. |
| Startup `ValueError` about context parallel | Drop CP for this feature. |
| No TTFT change | Prefix too short, no shared prefix, or prefix caching misses. |
| Log says checkpoint publish failed | Look for the `flashstate:` exception that follows; publish failures are logged and never abort the request. |
