# Suffix Speculative Decoding

Suffix decoding proposes draft tokens by matching the request history and
completed responses on the NPU, without an additional draft model.

## Model Runner V2 (experimental)

This path adapts upstream [vLLM #60339](https://github.com/vllm-project/vllm/pull/60339)
to Ascend. It requires a vLLM version containing the complete upstream change
(merge `190df0b90007eead441d97d3abf6270f84f2c63e` or a descendant), including
suffix configuration and the `BaseSpeculator` request lifecycle hooks. The
repository's older verified pin does not contain these interfaces. This change
does not update that pin or backport the upstream runner protocol.

Model Runner V2 is the only runner in current vLLM Ascend. An older upstream
version raises a dependency error; there is no V1 fallback.
NPU generation, async scheduling, TP consistency and throughput qualification
are still pending. On a compatible upstream version:

```bash
vllm serve /data/Qwen3-8B \
  --async-scheduling \
  --speculative-config '{"method":"suffix","num_speculative_tokens":8,"suffix_decoding_corpus_tokens":1048576}'
```

Arctic Inference is not used by the V2 speculator. Matching works as follows:

- Each request searches its own prompt and generated tokens for the longest
  matching suffix, up to `suffix_decoding_max_tree_depth` (default 24).
- Completed responses, with up to three preceding prompt tokens, are added to
  an NPU-resident ring. Two hash indexes use the last two and four tokens to
  locate candidates. Candidate matches are extended and their continuations
  are selected by prefix-consistent frequency voting. The longer history or
  corpus match wins; history wins ties.
- Finished requests are ingested before their state slots are released.
  Preempted requests are not ingested. Sorted request IDs and integer index
  updates preserve the same corpus state across TP ranks. Upstream warmup
  resets the corpus and acceptance statistics before serving.

The default corpus capacity is 1,048,576 tokens. Setting
`suffix_decoding_corpus_tokens: 0` or `suffix_decoding_max_cached_requests: 0`
disables cross-request retrieval. On V2 the latter is only a disable switch;
positive values do not limit the number of cached requests. Corpus matching
needs a maximum match depth of at least two; depth one uses history only.
Default corpus and index storage consume approximately 36 MiB, in addition to
dense token history (`max_num_seqs * max_model_len * 4` bytes) and scratch buffers.

V2 always returns `num_speculative_tokens` draft slots. Positions without a
continuation contain the last sampled token and are verified as ordinary
drafts. Upstream metrics include these filler slots; they are not a measure
of the number of genuine suffix matches. `suffix_decoding_max_spec_factor`
and `suffix_decoding_min_token_prob` apply only to V1.

Optional `enable_adaptive_verification: true` supplies per-position confidence
from online acceptance statistics grouped by draft source, match length and
position. It requires full decode graphs and an Ascend target attention backend
that supports variable query lengths. It is disabled by default. The NPU
observer reduces batch-local integer counts per statistics bucket rather than
using the GPU observer's contended int64 atomic additions. This implementation
has not yet been tested on NPU; backend compatibility and throughput remain
to be qualified. At high concurrency, verifying every draft can cost more than
the accepted tokens save.

The corpus is shared across requests, including different `cache_salt` values
and LoRA adapters. Disable the corpus where requests must not influence each
other's drafting. As upstream documents, a fixed sampling seed alone does not
make outputs reproducible with a shared corpus or adaptive verification. For
that use case, disable both. Under async scheduling, a finished request may
contribute tokens beyond the scheduler's final stop point; ingestion follows
upstream device-visible lengths.

The Ascend speculator inherits upstream `SuffixSpeculator` initialization,
slot bookkeeping, reset and finished-response ingestion. Its proposal wrapper
only guards empty batches before calling `super()`. NPU launch wrappers in
`ops/triton/spec_decode/suffix.py` own stride, grid and compilation-parameter
handling. The worker's `patch_triton.py` installs the four launch adapters at
startup; they remain bound for that worker process, including direct upstream
suffix calls. Per-call replacement, restoration and locking are unnecessary.
Older runtimes without the suffix module skip registration; selecting MRV2
suffix still requires its complete upstream protocol. Tests that directly
instantiate the speculator install the same worker bindings explicitly.
Remove these rebinds when upstream supports device kernel dispatch.
