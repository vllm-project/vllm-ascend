# Suffix Speculative Decoding

Suffix decoding proposes draft tokens by matching the request history and
completed responses on the NPU, without an additional draft model.

## Enable suffix decoding

This experimental feature is based on upstream
[vLLM #60339](https://github.com/vllm-project/vllm/pull/60339).
Set `method` to `suffix` in the speculative decoding configuration:

```bash
vllm serve /data/Qwen3-8B \
  --async-scheduling \
  --speculative-config '{"method":"suffix","num_speculative_tokens":8,"suffix_decoding_corpus_tokens":1048576}'
```

Matching works as follows:

- Each request searches its own prompt and generated tokens for the longest
  matching suffix, up to `suffix_decoding_max_tree_depth` (default 24).
- Completed responses, with up to three preceding prompt tokens, are added to
  an NPU-resident ring. Two hash indexes use the last two and four tokens to
  locate candidates. Candidate matches are extended and their continuations
  are selected by prefix-consistent frequency voting. The longer history or
  corpus match wins; history wins ties.
- Finished requests are ingested before their state slots are released.
  Preempted requests are not ingested. Warmup resets the corpus and acceptance
  statistics before serving.

The default corpus capacity is 1,048,576 tokens. Setting
`suffix_decoding_corpus_tokens: 0` or `suffix_decoding_max_cached_requests: 0`
disables cross-request retrieval. For `suffix_decoding_max_cached_requests`,
positive values do not limit the number of cached requests. Corpus matching
needs a maximum match depth of at least two; depth one uses history only.
Default corpus and index storage consume approximately 36 MiB, in addition to
dense token history (`max_num_seqs * max_model_len * 4` bytes) and scratch buffers.

Suffix always returns `num_speculative_tokens` draft slots. Positions without a
continuation contain the last sampled token and are verified as ordinary
drafts. Draft-slot metrics include these filler slots, so they do not count
only the tokens retrieved by suffix matching.

Optional `enable_adaptive_verification: true` supplies per-position confidence
from online acceptance statistics grouped by draft source, match length and
position. It requires full decode graphs and an Ascend target attention backend
that supports variable query lengths. It is disabled by default.

The corpus is shared across requests, including different `cache_salt` values
and LoRA adapters. Disable the corpus where requests must not influence each
other's drafting. As upstream documents, a fixed sampling seed alone does not
make outputs reproducible with a shared corpus or adaptive verification. For
that use case, disable both. Under async scheduling, a finished request may
contribute tokens beyond the scheduler's final stop point; ingestion follows
upstream device-visible lengths.
