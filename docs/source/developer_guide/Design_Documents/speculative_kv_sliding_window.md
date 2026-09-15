# Speculative Decoding KV Sliding Window

Speculative decoding draft models are typically trained on short contexts
(e.g. 4-8k tokens). At inference time, when the running context grows far
beyond the training length, the draft's attention over the full KV cache goes
out-of-distribution and acceptance collapses: on GLM-5.2 with an early DSpark
draft, the mean acceptance length dropped from ~5 to ~1 at 32k context, making
speculation a net loss.

The KV sliding window caps the **draft model's attention** to the most recent
`draft_window_size` tokens. The target model is untouched by this setting, so
generation quality is preserved; only the draft reads a shorter,
in-distribution window. On acceptance-collapsed drafts this
restores acceptance (the same GLM-5.2 draft recovered from ~1.0 to ~4.7-5.5
at 32k, up to +381% end-to-end throughput in the original evaluation).

## Configuration

The window is enabled through `additional_config` and is available for the
EAGLE3 / DFlash / DSpark draft paths on the default (MRV1) model runner:

```json
{"draft_window_size": 2048}
```

It is disabled by default (key absent). MTP force-disables it (MTP reuses the
target's own layers, so windowing the draft would window the target); for a
DSpark draft that was natively trained together with a DeepSeek-V4 target, a
warning is logged because the draft is already long-context stable and
windowing degrades acceptance.

For Qwen3-based DSpark drafts, this setting also changes the draft attention
layers' KV cache specs to sliding-window specs before cache groups are built.
The scheduler can therefore replace logical blocks outside the window with
the null block and return their physical blocks to the shared pool. This
physical eviction is limited to the DSpark draft KV groups; target-model KV
groups keep their original cache specs. Other draft families currently use
the setting only to limit the KV range read by attention unless their native
model already exposes a sliding-window cache spec.

## Window math

Let `W` be the window size, `b` the kernel block size, `K = future_offset`
the number of draft query positions lying beyond the `seq_lens` value the
adapter sees. The adapter computes, entirely on device:

```text
start        = floor(clamp(seq_lens + K - W, min=0) / b) * b   # block-aligned window start
windowed_len = seq_lens - start                                # <= W + b - 1
start_block  = start // b
needed       = ceil(windowed_len / b)  (capped at max_window_blocks = W//b + 1)
```

`future_offset` differs by draft family because their `seq_lens` semantics
differ:

| Method | future_offset | Why |
| --- | --- | --- |
| EAGLE3 | `num_speculative_tokens` | `seq_lens` is context-only; the K draft positions lie beyond it, so the window end must cover `seq_lens + K`. |
| DFlash / DSpark | `0` | the input-preparation kernel already bakes the query stretch (bonus + mask tokens) into `seq_lens`, so `final = seq_lens`. |

The block table is cropped by gathering columns
`start_block[i] + [0, max_window_blocks)` from the full table into a
zero-padded clone; columns past `needed` (and past the full-table width) are
zeroed. Zeroed columns are never read because every FIA KV-length input is
capped at the same `windowed_len`.

Two invariants hold throughout:

- **slot_mapping never uses the cropped attention table.** Draft KV writes go
  through the full *logical* block table at absolute positions. For Qwen3
  DSpark physical eviction, old logical entries can be null blocks, while the
  current tail always maps to live physical blocks. The configured window is
  fixed for the lifetime of the service and cannot be enlarged mid-request to
  recover already-evicted KV.
- **Only what FIA reads is capped.** The FIA kernel's KV-length inputs are the
  NPU `seq_lens` tensor plus CPU mirrors (`seq_lens_cpu`, `_seq_lens_cpu`,
  `seq_lens_cpu_upper_bound`); all of them are windowed with the same
  arithmetic so no path can read past the window.

## MRV1 implementation

`SlidingWindowAdapter` (`vllm_ascend/spec_decode/utils.py`) owns the math
above and a persistent `[max_num_reqs, max_window_blocks]` clone buffer. It is
constructed in `AscendSpecDecodeBaseProposer.__init__`
(`vllm_ascend/spec_decode/llm_base_proposer.py`) when `draft_window_size` is
set, then applied to the draft's `common_attn_metadata` **before** the
per-layer metadata is built.

There are two apply points because of how the drafts construct metadata:

- EAGLE3 / DFlash: applied once in `_propose` on the shared
  `common_attn_metadata` (`llm_base_proposer.py`).
- DSpark: that path is bypassed - the dspark branch of
  `build_draft_attn_metadata` overwrites
  `common_attn_metadata.block_table_tensor` with a per-group full table before
  building. The adapter is therefore applied **after** the per-group overwrite
  and before `builder.build_for_drafting`, so FIA reads the windowed clone of
  the per-group table instead of the full one.

For Qwen3 DSpark, `AscendQwen3DSparkForCausalLM` additionally translates
`draft_window_size` into the DFlash backbone's `use_swa` /
`swa_window_size` configuration before its attention layers are constructed.
Those layers consequently publish `SlidingWindowSpec` rather than
`FullAttentionSpec`, which makes the engine-core `SlidingWindowManager`
physically recycle blocks outside the committed-token window. Graph replay
remains address-stable because the KV pool tensor and the adapter's cropped
block-table buffer are persistent; only the block IDs copied into that buffer
change between steps.

## MRV2 status

Model Runner V2 is not yet covered by this feature. The MRV2 speculator
system bakes `seq_lens` / block tables into the final per-layer metadata
inside `build_attn_metadata` - there is no shared `common_attn_metadata` to
mutate after the build, so porting the window requires applying it to the
**builder inputs** (windowed `input_buffers.seq_lens`, gathered per-group
block-table clones, capped `seq_lens_cpu_upper_bound`) around the
`_build_draft_attn_metadata` call in
`vllm_ascend/worker/v2/spec_decode/dspark/speculator.py`. The window math
and the two invariants above carry over unchanged.

## Expected behavior

- Drafts trained on short contexts that collapse on long inputs (acceptance
  dropping with context length): the window restores acceptance up to the
  draft's short-context level. The gain grows with input length.
- Drafts already stable at long context (recently trained drafts, or drafts
  with their own SWA structure): the window provides no benefit and can hurt
  if set **below** the draft's training window - the draft expects to see
  more context than the window allows. Rule of thumb: keep
  `draft_window_size` at or above the draft's training/native window, or
  leave it off when acceptance is already flat across context lengths.
- End-to-end effect combines the acceptance change with the draft's reduced
  KV-read cost; with no acceptance gain the two roughly cancel.

## Related Files

- Shared window adapter: `vllm_ascend/spec_decode/utils.py`
- MRV1 proposer (apply points, config validation): `vllm_ascend/spec_decode/llm_base_proposer.py`
- Qwen3 DSpark physical KV eviction setup: `vllm_ascend/models/qwen3_dspark.py`
