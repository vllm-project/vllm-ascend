# Speculative Decoding KV Sliding Window

Speculative decoding draft models are typically trained on short contexts
(e.g. 4-8k tokens). At inference time, when the running context grows far
beyond the training length, the draft's attention over the full KV cache goes
out-of-distribution and acceptance collapses: on GLM-5.2 with an early DSpark
draft, the mean acceptance length dropped from ~5 to ~1 at 32k context, making
speculation a net loss.

The KV sliding window caps the **draft model's attention** to the most recent
`draft_window_size` tokens. The target model is untouched - it always attends
to the full context, so generation quality is preserved; only the draft reads
a shorter, in-distribution window. On acceptance-collapsed drafts this
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

- **slot_mapping is never windowed.** Draft KV writes still go through the
  full block table at absolute positions, so the cache stays coherent for the
  target model and for window sizes changed between steps.
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

## MRV2 implementation

MRV2 applies the window to fresh, per-group common metadata through the
existing `build_attn_metadata` callback, immediately before the actual backend
builder. It never temporarily replaces `input_buffers.seq_lens` or the
canonical Full tables. The factory resolves CONFIG_OFF and forced-OFF before
allocating any window runtime. The dedicated built-in DeepSeek V4 DSpark
category is identified from normalized model configuration, including Ascend's
V4.1 architecture normalization. This disables only the window; the original
speculative method is still constructed.

### Canonical reads and method phases

`DirectPrefixSource` borrows `AscendBlockTables.block_tables[gid].gpu` and
snapshots the current request-to-state-row mapping. Each build gathers a new
read table from that owner. It never clips a previous window result. The
canonical table is kernel-column addressed: physical allocator blocks may
have been expanded into several columns. `kernel_block_sizes[gid]`, not a
similarly named scheduler or physical block field, supplies the column span.

The first active group's span defines the historical alignment of `S`.
Other groups must represent that same logical start exactly. A larger LCM
alignment would change the feature's semantics and is not substituted. Each
group receives its own read table and backend-derived state. Initialized
column counts alone do not prove residency for recycled or offloaded caches.

DFlash and DSpark use their actual prepared query width and exact end; DSpark
may use `K` or `K+1` queries. Eagle3 freezes its proposal horizon and block
start at the first pass, then refreshes exact lengths at each AR step. The
startup window bound covers the configured rejection range. A continuing
prefill row keeps its full read view; an Eagle3 first prefill pass also keeps
full reads so long prompt queries are preserved. Per-row phase decisions use
computed prompt tokens, prompt length, and scheduled tokens excluding draft
placeholders.

`DraftReadView` changes only the table, device exact length, both exact CPU
mirrors, scheduling upper bound, and maximum length. Positions, query layout,
Full KV write slots, and target metadata keep their original ownership.
Backend builders run after this projection. The Eagle3 metadata update that
would otherwise restore Full CPU lengths is bypassed only while an enabled
window proposal owns the metadata.

### Actual consumers and parallel execution

Backend selection uses the actual backend, builder, spec, and layer names,
not the target model's architecture. For example, a DSA target can use a dense
MLA or GQA draft. Ordinary TP/EP and sequence-parallel hidden-state collectives
retain their original query/write coordinates. DSA-CP needs a distinct
provider for its local query ranges and global KV read coordinates.

Providers distinguish direct GQA, ordinary MLA, standard SFA with its indexer,
and uncompressed DSA/DSA-CP. MLA and SFA preserve full absolute rotary inputs
while taking ownership of scratch that would otherwise be overwritten.
DSA native-window reads must remain a subset of the baseline native read set;
the provider does not infer residency from stale prefix table entries.
Compressed, packed, offloaded, or unsupported context-parallel layouts require
specific providers and are rejected when the selected consumer lacks one.
An unsupported conditional layout is not evidence that a required model has
been validated through another backend.

### Asynchronous ownership

Every proposal owns independently allocated read views and all derived
metadata through a completion token. All Eagle3 steps belong to that proposal.
Mutable DSA builders are owned by lane and step. The owner enters its native
region before preparation can enqueue a copy or kernel. Successful completion
recording retains the owner until the event completes; capacity pressure may
wait for the oldest submitted proposal. Normal acquisition of a free lane
performs no global synchronization.

If host execution, event creation/recording, bookkeeping, or completion
query/wait fails after possible native submission, the pool retains the
backing and quarantines the execution context. Host return, process progress,
or a failed event cannot authorize reuse. A new pool is not a safe recovery
mechanism unless the executor separately establishes device completion. The
final event must cover all readers, including any secondary communication
stream; retaining a Python reference alone does not establish that boundary.

The current implementation obtains an exact CPU length snapshot once per
metadata build and shares it across groups. This adds a device-to-host
synchronization where the baseline could use an upper bound. Its effect on
actual scheduler overlap and performance requires device measurement. CPU
ownership tests are not proof of NPU completion or a throughput improvement.

### Validation boundaries

Host tests exercise the paired source interfaces, current mapping, exact vs
upper arithmetic, phase transitions, final writers, and partial-submit owner
failures. Device validation must compare fixed queries and logical KV against an
independent reference for the same window and preserve Full/query operands.
The included device tests require an NPU run; Host results do not establish
that this validation has completed.
Separate real-model tests must establish actual backend/gid identity, prefix
hits, multiple chunks, mixed arrival, row reuse, parallel layouts, and the
last-reader completion boundary. OFF and ON may produce different draft
proposals; naturally decoded token positions are not a same-state numerical
oracle. Performance is measured only after correctness, using the same source
candidate for both OFF and ON.

## Performance considerations

A shorter Draft context can improve acceptance for drafts trained on short
contexts, but the result depends on the checkpoint and workload. A window
below a draft's training or native window can reduce acceptance. Both effects
must be measured for the selected model rather than inferred from input length.

End-to-end performance combines acceptance, Draft KV-read cost, metadata
preparation, and scheduling overlap. A smaller read view alone does not prove
a speedup. Compare OFF and ON on the same candidate and runtime, retain short
context regressions, and separately compare candidate OFF to the clean base.

## Related Files

- Shared window adapter: `vllm_ascend/spec_decode/utils.py`
- MRV1 proposer (apply points, config validation): `vllm_ascend/spec_decode/llm_base_proposer.py`

- MRV2 policy and integration: `vllm_ascend/worker/v2/spec_decode/`
- MRV2 source/math/ownership: `draft_window_read.py`, `draft_window_plan.py`, `draft_window_lifetime.py`
- MRV2 builder seam: `vllm_ascend/worker/v2/attn_utils.py`
