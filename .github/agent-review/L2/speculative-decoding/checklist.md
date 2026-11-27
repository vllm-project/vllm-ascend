# L2 Speculative Decoding Domain Checklist

Rules for PRs changing draft proposal, target verification, rejection
sampling, or the state and parallelism used by those paths. Apply these
checks to shared runner, attention, cache, and model code when the changed
path affects speculative decoding. Entries use `L2-speculative-decoding#slug`.

## Dispatch and configuration

- `L2-speculative-decoding#upstream-hook-reuse`: For a new draft wrapper or
  override, trace the active hook in the vLLM revision pinned by
  `.github/vllm-main-verified.commit`, then compare the equivalent Ascend
  path. Flag duplicated proposal, sampling, or state handling when an
  existing hook can preserve the required inputs, outputs, and call order.
- `L2-speculative-decoding#dispatch-eligibility`: Trace the selector from
  the serving configuration to the chosen speculator or attention backend.
  If a guard or fallback changes, compare the base and head for the affected
  model, hardware, topology, and eager or graph mode. Flag unintended loss
  of an eligible path. Dispatch tests should assert the selected path in
  each affected mode, even when both paths produce valid tokens.
- `L2-speculative-decoding#config-default`: Follow omitted and explicit
  settings through parsing to the active caller. Check the documented
  default and preserve existing opt-in behavior unless the PR explicitly
  changes that contract. A default regression test should omit the setting.

## Draft loading and target interfaces

- `L2-speculative-decoding#draft-weight-loading`: Trace checkpoint names
  and aliases through the production draft loader to the parameter that
  consumes each weight. Check required shards, dtype, and quantization
  metadata. When draft embeddings or LM heads are shared with the target,
  verify that draft-only post-processing leaves the target weights intact.
- `L2-speculative-decoding#target-features`: When a draft consumes target
  hidden states, verify the capture points, layer order, normalization,
  and token rows against the draft's input contract. For pipeline parallelism,
  trace which stage produces each feature and how it reaches the draft.
  Verify feature row order across steps of chunked prefill when applicable.
- `L2-speculative-decoding#draft-vocabulary`: For changes to draft logits
  or candidate selection, trace draft and target token IDs through the LM
  head, any draft-to-target mapping, and verification. Cover identity and
  reduced-vocabulary paths where supported. Check that padded vocabulary
  entries from sharding are masked before candidate selection.

## Verification and state

- `L2-speculative-decoding#verification-state`: When proposal or verification
  changes, trace draft token counts, positions, and slot mappings through
  target verification to the next step. Check that accepted tokens and any
  sampled replacement or bonus token advance the right requests. The next
  step must use committed lengths and slots, and padding must leave request
  state intact.
- `L2-speculative-decoding#sampler-semantics`: When changing verification
  or rejection sampling, compare the applicable greedy and probabilistic
  branches with the reference contract. Check target logits processing,
  the first rejected position, replacement and bonus sampling, and any
  emitted logprobs. Check ordinary target-sampling behavior for requests
  with zero draft tokens, including in batches that also contain drafts.
- `L2-speculative-decoding#verification-budget`: When changing dynamic
  draft length or adaptive verification, follow per-request counts from
  scheduling through logit offsets, verification, and state updates.
  Check zero, partial, and maximum budgets and mixed budgets in a batch.
  Keep the scheduled, verified, and accepted counts consistent with their
  respective consumers.
- `L2-speculative-decoding#request-state-isolation`: When changing reused
  buffers or async updates, follow a request through batch reorder, a
  smaller next batch, or retirement and slot reuse. Check that fresh
  requests read their own tokens, lengths, and state, and that in-flight
  copies finish before the source or destination storage is reused.

## Graphs and parallelism

- `L2-speculative-decoding#draft-graph-replay`: When FULL graph capture or
  replay changes, trace the effective padded batch shape to the draft
  attention builder and kernel. Check that replay refreshes per-step
  metadata such as sequence lengths and slot mappings, while preserving
  the storage referenced by the captured graph.
- `L2-speculative-decoding#draft-parallelism`: When a layout, rank guard, or
  collective changes, trace which ranks own draft inputs and KV state and
  which consume the output. Verify that target parallelism preserves the
  existing draft placement unless the PR explicitly changes it. Check token
  order and per-rank shapes at any new gather or scatter. All participating
  ranks must enter collectives in the same order, including padded steps.

## Cache and disaggregated serving

- `L2-speculative-decoding#draft-cache-layout`: When changing draft cache
  allocation or layout, trace the cache group from its spec through block
  allocation, writes, and attention reads. Check that page size, block
  table, slot mapping, strides, and valid KV length agree across the
  producers and consumers that share the backing storage.
- `L2-speculative-decoding#prefix-cache-hit`: For changes to speculative
  prefix reuse, check which draft state is restored or rebuilt on a hit
  and which positions are appended during chunked prefill or decode. A
  regression test should prove that the warm request actually hits the
  cache and compare its output or committed state with the cold request.
- `L2-speculative-decoding#draft-transfer-lifecycle`: When changing
  disaggregated prefill or decode, check that decode becomes runnable after
  target KV and any required draft context are ready. Trace failure and
  cancellation through each participating rank. Keep destination buffers
  live until outstanding transfers finish and guard against stale
  completions updating a replacement request.

## Regression coverage

- `L2-speculative-decoding#regression-assertion`: Exercise the changed
  production path and assert the affected result, such as emitted tokens,
  accepted counts, or committed cache state. When adding or removing cases,
  check which distinct behavior each case covers. Keep separate mode cases
  when they exercise different branches or assert different results.
- `L2-speculative-decoding#ci-selection`: For a new or moved speculative
  E2E or nightly case, follow the workflow selection, test configuration,
  and skip conditions. Verify that the selected job runs the intended draft
  method, graph mode, and hardware or rank topology.
- `L2-speculative-decoding#acceptance-metrics`: When changing acceptance
  counters, verify which tokens enter the numerator and denominator after
  truncation, rejection, or padding. When a benchmark case changes, check
  that its baseline and candidate match outside the variable under test,
  including workload, model, NPU, topology, and speculative length.
  Keep acceptance, latency, and throughput claims tied to the metric measured.
