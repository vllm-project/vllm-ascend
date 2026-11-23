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

## Verification and state

- `L2-speculative-decoding#verification-state`: When proposal or verification
  changes, trace draft token counts, positions, and slot mappings through
  target verification to the next step. Check that accepted tokens and any
  sampled replacement or bonus token advance the right requests. The next
  step must use committed lengths and slots, and padding must leave request
  state intact.
- `L2-speculative-decoding#draft-graph-replay`: When FULL graph capture or
  replay changes, trace the effective padded batch shape to the draft
  attention builder and kernel. Check that replay refreshes per-step
  metadata such as sequence lengths and slot mappings, while preserving
  the storage referenced by the captured graph.
- `L2-speculative-decoding#draft-parallelism`: When a layout, rank guard, or
  collective changes, trace which ranks own draft inputs and KV state and
  which consume the output. Verify that target parallelism preserves the
  existing draft placement unless the PR explicitly changes it. Check token
  order and per-rank shapes at any new gather or scatter.

## Regression coverage

- `L2-speculative-decoding#regression-assertion`: Exercise the changed
  production path and assert the affected result, such as emitted tokens,
  accepted counts, or committed cache state. When adding or removing cases,
  check which distinct behavior each case covers. Keep separate mode cases
  when they exercise different branches or assert different results.
