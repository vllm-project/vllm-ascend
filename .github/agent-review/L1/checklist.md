# L1 Global Checklist

Cross-domain rules every PR is checked against. Owned by all maintainers
(multi-approval to change). Distilled from `AGENTS.md`; keep entries stable
IDs (`L1#slug`) so agent findings can cite them.

## Configuration

- `L1#no-new-env-vars`: Do not add new environment variables. User-facing
  configuration belongs in vLLM's additional config mechanism
  (`vllm_ascend/ascend_config.py`). An env var is a last resort (bootstrap
  values only), must follow `VLLM_ASCEND_*` naming, be defined in
  `vllm_ascend/envs.py`, and be documented.
- `L1#no-raw-dict-config`: New config keys need typed parsing in
  `ascend_config.py` following the sub-config pattern (`EplbConfig`,
  `RlConfig`), a table entry in the additional config docs, and a name that
  generalizes beyond the motivating workload.

## Code style

- `L1#top-level-imports`: Imports at the top of the file, except circular
  imports, lazy worker/isolation-process loading, and `TYPE_CHECKING` blocks.
- `L1#no-new-globals`: No new module-level mutable globals; pass dependencies
  explicitly. Allowed: `ALL_UPPER_CASE` constants and immutable config.
- `L1#no-magic-numbers`: Magic numbers become named constants.

## NPU hot paths

- `L1#no-item-sync`: `tensor.item()` on a device tensor forces an NPU→CPU
  sync and blocks AsyncScheduler in hot paths. Requires profiling evidence
  and a documented justification; prefer device-side ops or batching.
- `L1#no-cpu-transfer-hot-path`: Avoid CPU↔NPU transfers in hot paths; prefer
  in-place ops where safe.

## Architecture

- `L1#patch-justified`: New `vllm_ascend/patch/` entries must name the
  upstream component, stay minimal, and carry an upstream-contribution plan.
- `L1#model-runner-justified`: New `model_runner` behavior must explain why
  existing extension points cannot cover it, and its NPU performance and
  compatibility impact.
- `L1#model-registry-pattern`: Adapted models register in
  `vllm_ascend/models/__init__.py` following existing inheritance patterns;
  follow the model-adapter walkthrough for new models.

## PR hygiene

- `L1#release-note`: User-facing changes ship a release note fragment in
  `release_notes/fragments/<PR>.yaml`. Internal refactors, tests, CI and
  doc typos do not. Performance entries need a scenario and a measured
  effect; config changes need old → new + migration.
- `L1#commit-format`: Conventional Commits (`feat|fix|perf|refactor|test|docs|chore:`)
  with a sign-off; PR titles use bracketed tags (`[BugFix]`, `[Feature]`, `[Doc]`).
- `L1#tests-run`: Python changes run `bash format.sh ci`; unit tests live in
  `tests/ut` and run on CPU.
