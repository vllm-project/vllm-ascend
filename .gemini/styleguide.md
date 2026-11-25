# Review Guidance for vllm-ascend

vLLM Ascend is a hardware plugin integrating Ascend NPUs into vLLM. Formatting,
PR title tags, and PR description templates are enforced by CI (`format.sh ci`,
check gates) and by the coding agents developers already use — do NOT comment
on them.

Focus your review on issues that automated tooling cannot catch:

## NPU-specific correctness and performance
- `tensor.item()`, `.cpu()`, `.tolist()`, or data-dependent control flow on
  device tensors in hot paths: each forces a host-device synchronization and
  can stall AsyncScheduler. Suggest device-side alternatives
  (`torch.argmax`, `torch.sum`, keeping values on device, batching).
- Unnecessary CPU↔NPU transfers or avoidable copies in hot paths.
- Long-running-process memory concerns: fragmentation, tensors that should be
  freed, per-step allocations that could be reused.

## Architecture rules (see AGENTS.md at repo root)
- New environment variables need strong justification; user-facing config
  belongs in `--additional-config` via a typed sub-config in
  `vllm_ascend/ascend_config.py` (follow `EplbConfig`/`RlConfig` patterns).
  Flag scattered/hardcoded env var access instead of `vllm_ascend.envs`.
- New patches under `vllm_ascend/patch/`: check the patch is minimal, targets
  the right layer (`platform/` vs `worker/`), and is a candidate for upstream
  contribution to vLLM.
- Model runner changes (`vllm_ascend/worker/v2/model_runner.py`): check
  existing upstream extension points (inheritance, composition) were considered
  first, and flag device-side hot-path costs.
- No new module-level mutable globals; constants (`ALL_UPPER_CASE`) are fine.
- All imports at top of file, except circular imports, lazy worker-process
  loading, and `if TYPE_CHECKING:` blocks.

## Triton operators
- JIT specialization / recompilation risks from changing arguments.
- Warmup: required variants registered in the component's active registry
  window; runtime calls reuse the same wrapper as warmup.
- Device-applicability across Ascend 910B / 910C / 310P.

## Docs consistency
- Behavior changes (config keys, env vars, defaults) should be reflected in the
  relevant docs (`docs/source/user_guide/configuration/`, additional_config
  table). Flag docs that contradict the code in the same PR.

## Release notes
- Every user-facing PR must add `release_notes/fragments/<PR number>.yaml`
  following `release_notes/TEMPLATE.yaml`.
- Flag fragments that: describe internal refactors users cannot observe,
  state performance numbers without a scenario (model/workload/hardware) or an
  invented magnitude, or change config/env behavior without old→new plus the
  migration step.
- Bug-fix fragments should describe the user-visible symptom, not the internal
  root cause.

## Testing
- Changes to kernels, model runners, or config parsing should come with or
  update CPU-runnable unit tests under `tests/ut`.
- Do not ask authors for NPU/E2E/performance runs when they lack hardware —
  that evidence comes from CI; just say so.
