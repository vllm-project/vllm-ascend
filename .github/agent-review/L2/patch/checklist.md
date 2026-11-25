# Patch Domain Checklist

Rules for PRs touching `vllm_ascend/patch/`. The patch catalog in
`vllm_ascend/patch/__init__.py` is the source of truth: platform patches
apply globally before workers start, worker patches apply per worker.

- `L2-patch#registered`: Every new patch file must be registered in
  `vllm_ascend/patch/__init__.py` with the correct classification —
  platform (global) vs worker (non-global). An unregistered or
  misclassified patch is a MUST-FIX.
- `L2-patch#catalog-entry`: The catalog entry must carry targets, why,
  how and future_plan (+ related_pr when one exists). A new or changed
  patch without a catalog update, or with an empty why/future_plan, is a
  SHOULD-FIX.
- `L2-patch#upstream-plan`: Every patch carries a removal path ("remove
  when upstream merges X"). A patch with no related_pr and no future_plan
  means an unbounded lifetime — raise as MUST-FIX for discussion.
- `L2-patch#minimal-target`: Patches target the smallest upstream symbol
  set necessary. Broad-target patches (e.g. rewriting
  `torch.distributed.all_reduce` globally) need an explicit justification
  in the catalog why-field; flag patches that grew targets without one.
- `L2-patch#deprecated-channel`: A patch enabled via a deprecated env var
  (e.g. balance scheduling's `VLLM_ASCEND_BALANCE_SCHEDULING`) must keep
  additional_config as the primary switch and mark the env var deprecated;
  new patches must not add env-var-only switches.
- `L2-patch#behavior-audit`: Patches silently change upstream behavior for
  ALL users of the plugin, not only the opt-in feature. A patch that alters
  semantics without a feature switch (config/env) needs explicit
  justification in the PR.
