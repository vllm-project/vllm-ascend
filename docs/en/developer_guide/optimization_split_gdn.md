# Standalone GDN optimization branch

Branch: `codex/opt-gdn`. Extracted on 2026-09-24.

Common historical base: `2ee50f3847e4873e61d280911b2a24d85f5278d6`,
before the local 310P DFlash series. This is not a rebase to current upstream.
The integration was previously exercised with vLLM
`ee0da84ab9e04ac7610e28580af62c365e898389` (0.24 API).
No dependency on the sibling QBM branch is introduced.

## Included source commits

| Source | Scope |
| --- | --- |
| `655f8a097` | Compute-WY native operator, bindings, Python dispatch and tests |
| `f48bd99fd` | FwdO causal-mask correctness fix and regression coverage |
| `29bbb27c6` | Optimized compute-WY kernel and tiling |
| `d44c253f4` | Reuse existing host sequence boundaries during prefill |
| `51a71deaa` | Optional BS1 host-slot recurrent-state commit |
| `cb71fbdd5` | Optional single-sequence packing/copy elimination |
| `48450f331` | Optional shared input quantization across GDN projections |

Original authors and source hashes are retained in signed-off cherry-picks.
Conflicts in bindings, environment registration, metadata builder and patch
imports were resolved by retaining only the GDN changes. In particular, no
DFlash graph/proposer registrations or MLP fusion imports were carried over.
The host-boundary report's historical filename mentions DFlash because that
was the measurement workload, not because the selector requires its proposer.

## Runtime options

- `VLLM_ASCEND_GDN_PREFILL_HOST_COMMIT=1`: enable the guarded host-slot path.
- `VLLM_ASCEND_GDN_SINGLE_SEQUENCE_PACKING=1`: enable guarded packing reuse.
- `VLLM_ASCEND_GDN_SHARED_INPUT_QUANT=1`: experimental shared quantization;
  a reproducible end-to-end speedup was not established. Keep off by default.

All three options default to zero. Existing unsupported-shape and metadata
fallbacks remain. Shared quantization retains its explicit vLLM 0.24 check.
This extraction does not broaden its supported model or numerical contract.

## Verification and limitations

- CPU tests: shared quantization 4/4, packing 4/4, host-slot commit 7/7.
- Native compute-WY and FwdO source trees, chunk wrapper and GDN execution
  module match the pre-split integration at `c06388df9`.
- Python source syntax and diff whitespace checks pass.
- The repository `format.sh ci` check was attempted but could not run because
  `pre-commit` is not installed. No full-lint pass is claimed.
- Full plugin tests requiring vLLM/torch_npu and pytest could not run in the
  local CPU-only environment. Initial broad discovery reported missing
  dependencies; the targeted dependency-light tests above were run separately.
- No new CANN build, NPU correctness run or latency measurement was performed
  for this extracted branch. Earlier integrated results are not standalone
  performance or build-validation claims.

Excluded: Tree-MTP, DFlash changes, QBM, lm-head pruning/routing, unrelated
normalization experiments, last-layer prefill selection and early delivery.
