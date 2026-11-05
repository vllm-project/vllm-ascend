# GDN input projection overlap: local validation handoff

This is a local implementation for review, not a claim of hardware-qualified
support. Full GitHub PR/issue pagination remains unavailable, so absence of a
community PR has NOT been established. Recheck before submission.

## Usage and scope

Enable with `--additional-config '{"gdn_projection_overlap_max_tokens": 64}'`.
The default is 0 (disabled); 64 is a trial setting, not a measured 910C threshold.
Only BF16 plain unquantized QKVZ and BA projections, PCP=1, no LoRA/DBO,
no output gather or custom communication op, and ACLGraph capture outside
Dynamo tracing are eligible. Other paths remain serial. Quantized projections
and the split QKV/Z LoRA path are excluded. Both known upstream and Ascend
unquantized linear methods are accepted. Weight tensors are not repacked. Eligible layers share one model-owned stream
per NPU device, with separate events for each layer.

For whole-model evaluation, use MRV2 with an eager forward captured into a
full decode ACLGraph, for example
`--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'`.
`--enforce-eager` disables graph capture and therefore does not enable overlap.
Dynamo-compiled forward regions stay serial; confirm the chosen serving path
with a trace rather than assuming that the configuration flag proves overlap.

## Checks run locally

The standalone host test below passes on macOS without torch. It validates
host planning/control flow or event ownership; it does not validate Triton
lowering, NPU numerical accuracy or serving performance.

```bash
.venv/bin/python tests/ut/ops/test_gdn_projection_overlap.py
```

Focused Ruff lint/format checks, Python syntax checks and `git diff --check`
passed. Full `bash format.sh ci` exited because `pre-commit` is not installed.
The local environment also lacks torch and the Ascend runtime.

## Required 910C checks (not run locally)

Use the provided upstream vLLM checkout, a compatible CANN/torch_npu and
Triton-Ascend build, and install the affected Ascend checkout. Record exact
commits, software versions, model revision, TP, graph mode and all commands.

```bash
.venv/bin/python -m pytest -v tests/e2e/pull_request/one_card/test_gdn_projection_overlap.py
.venv/bin/python -m pytest -v tests/ut/ops/test_gdn.py
bash format.sh ci
```

Run relevant existing suites as well; for FLy include
`tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rejection_sample_v2.py`
and `tests/ut/worker/v2/test_entropy_verify.py`. For checkpoints include
`tests/ut/ops/test_gdn_attn_builder.py`, mixed prefill/decode and prefix-cache
end-to-end cases. The new focused NPU tests alone do not qualify whole-model
accuracy, PD, multi-rank execution or all graph sizes.

Compare clean baseline and changed trees on the SAME 910C node and workload:
Qwen3.5 BF16; concurrency 1/4/16/32; input 1K/8K/32K; output 512/2048;
TP1 then production TP; MTP off/on; eager and ACLGraph separately. Use distinct
prompts when measuring checkpoint creation, and a separate shared-prefix run
for cache reuse. Warm up, alternate A/B arms and repeat at least three times.
Record TTFT and TPOT p50/p95, output throughput, accepted draft length,
peak memory, CPU/NPU synchronization and request errors. Checkpoint tests must
include boundary+1, partial hits, CoW, cancellation/preemption and mixed batches.
Projection overlap needs a trace proving actual overlap and safe stream joins.

Run full model quality evaluation (e.g. GSM8K and a representative production
set). For overlap require unchanged output; for checkpoints compare cold/warm
continuation and recurrent states with dtype-appropriate tolerances; for FLy
report quality change relative to both target-only and standard speculation,
rather than claiming lossless equivalence. Disable the option if benefits or
accuracy are not demonstrated.

## Submission

No commit, push or PR was made. The release fragment uses PR 0 as a local
placeholder: rename it and replace the number when opening the PR. Include
failed/unrun checks, hardware results and AI assistance in the PR description.
