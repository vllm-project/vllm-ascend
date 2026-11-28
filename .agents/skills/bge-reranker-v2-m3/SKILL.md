---
name: bge-reranker-v2-m3-ascend-adapter
description: Validate BGE-Reranker-V2-M3 pooling inference on Ascend and prepare its tests and documentation.
---

# BGE-Reranker-V2-M3 Ascend Adapter

## Scope

This skill covers the small, encoder-only `BAAI/bge-reranker-v2-m3` model in
vLLM Ascend. The deliverables are the model configuration, pooling score E2E
tests, user tutorial, and validation evidence. The current vLLM Ascend branch
already contains the generic BERT/RoBERTa pooling path, so the task is focused
on reproducible coverage and documentation rather than a new model adapter.

## Repository Workflow

1. Inspect `tests/e2e/pull_request/one_card/pooling/test_scoring.py` and the
   model configuration directory before adding new coverage.
2. Use Transformers `AutoModelForSequenceClassification` in FP16 on the same
   NPU as `VllmRunner(..., runner="pooling")`, following the precision of the
   repository's cross-encoder tests. Use one model snapshot for both engines
   and apply sigmoid to the reference logits in FP32.
3. Cover single-pair, one-to-many, paired batches, different text lengths,
   empty input, mismatched input, score shape, and ranking. Keep the existing
   generic pooling tests and drive task-specific cases from the model YAML.
   Release the vLLM engine before loading the Transformers model on the NPU.
4. Record the exact hardware, CANN, PyTorch, torch-npu, vLLM, and vLLM Ascend
   versions for every NPU run.
5. Keep CPU/static checks separate from NPU evidence. For a failed model
   case, record the exact sample, scores, ranking, and leftover cases so the
   next run can distinguish a score regression from a coverage gap.

## Effective Prompt Pattern

Use a bounded exploration request before editing:

> Find the existing pooling score tests, model config conventions, tutorial
> structure, and repository navigation entry. Compare them with the BGE task
> acceptance criteria and list only the missing deliverables.

Then request a focused implementation:

> Add only the BGE config, dedicated score tests, tutorial, and SKILL.md. Keep
> the public API unchanged, compare against Transformers, and state which
> checks require real NPU hardware.

## Validation Commands

Static checks from a CPU-only checkout:

```bash
python -m compileall tests/e2e/models/test_bge_reranker_correctness.py
python -c "import yaml; yaml.safe_load(open('tests/e2e/models/configs/bge-reranker-v2-m3.yaml'))"
```

Run the model test on Ascend and save the output with the environment versions:

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn python -m pytest -sv \
  tests/e2e/models/test_bge_reranker_correctness.py \
  --config tests/e2e/models/configs/bge-reranker-v2-m3.yaml \
  --tp-size 1 \
  --report-dir ./benchmarks/bge-reranker-v2-m3/accuracy
```

The CI entry is `tests/e2e/pull_request/one_card/pooling/test_bge_reranker_ci.py`.
It uses the same YAML and checks without requiring `--config`. The standard
one-card route targets A2; record a separate run for A3 evidence.

## Known Runtime Notes

The task's historical Ascend validation used `runner="pooling"`,
`trust_remote_code=True`, and `enforce_eager=True`. Treat previous PR reports
as historical material until reproduced on the current image. The YAML metric
values of `1.0` are acceptance targets, not previously measured results. The
dedicated test writes scores, rankings, unpadded token hashes, timings, and
package versions under `--report-dir`. Keep results from later cases when
one score assertion fails. Use per-case timings to debug the test run.

On a hosted A3, record the vLLM Ascend and vLLM commits before running pytest.
In the 0.30.0 release image, a candidate score differed from Transformers CPU
FP32 by 1.80%. The token hashes agreed. With both engines using FP16 on A3,
that candidate differed by 0.0413% and the largest difference across all
seven pairs was 0.189%. The full suite passed within the existing 1% tolerance.
Keep both reports and use the CPU FP32 results when investigating changes
in numeric precision. Reuse cached weights and one engine across all cases
to reduce restart time.

`LLM.score([], [])`, an empty query list, an empty document list, and an
invalid N-to-N length mismatch must raise `VLLMValidationError` in the vLLM
API. The single-card `VllmRunner` forwards directly to `LLM.score()`. Only
`DPVllmRunner` normalizes an empty batch to `[]`; do not use that helper to
infer the single-card API contract. An empty string is different from an
empty input list.
