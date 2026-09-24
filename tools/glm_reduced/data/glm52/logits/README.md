# GLM5.x non-Flash intermediate-logits CI gate

The registered case is **GLM-5.2 W4A8, 11 main layers (0–10), one retained MTP
layer (78→11)**. MTP execution is disabled for this gate. Other non-Flash GLM5.x
models need their own pinned source, tokenizer inputs and full-model A1 reference;
this GLM-5.2 baseline must not be applied to them. Flash/hybrid models are rejected.

The metric is **argmax agreement with A1 ≥90% over all 20,693 prediction positions**,
not GSM8K answer accuracy. Every one of the fixed 200 questions is included,
including `gsm8k-test-0015`. At least 18,624 positions must agree. Missing samples,
wrong ordering/input hashes, invalid token IDs, missing positions, NaN/Inf logits,
and incomplete worker execution fail the gate. CI never recalibrates or retries.

`dataset.json` pins seed 20260922, the GSM8K test revision
`3101c7d5072418e28b9008a6636bde82a006892c`, question indices, prompt/gold-answer
token IDs and causal prediction positions. Raw question/answer text is not
stored in the fixture. Dataset provenance is
[openai/grade-school-math](https://github.com/openai/grade-school-math), MIT licensed.
`A1.json` contains argmax IDs extracted from the archived full-vocabulary float32
A1 arrays **after checking each original array's SHA-256 and finite values**.
It records original array and complete-manifest hashes. `case.json` pins both
fixture files, source metadata and the audited B1/B2 checkpoint metadata.

Observation: layer 10 output `(hidden, residual)` → select gold-answer prediction
rows → clone → final RMSNorm → shared LM head → full vocabulary → CPU float32
NumPy argmax. Sequence-parallel rows are gathered when needed. The collector runs
TP8+EP, eager, one unchunked request at a time, seed 1024, prefix cache off, and
the exact fixed engine settings in `run_logits_gate.py`. Generated answers are
unused. The projection observes cloned rows and does not change model execution.

The checked-in nightly YAML and matrix entry invoke the dedicated offline gate.
Provisioning uses the existing pinned `../source.json`, downloads required shards
into a persistent cache, and verifies a prefix-11 reduction including MTP. It does
not assume the lab server's personal path exists on the CI runner. `source_dir`
and `cache_dir` may be provided under `glm5x_logits_gate` for offline provisioning.
Reports (runtime versions, engine arguments, per-question IDs, counts, result,
and failure details) are uploaded even on failure.

To use the previously audited 11-layer artifact directly:

```bash
python -m tools.glm_reduced.run_logits_gate \
  --model /path/to/GLM-5.2-w4a8-L11-MTP \
  --report-dir benchmark_results/glm52-logits-once
```

Baseline runtime: vLLM `84030bbe3d74d99bad477a3d2e37a973ccd8865c`,
vllm-ascend `cb79183b7c04467c7358fe568ee88e85603ee965`, existing compiled libraries,
Ascend A3 on x86_64. No build is required by this gate. Candidate runtime versions
are recorded, not forced to equal the baseline, so the gate can detect regressions.
The registered nightly runner is A3 aarch64; its full workflow/provisioning requires
separate qualification. Lab evidence does not claim that workflow has run.

Archived B1/A1 agreement: 20,661/20,693 (99.845358%); B2/A1: 20,596/20,693
(99.531242%). Differences were confined to the first question, which remains in
the denominator. These values establish repeatability for the tested runtime;
they do not establish semantic reasoning quality or other models' accuracy.

Validation on 2026-09-22: the new collector ran once on the audited 11-layer
artifact using the above old runtime and existing native libraries, physical
devices 0–7. All 200 questions completed with exit 0 and 20,693/20,693 argmax
agreement (100%). CPU contract tests passed 20/20 in one run, including the
inclusive 90% boundary, fail-closed inputs and actual SingleNodeConfigLoader.
