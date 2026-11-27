### What this PR does / why we need it?

Fix the identified HY4 Target Attention, MoE, and IndexPool precision defects on **Hy4-flash-opti**, based on `363c1616b1dbbdcd12399301dfaa50b48753adc9`.

**For review only. Do not merge or enable auto-merge.**

- Preserve FP32 C8 KV/QK and use fused sink/softmax/PV to retain the required numerical boundaries. This replaces the previous CANN MLA FIA call; the final Attention tile is 64.
- Restore intermediate BF16 expert-sum rounding before CANN AddRMSNorm.
- Restore IndexPool key RoPE and valid dense history candidates for short contexts while keeping padding invalid. Retain the other Decode/Draft fusions, DFLY5, and Draft graph mode.

### Does this PR introduce _any_ user-facing change?

Yes. It fixes output precision defects in these paths without adding configuration switches. A release note is included. Remaining Engram rstd and iHC head numerical differences are outside this change.

### How was this patch tested?

#### Normal service startup and AIME2026

Start the service normally using the repository's `examples/hy4flash/pd_118/start_prefill.sh`, `start_decode.sh`, and `bin/vllm`, adapting only deployment paths and context capacity. P and all four D processes loaded this candidate in container `hy4-pr22839-363c1616b-1127`, from `/opt/hy4/validation/fia-moe-precision-20261127/source`.

- Evaluator: **EvalScope 1.15.0**, official **AIME2026 (`aime26`)** adapter and official model prompt/chat template.
- Context: **128K (`max_model_len=131072`)**.
- Concurrency: **16 (`eval_batch_size=16`)**.
- Sampling parameters: **`temperature=0.9`, `top_p=1`, `top_k=-1`**.
- No additional output length, generation seed, or repeats were specified.
- P: physical devices 0–3, TP4 eager. D: physical devices 4–7, TP1/DP4/EP4, with DFLY5, `hy4flash_dfly_dp_graph=true`, and normal graph sizes retained.

The run loaded all 30 questions through the official `local_path` interface, using dataset revision `evalscope/aime26@12ae93b463353acd4d272dbbc7dbecbcfb902642`. Cached integer answers were converted to strings for EvalScope compatibility; question text and answer values were unchanged.

**Latest partial results: 29/30 questions completed, all 29 officially scored correct.** The remaining question (zero-based index 14) was still generating and revisiting reasoning without converging. This submission does not wait for it and does not claim a final score for all 30 questions. It is not counted as correct, incorrect, or a service hang. After HTTP timeouts, evaluation resumed through the official cache with transport-only timeout/stream changes; prompts, sampling, scoring, and model service code remained unchanged.

`examples/hy4flash/experiments/20261127-precision-fix/aime2026-partial-results.json` records the 29 official per-question accuracy values and source review file SHA256. Heuristic checks found no mechanical repetition flags in completed responses; this does not replace manual analysis.

#### Regression and component validation

- All 11 new IndexPool regressions failed before the fix and passed afterward. Related tests recorded **73 passed, 72 skipped**; skipped cases are not NPU checks.
- Historical NPU component validation passed 34 cases on an earlier component revision. This is not a rerun of every final candidate file. The final candidate's model-level evidence is the normal-service evaluation above.
- Patch application, source consistency, Python syntax, Ruff, and before/after regression logs are in the [delivery archive](https://github.com/peterDengcx/vllm-ascend-118/tree/36171c25152120fff3c6cdcfc484610434e145d6/examples/hy4flash/experiments/20261125-verify-fusions/precision-fix-20261127).
- Local full-repository `bash format.sh ci` was blocked by missing pre-commit. Full repository checks and performance acceptance are not claimed; this PR makes no throughput improvement claim.

### Checklist

- [x] Cover IndexPool short rows, mixed lengths, padding, and candidate sets for batches 2/4/8.
- [x] Add regressions that reproduce failures before the fix.
- [x] Evaluate the real model through normal service startup and record AIME2026 partial results and configuration.
- [ ] Final 30-question report (this submission does not wait for the last question).
- [ ] Performance acceptance for the current candidate.
- [ ] Merge approval (hold; do not merge or enable auto-merge).
