# Qwen3.8-Flash-Next phase dispatch: implementation and validation

## Branch scope and status

This experimental branch extends `qwen38-gbsa-only-opt` at
`c57f949d2c836d670da594da91011f009a1f9425`, based on upstream PR #16981 at
`207a6753b4f34d3836acf186201529b1d49d7521`.

It preserves the optimized GBSA implementation and adds FIA with a selected-token
mask for Prefill. This is a branch snapshot, not a claim of completed accuracy
acceptance or upstream merge readiness. No existing remote branch is updated.
Add a release note fragment with the assigned PR number when preparing a PR.

## Request routing

| Request phase | Attention path |
| --- | --- |
| Pure Prefill | FIA with a selected-token mask when the chunk starts below 24K logical KV, including a chunk that crosses 24K; GBSA when the chunk starts at or above 24K |
| Pure Decode | Optimized GBSA, including multi-token Decode |
| Mixed Prefill and Decode | Route each Prefill chunk independently by its cached-prefix logical KV length; coalesce GBSA Prefill chunks and Decode rows |

The scheduler's CPU `is_prefilling` flags determine request phase. Within
Prefill, the per-request logical visible KV length selects the backend at the
24K (24576-token) chunk-start boundary. A Prefill chunk whose cached prefix is
shorter than 24K stays entirely on FIA even when the chunk crosses 24K. This length includes cached history and
the current Prefill chunk; it is not the padded physical cache capacity.
Prefill without history uses current K/V; chunked Prefill with history reads
paged KV. Request IDs and output row positions are preserved when splitting
mixed batches. Unknown phase metadata is rejected.

Decode uses the GBSA-only graph path. Capture does not freeze CPU KV lengths or
a Prefill decision into the graph. A Qwen4Exp-only Model Runner V2 guard prevents
one-token prompt tails from being relabeled as Decode and using a Decode graph.
Other model architectures retain the existing runner behavior.
The guard belongs in the runner because one-token Prefill reclassification occurs
before attention metadata construction; the attention backend alone cannot
recover the original phase after it has been relabeled.

The dense temporary mask is tiled along Q with a 64 MiB mask budget and at most
4096 query rows per tile. This budget describes mask dimensions, not total
temporary allocation. Prefill chunks starting at or above the 24K logical-KV threshold do not build
the FIA mask and instead use GBSA.

## Changed implementation

| File under `vllm_ascend/` | Purpose |
| --- | --- |
| `models/qwen4_exp/attention_dispatch.py` | CPU request plans and phase-specific attention dispatch |
| `models/qwen4_exp/fia.py` | Selected-token semantics, descriptor-backed paged KV views and tiled FIA calls |
| `ops/triton/qwen4_exp/fia.py` | Aligned mask construction and output writes, including empty selections |
| `models/qwen4_exp/qsa.py` | Metadata builder, phase plan and Decode capture integration |
| `worker/v2/model_runner.py` | Qwen4Exp-only one-token Prefill graph eligibility guard |

The GBSA adapter, fused GBSA preparation kernel and QSA pack/compression kernel
are unchanged from the GBSA-only parent. The change adds no environment variable,
custom operator binary, global patch, or other model-specific implementation.
The FIA helper retains batched-call support, but request dispatch never routes
Decode to FIA.

CPU regression coverage includes phase and 24K logical-KV boundaries, mixed request ordering,
one-token Prefill, multi-token Decode, padding, missing metadata, paged cache
strides, selected masks, residual groups and long-Prefill tiling.

## Observed validation on 2026-10-29

The selected CPU test set passed 545 tests. This is not a full-repository suite:

```bash
PYTHONUTF8=1 python -m pytest --noconftest -q \
  tests/ut/models/qwen4_exp/test_fia_dispatch.py \
  tests/ut/models/qwen4_exp/test_gbsa_preparation.py \
  tests/ut/models/qwen4_exp/test_gbsa_adapter.py \
  tests/ut/models/test_qwen4_exp_source_regression.py
```

Ruff 0.14.0 checks and formatting passed for the eight changed Python files;
Markdown lint and `git diff --check` also passed. The full `bash format.sh ci`
attempt ran in an independent review checkout, but was stopped during the
`actionlint` environment bootstrap after no progress; it did not complete the
full hook suite. Full repository formatting/CI is not claimed as passed.

Full-model runs used Qwen3.8-Flash-Next-w8a8, Ascend A3 physical devices 8-15
on host105, TP8/EP8, CANN 9.2.0, torch 2.10.0 and torch_npu 2.10.0.post7.
The pinned vLLM checkout was `544f182d9f0c9bfe8bd7645525951d5e3692a542`.
The configuration used chunked Prefill with 4096 scheduled tokens, maximum
sequence length 33024, maximum batch 32, seed 803, synchronous scheduling,
and disabled prefix caching. Decode graph settings were:

```json
{
  "cudagraph_capture_sizes": [1, 2, 4, 8, 12, 16, 20, 24, 28, 32],
  "cudagraph_mode": "FULL_DECODE_ONLY"
}
```

### Compatibility limitation

The available custom GBSA operator package supports 256 sparse slots, whereas
the model's default Indexer emits 512 candidates. Validation-only worker hooks
made both versions consume the same first 255 candidates plus residual capacity.
The Indexer still produced its default candidates. **This truncation is not in
production code. Default 512-candidate hardware validation remains outstanding.**

### Whole-model performance

These are offline `generate(max_tokens=1)` request times, including scheduling,
not individual operator times, independent chunk times, or server streaming TTFT.
Inputs were synthetic repeated token patterns. Each case used one warmup and
three profiling-off measurements; the table reports the median. Long requests
include all consecutive Prefill chunks.

| Batch | Input tokens per request | Optimized GBSA (s) | Phase dispatch (s) | Time reduction |
| --- | --- | --- | --- | --- |
| 1 | 128 | 0.390575 | 0.416592 | -6.66% |
| 1 | 4096 | 1.399694 | 1.136559 | 18.80% |
| 1 | 8192 | 2.780302 | 2.271511 | 18.30% |
| 1 | 16384 | 5.623510 | 4.608918 | 18.04% |
| 16 | 4096 each | 21.913479 | 17.845367 | 18.56% |
| 2 | 128 and 8192 | 3.082673 | 2.656834 | 13.81% |

The 128-token case was variable and did not establish a stable improvement.
These compatibility-configuration results do not establish performance for
default candidate counts, arbitrary workloads, or different hardware.

### Full-model profiling and graph replay

Both versions have full-model CPU/NPU profiles for all eight TP ranks, collected
with Level2, PMU PipeUtilization, `record_shapes=True` and `op_attr=True`.
All 16 `kernel_details.csv` files contain PMU columns and positive cycle records.
Each rank-0 trace confirms 15 `aclmdlRIExecuteAsync` and `Runtime@ModelExecute`
events. Each profiled generation contains one pure-Prefill step, three mixed
steps and 15 pure-Decode steps, plus an empty final model-step marker.

With 12 QSA layers, phase dispatch produced 48 FIA calls and 216 GBSA calls;
the GBSA-only baseline produced 228 GBSA calls. This agrees with the recorded
request phases. The profiles also include the rest of the model, including
linear attention, MoE/matmul, communication and sampling.

### Accuracy remains open

All six first-token timing cases produced equal first tokens. In a mixed
16-output-token generation, the long request matched, but the short request
diverged from output token 5. A same-process baseline/phase comparison with graph
replay disabled reproduced the discrepancy in repeated runs. Enabling or
disabling attention shadow checks did not change those outputs.

On rank0, 60 same-input attention comparisons gave:

| Metric across attention calls | Maximum |
| --- | --- |
| Absolute elementwise error | 0.015625 |
| RMSE | 0.0005521285 |
| Relative L2 error | 0.0009672444 (0.0967%) |

Each metric is computed on the entire attention output after converting both
BF16 outputs to FP32. Relative L2 is `norm(FIA - GBSA) / max(norm(GBSA), 1e-12)`.
The maxima can come from different calls. All checked outputs were finite.
Two early calls additionally compared sampled first/last query rows against a
dense FP32 attention calculation rounded back to BF16; that small sample is not
a full FP32 reference check for all calls.

Small attention errors do not prove final generation equivalence, and the cause
of the generation discrepancy has not been established. Full-model logits,
representative task accuracy, default candidate counts and broader integration
coverage remain necessary before treating this branch as ready to merge.
