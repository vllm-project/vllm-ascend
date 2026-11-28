# Rex-Omni (Qwen2.5-VL-3B) on Ascend 910B3: Deployment and Optimization Case Study

!!! note

    This page is a measured deployment and optimization case study, not an official
    support claim for the model. The numbers below were collected on a single
    **Ascend 910B3 (64 GB)** with **vLLM 0.14.1** and **vllm-ascend 0.14.0rc1**.
    The current main branch targets a newer vLLM release; the optimization levers
    and pitfalls are still representative, but re-validate the configuration on the
    version you deploy.

## 1 Introduction

Rex-Omni is an open-vocabulary perception model built on Qwen2.5-VL-3B. It accepts an
image plus a text prompt and emits structured detections (bounding boxes, points, or
polygons) for arbitrary object categories.

This document records how the offline batch workload was tuned on one 910B3, using the
Dense200 `box_eval` split (200 images) as the fixed benchmark. The goal was to raise image
throughput without degrading accuracy:

| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| Throughput | 0.472 img/s | **3.651 img/s** | **7.74x** |
| Wall time, 200 images | 424 s | **54.8 s** | -87.1% |
| F1@0.5 | 0.6896 | **0.7554** | +0.0658 |
| mIoU-F1 (IoU 0.5:0.95) | 0.5075 | **0.5486** | +0.0411 |

The accuracy improvements that can be attributed to specific changes come from
de-duplicating parsed results and from stopping degenerate repetition early; the runtime
optimizations alone move accuracy only within the run-to-run noise band. Section 9 breaks
the throughput gain down step by step.

## 2 Supported Features

| Feature | Status on this configuration | Remarks |
|---------|------------------------------|---------|
| Hardware | Single Atlas 800I A2 910B3, 64 GB | Multi-card and multi-node not evaluated |
| Precision | BF16 | W8A8 not tested (no msmodelslim in the environment) |
| Input modality | Image + text | Video and multi-image not evaluated |
| Backend | Offline `LLM.generate` batch | Online serving not evaluated |
| Async scheduling | Works | Logits processors require the fix described in Section 10 |
| Full ACL graph decode | Works (`FULL_DECODE_ONLY`) | Experimental in 0.14.0rc1; capture sizes must cover batch buckets |
| Tensor / data parallel | Not used | Single-card study |

## 3 Prerequisites

### 3.1 Model Weight

| Weight | Size | Source |
|--------|------|--------|
| `IDEA-Research/Rex-Omni` | ~7.6 GB (BF16) | [Hugging Face](https://huggingface.co/IDEA-Research/Rex-Omni) / [ModelScope](https://modelscope.cn/models/IDEA-Research/Rex-Omni) |

The evaluation split (`Rex-Omni-Eval`, Dense200 `box_eval`) comes from the same model
repository. The official `inference_text_prompt.py` hard-codes `.to("cuda")`; this study
uses the repository's `rex_omni.RexOmniWrapper` with the vLLM backend instead, and converts
its output to the official metric format.

### 3.2 Environment

| Component | Version |
|-----------|---------|
| Python | 3.11.14 |
| torch / torch_npu | 2.9.0 / 2.9.0 |
| vLLM | 0.14.1 |
| vllm-ascend | 0.14.0rc1 |
| CANN | Follow the image matching the versions above |

## 4 Installation

### 4.1 Container Image

Start the vllm-ascend image matching your CANN and driver versions, then install the model
repository into the same environment:

```bash
git clone https://github.com/IDEA-Research/Rex-Omni.git
cd Rex-Omni
pip install -e .
```

### 4.2 Source Installation of vllm-ascend

If you build vllm-ascend from source, follow the
[installation guide](../../getting_started/installation.md) and make sure the installed
vLLM version matches the branch you build.

## 5 Offline Batch Deployment

### 5.1 Recommended Configuration

| Parameter | Value | Why |
|-----------|-------|-----|
| Batch scope | Submit all 200 requests at once | Continuous batching removes the barrier between fixed chunks |
| `max_pixels` | 1,003,520 | 1M pixels cut prompt tokens by ~32% with no accuracy loss |
| `max_tokens` | 2,048 | Truncates degenerate repetition; 4,096 only produced duplicate boxes |
| `max_model_len` | 4,096 | Same reason; 8,192 did not improve accuracy |
| `repetition_penalty` | 1.05 | Higher values hurt recall |
| Parser | Drop duplicate boxes/points/polygons | Same compute, better precision |
| Early stop | Force EOS after 5 exact token-period repeats | Removes repetition loops at generation time |
| `async_scheduling` | `false` on 0.14.0rc1 | Logits processors see `-1` placeholders otherwise (Section 10) |
| ACL graph | `FULL_DECODE_ONLY`, capture sizes `[1, 8, 16, 32, 64, 128, 200, 256]` | Captures the whole decode forward |
| `max_num_batched_tokens` | 32,768 | 6 prefill chunks instead of 23 |
| Vision preprocessing | `REX_OMNI_VIS_WORKERS=8`, `--load-workers 8` | Parallelizes PIL decode/resize on the client side |

### 5.2 Launch Command

```bash
REX_OMNI_VIS_WORKERS=8 python3 run_eval.py \
  --jsonl /path/to/Rex-Omni-Eval/_annotations/box_eval/Dense200.jsonl \
  --image-root /path/to/Rex-Omni-Eval \
  --out /tmp/rex_omni_opt/answer.jsonl \
  --limit 200 --batch-size 200 \
  --max-pixels 1003520 --max-tokens 2048 --max-model-len 4096 \
  --gpu-mem 0.8 --load-workers 8 --early-stop-repeats 5 \
  --extra '{"async_scheduling": false,
            "max_num_batched_tokens": 32768,
            "compilation_config": {
              "cudagraph_mode": "FULL_DECODE_ONLY",
              "cudagraph_capture_sizes": [1, 8, 16, 32, 64, 128, 200, 256]}}'
```

`run_eval.py` is an evaluation harness, not part of the model repository. The same
configuration can be applied directly to `LLM(...)` and `SamplingParams(...)`; the important
part is the parameter set in Section 5.1.

## 6 Functional Verification

Start with a small smoke run (8-16 images) and check that every request returns parseable
boxes:

```bash
REX_OMNI_VIS_WORKERS=8 python3 run_eval.py \
  --jsonl .../Dense200.jsonl --image-root ... \
  --out /tmp/smoke/answer.jsonl --limit 8 --batch-size 8 \
  --max-pixels 1003520 --max-tokens 2048 --max-model-len 4096
```

Inspect `answer.jsonl`: each row should contain a non-empty detection list, and no output
should end with a long run of identical boxes. If outputs look truncated, compare with
`async_scheduling=true` disabled and with the early stop disabled to isolate the cause.

## 7 Accuracy Evaluation

Score the generated `answer.jsonl` with the model repository's official metric script,
using the same IoU thresholds as the baseline:

| Configuration | F1@0.5 | mIoU-F1 |
|---------------|--------|---------|
| Baseline: batch 64, 2M pixels, 4,096 tokens | 0.6896 | 0.5075 |
| + batch 200, 2,048 tokens, 1M pixels | 0.7064 | 0.5157 |
| + parser de-duplication | 0.7214 | 0.5248 |
| + generation-side early stop | 0.7481 | 0.5428 |
| + `FULL_DECODE_ONLY` ACL graph | 0.7474 | 0.5428 |
| + 32k prefill chunks | 0.7474 | 0.5426 |
| + parallel input preprocessing | **0.7554** | **0.5486** |

Greedy decoding is **not batch-invariant** on this stack: the same 200 images produce
completely identical raw outputs for only 16 of 200 samples when comparing batch 64 with
batch 200, and run-to-run F1 varies by 0.008-0.014. Differences below that band are not
evidence of an accuracy change.

## 8 Performance Evaluation

### 8.1 Throughput

| Configuration | Wall time (200 images) | Image throughput | Output tokens/s |
|---------------|------------------------|------------------|-----------------|
| Baseline | 424 s | 0.472 img/s | 291 |
| Recommended | **54.8 s** | **3.651 img/s** | 1,335 |

### 8.2 Per-request Latency

Latencies below are the job-level mean. Engine TTFT is measured inside the engine
(`queue + prefill`); offline `LLM.generate` numbers include the front-end add/output
backlog and should not be mixed with engine numbers.

| Metric | Baseline | Optimized |
|--------|----------|-----------|
| Engine TTFT, batch 1 | 0.126 s | 0.125 s |
| TPOT, batch 1 | 23.4 ms | **10.4 ms** |
| Engine TTFT, concurrency 200 | 4.16 s | **2.07 s** |
| TPOT mean, concurrency 200 | 123.7 ms | 109.9 ms |
| TPOT P90, concurrency 200 | 235.3 ms | **161.1 ms** |
| E2E, concurrency 200 | 36.2 s | **21.7 s** |

To collect per-request metrics, set `disable_log_stats: false`; otherwise
`RequestOutput.metrics` is `None`.

## 9 Performance Tuning

### 9.1 Gain Waterfall

| Step | Throughput | Wall time | Relative gain |
|------|------------|-----------|----------------|
| Baseline | 0.472 img/s | 424 s | - |
| Batch all 200 requests at once | 1.219 img/s | 164 s | 2.58x |
| `max_tokens=2048` | 1.82 img/s | 110 s | +49% |
| `max_pixels=1M` | 2.111 img/s | 95 s | +16% |
| Parser de-duplication | 2.111 img/s | 95 s | 0 (accuracy +0.015) |
| Early stop, async scheduling off | 2.441 img/s | 82 s | +17% |
| `FULL_DECODE_ONLY` ACL graph | 3.103 img/s | 64.5 s | +27% |
| 32k prefill chunks | 3.197 img/s | 62.6 s | +3% |
| Parallel input preprocessing | **3.651 img/s** | **54.8 s** | +14.2% |

The gains compound: each ratio is relative to the *previous* stage, and
`2.58 x 1.49 x 1.16 x 1.17 x 1.27 x 1.03 x 1.142 = 7.74x`.

### 9.2 What Each Step Does

- **Submit the whole batch at once.** Fixed chunks (64 images) force a barrier between
  chunks; continuous batching overlaps the tail of one chunk with the next. This alone is
  the largest single win (2.58x).
- **Cap output length.** Long outputs are dominated by degenerate repetition, not by
  additional detections. `max_tokens=2048` removes the repetition tails.
- **Lower input resolution to 1M pixels.** Prompt tokens drop from 1,378 to 942 on
  average; 800k gave no further gain.
- **De-duplicate parser output.** Identical box/point/polygon tuples were previously
  emitted repeatedly, creating false positives. This costs no compute but improves
  precision.
- **Stop repetition at generation time.** A logits processor detects an exactly periodic
  token tail (period 4-48, repeated 5 times) and forces the EOS token. Only degenerate
  samples are truncated.
- **Capture the decode step in one ACL graph.** `FULL_DECODE_ONLY` captures the whole
  decode forward, including attention and `ReshapeAndCache`. Step latency at batch 1 drops
  from 27.8 ms to 10.4 ms.
- **Raise the prefill chunk budget.** `max_num_batched_tokens=32768` reduces the number
  of prefill chunks from 23 to 6.
- **Parallelize client-side input preprocessing.** Rex-Omni's wrapper calls
  `process_vision_info()` serially (RGB conversion plus resize, ~36 ms per image). Both
  this and the PIL image loading happen before requests reach the engine, so they are on
  the critical path. A thread pool with 8 workers reduces that stage from 10.9 s to 1.6 s
  on 200 images; PIL releases the GIL during decode and resize.

### 9.3 Rejected Optimizations

| Attempt | Result | Conclusion |
|---------|--------|------------|
| `OMP_NUM_THREADS=8` for input preprocessing | 63.1 s vs 62.6 s | The engine already overlaps preprocessing; not on the critical path |
| `max_model_len=8192` | Accuracy unchanged, slower | The model repeats until the cap |
| `repetition_penalty=1.1 / 1.2` | No gain / recall loss | 1.05 is the sweet spot |
| `max_pixels=802,816` | F1 0.7339 vs 0.7483 | 1M is the sweet spot |
| W8A8 quantization | Not tested | No msmodelslim; profiling shows neither compute nor bandwidth saturation |

### 9.4 Where the Time Goes

At the final configuration (54.8 s for 200 images):

| Stage | Time | Share |
|-------|------|-------|
| Image loading (PIL, 8 threads) | ~0.5 s | 1% |
| Vision preprocessing (8 threads) | ~1.0 s | 2% |
| Engine start and first prefill chunk | ~0.3 s | <1% |
| Prefill (~188k prompt tokens, 6 chunks) | ~17.4 s | 32% |
| Decode (~70k output tokens) | ~36.0 s | 66% |

Decode is the remaining double-digit lever; speculative decoding is the next candidate.

## 10 Pitfalls and FAQ

**Logits processors see `-1` placeholders with async scheduling.** On vllm-ascend
0.14.0rc1, `NPUModelRunner._sample()` overrides the upstream method and did not call
`update_async_output_token_ids()`. With `--async-scheduling`, custom logits processors
therefore received placeholder token ids and could truncate correct outputs (the fix used
here was `async_scheduling=false`). Current main calls `update_async_output_token_ids()`;
see [#21219](https://github.com/vllm-project/vllm-ascend/pull/21219) for the equivalent
fix in the speculative-decoding branch.

**Use the config EOS id, not the tokenizer's.** The model vocabulary has 151,936 entries,
while the tokenizer maps `<|im_end|>` to id 152,645, which is outside the vocabulary and
cannot be sampled. To force a stop from a logits processor, use the config
`eos_token_id` (151,645).

**`FULL_DECODE_ONLY` is experimental.** Capture sizes must cover the batch buckets you
actually run; capturing too many sizes can cause OOM or a hang. Lower
`gpu_memory_utilization` or reduce the size list if that happens. Graph capture also
introduces small numerical differences, so greedy outputs are not bit-identical between
graph and non-graph runs.

**Do not compare accuracy across different batch compositions.** Greedy decoding on this
stack is not batch-invariant; fix the batch composition (or repeat runs) before drawing
conclusions.

## Appendix: Full Case Study (Chinese)

The complete write-ups are kept alongside this tutorial:

- [Conclusion-first case study (CASE_STUDY_V2.md)](rex-omni/CASE_STUDY_V2.md)
- [Methodology and ablation details (METHODOLOGY.md)](rex-omni/METHODOLOGY.md)
- [Final report (REPORT.md)](rex-omni/REPORT.md)
