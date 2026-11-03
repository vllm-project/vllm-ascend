# Kimi-K3 EP256 GSM8K accuracy regression

[中文版](README.zh-CN.md)

Updated: 2026-11-03. Branch: `test/kimi-k3-pruned-experts-accuracy`.
This document consolidates the operating instructions, change checklist and test report.

## 1. Scope and delivery status

**Single-node A5, eight NPUs, Kimi-K3 W4A8 pruned to 256 routed experts,
GSM8K first-500 accuracy regression.** One invocation evaluates one round.
Three historical rounds characterize variance; they are not three nightly invocations.

The pytest entry, pruning/scoring tools and nightly registration are implemented in source.
Eighteen CPU unit tests, Ruff checks and matrix parsing passed. The entry has not
been validated against the target NPU/AISBench runtime. Historical remote results
do not establish a pass for the current branch.

## 2. File inventory

Paths are relative to this document. Source files reside in `tools/kimi_k3`, E2E
configuration/entry under `tests/e2e/cases/models`, and unit tests under `tests/ut/tools/kimi_k3`.

| File | Responsibility |
| --- | --- |
| [prune_experts.py](prune_experts.py) | Byte-preserving pruning, planning and shard verification |
| [test_prune_experts.py](../../tests/ut/tools/kimi_k3/test_prune_experts.py) | Eight pruning tests |
| [serve_ep256.sh](serve_ep256.sh) | Shared eight-card service flags |
| [gsm8k_ep256_first500.py](../../tests/e2e/cases/models/configs/Kimi/gsm8k_ep256_first500.py) | AISBench first-500 config and original four-shot prompt |
| [accuracy_api.py](accuracy_api.py) | Non-streaming final-channel API adapter |
| [final_answer.py](final_answer.py) | Final-channel recovery and strict numeric extraction |
| [test_final_answer.py](../../tests/ut/tools/kimi_k3/test_final_answer.py) | Four final-answer tests |
| [accuracy_case.py](accuracy_case.py) | Model/data checks and independent result audit |
| [test_accuracy_case.py](../../tests/ut/tools/kimi_k3/test_accuracy_case.py) | Six input/result contract tests |
| [test_kimi_k3_ep256_accuracy.py](../../tests/e2e/cases/models/test_kimi_k3_ep256_accuracy.py) | Executable pytest regression entry |
| [nightly_config.yaml](../../.github/workflows/configs/nightly_config.yaml) | A5 nightly matrix registration |
| [nightly-plan.yaml](nightly-plan.yaml) | Case contract metadata, not an executable YAML |
| [summarize_accuracy.py](summarize_accuracy.py) | Three-round mean, standard deviations and range |
| [**init**.py](__init__.py) | Importable tools package |

## 3. Pruning and checkpoint verification

### Source checkpoint version used in this experiment

The pruning input was the **Kimi-K3 W4A8 MXFP QuaRot checkpoint**, locally
identified as `Kimi-K3-w4a8-mxfp-QuaRot`, with 93 layers and 896 routed experts
per MoE layer. The canonical source directory is
`/mnt/share/weights/Kimi-K3-w4a8-mxfp-QuaRot`; the published output was
`/mnt/share/weights/Kimi-K3-w4a8-mxfp-QuaRot-slim_ep_256xps`.

The experiment used a copy of this source checkpoint; the same original checkpoint
can be used directly. The pruning manifest records these source identity checksums:

| Source file | SHA256 |
| --- | --- |
| `config.json` | `07677f14bc468c16a788fc6517f7fa715701fa1c3ff74b3f7ef3449a5bc8c735` |
| `model.safetensors.index.json` | `b21a53e0b92c0bb6474be2da74ceb9261ba888f8880a8b8f84fb8b8d6ee6e4e1` |

No upstream repository revision or release tag is recorded for this local checkpoint.
The directory name and recorded checksums identify the experiment input; the checksums
cover configuration/index files, not every source weight byte. The vLLM/vllm-ascend
runtime versions in section 9 are not checkpoint versions.

Start from the original quantized checkpoint. In each of 92 MoE layers,
select `keep_ids[j] = floor(j * 896 / 256)`, `j=0..255`. Slice router rows and
correction bias identically and remap retained expert IDs to 0..255.
Copy retained quantized weights and scales byte-for-byte without requantization.
Retain 93 layers, top-k 16 and two shared experts. This uniform capacity pruning
is not importance-calibrated selection or an accuracy-preservation guarantee.

```bash
python -m tools.kimi_k3.prune_experts \
  --src /models/source --dst /models/ep256.partial --work /work/pruning \
  --keep-experts 256 --mode plan
python -m tools.kimi_k3.prune_experts \
  --src /models/source --dst /models/ep256.partial --work /work/pruning \
  --keep-experts 256 --mode run --workers 4
```

Example paths must be replaced with actual paths. Retain expert maps, copy plans
and shard verification records. Verify completion before publishing the partial
output as the final checkpoint. Private Docker publication/mount scripts are not included.
The historical pruning report records 358 shards, 144683 tensors and verification
of retained output bytes. Copy-plan digest:
`6ed94726e172e3b67c344901a70383451308446f6f346cd64213bd0094533f1e`.
This is not a full-checkpoint hash. The migrated tool has not reprocessed the full checkpoint.

## 4. Fixed execution and scoring contract

| Setting | Value |
| --- | --- |
| Hardware | One A5 node, eight available NPUs |
| Model | W4A8 MXFP QuaRot, 93 layers, 256 routed experts |
| Parallelism / runner | TP8 + EP, MRV2, eager |
| Context / sequences / batched tokens | 6144 / 100 / 2048 |
| Memory utilization | 0.85 |
| Prefix cache / flashcomm1 | Disabled / enabled |
| Dataset | Original GSM8K test rows `[0:500]`, unchanged four-shot prompt |
| Client concurrency / output limit | 60 / 1024 tokens |
| Sampling | temperature 0, top_p 1, seed 42, thinking_effort low |
| API | Non-streaming, model alias `kimi-k3-ep256` |

Original test.jsonl SHA256:
`3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`.
The loader also needs train.jsonl. No original GSM8K data was edited.

Score only an explicit final numeric answer or numeric-only final response.
Truncated responses and invalid/missing finals count wrong in the 500-question
denominator. Do not take the last number from reasoning. A complete XTML response
channel can be recovered; arbitrary reasoning cannot be treated as a final answer.
The adapter imports directly from this repository, without site-packages replacement.
Generic GSM8K presets are not an equivalent scoring procedure.

## 5. Execute the regression case

Use a compatible Linux A5 runtime with vLLM, vllm-ascend, AISBench and Kimi tokenizer
dependencies installed. Reserve eight NPUs. The bundle does not install packages.
From the repository root, provision all four variables:

| Variable | Required value |
| --- | --- |
| `KIMI_EP256_MODEL` | Absolute path to the prepared 93-layer EP256 checkpoint |
| `KIMI_GSM8K_DATASET` | Absolute path containing the original train.jsonl/test.jsonl |
| `KIMI_EP256_BASELINE` | Approved reference accuracy, percent in [0,100] |
| `KIMI_EP256_TOLERANCE` | Approved tolerance in percentage points, [0,100] |

```bash
python -m pytest -sv tests/e2e/cases/models/test_kimi_k3_ep256_accuracy.py
```

The entry checks inputs, starts its own service with a free port through the shared
`RemoteOpenAIServer`, executes AISBench, audits 500 unique IDs and original prompts,
recomputes accuracy, checks request success and applies
`abs(accuracy - baseline) <= tolerance`. Service cleanup uses the shared fixture.
This is a two-sided comparison; missing inputs/thresholds fail explicitly.
The historical mean and observed spread do not automatically set acceptance policy.

Outputs, logs and `accuracy-summary.json` are saved in a unique directory under
`benchmark_results/Kimi-K3-W4A8-EP256-A5/`. Archive it as CI/PR evidence. Request
failures invalidate a run; invalid final answers are counted incorrect.

## 6. Independent reproduction and repeated rounds

For manual operation, choose a free port and run from the repository root:

```bash
export KIMI_EP256_MODEL=/models/ep256
export KIMI_GSM8K_DATASET=/datasets/gsm8k
export KIMI_API_PORT=2000
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
bash tools/kimi_k3/serve_ep256.sh
```

```bash
ais_bench tests/e2e/cases/models/configs/Kimi/gsm8k_ep256_first500.py \
  --mode all --num-warmups 0 --max-num-workers 1 --dump-eval-details \
  --work-dir /results/round1
```

Run AISBench in another shell with the same exported variables. For three rounds,
repeat sequentially with fresh `/results/round2` and `/results/round3` work directories.
Keep the same service when investigating repeatability. Optional `KIMI_API_HOST`
defaults to 127.0.0.1 and `KIMI_API_PORT` to 2000. Do not use `--num-prompts` to
replace the explicit `[0:500]` selection. Replace TIMESTAMP/MODEL below with actual
paths; the summary consumes exactly three completed result files.

```bash
python -m tools.kimi_k3.summarize_accuracy \
  /results/round1/TIMESTAMP/results/MODEL/gsm8k.json \
  /results/round2/TIMESTAMP/results/MODEL/gsm8k.json \
  /results/round3/TIMESTAMP/results/MODEL/gsm8k.json \
  --output /results/three-round-summary.json
```

## 7. Nightly registration and local checks

Matrix: `.github/workflows/configs/nightly_config.yaml`, `a5.single_node.test_config`.

```yaml
- name: Kimi-K3-W4A8-EP256-A5
  os: linux-aarch64-a5-8
  tests: tests/e2e/cases/models/test_kimi_k3_ep256_accuracy.py
```

Selection: `/nightly Kimi-K3-W4A8-EP256-A5`, subject to the PR workflow and runner
availability. Provision the four required variables inside the CI test container.
`nightly-plan.yaml` is metadata, not the executed model YAML. The existing full-model
four-node K3 case remains unchanged. Source registration does not provision a runner.

```bash
python -m unittest discover -s tests/ut/tools/kimi_k3 -v
python -m ruff check tools/kimi_k3 tests/ut/tools/kimi_k3 \
  tests/e2e/cases/models/test_kimi_k3_ep256_accuracy.py \
  tests/e2e/cases/models/configs/Kimi/gsm8k_ep256_first500.py
```

Passed: eight pruning, four final-answer and six accuracy-contract tests; Ruff and
matrix entry resolution. These CPU checks do not establish NPU integration correctness.

## 8. Remaining validation and submission

Provision CI checkpoint/data locations, runtime dependencies and approved thresholds.
Then perform target-runtime import checks → current-branch service startup → a few
real-question end-to-end checks → verify the implemented nightly entry/registration →
accuracy validation with the final configuration. Complete these gaps before submission.

Submit the case, tools, related tests, matrix change and documentation. Attach raw
predictions/logs/result JSONs and manifests as artifacts. Do not commit model weights,
datasets, credentials or generated benchmark outputs. Record exact image digest,
vLLM/Ascend commits, runtime patches and checkpoint identifiers in the final evidence.

## 9. Historical three-round results and conclusions

The remote experiment used the older container: vLLM `0.30.0+empty`, vllm-ascend
`0.30.0rc2.dev304+geac9c0cf3.d20261031`, torch `2.10.0+cpu`, torch_npu
`2.10.0.post7`, transformers `5.14.1`. It did not validate current HEAD or the
separately downloaded formal v0.30.0-a5 image.

Each round was audited: 500 unique IDs 0..499, matching original questions, independent
score equal to AISBench. Excludes later 819-question continuation and the abandoned
zero-record third-round attempt.

| Round | Correct / 500 | Accuracy | Invalid final, counted wrong | Request failures |
| --- | --- | --- | --- | --- |
| 1 | 312 / 500 | 62.4% | 106 | 0 |
| 2 | 289 / 500 | 57.8% | 141 | 0 |
| 3 | 313 / 500 | 62.6% | 104 | 0 |

Mean: **60.93%** (914/1500 repeated answers). Range: **57.8%–62.6%**; spread:
**4.8 percentage points**. Population standard deviation: **2.22 pp**; sample:
**2.72 pp**. These are the same 500 questions repeated, not 1500 distinct questions.

Evidence paths below are relative to the first root line; wrapped directory lines
must be joined when locating the files:

```text
/mnt/share/c30087248/kimi-k3-slim_lp/component-repair-20261031-134927/watch_ep256/runs/
  ep256-19195a7468a7/accuracy_full/20261103_003449/results/
    kimi-k3-ep256-v2-eager-greedy-low-first500-o1024-c60-final/gsm8k.json
  ep256-19195a7468a7/first500_repeats/round2/20261103_011356/results/
    kimi-k3-ep256-v2-eager-greedy-low-first500-o1024-c60-final/gsm8k.json
  ep256-44f5b8764afc/accuracy_full/20261103_054335/results/
    kimi-k3-ep256-v2-eager-greedy-low-first500-o1024-c60-final/gsm8k.json
  ep256-19195a7468a7/first500_repeats/summary.json
```

Rounds 1/2 reused one service; round 3 restarted it. Variation remained with greedy
decoding and fixed seed, but its cause has not been established. Invalid finals
contribute materially to errors and need separate truncation/format/generation
analysis; they are not request failures. Without an original full-checkpoint
baseline, accuracy loss caused by pruning cannot be quantified. The observed spread
is not an approved tolerance. This evidence supports the historical experiment,
not a claim that the current branch or nightly has passed.
