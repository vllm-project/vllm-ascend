# Kimi-K3 EP256 A5 Accuracy Test Guide

Historical reference PR: [#21117](https://github.com/vllm-project/vllm-ascend/pull/21117).
This document now tracks `Task2_2_tcase_0.30.0_Kimi-K3_Experts256_A5_TCase`, based on
`releases/v0.30.0`. Historical results do not validate this release-branch case.
Objective: evaluate Kimi-K3 W4A8 MXFP/QuaRot pruned to 256 routed experts on a single A5 node
with eight NPUs, using the first 500 GSM8K test questions to measure answer accuracy.

## 1. Files Changed in This Branch

| File | Change |
| --- | --- |
| `.github/workflows/configs/nightly_config.yaml` | Register a single-node A5 test job with an ARM64 eight-NPU runner and the test YAML |
| `tests/e2e/cases/models/configs/Kimi/Kimi-K3-W4A8-EP256-A5.yaml` | Add model launch arguments, dataset settings, request parameters, and accuracy acceptance values |
| `tests/e2e/cases/models/configs/Kimi/Kimi-K3-EP256-A5-精度测试说明.md` | Document the change scope, test environment, parameters, and validation results in Chinese |
| `tests/e2e/cases/models/configs/Kimi/Kimi-K3-EP256-A5-Accuracy-Test-Guide.md` | Provide the English version of the test guide |

Test case: `Kimi-K3-W4A8-EP256-A5-GSM8K-First500`.
The case uses the shared entry point `tests/e2e/common/single_node/test_single_node.py`
and the existing AISBench integration. Complete model pruning and checkpoint preparation before running the test.

## 2. Environment Setup

### 2.1 Hardware and Software

The versions below come from the historical A5 accuracy test environment.

| Item | Configuration |
| --- | --- |
| Hardware / architecture | One A5 node, eight NPUs; ARM64 / aarch64 |
| Base image | `quay.io/ascend/vllm-ascend:nightly-releases-v0.30.0-a5` |
| Ascend driver / CANN Toolkit | `25.6.rc3` / `9.2.0` |
| Python / PyTorch / torch_npu | `3.12.13` / `2.10.0+cpu` / `2.10.0.post7` |
| vLLM / vllm-ascend | `0.30.0+empty` / `0.30.0rc2.dev304+geac9c0cf3.d20261031` |
| Transformers / safetensors | `5.14.1` / `0.8.0` |
| Triton / triton-ascend | `3.6.0` / `3.6.0.post2` |
| AISBench / datasets / evaluate | `3.1.0` / `3.6.0` / `0.4.6` |
| NumPy / PyArrow | `1.26.4` / `20.0.0` |
| Inference / evaluation container | `c30087248-kimi-k3-slim-container` / `c30087248-kimi-k3-eval-container` |

The historical setup ran inference and evaluation in separate containers. Evaluation used
`/opt/aisbench-venv` and accessed the inference API through host networking.
Weights and datasets were mounted read-only.

### 2.2 Historical Dependency Adjustments

| Component | Adjustment |
| --- | --- |
| FastAPI / Starlette | `0.123.10 → 0.136.3` / `0.50.0 → 1.0.1` |
| memcache-hybrid / memfabric-hybrid | Both changed from `1.3.0` to `1.2.0` |
| OpenCV headless / contourpy | Set to `4.11.0.86` / `1.3.3`, keeping NumPy `1.26.4` |
| FastAPI dependency declarations | Align vllm-ascend's `requirements.txt` and `pyproject.toml` with `>=0.133.0,<0.137.0` |
| triton-ascend source constraint | Align with the installed version, `3.6.0.post2` |
| OpenCV dependency declaration | Pin `opencv-python-headless==4.11.0.86` in vLLM's `requirements/common.txt` |
| Other dependencies | Install missing colorlog, Plotly, Matplotlib, msguard, expecttest, and FastAPI standard dependencies |

### 2.3 Weights, Data, and Environment Variables

| Item | Current YAML configuration |
| --- | --- |
| Model | Kimi-K3 W4A8 MXFP/QuaRot with 256 routed experts |
| Model path | `/root/.cache/modelscope/hub/models/CuiZ2007/Kimi-K3-w4a8-mxfp-QuaRot-experts256` |
| Dataset path | `/root/.cache/modelscope/hub/datasets/CuiZ2007/gsm8k` |
| Dataset files | `test.jsonl`: first 500 test questions; `train.jsonl`: four-shot examples |
| Service port / runner | `SERVER_PORT=8080` / `VLLM_USE_V2_MODEL_RUNNER=1` |
| Communication buffer / CPU threads | `HCCL_BUFFSIZE=800` / `OMP_NUM_THREADS=1` |
| Memory allocation | `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` |
| Engine timeout environment variables | `VLLM_ENGINE_READY_TIMEOUT_S=7200` / `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000` |
| Test harness service readiness wait | Default 2800 seconds in `RemoteOpenAIServer` |

The engine timeout environment variables do not override the test harness health-check timeout.
In `releases/v0.30.0`, the shared single-node entry point does not pass `max_wait_seconds`,
so `RemoteOpenAIServer` waits up to 2800 seconds for service readiness. Setting
`VLLM_ENGINE_READY_TIMEOUT_S=7200` in the YAML does not extend that wait to 7200 seconds.


Before execution, mount the weights and dataset at the paths specified in the YAML,
configure the target host's communication IP, network interfaces, and eight NPU devices,
and unset `HCCL_OP_EXPANSION_MODE`. Omit the trailing slash from the dataset directory path.

## 3. Test Parameters and Execution

| Category | Parameter | Configuration |
| --- | --- | --- |
| Service | Parallelism / execution mode | TP8 + EP / eager |
| Service | Data type / quantization | `bfloat16` / `ascend` |
| Service | Context length / sequence limit / batched-token limit | `6144` / `100` / `2048` |
| Service | Memory utilization / chunked prefill / prefix caching | `0.85` / enabled / disabled |
| Service | Tokenizer / reasoning parser | `kimi_k3` / `kimi_k3` |
| Service | Generation configuration / FlashComm1 | `vllm` / enabled |
| Request | Questions / client concurrency / output limit | `500` / `60` / `1024 tokens` |
| Request | temperature / top_p / seed / thinking_effort | `0` / `1.0` / `42` / `low` |
| Scoring | Current YAML scoring method | Native AISBench GSM8K four-shot CoT |
| Acceptance | Baseline / tolerance / passing range | `60.0%` / `±3.0` percentage points / `57.0%–63.0%` |

After preparing the test dependencies, run the following from the project root:

```bash
export CONFIG_BASE_PATH=tests/e2e/cases/models/configs/Kimi
export CONFIG_YAML_PATH=Kimi-K3-W4A8-EP256-A5.yaml
unset HCCL_OP_EXPANSION_MODE
python -m pytest -sv tests/e2e/common/single_node/test_single_node.py
```

## 4. Test Results

### 4.1 Three Historical A5 Accuracy Runs

The historical tests used a separate AISBench environment with custom final-answer scoring.
Each run covered the same 500 questions. Invalid or truncated final answers counted as incorrect.

| Run | Correct answers | Accuracy | Invalid final answers | Failed requests |
| --- | --- | --- | --- | --- |
| Run 1 | 312 / 500 | 62.4% | 106 | 0 |
| Run 2 | 289 / 500 | 57.8% | 141 | 0 |
| Run 3 | 313 / 500 | 62.6% | 104 | 0 |
| Summary | Mean 304.67 / 500 | Mean 60.93% | — | Total 0 |

Accuracy ranged from 57.8% to 62.6%, a spread of 4.8 percentage points.
Runs 1 and 2 shared a service instance; run 3 restarted the service.
Each run was checked for complete question IDs, correspondence to the original questions, and independent scoring.

### 4.2 Historical Validation Reference

The original test notes recorded successful static and lint checks and verified the
500-question subset with 7,473 training records. These checks belong to the earlier
change and have not been rerun as a complete validation suite for this release branch.

Historical nightly run: [42462410488](https://github.com/vllm-project/vllm-ascend/actions/runs/42462410488).
It failed during model path resolution before inference or evaluation. The YAML paths
were subsequently changed to container-visible `/root/.cache` paths.

Separate main-branch PR [#21284](https://github.com/vllm-project/vllm-ascend/pull/21284)
also has historical CI results. Its last two November 9 runs,
[42607268695](https://github.com/vllm-project/vllm-ascend/actions/runs/42607268695) and
[42613156488](https://github.com/vllm-project/vllm-ascend/actions/runs/42613156488),
reached model initialization but failed HCCL initialization with an invalid topology
path, `/home/wkf68687/server_8p_nomem.json`. They did not reach accuracy evaluation
and are not runs of this release branch.

### 4.3 Current Release-Branch Validation Status

| Check | Status |
| --- | --- |
| Test-file transfer | Four files synchronized from the source task branch |
| Nightly registration scope | Verified: only four Kimi registration lines added relative to the release baseline |
| Evaluation-field compatibility | Inspected the release AISBench implementation; configured fields are supported |
| Readiness timeout documentation | Clarified the harness default of 2800 seconds separately from engine environment variables |
| Model and dataset paths | Updated container paths retained; visibility on the target A5 CI runner remains to be confirmed |
| YAML loading, full lint and formatting checks | Pending rerun on this branch; historical passes are not carried forward |
| Target A5 image launch-argument validation | Pending |
| A5 model loading, generation and 500-question evaluation | Pending |
| Release-branch nightly CI | Not run for this branch as part of this update |
| Native-scoring acceptance calibration | Pending; current configuration remains 60.0% ±3.0 percentage points |

Record the tested commit, image digest and CI run link when completing these checks.
