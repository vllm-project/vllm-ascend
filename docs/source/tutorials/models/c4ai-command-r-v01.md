# CohereLabs/c4ai-command-r-v01

## Introduction

CohereLabs/c4ai-command-r-v01 is an open-weights instruction model optimized for reasoning, summarization, and question answering. It supports multilingual generation and long-context processing, and is suitable for enterprise-style assistant and RAG scenarios. This tutorial describes how to deploy and validate the model on vLLM-Ascend with Atlas A2.

## Supported Features

Refer to the [supported models matrix](../../user_guide/support_matrix/supported_models.md) for the feature support status of this model.

## Environment Preparation

### Model Weight

`CohereLabs/c4ai-command-r-v01` (BF16): requires **4 × Atlas 800I A2 (64G)** NPUs. [Model weight](https://huggingface.co/CohereLabs/c4ai-command-r-v01)

It is recommended to download the model weight to a shared cache directory (for example, `/root/.cache/`).

## Installation

Run in a Docker container:

```{code-block} bash
:substitutions:

export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
--name vllm-ascend \
--shm-size=1g \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-p 8000:8000 \
-it $IMAGE bash
```

Build from source:

```{code-block} bash
:substitutions:

# Install vLLM.
git clone --depth 1 --branch |vllm_version| https://github.com/vllm-project/vllm
cd vllm
VLLM_TARGET_DEVICE=empty pip install -v -e .
cd ..

# Install vLLM Ascend.
git clone --depth 1 --branch |vllm_ascend_version| https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git submodule update --init --recursive
pip install -v -e .
cd ..
```

## Deployment

Set environment variables before starting the service:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnrt/set_env.sh 2>/dev/null || true
unset ASCEND_DEVICE_ID DEVICE_ID ASCEND_VISIBLE_DEVICES
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HCCL_INTRA_ROCE_ENABLE=1
```

Startup command:

```{test} bash
:sync-yaml: tests/e2e/models/configs/c4ai-command-r-v01.yaml
:sync-target: test_cases[0].model test_cases[0].server_cmd
:sync-class: cmd

vllm serve "CohereLabs/c4ai-command-r-v01" \
  --served-model-name c4ai-command-r-v01 \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --port 8000
```

## Functional Verification

Open another terminal and run:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "c4ai-command-r-v01",
    "messages": [
      {"role": "user", "content": "Hello, please reply with: test successful"}
    ],
    "max_tokens": 32,
    "temperature": 0.2
  }'
```

Expected result:

- HTTP status is `200 OK`.
- Response JSON contains `choices`.
- `finish_reason` is `stop`.


## Accuracy Evaluation

Run the LM-Eval correctness test with the model config:

```bash
python -m pytest -sv tests/e2e/models/test_lm_eval_correctness.py \
  --config tests/e2e/models/configs/c4ai-command-r-v01.yaml \
  --tp-size 4 \
  --report-dir ./benchmarks/accuracy
```

Reference values (from `tests/e2e/models/configs/c4ai-command-r-v01.yaml`):

| Task | Metric | Expected (yaml) |
| --- | --- | --- |
| gsm8k | exact_match,strict-match | 0.20 |
| gsm8k | exact_match,flexible-extract | 0.15 |


## FAQ

### HCCL init failure (EI0010 / error code 5)

**Symptoms:**

- `hcclCommInitRootInfoConfig ... error code is 5`
- `P2P_Communication_Failed(EI0010)`

**Recommended checks:**

- Ensure `ASCEND_RT_VISIBLE_DEVICES` matches `--tensor-parallel-size`.
- Set `HCCL_INTRA_ROCE_ENABLE=1`.
- Keep device mapping consistent across worker processes.
- For baseline startup, try `--tensor-parallel-size 1` first.
