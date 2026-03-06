# Baichuan-M1-14B-Instruct

## Introduction

Baichuan-M1-14B-Instruct is a 14B-parameter instruction-tuned model optimized for medical scenarios. According to the public model card, it was released in February 2025, uses BF16 weights, supports up to 32K context, and is trained for strong medical terminology understanding, clinical reasoning, and healthcare question answering.

This tutorial provides a practical workflow to deploy and validate `Baichuan-M1-14B-Instruct` on vLLM Ascend, including environment preparation, service startup, medical-domain functional checks, and accuracy evaluation.

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## Environment Preparation

### Model Weight

- `Baichuan-M1-14B-Instruct` (BF16 version): [Download model weight](https://huggingface.co/baichuan-inc/Baichuan-M1-14B-Instruct)

The public model files use custom model and tokenizer code, so deployment should enable `--trust-remote-code`.

It is recommended to download the model weights to a shared local directory such as `/root/.cache/huggingface/` before deployment.

### Installation

You can use the official vLLM Ascend docker image to deploy `Baichuan-M1-14B-Instruct`.

:::::{tab-set}
:sync-group: install

::::{tab-item} A3 series
:sync: A3

```{code-block} bash
   :substitutions:

export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3
docker run --rm \
    --name vllm-ascend \
    --shm-size=1g \
    --net=host \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
```

::::
::::{tab-item} A2 series
:sync: A2

```{code-block} bash
   :substitutions:

export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
    --name vllm-ascend \
    --shm-size=1g \
    --net=host \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
```

::::
:::::

## Deployment

### Single-node Deployment

`Baichuan-M1-14B-Instruct` is a BF16 14B text generation model with a 32K maximum context length in its public configuration. A conservative first deployment is single-node, single-card or TP-enabled deployment depending on available NPU memory.

Create a script such as `deploy_baichuan_m1.sh` and adjust `ASCEND_RT_VISIBLE_DEVICES` and `--tensor-parallel-size` for your hardware:

```shell
#!/bin/sh
export ASCEND_RT_VISIBLE_DEVICES=0
export MODEL_PATH="baichuan-inc/Baichuan-M1-14B-Instruct"

vllm serve ${MODEL_PATH} \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name baichuan-m1-14b-instruct \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 32768
```

If a single card does not provide enough memory for your target batch size or context length, reduce `--max-model-len` or enable tensor parallel deployment.

### Multi-node Deployment

Single-node deployment is recommended for initial verification. Use multi-node deployment only after confirming the basic inference path and memory footprint on your target hardware.

### Prefill-Decode Disaggregation

Not verified in this tutorial.

## Functional Verification

After the service starts, verify the OpenAI-compatible endpoint with a medical-domain request.

```shell
curl http://<IP>:<Port>/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "baichuan-m1-14b-instruct",
        "messages": [
            {"role": "system", "content": "You are a careful medical assistant. Provide educational information and recommend professional evaluation for emergencies."},
            {"role": "user", "content": "A patient has had fever, productive cough, and shortness of breath for 3 days. List possible causes, red flags, and what information a clinician would want next."}
        ],
        "temperature": 0,
        "max_tokens": 256
    }'
```

A valid response should contain medically relevant differential ideas, explicit red-flag awareness, and a structured follow-up information request. For a broader smoke test, you can also try these medical-domain prompts:

- Drug safety check: ask for common contraindications and monitoring points for warfarin.
- Clinical triage: ask which symptoms in chest pain require emergency evaluation.
- Medical terminology: ask for a patient-friendly explanation of HbA1c and why it matters.

## Accuracy Evaluation

### Using LM Eval

An initial e2e config is provided at `tests/e2e/models/configs/Baichuan-M1-14B-Instruct.yaml`.

Run the accuracy job with:

```shell
pytest -sv tests/e2e/models/test_lm_eval_correctness.py \
    --config tests/e2e/models/configs/Baichuan-M1-14B-Instruct.yaml
```

The current config uses the `pubmedqa` benchmark as a medical-domain validation task.

## Performance

### Using vLLM Benchmark

After the serving endpoint is stable, run online or offline performance tests with the built-in benchmark tools.

```shell
vllm bench serve \
  --model baichuan-inc/Baichuan-M1-14B-Instruct \
  --trust-remote-code \
  --dataset-name random \
  --random-input 512 \
  --num-prompts 100 \
  --request-rate 1 \
  --save-result \
  --result-dir ./perf_results/
```

For first-pass benchmarking, start with moderate prompt length and concurrency, then increase `--max-model-len`, request rate, and parallel size after confirming memory stability on your Ascend platform.

## Known Limitations

- This tutorial documents a verification path and test artifact for `Baichuan-M1-14B-Instruct`, but actual supported feature coverage still depends on the vLLM Ascend version and target hardware.
- The public Hugging Face model uses custom code, so `trust_remote_code` is expected during deployment.
- Medical-domain prompts should be treated as informational checks, not as a substitute for clinical validation or regulatory review.
