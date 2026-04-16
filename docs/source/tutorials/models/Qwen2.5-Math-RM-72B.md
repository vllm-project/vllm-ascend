# Qwen2.5-Math-RM-72B

## Introduction

Qwen2.5-Math-RM-72B is a 72-billion parameter reward model designed for mathematical reasoning and evaluation. It is part of Alibaba Cloud's Qwen 2.5 series, specifically optimized for scoring and ranking mathematical problem solutions. The model supports a maximum context window of 128K tokens and delivers enhanced capabilities in mathematical computation, step-by-step reasoning evaluation, and solution quality assessment.

This document provides a detailed workflow for the complete deployment and verification of the model, including supported features, environment preparation, single-node deployment, functional verification, and performance evaluation.

The `Qwen2.5-Math-RM-72B` model is supported since `vllm-ascend:v0.9.0`.

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## Environment Preparation

### Model Weight

- `Qwen2.5-Math-RM-72B` (BF16 version):
    - With CPU offloading: requires at least 1 Atlas 910B4 (32G × 1) card or higher
    - Without CPU offloading: requires at least 4 Atlas 910B4 (32G × 4) cards or higher
  [Download model weight](https://modelscope.cn/models/Qwen/Qwen2.5-Math-RM-72B)

It is recommended to download the model weights to a local directory (e.g., `./Qwen2.5-Math-RM-72B/`) for quick access during deployment.

### Installation

You can use our official docker image to run `Qwen2.5-Math-RM-72B` directly. For optimal performance, it is recommended to use:

- PyTorch version: 2.10.0+cpu
- torch_npu version: 2.10.0rc2

These versions support multi-NPU deployment, allowing the model to utilize all available NPU devices (e.g., 4 NPUs) for improved performance.

Select an image based on your machine type and start the docker image on your node, refer to [using docker](../../installation.md#set-up-using-docker).

```{code-block} bash
   :substitutions:
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
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

## Deployment

### Single-node Deployment

Qwen2.5-Math-RM-72B supports single-node single-card deployment on the 910B4 platform. Follow these steps to start the inference service:

1. Prepare model weights: Ensure the downloaded model weights are stored in the `./Qwen2.5-Math-RM-72B/` directory.
2. Create and execute the deployment script (save as `deploy.sh`):

```shell
#!/bin/sh
export ASCEND_RT_VISIBLE_DEVICES=0
export MODEL_PATH="Qwen/Qwen2.5-Math-RM-72B"

vllm serve ${MODEL_PATH} \
          --host 0.0.0.0 \
          --port 8000 \
          --served-model-name qwen2.5-math-rm-72b \
          --trust-remote-code \
          --max-model-len 32768 \
          --task reward
```

:::{note}
The `--task reward` parameter is required to run the model in reward model mode for scoring mathematical solutions.
:::

### Multi-node Deployment

Single-node deployment is recommended for Qwen2.5-Math-RM-72B.

### Prefill-Decode Disaggregation

Not supported for reward models.

## Functional Verification

After starting the service, verify functionality using a `curl` request:

```shell
curl http://localhost:8000/v1/reward \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen2.5-math-rm-72b",
        "messages": [
            {"role": "system", "content": "You are a helpful math assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."}
        ]
    }'
```

A valid response (e.g., `{"reward_score": 1.69}`) indicates successful deployment.

### Batch Reward Scoring

You can also score multiple responses for comparison:

```shell
curl http://localhost:8000/v1/reward/batch \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen2.5-math-rm-72b",
        "conversations": [
            [
                {"role": "system", "content": "You are a helpful math assistant."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."}
            ],
            [
                {"role": "system", "content": "You are a helpful math assistant."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 5."}
            ]
        ]
    }'
```

## Accuracy Evaluation

### Using AISBench

Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

Results and logs are saved to `benchmark/outputs/default/`. A sample accuracy report is shown below:

| dataset | version | metric | mode | vllm-api-reward |
|----- | ----- | ----- | ----- |--------------|
| math_reward | - | accuracy | gen | 85.50  |

## Performance

### Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### Using vLLM Benchmark

Run performance evaluation of `Qwen2.5-Math-RM-72B` as an example.

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/contributing/benchmarks.html) for more details.

There are three `vllm bench` subcommands:

- `latency`: Benchmark the latency of a single batch of requests.
- `serve`: Benchmark the online serving throughput.
- `throughput`: Benchmark offline inference throughput.

Take the `serve` as an example. Run the code as follows.

```shell
vllm bench serve \
  --model ./Qwen2.5-Math-RM-72B/ \
  --dataset-name random \
  --random-input 200 \
  --num-prompts 200 \
  --request-rate 1 \
  --save-result \
  --result-dir ./perf_results/ \
  --task reward
```

After about several minutes, you can get the performance evaluation result.

## Troubleshooting

### Common Issues

**Issue**: Model fails to load with OOM error

**Solution**: Use `device_map="balanced"` with CPU offloading:

```python
model = AutoModel.from_pretrained(
    model_path,
    device_map="balanced",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    offload_folder="./offload"
).eval()
```

**Issue**: `pad_token_id` missing error

**Solution**: Add the following to `config.json`:

```json
{
  "pad_token_id": 151643
}
```

**Issue**: Version incompatibility with transformers

**Solution**: Install compatible versions:

```bash
pip install transformers==4.43.0 tokenizers==0.19.0
```

## References

- [Qwen2.5-Math Technical Report](https://arxiv.org/abs/2409.12122)
- [HuggingFace Model Card](https://huggingface.co/Qwen/Qwen2.5-Math-RM-72B)
- [vLLM Documentation](https://docs.vllm.ai/)
