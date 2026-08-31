# Qwen3 Dense on Atlas A3

This sample recipe shows the proposed recipe format for serving Qwen3 Dense
models on Atlas A3 with vLLM Ascend.

It is derived from the existing
[Qwen3 Dense tutorial](../tutorials/models/Qwen3-Dense.md). The tutorial remains
the source of truth for full tuning explanations, accuracy data, and performance
notes.

## Metadata

| Field | Value |
|-------|-------|
| Model | Qwen3 Dense, sample command uses `vllm-ascend/Qwen3-32B-W8A8`. |
| Hardware | Atlas 800 A3, four NPUs for the sample command. |
| Precision | W8A8 quantized model. |
| Parallelism | Tensor parallel size 4. |
| Image | `quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3`. |
| Source tutorial | [Qwen3 Dense tutorial](../tutorials/models/Qwen3-Dense.md). |
| Support matrix | [Supported models](../user_guide/support_matrix/supported_models.md). |
| vLLM recipes target | `Qwen/Qwen3.md` with an Ascend platform section. |
| Validation status | Prototype recipe; requires Atlas A3 hardware for runtime validation. |

## When to Use This Recipe

Use this recipe when you want a short serving path for Qwen3 Dense on Atlas A3
and already have the model weights available on the host.

For the full model list, accuracy results, and tuning discussion, use the
[Qwen3 Dense tutorial](../tutorials/models/Qwen3-Dense.md).

## Prerequisites

- Atlas 800 A3 environment with Ascend drivers installed.
- Qwen3 model weights downloaded to a host directory, for example
  `/root/.cache`.
- vLLM Ascend image available from
  [quay.io/ascend/vllm-ascend](https://quay.io/repository/ascend/vllm-ascend?tab=tags).

## Launch Container

```bash
export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3

docker run --rm \
  --name vllm-ascend-qwen3 \
  --shm-size=1g \
  --privileged=true \
  --net=host \
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
  -it "${IMAGE}" bash
```

## Serve the Model

Run the server inside the container.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve vllm-ascend/Qwen3-32B-W8A8 \
  --served-model-name qwen3 \
  --trust-remote-code \
  --quantization ascend \
  --distributed-executor-backend mp \
  --tensor-parallel-size 4 \
  --max-model-len 5500 \
  --max-num-batched-tokens 40960 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --additional-config '{"pa_shape_list":[48,64,72,80], "weight_prefetch_config":{"enabled":true}}' \
  --port 8113 \
  --block-size 128 \
  --gpu-memory-utilization 0.9
```

For non-quantized Qwen3 Dense models, remove `--quantization ascend` and update
the model path.

## Smoke Test

```bash
curl http://localhost:8113/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "messages": [
      {
        "role": "user",
        "content": "Give me a short introduction to large language models."
      }
    ],
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "max_completion_tokens": 256
  }'
```

## Benchmark or Evaluation Hooks

Use `vllm bench serve` for a quick serving benchmark.

```bash
vllm bench serve \
  --model vllm-ascend/Qwen3-32B-W8A8 \
  --served-model-name qwen3 \
  --port 8113 \
  --dataset-name random \
  --random-input 200 \
  --num-prompts 200 \
  --request-rate 1 \
  --save-result \
  --result-dir ./
```

For accuracy evaluation, see
[Using AISBench](../developer_guide/evaluation/using_ais_bench.md).

## Tuning Notes

- `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` is useful for tensor parallel serving in
  high-concurrency scenarios.
- `max-num-batched-tokens` should be tuned with memory usage and request shape.
- `pa_shape_list` and `weight_prefetch_config` are optional tuning knobs. See
  the [Qwen3 Dense tutorial](../tutorials/models/Qwen3-Dense.md) before using
  them for production tuning.
- `cudagraph_mode` is set to `FULL_DECODE_ONLY` to match the source tutorial.

## Promotion Checklist

- [ ] Runtime validated on Atlas A3.
- [ ] Smoke test completed.
- [ ] Benchmark or accuracy result recorded.
- [ ] Model support matrix entry checked.
- [ ] Ascend-specific settings isolated for a future `vllm-project/recipes`
      platform section.
