# Qwen2.5-1.5B-apeach

## Introduction

Qwen2.5-1.5B-apeach 是基于 Qwen2.5-1.5B-Instruct 进行微调的韩语优化大语言模型，由 jason9693 贡献。该模型保留了 Qwen2.5 系列的多语言能力，特别针对韩语场景进行了优化，适用于韩语对话、文本生成等任务。

本文档将展示该模型在 vLLM-Ascend 环境中的主要验证步骤，包括环境准备、单机部署、功能验证和精度评估。

Qwen2.5-1.5B-apeach 模型在 v0.13.0 版本中首次支持。本示例要求版本 **v0.13.0** 及以上。

## Supported Features

请参考 [Supported Features](../../user_guide/support_matrix/supported_models.md) 获取模型的特性支持矩阵。Qwen2.5 系列属于 Extended Compatible Models，通过上游 vLLM 内置的 Qwen2.5 架构支持。

请参考 [Feature Guide](../../user_guide/feature_guide/index.md) 获取特性配置信息。

## Environment Preparation

### Model Weight

- `Qwen2.5-1.5B-apeach`(BF16 version): 需要 1 张 Atlas 800I A2 (64G × 1) 卡或 1 张 Atlas 800 A3 (64G × 2) 卡。[下载模型权重](https://huggingface.co/jason9693/Qwen2.5-1.5B-apeach)

以上为推荐卡数，可根据实际情况调整。

建议将模型权重下载到多节点的共享目录中，例如 `/root/.cache/`

### Verify Multi-node Communication(Optional)

如需部署多节点环境，请根据 [verify multi-node communication environment](../../installation.md#verify-multi-node-communication) 进行通信验证。

### Installation

您可以使用官方 Docker 镜像来支持 Qwen2.5-1.5B-apeach 模型。
目前我们提供 all-in-one 镜像。[下载镜像](https://quay.io/repository/ascend/vllm-ascend?tab=tags)

#### Docker Pull (by tag)

```{code-block} bash
   :substitutions:

docker pull quay.io/ascend/vllm-ascend:|vllm_ascend_version|

```

#### Docker run

```{code-block} bash
   :substitutions:

# Update --device according to your device (Atlas A2: /dev/davinci[0-7] Atlas A3:/dev/davinci[0-15]).
# Update the vllm-ascend image according to your environment.
# Note you should download the weight to /root/.cache in advance.
# For Atlas A2 machines:
# export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
# For Atlas A3 machines:
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3
docker run --rm \
    --name vllm-ascend-env \
    --shm-size=1g \
    --net=host \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /etc/hccn.conf:/etc/hccn.conf \
    -v /root/.cache:/root/.cache \
    -it ${IMAGE} /bin/bash
```

## Online Service Deployment

### Single-Node Online Deployment

单节点部署在同一个节点内完成 Prefill 和 Decode，适用于单卡推理场景。

启动命令：

```bash
vllm serve /root/.cache/jason9693/Qwen2.5-1.5B-apeach \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.8 \
    --dtype auto
```

> 常见问题提示：如遇到启动问题，请参考 [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html) 进行排查。

服务验证：

```bash
curl http://127.0.0.1:8000/v1/models
```

预期结果：返回 HTTP 200，包含模型信息。

## Functional Verification

服务启动后，可以通过发送 prompt 来调用模型：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "jason9693/Qwen2.5-1.5B-apeach",
        "messages": [
            {"role": "user", "content": "안녕하세요, 자신을 소개해 주세요."}
        ],
        "max_tokens": 100,
        "temperature": 0
    }'
```

预期结果：返回 HTTP 200，JSON 响应中包含 `choices` 字段，模型以韩语回复。

## Accuracy Evaluation

### Using Language Model Evaluation Harness

以 `gsm8k` 数据集为例，在在线模式下运行 `Qwen2.5-1.5B-apeach` 的精度评估。

1. `lm_eval` 安装请参考 [Using lm_eval](../../developer_guide/evaluation/using_lm_eval.md)。
2. 运行 `lm_eval` 执行精度评估。

```bash
lm_eval \
  --model local-completions \
  --model_args model=/root/.cache/jason9693/Qwen2.5-1.5B-apeach,base_url=http://127.0.0.1:8000/v1/completions,tokenized_requests=False,trust_remote_code=True \
  --tasks gsm8k \
  --num_fewshot 5 \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --output_path ./
```

### Using AISBench

详情请参考 [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md)。

## Performance Tuning

### Recommended Configurations

> **Note**: 以下配置在特定测试环境中验证，仅供参考。最优配置取决于最大输入/输出长度、prefix cache 命中率、精度要求和部署机器配比等因素。

| Scenario | TP | max-model-len | gpu-memory-utilization | Key Considerations |
|----------|----|---------------|------------------------|---------------------|
| Default | 1 | 4096 | 0.8 | 单卡推理，适合一般场景 |
| Long Context | 1 | 8192 | 0.85 | 长文本场景，适当增加内存 |

### General Tuning Reference

请参考 [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) 进行调优。
请参考 [Feature Guide](../../user_guide/support_matrix/feature_matrix.md) 获取详细的特性说明。

## FAQ

> 对于常见的环境、安装和通用参数问题，请参考 [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html)；本章节仅涵盖模型特有问题。

### Q: 模型是否需要 `trust_remote_code`？

A: 是的。Qwen2.5 系列模型需要 `--trust-remote-code` 参数才能正常加载其自定义建模代码。

### Q: 该模型支持哪些 NPU 硬件？

A: 该模型支持 Atlas 800I A2 和 Atlas 800 A3 系列 NPU。作为 1.5B 参数的小型模型，仅需 1 张卡即可运行。
