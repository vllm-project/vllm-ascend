# TeleChat2-3B

## 简介

TeleChat2-3B 是中国电信星辰大模型推出的 30 亿参数语言模型，基于标准全注意力架构（MHA，无 GQA），支持最长 32768 token 上下文。模型采用 RoPE + NTK 动态扩展，词表大小 131072。

该模型首次在 `vllm-ascend` 中通过 vLLM 内置 `telechat2` 适配器支持（继承自 `LlamaForCausalLM`，无需额外补丁）。

## 支持特性

| 特性 | 状态 |
|------|------|
| ACLGraph | ✅（eager 模式已验证，ACLGraph 待图编译开启后验证）|
| Expert Parallel (EP) | N/A（非 MoE 模型）|
| flashcomm1 | N/A（非 MoE 模型）|
| MTP | N/A（checkpoint 不包含 MTP 权重）|
| 多模态 | N/A（纯语言模型）|

## 环境准备

### 模型权重

- ModelScope：[TeleAI/TeleChat2-3B](https://www.modelscope.cn/models/TeleAI/TeleChat2-3B)

建议将模型权重下载至本地路径，例如 `/models/TeleChat2-3B`。

### 安装

使用官方 vllm-ascend Docker 镜像直接运行。

:::::{tab-set}
:sync-group: install

::::{tab-item} A2 系列
:sync: A2

```{code-block} bash
   :substitutions:

export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a2
docker run --rm \
    --name vllm-ascend \
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
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /models:/models \
    -it $IMAGE bash
```

::::

:::::

## 部署

```bash
OMP_NUM_THREADS=1 \
HCCL_OP_EXPANSION_MODE=AIV \
vllm serve /models/TeleChat2-3B \
    --served-model-name TeleAI/TeleChat2-3B \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --tensor-parallel-size 8 \
    --max-num-seqs 16 \
    --port 8000
```

> 注意：运行前需设置 `OMP_NUM_THREADS=1` 以避免多进程 worker 中 OpenMP 线程池初始化失败。

## 功能验证

```bash
# 查询模型列表
curl http://127.0.0.1:8000/v1/models

# 文本推理
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "TeleAI/TeleChat2-3B",
    "messages": [{"role": "user", "content": "你好，请介绍一下你自己。"}],
    "temperature": 0,
    "max_tokens": 64
  }'
```

**实测输出（真实权重，TP=2，eager 模式）：**

```
你好，很高兴能和你聊天。我是李华，是一名大学生，专业是计算机科学。我平时喜欢参加各种科技竞赛和参加编程俱乐部，同时
```

## 精度评估

精度评估结果待补充（当前 gsm8k 基准值尚未运行，占位为 0.0）。

## 性能

| 配置 | 吞吐量 |
|------|--------|
| TP=2，eager，max-model-len=8192，bs=1 | 输入 3.14 tok/s，输出 16.74 tok/s |
