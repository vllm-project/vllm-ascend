# BGE-Reranker-V2-M3

## Introduction

`BAAI/bge-reranker-v2-m3` 是 BAAI 发布的多语言 RoBERTa cross-encoder
重排序模型。模型对 query-document 对直接计算相关性分数，适合 RAG
召回后的重排序。它是 encoder-only pooling 模型，使用 vLLM 的 `score()`
接口，不需要聊天模板或模型专用补丁。

## Supported Features

当前任务覆盖以下行为：

- 1-to-1、1-to-N 和 N-to-N 打分；
- 不同文本长度的成对 batch；
- 分数形状、排序和 Transformers 参考分数；
- 空列表、批次长度不匹配、非法类型和超长输入的异常，以及拒绝后继续处理有效输入。

模型的硬件支持矩阵见[支持的模型](../../user_guide/support_matrix/supported_models.md)。

## Environment Preparation

下载模型权重 `BAAI/bge-reranker-v2-m3` 到共享模型目录，或者在环境可以访问
Hugging Face/ModelScope 时直接使用模型 ID。建议使用与当前 vLLM Ascend 版本匹配的
官方镜像。

=== "Atlas A2"

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:nightly-main
    export DEVICE=/dev/davinci0
    docker run --rm --name bge-reranker-a2 --shm-size=1g \
      --device "$DEVICE" --device /dev/davinci_manager \
      --device /dev/devmm_svm --device /dev/hisi_hdc \
      -v /root/.cache:/root/.cache -it "$IMAGE" bash
    ```

=== "Atlas A3"

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:nightly-main-a3
    export DEVICE=/dev/davinci0
    docker run --rm --name bge-reranker-a3 --shm-size=1g \
      --device "$DEVICE" --device /dev/davinci_manager \
      --device /dev/devmm_svm --device /dev/hisi_hdc \
      -v /root/.cache:/root/.cache -it "$IMAGE" bash
    ```

## Deployment

在容器内用 `LLM.score()` 在单卡上离线打分。示例使用 FP16 和 eager 模式。

```python
from vllm import LLM

llm = LLM(
    model="BAAI/bge-reranker-v2-m3",
    runner="pooling",
    dtype="half",
    trust_remote_code=True,
    enforce_eager=True,
    max_model_len=8192,
    tensor_parallel_size=1,
    enable_chunked_prefill=False,
)

query = "What is the capital of France?"
documents = [
    "Berlin is the capital of Germany.",
    "Paris is the capital of France.",
    "France is a country in Europe.",
]
outputs = llm.score(query, documents)
scores = [output.outputs.score for output in outputs]
print(scores)
print(sorted(range(len(scores)), key=scores.__getitem__, reverse=True))
```

## Functional Verification

`scores` 是按照文档输入顺序排列的浮点分数列表，每篇文档对应一个分数。示例的预期排序
下标为 `[1, 2, 0]`，下标 `1` 对应 Paris。把等长的 query 列表和 document 列表一起传入时，
每个 query 只与相同下标的 document 配对。

离线 `LLM.score()` 对空 query 或 document 列表、无法配对的批次、非法输入类型及超过
模型上下文上限的输入抛出 `VLLMValidationError`。单个 query 可以与多个 document 配对。

## Accuracy Evaluation

在真实 Ascend 权重环境运行专用测试：

```bash
cd /vllm-workspace/vllm-ascend
VLLM_WORKER_MULTIPROC_METHOD=spawn python -m pytest -sv \
  tests/e2e/models/test_bge_reranker_correctness.py \
  --config tests/e2e/models/configs/bge-reranker-v2-m3.yaml \
  --tp-size 1 \
  --report-dir ./benchmarks/bge-reranker-v2-m3/accuracy
```

单卡测试先用 vLLM 收集分数和异常处理结果，释放引擎后在同一张 NPU 上以 FP16 运行
Transformers `AutoModelForSequenceClassification`。两者使用相同权重和 tokenizer。
参考端将 logit 转为 FP32 后应用 Sigmoid。测试要求实际输入 token 一致、分数相对误差
在 1% 以内，且 1-to-N 的候选排序一致。JSON 报告记录分数、形状、排序和 token 哈希。

在 HiDevLab 单卡 A3 的 vLLM Ascend 0.30.0 配套镜像上，4 组场景中的 7 对分数均满足
1% 门槛，最大相对误差为 0.189%。三候选按 YAML 输入顺序的下标排序为 `[1, 2, 0]`，
6 组非法输入及拒绝后的恢复检查均通过。CPU FP32 可作为排查时的额外参考。

## Known Limitations

- 当前分支已有 pooling/cross-encoder 执行路径，本任务不增加模型专用算子或环境变量。
- 8192 是示例运行上限，计算长度时需计入 query、document 和特殊 token。提前截断文本会
  改变模型输入和分数。
- CPU 静态检查不能证明 NPU 运行正确；最终验收必须记录真实权重、硬件、CANN、
  PyTorch、torch-npu、vLLM 和 vLLM Ascend 版本。
