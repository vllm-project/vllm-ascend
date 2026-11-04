# DeepSeek-R1-Distill-Llama-70B

## 模型介绍

[DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B)
使用 `LlamaForCausalLM` 架构，可复用 vLLM 的 Llama 实现。本文面向原始 BF16
权重的低并发部署：4 张 Atlas A2 64 GB NPU，单机 TP=4、DP=1、PP=1。
权重约 141 GB，运行时还需要算子工作区与 KV cache 空间。

该 checkpoint 的旧版 `LlamaTokenizerFast` 声明在 Transformers 5 中会选择
SentencePiece 风格的分词器，覆盖原始 ByteLevel 编解码设置，造成空格、换行及中文异常。
使用 `--tokenizer-mode deepseek_r1_distill_llama` 显式加载 checkpoint 中序列化的
分词器，保留其编解码规则及特殊 token。

## 功能范围

| 功能 | 本次验证范围 |
| --- | --- |
| BF16、TP=4、文本推理 | 原始完整权重，数学、英文及中文请求通过 |
| ACLGraph | 完整 80 层真实权重，图捕获与推理通过 |
| Eager | 图模式出现问题时的隔离路径 |
| EP、flashcomm1 | 不适用：本模型为 dense 架构 |
| 多模态 | 不适用：本模型仅接受文本 |
| MTP | checkpoint 未提供：配置及权重索引均无相应预测层 |

模型配置的最大上下文为 131072 tokens。本次实测覆盖 8192 tokens 上下文边界和双请求并发。

## 环境准备

使用与开发分支配套的 vLLM、CANN、torch-npu 和 Triton Ascend，参考
[安装指南](../../getting_started/installation.md)。基础 CANN 镜像需补齐推理依赖及对应硬件的 CANN Ops 包。

Docker 环境应选择对应硬件的镜像；已有基础容器可以补齐依赖后安装源码。

=== "Atlas A2"

    按安装指南准备 Atlas A2 Docker 环境，并安装要验证的 vLLM Ascend 源码提交。
    本次实际使用已有的 openEuler 24.03 容器，补齐依赖后进行 editable 安装。

=== "Atlas A3"

    A3 部署需使用对应硬件镜像，并在目标设备上运行下述验证流程；本次实测设备为 A2。

按安装指南映射 NPU 设备及驱动目录，并将模型目录挂载到容器中的 `/data/models`。
本次真实权重验收环境如下：

| 组件 | 版本 |
| --- | --- |
| Python | 3.11.15 |
| CANN Toolkit / 910B Ops / NNAL | 9.2.1 |
| vLLM | 0.32.0，源码安装 |
| PyTorch / torch-npu | 2.10.0+cpu / 2.10.0.post7 |
| Triton Ascend | 3.6.0.post2 |
| Transformers | 5.19.0 |
| FLA NPU A2 | 26.9.1 |
| lm-eval | 0.4.12 |

主线工作进程初始化会导入 FLA NPU，即使目标为 Llama 也需安装该依赖。
若镜像站链接失效，可从
[FLA NPU 官方发布](https://github.com/flashserve/flash-linear-attention-npu/releases/tag/v26.9.1-release)
获取匹配架构的安装包，并核对发布资产的校验摘要。

下载前先验证设备和通信：

```bash
npu-smi info
npu-smi info -t topo
python - <<'PY'
import torch
import torch_npu

assert torch.npu.device_count() >= 4
result = torch.ones(2, device="npu:0") + 1
torch.testing.assert_close(result.cpu(), torch.tensor([2.0, 2.0]))
print("NPU operator check passed")
PY
```

另行运行四 rank HCCL collective 测试。若分配的设备跨通信平面，需要确认实例支持
机内 RoCE；本次实例设置 `HCCL_INTRA_ROCE_ENABLE=1` 后 all-reduce 通过。
该设置根据实例通信拓扑确定。

```bash
pip install modelscope
modelscope download \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
    --local_dir /data/models/DeepSeek-R1-Distill-Llama-70B
```

持久化磁盘需同时容纳权重、临时下载文件及编译环境。应先确认下载完整，再启动真实权重服务。

## 部署

以下是已验证的低并发启动配置：

```bash
cd /workspace
vllm serve /data/models/DeepSeek-R1-Distill-Llama-70B \
    --served-model-name deepseek-r1-distill-llama-70b \
    --tokenizer-mode deepseek_r1_distill_llama \
    --dtype bfloat16 \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 1 \
    --data-parallel-size 1 \
    --max-model-len 8192 \
    --max-num-seqs 2 \
    --gpu-memory-utilization 0.9 \
    --host 127.0.0.1 \
    --port 8000
```

图模式出错时可增加 `--enforce-eager` 进行隔离，并保留失败日志。
下载期间可使用 `--load-format dummy` 检查完整模型结构和接口；下载完成后移除此参数，运行真实权重验证。

## 功能验证

```bash
curl --fail http://127.0.0.1:8000/v1/models
curl --fail http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "deepseek-r1-distill-llama-70b",
      "messages": [{"role": "user", "content": "What is 2 + 3?"}],
      "temperature": 0,
      "max_tokens": 512
    }'
```

两个接口都应返回 HTTP 200，生成请求应包含可读且符合提示的文本。
输入和输出总长度应不超过 8192 tokens。本次数学、英文及中文请求均通过。
上下文边界测试中，8064 输入＋128 输出和 8191 输入＋1 输出均成功；
8192 输入＋1 输出及 8193 输入＋1 输出均返回预期的 HTTP 400。
两个 8064 输入＋128 输出请求同时执行也均成功，整组耗时 13.381 秒。

## 准确率评测

使用仓库配置 `tests/e2e/models/configs/DeepSeek-R1-Distill-Llama-70B.yaml`：
CEval accountant 的完整 val 集 49 题、dev 集 5-shot、无聊天模板、batch size 1。
该测试通过比较各选项的条件对数似然选择答案。

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn HF_ENDPOINT=https://hf-mirror.com python -m pytest -sv \
    tests/e2e/models -k "lm_eval_correctness or distill_llama_preserves_serialized_tokenizer" \
    --ignore=tests/e2e/models/test_asr_eval_correctness.py \
    --ignore=tests/e2e/models/test_rm_eval_correctness.py \
    --config tests/e2e/models/configs/DeepSeek-R1-Distill-Llama-70B.yaml \
    --tp-size 4
```

该命令选择文本准确率和分词器用例，跳过 ASR 与奖励模型模块的收集。
使用本地权重时，在配置副本中将 `model_name` 改为已完整下载的目录，避免重复下载。
参考值来自上述评测设置，正式 pytest 复测结果一致。Greedy 输出另按下一节与 Transformers 参考实现对齐。

| 任务 | 样本数 | Few-shot | 指标 | 实测值 |
| --- | --- | --- | --- | --- |
| ceval-valid_accountant | 49 | 5 | acc,none | 0.5510（27/49）|

## Greedy 输出与参考实现对齐

参考脚本 `tests/e2e/models/distill_llama_reference.py` 使用相同 BF16 权重、聊天模板和
输入 token，依次运行 Transformers 与 vLLM。Transformers 使用 eager attention，
通过 Accelerate 1.15.0 将 80 层模型按每卡 20 层分布在四张 NPU；vLLM 使用
TP=4、DP=1、PP=1 和 ACLGraph。每次只运行一个引擎。

在已配置的四卡环境中执行：

```bash
MODEL_DIR=/data/models/DeepSeek-R1-Distill-Llama-70B
RESULT_DIR=/data/distill-llama-reference
for phase in hf graph teacher check; do
    VLLM_WORKER_MULTIPROC_METHOD=spawn python tests/e2e/models/distill_llama_reference.py \
        "$phase" --model "$MODEL_DIR" --output-dir "$RESULT_DIR"
done
```

脚本保存提示词、输入 token、32-token greedy 输出、top-20 logprob、完整答案及
`comparison.json`。`teacher` 阶段将 Transformers 生成的 token 接在原始输入后，
取得相同前缀下的 vLLM 概率，避免分支差异影响后续数值比较。
概率绝对误差阈值沿用 `tests/e2e/pull_request/utils.py::compare_logprobs`：
prefill 为 0.0689，decode 为 0.1378。

| 提示词 | 输入 tokens | 对齐范围 | 完整答案检查 |
| --- | --- | --- | --- |
| 英文短提示 | 18 | 32-token greedy 与同前缀概率 | — |
| 中文短提示 | 18 | 32-token greedy 与同前缀概率 | — |
| 分步算术 reasoning | 44 | 32-token greedy 与同前缀概率 | 最终答案为 7 |
| 长文本检索 | 2558 | 32-token greedy 与同前缀概率 | 提取 MAPLE-742 |

实测中文短提示、reasoning 和长文本的 32-token greedy 输出完全一致。
英文短提示在第 18 个 token 出现等概率候选分支；首次分歧的概率误差为 0.06218。
四类提示词的同前缀概率检查共覆盖 128 个 token，prefill 最大误差 0.03409，
decode 最大误差 0.11663，均在上述容差内。Reasoning 和长文本完整回答均正常结束，
最终答案分别为 7 和 MAPLE-742。

## 性能与稳定性

使用自定义流式 HTTP 基准脚本，在服务器回环地址发送精确 token ID 输入，
设置 temperature=0、ignore_eos=true 和固定 128-token 输出。每个请求使用唯一前缀，
每组预热 2 次后测量 20 次。四组共 80 个请求全部成功。

| 输入／输出 tokens | 并发 | TTFT P50/P95（ms）| TPOT P50/P95（ms/token）| 总输出吞吐（tokens/s）|
| --- | --- | --- | --- | --- |
| 512/128 | 1 | 228.5/234.0 | 57.28/57.40 | 17.05 |
| 512/128 | 2 | 390.7/496.7 | 59.34/59.89 | 32.25 |
| 2048/128 | 1 | 685.2/686.9 | 56.65/56.65 | 16.25 |
| 2048/128 | 2 | 845.1/1420.3 | 64.51/64.54 | 28.33 |

TTFT 从发起请求计时至首个非空文本事件。TPOT 按“首个文本事件至流结束的时间／
（输出 token 数−1）”计算，包含流结束开销；总输出吞吐按整组输出 token 数除以
整组耗时计算。P95 基于每组 20 个样本。

持续双并发测试采用 512 输入／128 输出，运行 30.02 分钟，454/454 请求成功，
总输出吞吐为 32.26 tokens/s。TTFT P50/P95 为 374.0/494.9 ms，
TPOT P50/P95 为 58.78/59.91 ms/token。每 30 秒采样的四卡 HBM 占用无持续增长。

上述结果构成 Atlas A2、BF16、TP=4、8192 上下文、并发 1–2 配置的性能基线。

## 故障排查

| 现象 | 检查与处理 |
| --- | --- |
| 空格、换行或中文异常，输出出现 `Ġ`、`Ċ` | 确认使用 `--tokenizer-mode deepseek_r1_distill_llama`，并检查运行时加载的 vLLM Ascend 源码版本 |
| NPU 可枚举，但基础运算报算子缺失 | 核对 CANN Toolkit、910B Ops 及对应环境初始化脚本，先运行基础算子检查 |
| 四卡初始化超时或 HCCL 报错 | 使用 `npu-smi info -t topo` 检查设备拓扑，先验证四 rank all-reduce；跨平面实例需确认机内 RoCE 配置 |
| pytest 报 NPU 无法在 fork 子进程重复初始化 | 设置 `VLLM_WORKER_MULTIPROC_METHOD=spawn` 后重新运行 |
| 加载或推理 OOM | 确认四张 64 GiB 卡可用且无其他模型进程；从本文 BF16、TP=4、8192 上下文、2 个序列配置开始，结合空闲显存调整序列数和缓存预算 |

环境与验收记录应包含 NPU 型号和卡数、TP/DP/PP、dtype、上下文与并发上限，以及
vLLM、vLLM Ascend、CANN、PyTorch、torch-npu、Transformers 的版本。
