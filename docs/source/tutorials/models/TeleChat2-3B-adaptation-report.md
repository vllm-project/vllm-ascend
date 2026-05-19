# TeleChat2-3B Ascend NPU 适配报告

## 基本信息

| 项目 | 内容 |
|------|------|
| 模型 | TeleAI/TeleChat2-3B |
| checkpoint 路径 | `/home/cmq/cache/modelscope/models/TeleAI/TeleChat2-3B` |
| served-model-name | `TeleAI/TeleChat2-3B` |
| TP size | 2 |
| vLLM 源码 | `/home/cmq/code/vllm-cpu/vllm`（v0.17.1rc1.dev175） |
| vllm-ascend 源码 | `/home/cmq/code/vllm-ascend` |
| 适配日期 | 2026-04-02 |
| Commit | `e94a71855` (branch: `model_adaption_skill`) |

---

## 模型分析

### 架构

- **类型**：LLM（纯语言模型，非 VLM / ASR / MoE）
- **架构类**：`TeleChatForCausalLM`，继承自 `LlamaForCausalLM`
- **注意力机制**：标准全注意力（MHA），无 GQA
- **上下文长度**：32768（训练），本次验证使用 8192
- **位置编码**：RoPE + NTK 动态扩展
- **词表大小**：131072
- **量化**：无（BF16）
- **多模态**：无

### 算子分析（Step 3）

| 算子类型 | 结论 |
|----------|------|
| 全部为 Torch 原生算子 | ✅ Ascend 兼容，无 CUDA/Triton 专属 kernel |

无 CUDA operator early-exit 触发，无 Triton operator 验证需求。

### 框架侧代码分析（Step 4）

- `TeleChatForCausalLM` 已在上游 vllm `registry.py` 中注册
- 无新增 vllm 框架模块变更（scheduler / attention backend / sampler / weight loader 均无改动）
- vllm-ascend 无需新增 patch 或 override

**结论：无需任何 vllm 或 vllm-ascend 代码变更即可支持该模型。**

---

## 适配策略（Step 5）

复用上游 vllm 已有 `telechat2` 架构适配器，无需新增 modeling 文件或 registry 注册。

---

## 代码变更

### vllm-ascend（交付仓库）

无模型适配代码变更。交付内容为文档与测试配置：

| 文件 | 类型 |
|------|------|
| `docs/source/tutorials/models/TeleChat2-3B.md` | 新增 |
| `tests/e2e/models/configs/TeleChat2-3B.yaml` | 新增 |
| `docs/source/tutorials/models/index.md` | 更新 |

### vllm 上游（本地修复，需提 PR）

| 文件 | 变更 | 说明 |
|------|------|------|
| `vllm/v1/worker/gpu/block_table.py` | `+import numpy as np` | 缺失 import，导致 worker subprocess 启动失败 |

---

## 中间单元测试（Step 6.5）

本模型无新算子，无框架模块变更，跳过专项 NPU 单元测试。直接进入两阶段验证。

---

## 两阶段验证结果

### Stage A：dummy 权重（快速门控）

```bash
OMP_NUM_THREADS=1 python /tmp/telechat2_offline_test.py
```

| 项目 | 结果 |
|------|------|
| EngineCore 启动 | ✅ |
| TP=2 workers 初始化 | ✅（NPU 0 & NPU 1）|
| KV cache | 51.12 GiB，372,096 tokens |
| 推理请求 | ✅ 非空输出 |
| 输出样例 | `' comfortably comfortably comfortably ...'`（dummy 权重，符合预期）|

### Stage B：真实权重（必过门控）

```bash
OMP_NUM_THREADS=1 python /tmp/telechat2_real_test.py
```

| 项目 | 结果 |
|------|------|
| 权重加载 | ✅ 14.84s，2.8659 GB/worker（4 shards）|
| KV cache | 51.12 GiB，372,224 tokens |
| 推理请求 | ✅ HTTP 200，非空中文输出 |
| 输出样例 | `'你好，很高兴能和你聊天。我是李华，是一名大学生，专业是计算机科学。我平时喜欢参加各种科技竞赛和参加编程俱乐部，同时'` |
| 输入速度 | 3.14 tok/s |
| 输出速度 | 16.74 tok/s |

---

## 特性矩阵

| 特性 | 状态 | 说明 |
|------|------|------|
| 基础推理（eager） | ✅ 通过 | Stage A + B 均验证 |
| ACLGraph | ⚠️ 未验证 | enforce_eager=True，图编译模式待开启后验证 |
| Expert Parallel (EP) | N/A | 非 MoE 模型 |
| flashcomm1 | N/A | 非 MoE 模型 |
| MTP | N/A | checkpoint 不含 MTP 权重 |
| 多模态 | N/A | 纯语言模型 |

---

## 已知问题

### 1. OMP_NUM_THREADS 必须提前设置

**现象**：未设置 `OMP_NUM_THREADS` 时，worker subprocess 在 `multiproc_executor.py:1014` 调用 `torch.set_num_threads(1)` 失败：

```
INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/ParallelOpenMP.cpp":64
Invalid thread pool!
```

**根因**：子进程 fork 后尝试修改 OpenMP 线程池，该内核版本不支持此操作。

**解决方案**：启动前设置 `OMP_NUM_THREADS=1`（或任意值），使线程池在 fork 前已完成初始化。

**上游状态**：vllm 上游 bug，建议提 issue。

### 2. vllm block_table.py 缺失 numpy import

**文件**：`vllm/v1/worker/gpu/block_table.py`

**现象**：`NameError: name 'np' is not defined`

**修复**：添加 `import numpy as np`（已本地修复）

**上游状态**：需向 vllm 上游提 PR。

---

## 启动命令参考

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
