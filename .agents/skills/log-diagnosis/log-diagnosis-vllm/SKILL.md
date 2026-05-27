---
name: log-diagnosis-vllm
description: vLLM 日志诊断调度器。接收 vLLM 相关日志，自动判断问题方向并调度对应子skill进行诊断。触发词：vllm日志、vllm诊断、vllm日志分析、vllm推理、vllm问题定位、vllm故障。
---

# vLLM 日志诊断调度器

接收 vLLM 相关日志，自动判断方向，分发给对应子 skill 进行诊断，并汇总输出结论。

---

## 使用方式

**方式一：直接调度（用户已告知方向）**
```
/skill log-diagnosis-vllm
[粘贴日志]
> 推理超时问题
```

**方式二：全自动（用户不知道什么方向）**
```
/skill log-diagnosis-vllm
[粘贴日志]
> 帮我看看是什么问题
```

---

## 诊断流程

```
Step 1：解析输入
Step 2：判断方向
Step 3：加载子 skill 并执行
Step 4：汇总输出
```

### Step 1：解析输入

同时解析两个输入：
- **原始日志**：用户粘贴的完整或部分日志
- **用户提示**（可选）：用户对可能方向的猜测

### Step 2：判断方向

三种路径，优先级从高到低：

| 路径 | 条件 | 处理方式 |
|------|------|----------|
| **直接调度** | 用户主动提示方向 | 根据提示直接加载对应子 skill |
| **错误特征分析** | 日志中有 error/warn 日志 | 分析错误语义，映射到对应方向 |
| **零样本分类** | 以上都没有命中 | LLM 根据日志特征判断最可能的方向 |

**错误特征 → 方向 映射表：**

| 错误特征 | 映射方向 | 对应子 skill |
|----------|----------|--------------|
| RPC timed out / sample_tokens timed out | 推理超时 | `log-diagnosis-vllm-inference-timeout` |
| EngineDeadError / EngineCore fatal error | 引擎崩溃 | `log-diagnosis-vllm-inference-timeout` |
| 500 Internal Server Error + 超时 | 推理超时 | `log-diagnosis-vllm-inference-timeout` |
| 模型加载失败 / weight loading failed / checkpoint error | 模型初始化 | `log-diagnosis-vllm-model-init`（待创建）|
| OOM / out of memory / NPU memory | NPU 内存问题 | `log-diagnosis-vllm-npu-oom`（待创建）|
| NCCL / nccl error / 通信失败 | 多节点通信 | `log-diagnosis-vllm-multi-node`（待创建）|
| CUDA error / GPU error | GPU/NPU 硬件问题 | 结合系统日志 |

**零样本分类 prompt（备用）：**

```
你是一个 vLLM 推理服务的日志诊断专家。以下是一段服务日志，请判断最可能属于哪类问题：

日志：
---
{raw_logs}
---

选项：
A. 推理超时问题（RPC 超时、sample_tokens 超时、引擎卡住）
B. 模型初始化问题（权重加载失败、模型加载超时）
C. NPU/GPU 内存问题（OOM、内存不足）
D. 多节点通信问题（NCCL 错误、节点间通信失败）
E. 其他 vLLM 异常

请只输出选项字母和简短理由（1-2句话）。
```

### Step 3：加载子 skill 并执行

根据 Step 2 的判断结果，加载对应子 skill：

```
if 方向 == "推理超时" or "引擎崩溃":
    skill_view(name="log-diagnosis-vllm-inference-timeout")
    执行子 skill 的诊断流程
elif 方向 == "模型初始化":
    skill_view(name="log-diagnosis-vllm-model-init")
    执行子 skill 的诊断流程（待创建）
elif 方向 == "无法判断" or 命中多个方向:
    尝试所有相关子 skill，汇总各自发现
```

### Step 4：汇总输出

统一格式输出：

```
## vLLM 日志诊断结论

**判断方向**：xxx
**诊断依据**：...
**根因定位**：...
**修复建议**：...

---
## 详细诊断过程

[子 skill 的完整输出]

---
## 下一步

1. ...
2. ...
```

---

## 子 skill 清单

| 子 skill | 状态 | 说明 |
|----------|------|------|
| `log-diagnosis-vllm-inference-timeout` | ✅ 已就绪 | 推理超时/引擎崩溃诊断 |
| `log-diagnosis-vllm-model-init` | ⏳ 待创建 | 模型加载/初始化失败 |
| `log-diagnosis-vllm-npu-oom` | ⏳ 待创建 | NPU 内存不足 |
| `log-diagnosis-vllm-multi-node` | ⏳ 待创建 | 多节点通信问题 |

---

## 扩展方式

新增诊断方向时：

1. 在 `log-diagnosis-vllm/` 下创建子 skill 目录
2. 在上方「错误特征 → 方向映射表」中添加对应条目
3. 在「子 skill 清单」中更新状态

详见 `log-diagnosis-framework` 框架规范。
