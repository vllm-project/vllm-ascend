---
name: log-diagnosis-vllm-inference-timeout
description: vLLM 推理超时异常的日志诊断。当 vLLM 引擎在推理过程中出现 RPC 超时、EngineDeadError、sample_tokens 超时等问题时，通过日志定位根因。触发词：vllm超时、推理超时、EngineDeadError、RPC timed out、sample_tokens timed out、vllm日志分析、vllm推理异常。
---

# vLLM 推理超时问题日志诊断

## 问题描述

**vLLM 推理超时**是指在 vLLM 多节点或单节点推理过程中，引擎核心进程（EngineCore）因 RPC 调用超时、模型 forward 卡住等原因导致服务不可用的问题。

**涉及组件/文件：**
- `vllm/v1/engine/core.py` — 引擎核心，run_busy_loop
- `vllm/v1/executor/multiproc_executor.py` — 多进程执行器，RPC 调用
- `vllm/distributed/device_communicators/shm_broadcast.py` — 共享内存通信
- `vllm/v1/engine/exceptions.py` — EngineDeadError

**典型异常表现：**
- `TimeoutError: RPC call to sample_tokens timed out`
- `vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue`
- ApiServer 全部返回 500 Internal Server Error
- 多节点场景下部分 worker 进程集体挂掉

**关联问题：**
- 可能是 NPU 资源耗尽导致 forward 卡住
- 可能是共享内存通信故障
- 可能是模型加载/初始化异常的前兆

---

## 诊断入口日志

| 入口日志 | 组件 | 含义 | grep 命令 |
|----------|------|------|-----------|
| `TimeoutError: RPC call to sample_tokens timed out` | multiproc_executor | worker RPC 超时 | `grep "RPC call to.*timed out" log` |
| `EngineDeadError` | async_llm | 引擎核心崩溃 | `grep "EngineDeadError" log` |
| `EngineCore encountered a fatal error` | engine core | 引擎遇到致命错误 | `grep "fatal error" log` |
| `500 Internal Server Error` | ApiServer | 推理请求全部失败 | `grep "500 Internal Server Error" log` |

---

## 诊断决策树

```
[发现 TimeoutError: RPC call to sample_tokens timed out]
├── 是 → 检查 dequeue 超时的具体位置
│        ├── shm_broadcast.py dequeue 超时 → 共享内存通信问题
│        │   ├── 检查是否多节点环境
│        │   ├── 检查 SHM 大小配置
│        │   └── 检查 NPU 通信是否正常
│        └── 多节点 RPC 超时 → 网络或 NPU 资源问题
│            ├── 检查模型 forward 是否卡住
│            ├── 检查 NPU 利用率
│            └── 检查是否有 OOM
└── 否 → 检查其他 EngineDeadError 原因
         ├── 模型加载失败 → log-diagnosis-vllm-model-init
         └── 其他 → 汇总输出
```

---

## 诊断执行流程

### 步骤 1：确认超时类型

执行以下检查，收集证据：

| 检查项 | 命令 | 命中条件 |
|--------|------|----------|
| 超时的 RPC 方法 | `grep "RPC call to.*timed out"` | 记录方法名（如 sample_tokens）|
| dequeue 超时 | `grep "shm_broadcast.*dequeue\|acquire_read"` | 确认是共享内存通信超时 |
| 引擎致命错误 | `grep "EngineCore encountered a fatal error"` | 确认引擎核心崩溃 |
| 请求级联失败 | `grep "500 Internal Server Error"` | 确认所有 ApiServer 都失败 |

### 步骤 2：定位超时根因

根据步骤1的证据，选择对应分支：

**分支A：dequeue 超时（共享内存问题）**
- 证据：`shm_broadcast.py dequeue` 超时
- 继续查：
  - 多节点？`grep "multi-node\|TP\|tensor_parallel"`
  - SHM 大小？`grep "shm\|shared_memory\|HBM"`
  - NPU 通信？`grep "NCCL\|hccl\|communication"`

**分支B：RPC 超时但无 dequeue 超时（模型计算卡住）**
- 证据：超时发生在 model forward 阶段
- 继续查：
  - NPU 内存？`grep -E "OOM|out of memory|NPU.*memory"`
  - 计算阻塞？`grep -E "blocked|stuck|waiting.*lock"`

**分支C：EngineDeadError 但无明确超时（进程异常退出）**
- 证据：引擎进程直接崩溃
- 继续查：
  - 核心转储？`grep "core\|signal\|SIGSEGV"`
  - 内存越界？结合系统 dmesg

### 步骤 3：输出诊断报告

按以下格式输出：

```markdown
## vLLM 推理超时诊断报告

### 诊断结论
**方向**: 推理超时
**根因**: [一句话描述根因]

### 证据链
| # | 日志位置 | 日志内容 | 推导结论 |
|---|----------|----------|----------|
| 1 | 行6489 | `TimeoutError` at shm_broadcast.py:631 | dequeue 等待超时 |
| 2 | 行6518 | `RPC call to sample_tokens timed out` | worker 进程卡在 sample_tokens |
| 3 | 行6520 | `EngineCore encountered a fatal error` | 引擎核心进程崩溃 |

### 根因分析
[2-3句话分析：为什么这些证据指向这个根因]

### 下一步行动
| 优先级 | 方向 | 具体操作 | 可信度 |
|--------|------|----------|--------|
| P0 | 验证根因 | [检查命令或配置] | 高/中/低 |
| P1 | 继续排查 | [如果可信度不够高，还需要查什么] | 中/低 |

### 修复建议
[如果根因确定，给出具体修复步骤或配置修改建议]
```

---

## 快速诊断表

| 日志现象 | 根因 | 修复方法 | 优先级 |
|----------|------|----------|--------|
| sample_tokens 超时 + dequeue 超时 | 共享内存通信故障 | 检查 SHM 大小配置，增加 `enforce_eager` | P1 |
| sample_tokens 超时 + NPU OOM | NPU 内存不足 | 减小 batch_size，减少 max_num_seqs | P0 |
| sample_tokens 超时 + forward 卡住 | 模型计算超时 | 检查 NPU 利用率和通信 | P1 |
| EngineDeadError + 多节点 | 节点间通信失败 | 检查网络配置和 NCCL | P0 |

---

## 关联问题

| 关联问题 | 关联方向 | 对应 Skill |
|----------|----------|------------|
| vLLM 模型加载异常 | 上游可能 | `log-diagnosis-vllm-model-init`（待创建） |
| NPU 资源耗尽 | 资源层 | 需结合系统日志 |

---

## 扩展方向

本 skill 覆盖 "推理超时" 这一个具体场景。如需扩展，可按以下方向新增：
- `log-diagnosis-vllm-model-init` — 模型加载/初始化失败
- `log-diagnosis-vllm-engine-dead` — EngineDeadError 其他原因
- `log-diagnosis-vllm-npu-oom` — NPU 内存不足
- `log-diagnosis-vllm-multi-node` — 多节点通信问题
