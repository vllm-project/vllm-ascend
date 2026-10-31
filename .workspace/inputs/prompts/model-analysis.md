# 模型 NPU 适配分析 Prompt 模板

## 使用方式

复制此模板，替换 `{MODEL_NAME}`、`{MODEL_PATH}` 等占位符后，作为 agent 的输入。

---

## 任务

分析 {MODEL_NAME} 在 vLLM + vLLM-Ascend 上的 NPU 适配实现。

## 分析维度

### 1. 模型架构

- 注意力类型：MLA / GQA / MHA / Linear Attention？
- MoE：有无专家并行，Router 机制？
- 特殊 Norm：TP 感知的 RMSNorm / LayerNorm？
- 位置编码：RoPE / ALiBi / YaRN？

### 2. 上游 vLLM 实现

- 模型文件路径：`vllm/model_executor/models/{model_file}.py`
- 核心类：Attention、MoE、DecoderLayer
- 与标准 Transformer 的差异点

### 3. vLLM-Ascend NPU 适配

- Patch 位置：`vllm_ascend/patch/worker/` 和 `vllm_ascend/patch/platform/`
- 算子替换：哪些 GPU 算子被替换为 NPU 算子？
- 融合优化：有哪些 kernel fusion？
- 量化处理：FP8 checkpoint 如何处理？

### 4. 关键代码路径

- 标注公式与代码的对应关系
- 标注 NPU 特有路径 vs 通用路径的分支条件
- 标注通信优化（all_reduce、TP、EP）

### 5. 对比分析

- 与同类型模型的实现差异
- NPU vs CUDA 路径差异
- MindIE / SGLang / vLLM-Ascend 三方差异（如适用）

## 输出格式

将分析结果写入 `.workspace/outputs/model-analysis/{model_name}-npu.md`，包含：
- 架构图（ASCII）
- 核心代码片段 + 注释
- 差异对比表
