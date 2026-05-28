# 基于XXX模型部署教程的技术文档模板

<p align="center">
  <a href="Model-Deployment-Tutorial-Template.md"><b>English</b></a> | <a href="Model-Deployment-Tutorial-Template.zh.md"><b>中文</b></a>
</p>

本模板基于DeepSeek-V3.2、Qwen-VL-Dense等部署教程，旨在为技术文档撰写提供参考。使用者可遵循模板指引，系统性完成相关技术文档的构建工作。

## 1 简介（Introduction）

**资料写作要求：**

- 一句话介绍模型的基本架构、核心特性及主要应用场景。
- 一句话写清楚文档要干什么，要达成的目的。
- 说明文档使用的vLLM-Ascend版本及模型的版本支持情况。

**示例1：模型介绍**  

DeepSeek-V3.2 是一种稀疏注意力模型。其主要架构与 DeepSeek-V3.1 类似，但采用了稀疏注意力机制，旨在探索和验证在长上下文场景下训练和推理效率的优化方案。

**示例2：文档目的**  

本文档将展示模型的主要验证步骤，包括支持的功能、功能配置、环境准备、单节点和多节点部署、准确性和性能评估。

**示例3：版本信息**  

本文档基于 **vLLM-Ascend v0.13.0** 版本进行验证和编写。当前模型（XXX）在该版本中已完整支持，**v0.13.0 及更高版本**均可稳定运行。如需使用最新特性（如PD分离、MTP等），建议使用v0.13.0或以上版本。

## 2 支持的特性（Supported Features）

介绍该模型支持的特性，包括支持的硬件、量化方式、数据并行、长序列特性等。

**资料写作要求：**  

- 采用表格形式，呈现模型和特性的支持情况。
- 或提供可跳转的交叉引用（推荐）。

**示例1：特性支持列表**  

| 模型名称 | 支持状态 | 备注 | BF16 | 支持的硬件 | W8A8 | 分块预填充 | 自动前缀缓存 | LoRA | 推测解码 | 异步调度 | 张量并行 | 流水线并行 | 专家并行 | 数据并行 | Prefill-Decode分离 | 分段式ACL图执行 | 整图ACL图执行 | 最大模型长度 | MLP权重预取 | 文档 |
| ------ | ---------- | ------ | ------ | ---------- | ------ | ------------ | -------------- | ------ | ---------- | ---------- | ---------- | ------------ | ---------- | ---------- | ------------------- |----------- | ----------- | ------------- | ------------- | ---------- |
| DeepSeek V3/3.1 | ✅ | | ✅ | Atlas 800I A2:<br>最低卡数要求为xx | ✅ | ✅ | ✅ | | ✅ | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 240k | | [DeepSeek-V3.1](../../tutorials/models/DeepSeek-V3.1.md) |
| DeepSeek V3.2 | ✅ | | ✅ | Atlas 800I A2:<br>最低卡数要求为xx | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 160k | ✅ | [DeepSeek-V3.2](../../tutorials/models/DeepSeek-V3.2.md)|
| Qwen3 | ✅ | | ✅ | Atlas 800I A2:<br>最低卡数要求为xx | ✅ | ✅ | ✅ | | | ✅ | ✅ | | | ✅ | | ✅ | ✅ | 128k | ✅ | [Qwen3-Dense](../../tutorials/models/Qwen3-Dense.md) |

>**注意**：此为简化示例，完整表格请参考完整特性矩阵。

**示例2：引用**  

请参考[支持的功能列表](../user_guide/support_matrix/supported_models.md)，获取模型支持的功能矩阵。

请参考[特性指南](../user_guide/feature_guide/index.md)获取功能配置信息。

## 3 前置准备（Prerequisites）

### 3.1 模型权重（Model Weight）

**资料写作要求：**  说明部署所需的硬件资源、软件环境和模型文件。

**示例：**

- `DeepSeek-V3.2-Exp-W8A8`（量化版）：需要 1 台 Atlas 800 A3（64G × 16）节点或 2 台 Atlas 800 A2（64G × 8）节点。 [模型权重](https://www.modelscope.cn/models/vllm-ascend/DeepSeek-V3.2-Exp-W8A8)
- `DeepSeek-V3.2-w8a8`（量化版）：需要 1 台 Atlas 800 A3（64G × 16）节点或 2 台 Atlas 800 A2（64G × 8）节点。 [模型权重](https://www.modelscope.cn/models/vllm-ascend/DeepSeek-V3.2-W8A8/)

建议将模型权重下载至多节点共享目录。

### 3.2 验证多节点通信（可选）（Verify Multi-node Communication(Optional)）

**示例：**

若需部署多节点环境，请依据[验证多节点通信环境](../installation.md#verify-multi-node-communication)指南进行通信验证。

## 4 安装（Installation）

**资料写作要求：**

- 提供具体的安装步骤与命令（参数需解释含义、取值范围、单位等）。
- 版本号书写规范：优先使用占位符（值统一配置）；若使用固定值且该值与文档验证版本不一致，须加注释“请按实际版本替换”。
- 提供验证命令及预期状态：指导用户通过执行命令（如 docker ps）检查安装结果，说明成功时的状态码或输出特征。

### 4.1 Docker镜像安装

**示例：**  略

### 4.2 源码安装

**示例：**  略

## 5 在线服务化部署（Online service deployment）

### 5.1 单机在线部署

**资料写作要求：**

- 说明单机部署的架构特点与适用场景
- 提供启动命令模板和关键参数说明
- 提供服务验证方法（如 curl 命令）及预期结果，说明成功特征（如 200 OK）。
- 在启动命令下方提供常见问题指引，如公共FAQ中已有描述可直接链接呈现。

**示例：**

单机部署将Prefill与Decode在同一节点内完成，适用于XXX场景。

启动命令：

```bash
# 略
```

常见问题提示：如遇xxx问题，请参考[公共FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html)进行检查。

服务验证：

```bash
# 略
```

预期结果：略（按实际输出书写即可）。

### 5.2 多机PD分离部署

**资料写作要求：**

- 说明PD分离架构的原理与适用场景。
- 提供启动流程、关键配置及**部署验证说明**。
- 注明性能指标。
- 在启动命令下方提供常见问题指引，如公共FAQ中已有描述可直接链接呈现。

**示例：** 略

### 5.3 特殊部署形态（可选）

**资料写作要求：**

- 若模型存在非标准部署形态（如embedding模型的离线批处理、reranker模型的低延迟在线服务等），需在文档中明确体现对应部署方案。
- 可参考本章5.1和5.2节进行扩展。

## 6 功能验证（Functional Verification）

**资料写作要求：**

- 指导用户如何在服务启动后，通过简单接口测试模型的基本功能是否正常。
- 提供预期结果，说明成功特征（如 HTTP 200、返回包含 choices 字段的 JSON）。

**示例：**

服务启动后，即可通过发送提示词来调用模型：

```shell
       curl http://<node0_ip>:<port>/v1/completions \
           -H "Content-Type: application/json" \
           -d '{
               "model": "deepseek_v3.2",
               "prompt": "The future of AI is",
               "max_tokens": 50,
               "temperature": 0
           }'
```

预期结果：略（按实际输出书写即可）。

## 7 精度评估（Accuracy Evaluation）

**资料写作要求：** 介绍评估模型输出质量（精度）的标准化方法及工具，以下提供两种精度评估方法作为示例；或直接链接现有文档进行呈现。

### AISBench的使用

详情请参考[Using AISBench](../developer_guide/evaluation/using_ais_bench.md)。  

### Language Model Evaluation Harness的使用

以`gsm8k`数据集作为测试数据集为例，在线模式下运行`DeepSeek-V3.2-W8A8`的精度评估。

1. `lm_eval`安装请参考[Using lm_eval](../developer_guide/evaluation/using_lm_eval.md)。
2. 运行`lm_eval`执行精度评估。

```shell
lm_eval \
  --model local-completions \
  --model_args model=/root/.cache/Eco-Tech/DeepSeek-V3.2-w8a8-mtp-QuaRot,base_url=http://127.0.0.1:8000/v1/completions,tokenized_requests=False,trust_remote_code=True \
  --tasks gsm8k \
  --output_path ./
```

## 8 性能（Performance）

略，要求同精度评估

## 9 最佳实践

**资料写作要求：**

每个模型给三个场景的推荐配置（长序列、低时延、高吞吐），可以跑出最佳性能的，但是不提供具体的性能数据。

**示例：**

### 最佳实践配置参考

### 表1：场景概览

| 场景 | 部署形态 | *总卡数 | 权重版本 | 优化思路 |
|------|------|---------|----------|----------|
| 高吞吐<br>(32K推1K) | 1P1D部署 | 16（A3） | glm5.1w4a8 | 短序列高吞吐情况下，尝试调整xxx参数 |
| 长序列 |  |  |  |  |
| 低时延 |  |  |  |  |

> **说明**：`*总卡数` 表示所有节点使用的 NPU 总数。

### 表2：节点详细配置

| 场景 | 配置 | 卡数 | TP | DP | BS | 并发 | 最大上下文 | MTP投机数 | FUSED_MC2 | EP开关 | FC+CP开关 | 异步调度 |
|------|------|------|----|----|----|------|----------|---------|---------------|--------|-------|------|
| 高吞吐(32K推1K) | 服务端-P节点/单机 | 8 | 8 | 2 | 32 | 64 | 30k | 3 | 关 | 开 | 开 | 开 |
| 高吞吐(32K推1K) | 服务端-D节点 | 8 | 2 | 8 | 8 | 64 | 30k | 12 | 关 | 开 | 关 | 开 |
| 长序列 | 服务端-P节点/单机 |  |  |  |  |  |  |  |  |  |  |  |
| 长序列 | 服务端-D节点 |  |  |  |  |  |  |  |  |  |  |  |
| 低时延 | 服务端-P节点/单机 |  |  |  |  |  |  |  |  |  |  |  |
| 低时延 | 服务端-D节点 |  |  |  |  |  |  |  |  |  |  |  |

## 10 性能调优 （可选）（Performance Tuning）

**资料写作要求：**

- 总结针对该模型的关键优化技术和调参经验，帮助用户在特定场景下达到最佳性能。应包括优化技术说明、启用方式、参数调优建议和典型配置示例。
- 可通过超链接跳转到features guide，以便用户查看具体特性的详细说明。

### 10.1 关键优化点

在本节中，我们将介绍能够显著提升XX模型性能的关键优化点。这些技术旨在提升各种场景下的吞吐量和效率。

#### 10.1.1 基础优化

**示例：**

以下优化默认启用，无需额外配置：

| 优化技术 | 技术原理 | 性能收益 |
| --------- | --------- | --------- |
| Rope优化 | 位置编码的cos_sin_cache及索引操作仅在第一层执行，后续层直接复用 | 减少解码阶段重复计算，加速推理 |
| AddRMSNormQuant融合 | 将逐地址多尺度归一化与量化操作合并为单算子 | 优化内存访问模式，提升计算效率 |
| Zero-like Elimination | 移除Attention前向中的非必要零张量操作 | 减少内存占用，提高矩阵运算效率 |
| FullGraph优化 | 通过`compilation_config={"cudagraph_mode":"FULL_DECODE_ONLY"}`将整个解码图一次性捕获重放 | 显著降低调度延迟，稳定多设备性能 |

#### 10.1.2 高级优化（需显式开启）

**示例：**

| 优化技术 | 适用场景 | 启用方式 | 技术原理 | 注意事项 |
| --------- | --------- | --------- | --------- | --------- |
| FlashComm_v1 | 大并发、张量并行(TP)场景 | `export VLLM_ASCEND_ENABLE_FLASHCOMM1=1` | 将传统Allreduce分解为Reduce-Scatter和All-Gather，减少RMSNorm计算维度 | 阈值保护：仅当实际token数超过阈值时生效，避免小并发场景性能倒退|
| Matmul-ReduceScatter融合 | 大型分布式环境 | 启用FlashComm_v1后自动开启 | 将矩阵乘法与Reduce-Scatter操作融合，实现流水线并行处理 | 同FlashComm_v1，有阈值保护 |
| 权重预取 | MLP密集型场景（Dense模型）| `export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1` | 利用向量计算时间，提前将MLP权重加载到L2 Cache | 需配合预取缓冲区大小调整 |
| 异步调度 | 大规模模型、高并发场景 | `--async-scheduling` | 非阻塞任务调度，提升并发处理能力 | 与FullGraph优化协同使用 |

### 10.2 优化亮点

**资料写作要求：**

概括实际调优过程中最值得关注的优化要点，提炼核心经验，为读者提供快速上手的调优思路。

**示例：**

在实际调优过程中，以下要点对性能提升最为关键：预取缓冲区大小需通过实测找到计算与预取的最佳重叠点；`max-num-batched-tokens`的设置需平衡吞吐与显存，避免分块过多或OOM风险；`cudagraph_capture_sizes`必须手动指定并覆盖目标并发，启用FlashComm_v1时还需保证值为TP倍数；`pa_shape_list`作为临时调优参数，仅在特定batch size下生效，需关注版本演进及时调整。以上参数与环境变量的协同配置，是实现极致性能的关键。

## 11 FAQ

**资料写作要求：**

- 在章节开头添加说明：常见环境、安装、通用参数问题请参考[公共FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html)；本章仅收录本模型特有疑难问题。
- 针对**本模型特有疑难问题** ，提供以下要素：问题现象描述、原因分析、解决措施。
