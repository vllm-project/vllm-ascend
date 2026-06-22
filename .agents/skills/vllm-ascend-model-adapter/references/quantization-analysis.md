# Quantization Analysis

本文档分析 vLLM Ascend 当前在量化层的实现能力，聚焦已注册的 quant schemes、linear/MoE/attention 三类量化接线、KV C8 路径，以及当前量化能力的边界。

## 1. 这一层解决什么问题

量化层当前主要解决：

- checkpoint quant 如何接到具体层
- runtime quant kernel 如何执行
- linear / MoE / attention 三类层如何选 quant scheme
- KV cache quant 如何与 attention backend 协同

## 2. 当前能力总览

当前量化实现已经形成独立体系，核心在：

- `quantization/methods/`
- `quantization/modelslim_config.py`
- `ops/cv_linear.py`
- `attention/attention_v1.py` 的 C8 路径

注册总览见：

- [quantization/methods/__init__.py](/home/cmq/code/vllm-ascend/vllm_ascend/quantization/methods/__init__.py)

## 3. 当前实现的关键能力

### 3.1 量化 scheme 已覆盖 linear / MoE / attention

从 registry 可见，当前量化不仅覆盖 linear，还覆盖：

- MoE fused method
- attention quant method

典型类包括：

- `AscendW8A8LinearMethod`
- `AscendW8A8DynamicLinearMethod`
- `AscendW4A8DynamicFusedMoEMethod`
- `AscendFAQuantAttentionMethod`
- 多类 MXFP / FP8 变体

这说明量化当前不是“某几个临时 patch”，而是已经按 layer type 建立注册体系。

### 3.2 linear quant 路径较成熟

当前 linear quant 主路径包括：

- `W8A8 static`
- `W8A8 dynamic`
- `W8A16`
- `W4A8`
- `W4A4`
- MXFP 变体

其中 `cv_linear.py` 显式把某些 linear 拆成：

1. `npu_dynamic_quant`
2. `npu_quant_matmul`

见：

- [ops/cv_linear.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/cv_linear.py)

这说明当前量化 linear 已经不是只会“调 quant_method.apply”，还存在更低层的执行拆分优化。

### 3.3 MoE quant 路径已是正式能力

当前 MoE 量化路径覆盖：

- `W8A8 dynamic`
- `W4A8 dynamic`
- `W4A16`
- `PDMix`
- `FP8/MXFP` 变体

并且与 MoE 的 `w13/w2` 布局深度耦合。

这表明当前量化能力在 MoE 上并不只是“兼容”，而是重要功能面。

### 3.4 C8 KV cache 是 attention 级量化正式路径

`attention_v1.py` 中已存在完整 C8 INT8 KV 路径，包括：

- scale/offset 准备
- paged INT8 KV 直接 decode
- chunked prefill 混合路径
- gather + dequant 到 dense 的 fallback

见：

- [attention_v1.py](/home/cmq/code/vllm-ascend/vllm_ascend/attention/attention_v1.py:1497)
- [quantization/methods/kv_c8.py](/home/cmq/code/vllm-ascend/vllm_ascend/quantization/methods/kv_c8.py)

因此当前量化不能只看 linear/MoE；KV quant 已经直接进入 backend 层。

### 3.5 compressed tensors / modelslim 已被显式支持

`modelslim_config.py` 和若干 quant method 中可见：

- `compressed_tensors`
- 新旧 quant version 差异
- 对 scale/bias/packed weight 的后处理

说明当前实现已经承认“checkpoint 量化格式”本身就是一层独立复杂性。

## 4. 当前结构假设

当前量化层隐含这些假设：

- layer type 决定 quant scheme 的大类
- linear、MoE、attention 的量化不能混为一谈
- 动态量化和静态量化是两套不同执行约定
- KV quant 必须与 attention backend 联动

## 5. 已知边界与风险

当前主要边界有：

- 并非所有 quant format 都天然能落到现有 scheme
- 某些 FP8 路径更适合 load-time dequant，而不是强推 runtime fp8 kernel
- quant 加载、quant 执行、quant 通信常常耦合在一起
- dummy 验证对量化问题的覆盖明显不足

## 6. 分析这一层时应该看什么

建议优先看：

- `quantization_config`
- `quant_description`
- layer type 是 linear / moe / attention 哪一类
- 是否存在 KV quant / C8
- 是否使用 compressed_tensors / modelslim 格式

## 7. 相关代码

- [quantization/methods/__init__.py](/home/cmq/code/vllm-ascend/vllm_ascend/quantization/methods/__init__.py)
- [quantization/modelslim_config.py](/home/cmq/code/vllm-ascend/vllm_ascend/quantization/modelslim_config.py)
- [ops/cv_linear.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/cv_linear.py)
- [attention/attention_v1.py](/home/cmq/code/vllm-ascend/vllm_ascend/attention/attention_v1.py)
- [quantization-baseline.md](/home/cmq/code/vllm-ascend/.agents/skills/vllm-ascend-model-adapter/references/quantization-baseline.md)
- [fp8-on-npu-lessons.md](/home/cmq/code/vllm-ascend/.agents/skills/vllm-ascend-model-adapter/references/fp8-on-npu-lessons.md)
