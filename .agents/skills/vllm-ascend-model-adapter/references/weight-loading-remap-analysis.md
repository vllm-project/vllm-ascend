# Weight Loading / Remap Analysis

本文档分析 vLLM Ascend 当前在模型注册、权重加载、命名 remap、TP/KV shard 特例方面的现有能力。这里讨论的是“当前能怎么接 checkpoint”，不是针对某个新模型制定适配动作。

## 1. 这一层解决什么问题

这一层主要回答：

- architecture 入口是怎么识别的
- 当前权重命名 remap 已覆盖哪些常见模式
- TP / KV replication / norm shard 有哪些现成处理
- checkpoint scale/offset 与 quant 元信息怎么接入

## 2. 当前能力总览

当前权重加载与 remap 能力并不是集中在一个文件里，而是散落在：

- 各模型文件自己的 `load_weights` / rename 逻辑
- `quantization/modelslim_config.py`
- worker patch
- quant weight loader

现有 baseline 文档见：

- [model-adapter-and-weight-loading-baseline.md](/home/cmq/code/vllm-ascend/.agents/skills/vllm-ascend-model-adapter/references/model-adapter-and-weight-loading-baseline.md)

## 3. 当前实现的关键能力

### 3.1 OOT custom op 已覆盖大量基础层

`utils.py` 中的 `REGISTERED_ASCEND_OPS` 已把多数关键层替换为 Ascend 版本，包括：

- embedding
- lm_head
- linear
- rotary
- norm
- fused MoE

见：

- [utils.py](/home/cmq/code/vllm-ascend/vllm_ascend/utils.py:684)

这意味着很多 checkpoint 只要能映射到 upstream/vLLM 认得的结构，就能自动落到 Ascend 层实现。

### 3.2 remap 规则已覆盖多种命名漂移

当前仓库显式处理过很多重命名模式，例如：

- `head -> lm_head`
- `embed -> embed_tokens`
- `.w1. -> .gate_proj.`
- `.w2. -> .down_proj.`
- `.w3. -> .up_proj.`
- `.ffn. -> .mlp.`
- `.attn_norm. -> .input_layernorm.`
- `.ffn_norm. -> .post_attention_layernorm.`

见：

- [models/deepseek_v4.py](/home/cmq/code/vllm-ascend/vllm_ascend/models/deepseek_v4.py:1337)
- [models/deepseek_v4_mtp.py](/home/cmq/code/vllm-ascend/vllm_ascend/models/deepseek_v4_mtp.py:326)
- [quantization/modelslim_config.py](/home/cmq/code/vllm-ascend/vllm_ascend/quantization/modelslim_config.py:274)

这表明当前仓库对“结构相近但命名不同”的 checkpoint 已经有较成熟经验。

### 3.3 quant weight loader 已深入到层级

当前很多量化路径不是靠通用 `load_state_dict`，而是自定义 `weight_loader`：

- embedding / lm_head
- C8 KV scale/offset
- FA quant
- 各种 quant linear / fused MoE 权重

典型入口：

- [vocab_parallel_embedding.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/vocab_parallel_embedding.py:126)
- [quantization/methods/kv_c8.py](/home/cmq/code/vllm-ascend/vllm_ascend/quantization/methods/kv_c8.py)

### 3.4 TP / KV replication 特例已不是空白

当前仓库已经显式处理过：

- `q_norm` / `k_norm` shard
- `num_kv_head_replicas`
- k_norm 在 KV heads 小于 TP world 时的切分

见：

- [patch/worker/patch_minimax_m2.py](/home/cmq/code/vllm-ascend/vllm_ascend/patch/worker/patch_minimax_m2.py)
- [patch/worker/patch_minimax_m2_linear_attn.py](/home/cmq/code/vllm-ascend/vllm_ascend/patch/worker/patch_minimax_m2_linear_attn.py)

这说明当前加载能力已经不只是“名字对上就行”，还包含 head topology 相关的 shard 语义。

### 3.5 fp8 / scale pairing 已被正式考虑

现有经验文档和代码都承认：

- fp8 checkpoint 需要 weight + scale pairing
- dummy run 不能替代真实权重验证

见：

- [fp8-on-npu-lessons.md](/home/cmq/code/vllm-ascend/.agents/skills/vllm-ascend-model-adapter/references/fp8-on-npu-lessons.md)

## 4. 当前结构假设

当前 weight loading 层隐含这些假设：

- 最理想的情况是复用 upstream adapter，只做最小 remap
- 很多模型差异本质是命名/布局差异，而不是 backend 能力差异
- TP/KV/norm shard 是加载层正式组成部分
- quant checkpoint 需要专门 weight loader 与后处理

## 5. 已知边界与风险

当前主要边界有：

- registry 缺失仍属于 upstream/vLLM 层问题，不由 Ascend 文档自动解决
- 模型结构若偏离现有 adapter 太远，现有 remap 价值会下降
- 很多 remap 逻辑散落在模型和量化文件中，不是单一中心化框架
- 真正困难的路径常常不是“重命名”，而是“重命名 + shard + quant”叠加

## 6. 分析这一层时应该看什么

建议优先看：

- `architectures` / `model_type`
- checkpoint key 前缀
- `head/embed/ffn/norm` 的命名风格
- TP / KV / qk norm 的 shard 行为
- scale/offset 是否成对出现

## 7. 相关代码

- [models/deepseek_v4.py](/home/cmq/code/vllm-ascend/vllm_ascend/models/deepseek_v4.py)
- [models/deepseek_v4_mtp.py](/home/cmq/code/vllm-ascend/vllm_ascend/models/deepseek_v4_mtp.py)
- [quantization/modelslim_config.py](/home/cmq/code/vllm-ascend/vllm_ascend/quantization/modelslim_config.py)
- [patch/worker/patch_minimax_m2.py](/home/cmq/code/vllm-ascend/vllm_ascend/patch/worker/patch_minimax_m2.py)
- [model-adapter-and-weight-loading-baseline.md](/home/cmq/code/vllm-ascend/.agents/skills/vllm-ascend-model-adapter/references/model-adapter-and-weight-loading-baseline.md)
