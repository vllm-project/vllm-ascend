# LM Head / Output Analysis

本文档分析 vLLM Ascend 当前在 `lm_head` / output 层的实现能力，聚焦 `ParallelLMHead`、logits gather、`lmhead TP`、量化输出头以及 embedding/output 的共享 vocab 假设。

## 1. 这一层解决什么问题

output 层当前主要承担：

- hidden states 到 logits 的最终 matmul
- logits 在并行组上的 gather/all-to-all
- vocab padding 的截断
- output head 的量化 apply
- 与 embedding 共用 vocab 分片约束

## 2. 当前能力总览

当前 output 层已通过 custom op 正式接管：

- `ParallelLMHead -> AscendParallelLMHead`
- `LogitsProcessor -> AscendLogitsProcessor`

见：

- [utils.py](/home/cmq/code/vllm-ascend/vllm_ascend/utils.py:697)
- [vocab_parallel_embedding.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/vocab_parallel_embedding.py:209)

因此当前 output 层并不是单纯沿用 upstream logits path，而是有 Ascend 专门的 gather 与并行行为。

## 3. 当前实现的关键能力

### 3.1 `ParallelLMHead` 与 embedding 共享基础实现

`AscendParallelLMHead` 直接复用 `AscendVocabParallelEmbedding` 的初始化逻辑。

见：

- [vocab_parallel_embedding.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/vocab_parallel_embedding.py:213)

这意味着当前 output 层默认与 embedding 层共享：

- vocab shard 逻辑
- padded vocab 逻辑
- quant method 接入方式

### 3.2 `lmhead TP` 是独立特性

当前实现支持 `lmhead_tensor_parallel_size`，并显式区分：

- 普通 logits path
- `lmhead_tp` logits path

入口见：

- [utils.py](/home/cmq/code/vllm-ascend/vllm_ascend/utils.py:819)
- [vocab_parallel_embedding.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/vocab_parallel_embedding.py:254)

在 `lmhead_tp` 下，hidden states 会先 `all_gather`，再由 `lm_head.quant_method.apply(...)` 计算 logits，随后按配置选择是否 `all_to_all`。

### 3.3 logits gather 与 padding 截断已内建

无论是普通路径还是 `lmhead_tp` 路径，当前实现都会在最终阶段：

- 根据 `enable_reduce_sample` 决定是否 gather
- 截掉 vocab padding

见：

- [vocab_parallel_embedding.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/vocab_parallel_embedding.py:259)
- [vocab_parallel_embedding.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/vocab_parallel_embedding.py:280)

说明当前 output 层已经把“计算 logits”和“整理最终 vocab 视图”视作同一能力的一部分。

### 3.4 量化 output head 已纳入正式路径

当前 logits 计算统一依赖：

- `lm_head.quant_method.apply(...)`

这意味着 output head 量化并不是外挂逻辑，而是主路径的一部分。

这也带来一个结构特征：只要 quant method 正确实现 output head 所需接口，output 层大多可直接复用。

### 3.5 `lm_head` 重命名映射已多次出现

在 DeepSeek V4、ModelSlim 相关逻辑中，仓库已显式处理：

- `head -> lm_head`
- `model.lm_head -> lm_head`
- `embed -> embed_tokens`

见：

- [models/deepseek_v4.py](/home/cmq/code/vllm-ascend/vllm_ascend/models/deepseek_v4.py:1337)
- [quantization/modelslim_config.py](/home/cmq/code/vllm-ascend/vllm_ascend/quantization/modelslim_config.py:497)

这说明 output 层命名漂移是仓库里反复处理过的问题。

## 4. 当前结构假设

当前 output 层隐含这些假设：

- 输出头仍然可以建模为 vocab-parallel matmul
- output 侧 vocab shard 与 embedding 侧 vocab shard 在大方向上兼容
- logits 规约与 vocab padding 截断需要统一处理
- output head 若量化，仍应沿用 quant method 主接口

## 5. 已知边界与风险

当前需要注意的边界有：

- tied/untied 共享关系更多发生在模型定义层，不完全由 output 层自行判断
- 非标准 output head、额外 logits scale / soft cap / output norm 可能在模型文件或 processor 层处理
- `lmhead_tp` 是能力增强路径，不代表所有模型默认走这条路
- output 层命名映射经常和 loader/quant config 耦合

## 6. 分析这一层时应该看什么

建议优先看：

- 是否使用 `ParallelLMHead`
- 是否启用 `lmhead_tensor_parallel_size`
- output head 是否量化
- vocab padding 是否存在
- checkpoint 中是 `head` 还是 `lm_head`

## 7. 相关代码

- [ops/vocab_parallel_embedding.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/vocab_parallel_embedding.py)
- [utils.py](/home/cmq/code/vllm-ascend/vllm_ascend/utils.py)
- [quantization/modelslim_config.py](/home/cmq/code/vllm-ascend/vllm_ascend/quantization/modelslim_config.py)
- [models/deepseek_v4.py](/home/cmq/code/vllm-ascend/vllm_ascend/models/deepseek_v4.py)
