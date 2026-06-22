# Embedding Analysis

This document analyzes the current `embedding`-layer capabilities in vLLM Ascend, focusing on token embedding, embedding tensor parallelism, quantized embedding, and the wiring relationship with `lm_head`.

## 1. What this layer is responsible for

The embedding layer is not just a table lookup. In the current implementation it also covers:

- vocab sharding and padding
- embedding TP versus default TP group selection
- quantized embedding integration
- prompt-token to hidden-state aggregation
- shared vocab constraints with `lm_head`

This layer is mainly about how input representations enter the model, not how attention or the backend executes.

## 2. Current capability overview

Embedding support on Ascend is primarily provided through custom OOT operator registration:

- `VocabParallelEmbedding -> AscendVocabParallelEmbedding`
- `ParallelLMHead -> AscendParallelLMHead`
- `LogitsProcessor -> AscendLogitsProcessor`

Key entry points:

- [utils.py](/home/cmq/code/vllm-ascend/vllm_ascend/utils.py:684)
- [vocab_parallel_embedding.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/vocab_parallel_embedding.py:44)

That means embedding is already part of the default Ascend custom-op replacement path, not a loose patch.

## 3. Key implementation properties

### 3.1 Vocab sharding and padding

`AscendVocabParallelEmbedding` inherits upstream `VocabParallelEmbedding` and keeps the vocab-dimension sharding model. During initialization it explicitly manages:

- `org_vocab_size`
- `num_added_embeddings`
- `org_vocab_size_padded`
- `num_embeddings_padded`
- `num_embeddings_per_partition`

See:

- [vocab_parallel_embedding.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/vocab_parallel_embedding.py:74)

So the current implementation still assumes the upstream-style model of "shard by vocab, then do masked lookup inside each shard."

### 3.2 Embedding TP is separated from default TP

The implementation can detach embedding from the default TP group and use `embedding_tensor_parallel_size` independently:

- `embedding_tp_enable()`
- `get_embed_tp_group()`
- `forward_type == "embed_tp"`

See:

- [utils.py](/home/cmq/code/vllm-ascend/vllm_ascend/utils.py:823)
- [vocab_parallel_embedding.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/vocab_parallel_embedding.py:63)
- [parallel_state.py](/home/cmq/code/vllm-ascend/vllm_ascend/distributed/parallel_state.py)

In the `embed_tp` path, the input is `all_gather`-ed first, looked up locally, and then `reduce_scatter`-ed back.

### 3.3 Masked lookup remains the core contract

The basic contract is still:

1. Compute which token IDs belong to the local shard.
2. Mask out invalid IDs.
3. Run local embedding lookup.
4. Zero out masked positions.
5. Reduce across ranks.

Relevant methods:

- `_get_masked_input_and_mask`
- `_forward_embed_tp`
- `_forward_origin`

## 4. Current assumptions and boundaries

The current implementation assumes:

- token embedding is still shaped like standard `VocabParallelEmbedding`
- vocab can be cleanly partitioned by shard boundaries
- vocab padding remains legal
- quantized embedding methods must implement the embedding interface explicitly

It is strongest for standard decoder-only LLM token embeddings. It is weaker for heterogeneous multi-branch embedding structures.

## 5. Related code

- [vocab_parallel_embedding.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/vocab_parallel_embedding.py)
- [utils.py](/home/cmq/code/vllm-ascend/vllm_ascend/utils.py)
- [distributed/parallel_state.py](/home/cmq/code/vllm-ascend/vllm_ascend/distributed/parallel_state.py)
- [quantization/modelslim_config.py](/home/cmq/code/vllm-ascend/vllm_ascend/quantization/modelslim_config.py)
- [vocab_parallel_embedding.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/vocab_parallel_embedding.py:63)
- [parallel_state.py](/home/cmq/code/vllm-ascend/vllm_ascend/distributed/parallel_state.py)

在 `embed_tp` 路径里，输入会先 `all_gather`，查表后再 `reduce_scatter`，而不是直接走默认 TP reduce 逻辑。

### 3.3 masked lookup 仍是当前主契约

当前实现的核心仍是：

1. 根据 shard 边界计算本 rank 有效 token
2. 对无效 token 做 mask
3. 本地 embedding lookup
4. 对 masked 位置置零
5. 跨 rank 规约

对应实现见：

- `_get_masked_input_and_mask`
- `_forward_embed_tp`
- `_forward_origin`

见：

- [vocab_parallel_embedding.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/vocab_parallel_embedding.py:136)

这说明当前 embedding 层并没有引入 Ascend 专属的新数据结构，本质还是在复用 upstream 的 shard + mask 模型。

### 3.4 量化 embedding 已进入正式路径

`AscendVocabParallelEmbedding` 初始化时会从 `quant_config` 获取 quant method；如果 quant method 不支持 embedding 接口，会直接拒绝。

对应实现见：

- [vocab_parallel_embedding.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/vocab_parallel_embedding.py:93)

因此当前实现对 embedding 量化的基本假设很明确：

- 量化方法必须显式实现 `embedding(...)`
- 不能把只支持 linear matmul 的 quant method 直接套到 embedding

### 3.5 多模态 embedding 不是这篇文档的主体

当前仓库存在：

- `prompt_embeds`
- PCP 下的 multimodal embedding gather
- Qwen2/Qwen3 VL patch

但这些属于多模态输入接线，不属于纯 `embedding` 层主能力。这里仅记录当前 embedding 层能与这些路径共存，不展开多模态能力本身。

相关入口：

- [worker/pcp_utils.py](/home/cmq/code/vllm-ascend/vllm_ascend/worker/pcp_utils.py)
- [worker/npu_input_batch.py](/home/cmq/code/vllm-ascend/vllm_ascend/worker/npu_input_batch.py)

## 4. 当前结构假设

当前 embedding 层隐含了这些结构假设：

- token embedding 仍是标准的 `VocabParallelEmbedding` 风格
- vocab 维度可按 partition 边界干净切分
- vocab padding 仍然是合法做法
- 输入 token 可通过 mask 方式归并到本 rank shard
- 若启用 quant，quant method 必须为 embedding 提供专用实现

这意味着当前实现更适合：

- 标准 decoder-only LLM 的 token embedding
- 与 upstream 相近的 tied/untied 输出头结构
- 没有额外复杂 token-branch 的 embedding 结构

## 5. 已知边界与风险

当前 embedding 层值得注意的边界有：

- `embed_tp` 是显式特性，不是所有模型默认开启
- 多 embedding 分支、额外 modality embedding、或特殊 lookup 逻辑，不在纯 embedding 层主路径内
- 量化 embedding 依赖 quant method 自己实现 embedding 接口
- vocab 改名、embedding 权重重映射、特殊 checkpoint 命名，不由这一层单独解决

也就是说，embedding 层当前能力偏强的是“执行与分布式查表”，偏弱的是“异构 embedding 结构解释”。

## 6. 分析这一层时应该看什么

建议优先看：

- `embed_tokens` / `lm_head` 是否沿用标准命名
- 是否使用 `VocabParallelEmbedding`
- 是否启用了 `embedding_tensor_parallel_size`
- quant method 是否实现 `embedding`
- vocab 是否存在 added embeddings / resize / padding

## 7. 相关代码

- [vocab_parallel_embedding.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/vocab_parallel_embedding.py)
- [utils.py](/home/cmq/code/vllm-ascend/vllm_ascend/utils.py)
- [distributed/parallel_state.py](/home/cmq/code/vllm-ascend/vllm_ascend/distributed/parallel_state.py)
- [quantization/modelslim_config.py](/home/cmq/code/vllm-ascend/vllm_ascend/quantization/modelslim_config.py)
