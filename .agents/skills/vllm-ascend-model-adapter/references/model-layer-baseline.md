# Model Layer Baseline

本文档定义 **按模型内部层组件逐层分析** 的当前能力基线。这里的“层”指模型结构层，而不是适配工作层。

目标是让每次模型适配先回答：

- 这个模型由哪些关键层组成；
- 每一层在当前 vLLM / vllm-ascend 中是否已有可复用能力；
- 每一层的差异具体落在哪；
- 哪些层需要改，哪些层只需要验证接线。

## 1. 当前只要求的模型类型

当前 skill 先只要求对两类模型使用逐层模板：

1. `dense llm`
2. `moe llm`

其他类型（VLM、Whisper、pooling model 等）后续再扩展。

## 2. 为什么要逐层分析

很多适配失败不是“整体不支持”，而是某一个层组件与现有假设不一致，例如：

- embedding 权重命名不同；
- rope / position id 组织方式不同；
- MLP 是 gated / fused / parallel 变体；
- MoE expert gate / routed expert 结构不同；
- norm 在 TP/KV replication 下 shard 方式不同；
- lm_head 与 embedding tied / untied 方式不同；
- multimodal projector 或 encoder 输入格式不同。

如果不逐层拆开，很容易把 loader 问题误判成 attention 或 operator 问题。

## 3. 当前能力基线应该怎么理解

“当前能力基线”不是说每层都已经显式有参考文档，而是要先用仓库现有实现回答：

- 当前 vLLM 里有没有相近模型已经实现这一层；
- 当前 vllm-ascend 对应执行路径是否已有支持；
- 这个层的差异是结构差异、权重命名差异，还是 backend 差异。

对于常见层，默认优先这样判断：

### 3.1 Embedding

优先看：

- token embedding 命名
- tied embedding / untied lm_head
- vocab resize / padding id / special token 处理

高频问题：

- 权重名不同；
- lm_head 复用 embedding 的方式不同；
- 多 embedding 分支或额外 modality embedding。

### 3.2 Positional / RoPE

优先看：

- rope scaling
- rope type / interleaving
- mrope / dynamic rope / longrope 之类变体
- position ids 的组织方式

高频问题：

- rope 参数位置不同；
- qk rope 维度与现有假设不一致；
- position 生成逻辑和现有 model runner 假设不一致。

### 3.3 Attention

这是最复杂的一层，但不是唯一要看的一层。详细基线看：

- `references/attention-v1-analysis.md`

### 3.4 MLP / FFN

优先看：

- gated MLP / SwiGLU / GeGLU / 并行 MLP
- 权重切分方式
- 中间维度命名

高频问题：

- gate/up/down proj 命名和装载方式不同；
- 与 TP shard 规则不一致；
- fused linear 假设不成立。

### 3.5 MoE

优先看：

- expert 数量
- gate / router 结构
- shared expert 是否存在
- routed expert 权重布局
- EP / flashcomm / MoE 通信路径是否相关

详细基线看：

- `references/moe-fused-analysis.md`

高频问题：

- 非标准 expert 命名；
- gate / router 权重映射不一致；
- EP-only 路径与 TP-only 路径行为不同。

### 3.6 Norm

优先看：

- RMSNorm / LayerNorm / 其他变体
- q_norm / k_norm / post-attn norm / pre-ffn norm
- TP / replicated KV heads 下 shard 规则

高频问题：

- norm 权重需要本地 shard，而不是普通均匀切分；
- qk norm 跟 head topology 强相关。

### 3.7 LM Head / Output Head

优先看：

- 是否 tied to embedding
- 是否额外有 logits scale / soft cap / output norm
- 生成头是不是单独模块

高频问题：

- embedding 与 lm_head 共享关系没有正确接线；
- output 层 remap 不正确。

## 4. 按模型类型选择模板

先根据 `CLASSIFICATION_SUMMARY` 选择模板。

### 4.1 Dense LLM 模板

适用条件：

- 非 multimodal
- 非 encoder-decoder
- 非 MoE
- 主体是标准 decoder-only dense LLM

固定模板：

```markdown
## Layer-by-Layer Compatibility Matrix

| Layer | Current capability | Model requirement | Gap | Adaptation plan |
| --- | --- | --- | --- | --- |
| embedding |  |  |  |  |
| positional/rope |  |  |  |  |
| attention |  |  |  |  |
| mlp/ffn |  |  |  |  |
| norm |  |  |  |  |
| lm_head/output |  |  |  |  |
```

### 4.2 MoE LLM 模板

适用条件：

- decoder-only LLM
- 含 routed experts / shared experts / MoE router

固定模板：

```markdown
## Layer-by-Layer Compatibility Matrix

| Layer | Current capability | Model requirement | Gap | Adaptation plan |
| --- | --- | --- | --- | --- |
| embedding |  |  |  |  |
| positional/rope |  |  |  |  |
| attention |  |  |  |  |
| moe router/gate |  |  |  |  |
| moe experts |  |  |  |  |
| shared expert / residual mlp | N/A or ... |  |  |  |
| norm |  |  |  |  |
| lm_head/output |  |  |  |  |
```

## 5. 填写要求

无论使用哪种模板，都满足以下要求：

- `Current capability` 必须引用当前仓库已有实现或已有路径，不要空泛写“supported”。
- `Model requirement` 必须来自 `config.json`、modeling code、checkpoint key、或运行证据。
- `Gap` 必须是具体差异，不要写成“可能不兼容”。
- `Adaptation plan` 必须说清楚是：
  - 复用现有路径；
  - 改模型 adapter / loader；
  - 改 processor；
  - 改 framework 接线；
  - 验证已有 Ascend backend；
  - 或停止升级。

### 5.1 Dense LLM 至少回答这些点

- embedding 与 lm_head 是否 tied
- rope 类型与 scaling
- attention 子类型
- MLP 是普通 FFN 还是 gated FFN
- norm 类型与 shard 风险

### 5.2 MoE LLM 额外必须回答这些点

- router/gate 的实现与权重命名
- expert 权重布局
- 是否存在 shared expert
- EP / TP 路径差异
- MoE 层是否还带 attention 外的特殊通信假设

## 6. 与其他基线文档的关系

- attention 行优先参考 `attention-v1-analysis.md`
- quantized 层同时参考 `quantization-baseline.md`
- framework 行为异常时，再结合 `framework-integration-baseline.md`
- 如果未来扩展到多模态，再接 `processor-and-multimodal-baseline.md`

## 7. 使用顺序建议

推荐顺序：

1. 先做 `Layer-by-Layer Compatibility Matrix`
2. 再做 `Model Adapter Gap Analysis`
3. 再按需要展开 attention / multimodal / operator / framework / quantization 各专项 gap analysis

这样可以先建立模型整体结构图，再进入复杂专项。
