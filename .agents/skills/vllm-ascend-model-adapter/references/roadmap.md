# References Roadmap

本文档用于整理 `vllm-ascend-model-adapter/references` 目录后续应补充的分析文档，目标是把“已有 baseline/checklist”与“缺少深度分析的特性”区分开，并给出一条可执行的补齐路线。

当前目录中，真正偏“深度分析”的文档主要是：

- `attention-v1-analysis.md`
- `moe-fused-analysis.md`

其余大多是 baseline、checklist、lessons learned 或固定输出模板，尚不足以支撑“按模型结构逐层分析”的系统化特性覆盖。

## 1. 规划原则

后续补文档时，优先遵循这几个原则：

1. 先补 `dense llm` 和 `moe llm` 的逐层主骨架，再补跨层专题。
2. 文档聚焦“当前能力分析”，不承担具体模型 gap analysis、适配决策或改造方案设计。
3. 优先写“结构差异最常导致误判”的层，避免把 loader 问题误判成 attention/backend 问题。
4. 每篇文档尽量绑定现有代码路径，不写成抽象模型科普。
5. 文档内容优先回答这些问题：
   - 当前实现覆盖了什么能力
   - 当前实现依赖哪些结构假设
   - 当前实现有哪些边界、限制和风险
   - 分析这一层时应该去看哪些代码和运行证据
6. 与已有文档边界清晰：
   - attention 细节优先落到 `attention-v1-analysis.md`
   - MoE 执行路径细节优先落到 `moe-fused-analysis.md`
   - roadmap 中新增文档应更多补“逐层结构分析”和“跨层适配专题”

## 2. 当前覆盖与主要缺口

按模型结构主线看，当前明显缺少这些分析：

- `embedding`
- `positional / rope`
- `mlp / ffn`
- `norm`
- `lm_head / output`
- `moe router / gate`
- `moe experts`
- `shared expert / residual mlp`

按跨层适配专题看，当前还缺少这些深度分析：

- `weight loading / remap`
- `quantization`
- `operator compatibility`
- `framework integration`

按模型类型看，当前模板还没有系统覆盖：

- `vlm`
- `whisper / asr`
- `pooling / reranker / embedding model`

## 3. 推荐补齐顺序

建议按下面顺序推进：

1. 先补齐 `dense llm` / `moe llm` 逐层主骨架。
2. 再补跨层专题，解决适配时最常见的误判点。
3. 最后扩展到 `vlm`、`whisper`、`pooling` 等新模型类型模板。

推荐优先级：

1. `embedding-analysis.md`
2. `positional-rope-analysis.md`
3. `mlp-ffn-analysis.md`
4. `norm-analysis.md`
5. `lm-head-output-analysis.md`
6. `moe-router-analysis.md`
7. `moe-experts-analysis.md`
8. `shared-expert-residual-mlp-analysis.md`
9. `weight-loading-remap-analysis.md`
10. `quantization-analysis.md`
11. `operator-analysis.md`
12. `framework-integration-analysis.md`
13. `vlm-layer-baseline.md`
14. `whisper-layer-baseline.md`
15. `pooling-layer-baseline.md`

## 4. 文档清单与建议大纲

### 4.1 `embedding-analysis.md`

定位：
- 分析 token embedding、special token、embedding/lm_head 接线与装载差异。

建议大纲：
- 这一层解决什么问题
- 当前 vLLM / vllm-ascend 可复用能力
- 代码范围
- 常见结构变体
- 当前实现中的权重命名与接线模式
- 当前实现如何处理 tied embedding / untied lm_head
- 当前实现中的 vocab resize / padding id / special token 假设
- 已知限制与风险
- 分析时应收集的证据
- 典型失败信号
- 参考代码与文档

### 4.2 `positional-rope-analysis.md`

定位：
- 分析位置编码、RoPE 变体以及它们与 head 拓扑、position ids、runtime metadata 的关系。

建议大纲：
- 这一层解决什么问题
- 当前可复用能力
- 代码范围
- rope type / scaling / interleave / mrope / dynamic rope / longrope 分类
- 当前实现支持哪些 rope 类型与 scaling 路径
- `position_ids` 生成与 model runner 的现有假设
- q/k rope 维度与 head topology 的现有约束
- 已知限制与风险
- 分析时应收集的证据
- 典型失败信号
- 与 attention / weight loading 的边界
- 参考代码与文档

### 4.3 `mlp-ffn-analysis.md`

定位：
- 分析 dense MLP/FFN 结构差异、权重布局差异和 TP shard 风险。

建议大纲：
- 这一层解决什么问题
- 当前可复用能力
- 代码范围
- FFN 结构分类：plain / gated / SwiGLU / GeGLU / parallel
- 当前实现中 `gate_proj` / `up_proj` / `down_proj` 的命名与布局模式
- 当前 fused linear 假设
- 当前 TP shard / pack-unpack 路径
- 已知限制与风险
- 分析时应收集的证据
- 典型失败信号
- 参考代码与文档

### 4.4 `norm-analysis.md`

定位：
- 分析各种 norm 层、qk norm、kv norm 及其 shard 行为。

建议大纲：
- 这一层解决什么问题
- 当前可复用能力
- 代码范围
- RMSNorm / LayerNorm / 其他 norm 变体
- 当前实现中的 pre-attn / post-attn / pre-ffn norm 模式
- 当前 `q_norm` / `k_norm` / `kv_norm` 支持方式
- 当前 TP / KV replication 下的 shard 约束
- 已知限制与风险
- 分析时应收集的证据
- 典型失败信号
- 参考代码与文档

### 4.5 `lm-head-output-analysis.md`

定位：
- 分析输出头结构、embedding 共享关系以及 logits 前后的额外处理。

建议大纲：
- 这一层解决什么问题
- 当前可复用能力
- 代码范围
- 当前 tied / untied output head 接线模式
- 当前 logits scale / soft cap / output norm 处理方式
- 当前输出层 remap 与 shard 方式
- 已知限制与风险
- 分析时应收集的证据
- 典型失败信号
- 参考代码与文档

### 4.6 `moe-router-analysis.md`

定位：
- 从“层结构”角度分析 MoE router/gate，而不是只看 fused 执行流水线。

建议大纲：
- 这一层解决什么问题
- 与 `moe-fused-analysis.md` 的边界
- 当前可复用能力
- router 结构分类：top-k / score fn / normalization / auxiliary logic
- 当前 gate 权重命名与布局模式
- 当前 router 输出契约与 runtime metadata 约束
- 当前 EP / TP 对 router 路径的影响
- 已知限制与风险
- 分析时应收集的证据
- 典型失败信号
- 参考代码与文档

### 4.7 `moe-experts-analysis.md`

定位：
- 从专家层结构、命名和布局角度分析 MoE experts。

建议大纲：
- 这一层解决什么问题
- 当前可复用能力
- 与 `moe-fused-analysis.md` 的边界
- expert MLP 结构分类
- 当前 `w13` / `gate_up` / `w2` 等权重布局模式
- 当前 grouped experts / packed weights / shard 方式
- 当前 expert 与 quant / comm 的耦合点
- 已知限制与风险
- 分析时应收集的证据
- 典型失败信号
- 参考代码与文档

### 4.8 `shared-expert-residual-mlp-analysis.md`

定位：
- 分析 shared expert、residual MLP、并联或串联残差分支。

建议大纲：
- 这一层解决什么问题
- 当前可复用能力
- shared expert 是否存在的判定方法
- 当前 residual MLP 接线模式
- 当前 shared expert 与 routed expert 的组合方式
- 当前权重命名 / shard 方式
- 已知限制与风险
- 分析时应收集的证据
- 典型失败信号
- 参考代码与文档

### 4.9 `weight-loading-remap-analysis.md`

定位：
- 把 `model-adapter-and-weight-loading-baseline.md` 从 baseline 扩展成“当前装载能力分析”。

建议大纲：
- 这一层解决什么问题
- 当前可复用能力
- architecture 注册与 adapter 接线现状
- 当前 checkpoint key 前缀与层命名模式
- 当前 shard / remap / safetensors index 处理模式
- 当前 fp8 scale pairing 支持方式
- 当前 KV replication / TP 下的本地 shard 特例
- 已知限制与风险
- 分析时应收集的证据
- 典型失败信号
- 参考代码与文档

### 4.10 `quantization-analysis.md`

定位：
- 将量化问题按 checkpoint、runtime kernel、KV cache 三层拆开分析。

建议大纲：
- 这一层解决什么问题
- 与 `quantization-baseline.md`、`fp8-on-npu-lessons.md` 的边界
- 当前 checkpoint quant 分类与支持现状
- 当前 runtime quant 路径分类
- 当前 KV quant / C8 KV 的特殊路径
- 当前 W8A8 / compressed-tensors 覆盖边界
- dummy 与 real 的验证边界
- 已知限制与风险
- 分析时应收集的证据
- 典型失败信号
- 参考代码与文档

### 4.11 `operator-analysis.md`

定位：
- 将算子兼容性从“基线判断”扩展成“按算子类别和约束维度分析”的文档。

建议大纲：
- 这一层解决什么问题
- 与 `operator-compatibility-baseline.md` 的边界
- Torch / Triton / CUDA / `torch_npu` / `aclnn` 分类
- 当前 dtype / shape / layout / contiguous / graph-mode 约束
- 当前 fallback 与 early-exit 边界
- HiAscend 文档查证要求
- 已知限制与风险
- 分析时应收集的证据
- 典型失败信号
- 参考代码与文档

### 4.12 `framework-integration-analysis.md`

定位：
- 分析模型结构变化如何传导到 scheduler、model runner、sampler、backend selector 等框架层。

建议大纲：
- 这一层解决什么问题
- 与 `framework-integration-baseline.md` 的边界
- 当前可复用覆盖范围
- scheduler / worker / sampler / backend selector / kv cache group 分层分析
- 当前 upstream drift 敏感点
- 当前 metadata / interface mismatch 模式
- 当前 graph capture / feature flag 风险
- 已知限制与风险
- 分析时应收集的证据
- 典型失败信号
- 参考代码与文档

### 4.13 `vlm-layer-baseline.md`

定位：
- 为 VLM 建立一版逐层模板，补足 `model-layer-baseline.md` 当前只覆盖 LLM 的缺口。

建议大纲：
- 适用范围
- 为什么 VLM 必须独立逐层分析
- 建议层级划分：
- vision encoder
- projector / resampler
- multimodal embedding merge
- decoder attention
- mlp / moe
- norm
- lm_head / output
- 与 processor/multimodal baseline 的关系
- 当前分析关注点
- 参考代码与文档

### 4.14 `whisper-layer-baseline.md`

定位：
- 为 Whisper/ASR 建立 encoder-decoder 逐层模板。

建议大纲：
- 适用范围
- 为什么 Whisper 不能直接套 decoder-only 模板
- 建议层级划分：
- audio feature extractor / processor
- encoder embedding / positional
- encoder attention / ffn
- decoder self-attention
- cross-attention
- decoder ffn / norm / lm_head
- 当前分析关注点
- 参考代码与文档

### 4.15 `pooling-layer-baseline.md`

定位：
- 为 pooling / reranker / embedding model 建立结构分析模板。

建议大纲：
- 适用范围
- 为什么 pooling 模型需要单独模板
- 建议层级划分：
- embedding
- positional/rope
- encoder/decoder blocks
- pooling head
- score / classification head
- non-causal attention 风险
- 当前分析关注点
- 参考代码与文档

## 5. 与现有文档的整合建议

为避免目录继续失控，建议把后续新增文档分成三层：

1. 逐层主骨架
- `embedding-analysis.md`
- `positional-rope-analysis.md`
- `mlp-ffn-analysis.md`
- `norm-analysis.md`
- `lm-head-output-analysis.md`
- `moe-router-analysis.md`
- `moe-experts-analysis.md`
- `shared-expert-residual-mlp-analysis.md`

2. 跨层适配专题
- `weight-loading-remap-analysis.md`
- `quantization-analysis.md`
- `operator-analysis.md`
- `framework-integration-analysis.md`

3. 模型类型模板
- `vlm-layer-baseline.md`
- `whisper-layer-baseline.md`
- `pooling-layer-baseline.md`

## 6. 最小落地建议

如果只做一轮最小补齐，建议先完成下面 5 篇：

1. `embedding-analysis.md`
2. `positional-rope-analysis.md`
3. `mlp-ffn-analysis.md`
4. `norm-analysis.md`
5. `lm-head-output-analysis.md`

这样至少可以让 `dense llm` 的逐层分析不再只停留在模板层。

如果第二轮继续推进，再补：

1. `moe-router-analysis.md`
2. `moe-experts-analysis.md`
3. `shared-expert-residual-mlp-analysis.md`
4. `weight-loading-remap-analysis.md`

这样 `moe llm` 主骨架也会完整很多。

## 7. 后续使用建议

后续做模型适配时，建议按下面顺序引用这些文档：

1. 先读 `model-layer-baseline.md`
2. 再按模型结构读取对应层文档
3. 如果发现是跨层问题，再进入：
- `weight-loading-remap-analysis.md`
- `quantization-analysis.md`
- `operator-analysis.md`
- `framework-integration-analysis.md`
4. 如果是 VLM / Whisper / pooling 模型，再切换到对应类型模板

这样可以先建立“当前能力结构化视图”，再由模型适配 skill 去做需求分析、gap 判断和适配决策，减少漏项和误判。
