# Model Adapter And Weight Loading Baseline

本文档定义 `vllm-ascend-model-adapter` 在 **模型注册、模型接线、权重映射、weight loading** 这一层的当前能力基线，用于和新模型需求做对比，输出 gap analysis。

## 1. 这一层解决什么问题

这一层回答的问题不是 “Ascend 算子能不能跑”，而是：

- vLLM 是否已经认识这个 architecture；
- 模型是否已经有可复用的 adapter；
- checkpoint 的 key 结构是否能映射到 vLLM 当前实现；
- TP / KV head / norm / rope / fp8 scale 这些权重装载规则是否已经被现有实现覆盖。

如果这一层没打通，通常在 attention/operator 执行前就会失败。

## 2. 当前能力基线

### 2.1 已有默认能力

当前 skill 假设以下能力已经存在或可快速复用：

- 通过 `config.json` 的 `architectures` 字段判断模型入口；
- 通过 `vllm/model_executor/models/registry.py` 判断 architecture 是否已注册；
- 在 `/vllm-workspace/vllm/vllm/model_executor/models/` 中新增或复用 model adapter；
- 在需要时新增 processor；
- 通过显式 weight remap 规则解决 checkpoint key 与 vLLM 层名不一致的问题；
- 保持实现主要发生在 `vllm`，而不是 `vllm-ascend`。

### 2.2 默认适配位置

这一层默认应优先落在：

- `vllm/model_executor/models/<new_model>.py`
- `vllm/model_executor/models/registry.py`
- `vllm/transformers_utils/processors/<new_model>.py`（如果需要 processor）

只有确认是 Ascend backend 层面的 incompatibility，才继续往 `vllm-ascend` 分析。

### 2.3 已知高频能力边界

当前 skill 已经默认把下面这些视为这一层的高频适配项：

- architecture 注册缺失；
- remote code 与原生 vLLM adapter 不兼容；
- qkv / o_proj / gate / moe 层命名不一致；
- q_norm / k_norm / kv_norm / rope 相关权重装载规则不一致；
- KV head replication 或 TP shard 方式导致 norm / head 维度不匹配；
- fp8 checkpoint 需要 weight + scale 成对加载；
- safetensors index 或 shard key 模式与现有 loader 假设不一致。

## 3. 这一层的典型输入证据

做 gap analysis 时，优先看：

- `config.json`
- `architectures`
- `model_type`
- `auto_map`
- modeling / remote code 的模块命名
- safetensors index
- checkpoint key 前缀与层命名模式
- load 阶段报错（missing keys / unexpected keys / shape mismatch）

## 4. 这一层的典型失败信号

以下症状通常优先归到这一层：

- architecture not recognized
- registry 未命中
- 模型类导入成功但 `load_weights` 失败
- missing / unexpected key 大量出现
- 某些层 shape mismatch，但 attention/operator 还没真正执行
- qk norm / kv norm / tp shard 维度不匹配
- fp8 scale / weight 配对不完整

## 5. 适配判断原则

### 5.1 优先复用已有 adapter

如果新模型只是某个已支持架构的轻微变体，优先：

- 复用现有 adapter；
- 增加最小 remap / conditional path；
- 避免新建一整套模型实现。

### 5.2 新建 adapter 的条件

只有在下面情况同时成立时，才倾向新建 adapter：

- registry 中没有合适架构；
- 现有 adapter 的输入/层结构与新模型差异较大；
- 通过 remap 无法干净复用；
- processor / weight layout / execution semantics 都明显不同。

### 5.3 不要把模型适配问题错误下沉到 Ascend backend

如果问题本质是：

- architecture 注册缺失；
- 权重键映射不对；
- processor 绑定错误；
- vLLM 模型文件里层接线不匹配；

那么应先修 `vllm` 侧，而不是直接改 `vllm-ascend`。

## 6. 这一层的固定输出模板

每次涉及这一层时，先写：

```markdown
## Model Adapter Gap Analysis

### 1. Current Capability
- Existing registered architecture:
- Reusable adapter path:
- Existing weight loading assumptions:
- Existing shard/remap support:

### 2. Model Requirement
- `architectures` / `model_type`:
- Adapter structure needed:
- Checkpoint key patterns:
- TP / KV / norm / rope / scale loading needs:

### 3. Gap
- Registration gap:
- Adapter gap:
- Weight mapping gap:
- Loader / shard gap:

### 4. Adaptation Plan
- Fix location:
- Minimal files to touch:
- Validation focus:
- Stop / escalate condition:
```

## 7. 最常见的适配动作

- 在 `registry.py` 补 architecture 映射；
- 复用邻近模型 adapter；
- 增加 `load_weights` remap；
- 为 KV/QK norm、replicated KV heads、rope 变体补 shard 规则；
- 为 fp8 checkpoint 增加 scale pairing / dequant load path；
- 在 processor 存在强依赖时，再继续扩展 processor 层适配。

## 8. 什么时候停止并升级

如果你已经确认：

- 模型 adapter 正确；
- 权重映射正确；
- load 阶段通过；
- 失败只在 Ascend backend/operator 层出现；

再进入 attention/operator/framework 层分析。

如果模型必须在 `vllm-ascend` 中新增 modeling 文件才能工作，按当前 skill 约束，应停止并升级分析，不直接这么做。
