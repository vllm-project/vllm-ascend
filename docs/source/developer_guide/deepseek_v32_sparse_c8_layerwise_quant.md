# DeepSeek V3.2 Sparse C8 按层量化方案

## 背景

DeepSeek V3.2 的 sparse MLA attention indexer int8 量化，目前通过 `additional_config["enable_sparse_c8"]` 打开。

原有行为是全局开关式的：

- 一旦 `enable_sparse_c8=True`，所有 sparse MLA 层都会进入 Sparse C8 路径。
- 无法表达“部分层量化、部分层回退”。

为了支持混合部署，更合理的做法是让权重包里的 `quant_model_description.json` 成为唯一真值来源，而不是再额外维护一份 runtime 层列表。

## 目标

用 `quant_model_description.json` 中的 layerwise 字段决定某一层是否启用 indexer int8 量化，例如：

```json
{
  "model.layers.0.self_attn.indexer.quant_type": "INT8_DYNAMIC"
}
```

规则如下：

- 某层存在 `*.indexer.quant_type = "INT8_DYNAMIC"`，则该层启用 Sparse C8。
- 某层没有这个字段，则该层回退到非 C8 的 sparse MLA 路径。
- 如果文件里存在 layerwise `*.indexer.quant_type` 字段，但没有任何一层是 `INT8_DYNAMIC`，则所有 sparse MLA 层都回退到非 C8 路径。
- 仅修改 `vllm-ascend`。

## 设计

### 1. 单一配置来源

直接复用已经由 `ModelSlim` 读入的 `vllm_config.quant_config.quant_description`，不再引入新的 runtime 选层配置。

Sparse C8 的层选择规则为：

- key 后缀为 `.indexer.quant_type`
- value 等于 `INT8_DYNAMIC`

### 2. Runtime 判定链路

保留 `additional_config["enable_sparse_c8"]` 作为总开关，但“哪些层开启 Sparse C8”完全由 `quant_model_description.json` 决定。

判定流程：

1. `AscendConfig` 解析 `quant_description`
2. `AscendConfig.is_sparse_c8_layer(layer_name)` 成为统一的按层判定接口
3. `AscendSFAImpl` 通过该接口决定当前层是否走 Sparse C8
4. `NPUModelRunner.get_kv_cache_spec()` 将结果写入每层的 `MLAAttentionSpec.cache_sparse_c8`
5. KV cache 的分配与 reshape 按 layer spec 执行，因此量化层与回退层可以并存

### 3. ModelSlim attention 量化识别保持一致

只改 runtime 还不够，因为 attention 量化方法本身也需要识别 layerwise `indexer.quant_type`。

因此同步修改 `AscendModelSlimConfig`：

- attention quant type 获取时，优先读取 `"{prefix}.indexer.quant_type"`
- `is_indexer_quant_layer(prefix)` 改为识别 layerwise key
- 如果只有旧的全局 `indexer_quant_type`，且完全没有 layerwise key，则保持原有“全层量化”行为，兼容旧模型

这样可以避免出现下面这种不一致：

- runtime 希望某层回退
- 但量化加载阶段仍然给该层挂上了全局 attention quant method

## 代码修改点

### `vllm_ascend/ascend_config.py`

- 不再依赖 `additional_config["sparse_c8_layers"]`
- 从 `vllm_config.quant_config.quant_description` 解析 Sparse C8 层信息
- 将 `*.indexer.quant_type = INT8_DYNAMIC` 视为按层启用信号
- 如果存在 layerwise key，则严格按 layerwise 配置执行
- 如果完全不存在 layerwise key，则保留旧的全层 Sparse C8 行为

### `vllm_ascend/quantization/modelslim_config.py`

- `get_quant_type_for_layer(..., layer_type="attention")` 优先读取 layerwise key：
  `"{prefix}.indexer.quant_type"`
- `_add_kvcache_quant_metadata()` 增加 layerwise indexer quant 元数据解析
- `is_indexer_quant_layer(prefix)` 支持两种模式：
  - layerwise 模式：只有标记层量化
  - legacy 模式：只有全局配置时仍按全层量化

### 复用已有 runtime 改造

下面这些改造已经具备按层生效能力，这次无需重做：

- `vllm_ascend/attention/sfa_v1.py`
- `vllm_ascend/worker/model_runner_v1.py`
- `vllm_ascend/patch/platform/patch_kv_cache_interface.py`

它们已经统一依赖按层 predicate / spec，因此天然支持“部分层量化、部分层回退”。

## 示例

如果 `quant_model_description.json` 中包含：

```json
{
  "model.layers.0.self_attn.indexer.quant_type": "INT8_DYNAMIC",
  "model.layers.1.self_attn.indexer.quant_type": "INT8_DYNAMIC"
}
```

则行为为：

- layer 0 走 Sparse C8
- layer 1 走 Sparse C8
- 其他 sparse MLA 层回退到非 C8 路径

runtime 只需要保留总开关：

```python
additional_config = {
    "enable_sparse_c8": True,
}
```

不再需要额外的 `sparse_c8_layers` 列表。

## 兼容性

- 推荐模式：
  使用 layerwise `*.indexer.quant_type = "INT8_DYNAMIC"` 控制按层 Sparse C8。
- 兼容模式：
  如果 `quant_model_description.json` 只有全局 `indexer_quant_type`，且没有任何 layerwise key，则继续保持原有全层 Sparse C8 行为。

## 测试覆盖

这次补充/调整了以下单测方向：

- `AscendConfig` 从 `quant_description` 解析 layerwise Sparse C8
- `AscendConfig` 在存在 layerwise key 但无 `INT8_DYNAMIC` 时正确回退
- `AscendModelSlimConfig` 优先读取 layerwise attention quant type
- `AscendModelSlimConfig` 的 layerwise indexer 量化识别
- 旧的全局 `indexer_quant_type` 兼容路径
