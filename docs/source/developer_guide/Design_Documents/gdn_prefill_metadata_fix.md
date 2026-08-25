# V2 投机推理 prefill 阶段 GDN metadata 过滤修复

## 1. 问题概述

在 Model Runner V2 的投机推理场景中，混合注意力模型可能同时包含：

- 全注意力层，例如 GQA/FIA；
- 线性注意力层，例如 GDN；
- MTP 或 EAGLE 草稿层。

GDN 层使用 `GDNAttentionMetadata`。该 metadata 用于描述递归状态、state index
以及投机推理相关的 state 信息，本身没有全注意力 metadata 中的 `seq_lens` 字段。

Ascend speculator 的 `_run_model()` 在一次草稿模型前向完成后，会调用：

```python
self._ascend_update_seq_lens(attn_metadata)
```

原实现假设传入字典中的所有对象都是全注意力 metadata，并无条件执行：

```python
attn_meta.seq_lens = attn_meta.seq_lens + 1
attn_meta.seq_len_list = attn_meta.seq_lens.tolist()
```

当 GDN metadata 被传入该路径时，就会触发：

```text
AttributeError: 'GDNAttentionMetadata' object has no attribute 'seq_lens'
```

本次修复的目标是：**在 draft prefill 入口处，不再把 target-only 的 attention
metadata 传给草稿模型。**

## 2. prefill 原来的调用链

相关代码位于：

```text
vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py
```

Ascend speculator 没有重写 `propose()`，因此调用的是上游
`AutoRegressiveSpeculator.propose()`。

原调用关系如下：

```text
GPUModelRunner
  |
  | 生成 target 模型的 attn_metadata
  |
  v
AutoRegressiveSpeculator.propose()
  |
  | _prefill(..., attn_metadata, slot_mappings, ...)
  | 这里的 attn_metadata 是 target 模型的完整字典
  v
AutoRegressiveSpeculator._prefill()
  |
  v
AscendAutoRegressiveSpeculator._run_model()
  |
  | super()._run_model(..., attn_metadata, ...)
  |
  v
草稿模型 forward
  |
  v
AscendAutoRegressiveSpeculator._ascend_update_seq_lens(attn_metadata)
```

上游 prefill 路径明确复用了 target 模型的 metadata。上游代码中的设计依据是：

1. target 和 draft 使用相同的 batch 形状；
2. target 和 draft 使用相同的 KV cache 布局；
3. 草稿模型 forward 实际上只会按自身的 layer name 读取 metadata。

因此，“复用 metadata 对象”本身不是问题，问题在于传入的字典范围过大：
target-only 的 metadata 也被一并传入了 Ascend 的通用更新逻辑。

## 3. 哪些 metadata 被直接复用了

以混合 GDN + MTP 模型为例，target runner 构建出的 metadata 可能类似于：

```text
attn_metadata = {
    "model.layers.0.self_attn":  Ascend full-attention metadata,
    "model.layers.1.linear_attn": GDNAttentionMetadata,
    "model.layers.2.self_attn":  Ascend full-attention metadata,
    "mtp.layers.0.self_attn":     MTP draft metadata,
}
```

原 prefill 路径把整个字典直接传入：

```python
self._prefill(
    num_reqs,
    prefill_batch_desc.num_tokens,
    attn_metadata,
    slot_mappings,
    num_tokens_across_dp=num_tokens_across_dp,
    cudagraph_runtime_mode=prefill_batch_desc.cg_mode,
    mm_inputs=mm_inputs,
)
```

也就是说，以下对象都可能进入草稿模型的 forward context：

| metadata | 是否是草稿层需要的对象 | 原行为 |
|---|---:|---|
| target 全注意力 metadata | 否 | 直接复用 |
| target GDN metadata | 否 | 直接复用，并被后置逻辑访问 |
| MTP draft metadata | 是 | 直接复用 |

草稿模型 forward 本身通常只查找 `mtp.layers.*` 等草稿层对应的 key，
所以 target-only 条目可能不会在模型计算中被读取。但是，Ascend 的
`_run_model()` 在 forward 完成后会遍历传入的整个字典，导致 GDN 条目仍然被访问。

## 4. `draft_attn_layer_names` 的来源

上游 `DraftModelSpeculator.load_model()` 会计算草稿层名称集合：

```python
target_attn_layer_names = set(
    get_layers_from_vllm_config(
        self.vllm_config,
        AttentionLayerBase,
    ).keys()
)

self.model = self.load_draft_model(target_model, target_attn_layer_names)

all_attn_layers = set(
    get_layers_from_vllm_config(
        self.vllm_config,
        AttentionLayerBase,
    ).keys()
)
self.draft_attn_layer_names = all_attn_layers - target_attn_layer_names
```

因此，`draft_attn_layer_names` 表示草稿模型独有的 attention layer name，
例如 MTP 草稿层通常属于这一集合。

Ascend speculator 在 decode metadata 构建时已经使用了相同的过滤原则：

```python
attn_metadata = {
    name: metadata
    for name, metadata in attn_metadata.items()
    if name in self.draft_attn_layer_names
}
```

原问题是该过滤只出现在部分 decode/graph metadata 构建路径，target metadata
直接复用的 prefill 路径没有使用它。

## 5. 本次修复的位置

修复提交在：

```text
vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py
```

新增 `AscendAutoRegressiveSpeculator._prefill()` 覆盖方法，在调用上游
`_prefill()` 之前过滤 metadata：

```python
def _prefill(
    self,
    num_reqs: int,
    num_tokens: int,
    attn_metadata: dict[str, Any] | None,
    slot_mappings: dict[str, torch.Tensor] | None,
    num_tokens_across_dp: torch.Tensor | None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
) -> None:
    if attn_metadata is not None and self.draft_attn_layer_names is not None:
        attn_metadata = {
            name: metadata
            for name, metadata in attn_metadata.items()
            if name in self.draft_attn_layer_names
        }

    super()._prefill(
        num_reqs,
        num_tokens,
        attn_metadata,
        slot_mappings,
        num_tokens_across_dp,
        cudagraph_runtime_mode,
        mm_inputs,
    )
```

这里创建的是一个新的字典容器，metadata 对象本身没有深拷贝，也没有重建。
这样可以保持最小改动，并保留原有 target/draft KV cache 和 batch 对齐关系。

## 6. 修复后的调用链

修复后流程变为：

```text
target runner 构建完整 attn_metadata
  |
  v
AutoRegressiveSpeculator.propose()
  |
  v
AscendAutoRegressiveSpeculator._prefill()
  |
  | 按 draft_attn_layer_names 过滤
  v
draft_attn_metadata = {
    "mtp.layers.0.self_attn": MTP draft metadata,
    ...
}
  |
  v
super()._prefill(..., draft_attn_metadata, ...)
  |
  v
AscendAutoRegressiveSpeculator._run_model()
  |
  v
_ascend_update_seq_lens(draft_attn_metadata)
```

对于 target GDN 条目：

```text
"model.layers.1.linear_attn" not in draft_attn_layer_names
```

因此它在进入上游 `_prefill()` 之前已经被移除，不会进入草稿模型 forward
的 `attn_metadata` 参数，也不会进入 `_ascend_update_seq_lens()`。

## 7. 为什么修改 `_prefill()` 而不是只修改 `propose()`

如果只在 `propose()` 中增加局部过滤，可能只覆盖普通 eager 调用点，
但 prefill 还可能被 CUDA/ACL graph capture 路径调用。

将过滤放在 Ascend speculator 的 `_prefill()` 覆盖方法中有两个好处：

1. 所有进入上游 prefill 的 metadata 都经过同一处过滤；
2. 不需要修改上游 vLLM 代码，也不需要重写完整的 `propose()`。

这是当前修复中改动范围最小、作用边界最清晰的位置。

## 8. 为什么不删除 GDN metadata 定义或修改 GDN builder

GDN metadata 没有 `seq_lens` 是设计上的契约差异，不是字段缺失：

- 全注意力 kernel 依赖显式 KV sequence length；
- GDN kernel 使用递归 state 和 state index；
- GDN 的 speculative 信息在 metadata build 阶段已经生成。

因此不应给 `GDNAttentionMetadata` 强行添加 `seq_lens`，也不应修改 GDN builder
来伪造全注意力字段。正确做法是避免将它传入只适用于全注意力的后置逻辑。

## 9. 修改范围与未修改内容

本次代码提交只修改一个文件：

```text
vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py
```

只新增一个 `_prefill()` 方法，未修改：

- `GDNAttentionMetadata` 定义；
- GDN metadata builder；
- V1 model runner；
- decode metadata 构建逻辑；
- `slot_mappings` 内容；
- `_ascend_update_seq_lens()` 本身；
- target 模型的全量 metadata 构建逻辑。

`slot_mappings` 仍按原路径传递，因为本次异常的直接来源是 attention metadata
字典中包含 target GDN 对象，而不是 slot mapping 字典的内容。

## 10. 适用范围与验证建议

该修复适用于以下条件：

1. 使用 Model Runner V2；
2. 使用 MTP/EAGLE 等 `AscendAutoRegressiveSpeculator` 派生实现；
3. target metadata 同时包含 target GDN 层和 draft 层 metadata；
4. `draft_attn_layer_names` 正确包含草稿层名称。

建议使用以下场景验证：

- Qwen3.6-27B MTP；
- Qwen3.6-35B-A3B MTP；
- eager prefill；
- ACL/CUDA graph prefill；
- 多请求 batch；
- 纯 decode 多步 draft；
- 输出 token 一致性和 speculative acceptance rate。

调试时可以临时检查过滤结果：

```python
unexpected_names = set(attn_metadata) - set(self.draft_attn_layer_names)
```

过滤完成后，传入上游 `_prefill()` 的字典不应再包含 target-only 的 GDN layer name。
