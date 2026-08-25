# v2 投机推理中 `GDNAttentionMetadata` 缺少 `seq_lens` 报错分析

## 现象

Model Runner V2 + 投机解码（MTP/EAGLE），目标模型为混合注意力架构
（如 Qwen3.5，`layer_types` 同时含 `full_attention` 与 `linear_attention`），
draft prefill 阶段报错：

```text
AttributeError: 'GDNAttentionMetadata' object has no attribute 'seq_lens'
```

出错位置：`vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py`
的 `AscendAutoRegressiveSpeculator._ascend_update_seq_lens`。

## 三个叠加原因

### 原因一：目标模型的 attn_metadata 是全量、未过滤的 dict

Model runner 为 forward context 里**所有**注册的 attention 层各建一条
metadata，不区分 target/draft。Qwen3.5 场景下这个 dict 形如：

| key | metadata 类型 | 有无 `seq_lens` |
| --- | --- | --- |
| `model.layers.{i}.self_attn` | Ascend FIA metadata | 有 |
| `model.layers.{j}.linear_attn` | `GDNAttentionMetadata` | **无（设计如此）** |
| `mtp.layers.0.self_attn` | Ascend FIA metadata | 有 |

GDN 是状态空间（递归）注意力，寻址靠 state index / block table，
不需要 `seq_lens`——这是元数据契约差异，不是字段遗漏。

### 原因二：draft prefill 设计上直接复用 target 的 metadata（复用非 bug）

投机解码 speculator 的第一遍 draft forward（prefill）**不做任何过滤**，
直接把 target 的完整 `attn_metadata` 传入（上游注释明确：batch 形状与
KV 布局一致，可复用）。draft 模型前向只按自己的 key（如
`mtp.layers.0.self_attn`）从 forward context 取条目，dict 里 target 的
条目（含 GDN）是惰性的，**前向本身不会读它们，所以复用是安全的**。

对比：decode 多步路径的 metadata 由 speculator 自己的 `attn_groups`
构建，`init_attn_backend` 传了 `active_layer_names=draft_attn_layer_names`
做交集过滤，天然不含 GDN——所以**崩溃只发生在 prefill 这条复用路径**，
层过滤机制本身工作正常。

### 原因三：Ascend 后置钩子假设 dict 内所有 metadata 同构（真正缺陷）

`_run_model` 每次前向后调用 `_ascend_update_seq_lens(attn_metadata)`，
原实现无条件遍历 dict 的**每一个** value 执行：

```python
attn_meta.seq_lens = attn_meta.seq_lens + 1
attn_meta.seq_len_list = attn_meta.seq_lens.tolist()
```

它默认所有条目都是带 `seq_lens` 的 FIA 风格 metadata。当传入的是
prefill 路径复用的 target 全量 dict 时，循环撞上 GDN 条目，
`AttributeError`。

函数开头的 `if self.attn_architecture in ("DSA", "SFA"): return`
说明作者踩过 DSA/SFA 的同类契约问题，但用架构级特判兜底，
没覆盖"dict 内混入其他类型 metadata"（hybrid 模型 GDN、mamba）的情况。

## 附带的潜在语义问题

即使没有 GDN，这个钩子也在把 **target full-attention 层的条目** `+1`：
speculator 篡改了不属于它的 target 状态。目前没出事只是因为
`propose` 返回后没有代码再读 target 的 metadata。这是比崩溃更隐蔽的
越权写入，GDN 崩溃只是把这个语义错误显性化了。

## 责任链总结

| 环节 | 有无问题 |
| --- | --- |
| target metadata 全量构建 | 无问题（本应全量） |
| GDN metadata 无 `seq_lens` | 无问题（契约设计） |
| draft prefill 复用 target metadata | 无问题（上游设计，惰性条目无人读） |
| decode 路径 `active_layer_names` 过滤 | 无问题（工作正常） |
| `_ascend_update_seq_lens` 无差别遍历+写入 | **问题所在**：假设单一 metadata 形状，且越权修改 target 条目 |

## 触发条件

三个条件同时满足才触发：

1. Model Runner V2 + 投机解码走 `AscendAutoRegressiveSpeculator`；
2. 目标模型是 hybrid 架构，target metadata 中存在 GDN/linear-attn 条目；
3. attn_architecture 不是 DSA/SFA（否则被开头特判提前 return）。
