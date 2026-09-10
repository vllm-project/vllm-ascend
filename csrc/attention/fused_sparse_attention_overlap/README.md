# FusedSparseAttentionOverlap

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|√|
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|√|
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|√|
|<term>Atlas 200I/500 A2 推理产品</term>|×|
|<term>Atlas 推理系列产品</term>|×|
|<term>Atlas 训练系列产品</term>|×|

## 功能说明

- API功能：`fused_sparse_attention_overlap`面向大序列稀疏注意力推理场景，在一个算子中完成Selection KV Cache h2d更新和Sparse Flash Attention计算。算子根据TopK索引及缓存元数据判断KV是否命中Selection KV Cache；未命中的KV从Full KV Cache加载并写入Selection KV Cache，同时通过流水调度将KV搬运与注意力计算重叠，减少独立Gather和Sparse Flash Attention串行执行的开销。

- 计算公式：

    $$
    \text{softmax}(\text{scaleValue} \cdot (Q_{NoPE}@\tilde{K}_{NoPE}^{T} + Q_{RoPE}@\tilde{K}_{RoPE}^{T}))@\tilde{V}
    $$

    其中，$\tilde{K}$和$\tilde{V}$由`selectionTopkIndices`指定。命中项从Selection KV Cache读取，未命中项从Full KV Cache读取。算子执行后会原地更新`selectionKRoPE`、`selectionKvCache`、`selectionKvBlockStatus`和`selectionMembershipMap`，供后续Token复用。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1080px"><colgroup>
<col style="width: 200px">
<col style="width: 150px">
<col style="width: 480px">
<col style="width: 150px">
<col style="width: 100px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出/属性</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>query</td>
    <td>输入</td>
    <td>MLA结构的Query输入，不支持空tensor和非连续。当前仅支持TND排布，shape为(T1,N1,D+Dr)。最后一维依次存放D维NoPE数据和Dr维RoPE数据。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>selectionKRoPE</td>
    <td>输入/输出（原地更新）</td>
    <td>NPU侧Selection K RoPE缓存，采用PageAttention排布，shape为(selection_block_num,selection_block_size,Dr)。未命中的RoPE数据会从fullKRoPE加载并写入该tensor。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>selectionKvCache</td>
    <td>输入/输出（原地更新）</td>
    <td>NPU侧Selection KV Cache，采用PageAttention排布，shape为(selection_block_num,selection_block_size,D)。未命中的KV数据会从fullKvCache加载并写入该tensor。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>selectionKvBlockTable</td>
    <td>输入</td>
    <td>Selection KV Cache的逻辑块到物理块映射表，shape为(B,selection_block_count)。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>selectionKvBlockStatus</td>
    <td>输入/输出（原地更新）</td>
    <td>Selection KV Cache的驻留Token状态，shape为(B,N2,status_stride)。前TopK个位置记录驻留Token ID，并使用附加位置记录有效数量。status_stride需要满足算子对齐要求。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>selectionMembershipMap</td>
    <td>输入/输出（原地更新）</td>
    <td>Selection KV Cache的Token到Slot映射及紧凑搬运计划，shape为(B,N2,membership_stride)。既可由算子内部Planner维护，也可承载框架生成的External Plan。</td>
    <td>INT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>selectionTopkIndices</td>
    <td>输入</td>
    <td>Indexer输出的TopK Token索引，不支持空tensor。支持三维(T1,N2,TopK)或可展平为该排布的四维输入。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>fullKRoPE</td>
    <td>输入</td>
    <td>完整K RoPE缓存，采用PageAttention排布，shape为(full_block_num,block_size,Dr)。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>fullKvCache</td>
    <td>输入</td>
    <td>完整KV Cache，采用PageAttention排布，shape为(full_block_num,block_size,D)。作为Selection KV Cache未命中时的数据源。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>fullKvBlockTable</td>
    <td>输入</td>
    <td>Full KV Cache的逻辑块到物理块映射表，shape为(B,full_block_count)。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>fullKvActualSeq</td>
    <td>输入</td>
    <td>各Batch的Full KV有效序列长度，shape为(B)。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>fullQActualSeq</td>
    <td>输入</td>
    <td>TND排布下Query的有效序列信息，元素个数需要与query第一维对应。Decode场景中每个Query Token的长度为1。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>scaleValue</td>
    <td>属性</td>
    <td>Query与Key乘积的缩放系数，通常设置为1/sqrt(D+Dr)。</td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>sparseBlockSize</td>
    <td>属性</td>
    <td>稀疏选择粒度。当前融合路径仅支持Token-wise稀疏化，取值为1。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>selectionTopkBlockSize</td>
    <td>属性</td>
    <td>TopK索引的选择粒度，当前仅支持1。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>layoutQuery</td>
    <td>属性</td>
    <td>query的数据排布格式，当前仅支持"TND"。</td>
    <td>STRING</td>
    <td>-</td>
  </tr>
  <tr>
    <td>layoutKv</td>
    <td>属性</td>
    <td>Full KV Cache的数据排布格式，当前仅支持PageAttention排布"PA_BSND"。</td>
    <td>STRING</td>
    <td>-</td>
  </tr>
  <tr>
    <td>sparseMode</td>
    <td>属性</td>
    <td>稀疏模式。当前Decode融合路径使用3，表示rightDownCausal模式。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>attentionOut</td>
    <td>输出</td>
    <td>Attention计算结果，shape为(T1,N1,D)。输出不包含Query的RoPE维度。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## 约束说明

- 该接口仅支持推理场景，并支持图模式。
- 当前编译的模板仅支持GLM5.2 Decode使用的组合：`FLASH_DECODE=0`、`layoutQuery="TND"`、`layoutKv="PA_BSND"`、`TEMPLATE_MODE=V_TEMPLATE`、`IS_SPLIT_G=0`。
- 当前支持FLOAT16和BFLOAT16，`query`、Full KV Cache及Selection KV Cache的数据类型必须保持一致。
- 当前仅支持MLA-absorb形态，NoPE维度D为512，RoPE维度Dr为64；`query`最后一维为576，`attentionOut`最后一维为512。
- 当前验证的分组头配置为N1=4、N2=1，PageAttention的`block_size`为128。
- `sparseBlockSize`和`selectionTopkBlockSize`必须为1，TopK最大支持2048。
- `selectionKvBlockStatus`最后一维至少容纳TopK个Token ID和一个有效数量字段，并按8个INT32元素对齐。
- 当前`selectionMembershipMap`每行包含16376个Token映射位置、8个控制字段及对齐空间，`membership_stride`为16400。
- `selectionTopkIndices`中的有效Token ID必须在对应Batch的`fullKvActualSeq`范围内。
- Selection KV Cache容量不得小于TopK，Full KV Cache和Selection KV Cache的Block Table必须覆盖所有被访问的逻辑块。

## 调用示例

```python
import math

import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

output = torch.ops._C_ascend.npu_fused_sparse_attention_overlap(
    query=query,
    selection_k_rope=selection_k_rope,
    selection_kv_cache=selection_kv_cache,
    selection_kv_block_table=selection_kv_block_table,
    selection_kv_block_status=selection_kv_block_status,
    selection_membership_map=selection_membership_map,
    selection_topk_indices=selection_topk_indices,
    full_k_rope=full_k_rope,
    full_kv_cache=full_kv_cache,
    full_kv_block_table=full_kv_block_table,
    full_kv_actual_seq=full_kv_actual_seq,
    full_q_actual_seq=full_q_actual_seq,
    scale_value=1.0 / math.sqrt(576),
    sparse_block_size=1,
    selection_topk_block_size=1,
    layout_query="TND",
    layout_kv="PA_BSND",
    sparse_mode=3,
)
```

完整的Tensor构造、首次Miss及后续All-Hit精度校验请参考
[test_fused_sparse_attention_overlap.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_sparse_attention_overlap.py)。
