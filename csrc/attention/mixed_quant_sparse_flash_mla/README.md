# MixedQuantSparseFlashMla

## 产品支持情况

| 产品                                                         |  是否支持 |
| :----------------------------------------------------------  | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>                        |     √    |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>        |     ×    |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>        |     ×    |
|<term>Atlas 200I/500 A2 推理产品</term>                        |     ×    |
|<term>Atlas 推理系列产品</term>                                |     ×    |
|<term>Atlas 训练系列产品</term>                                |     ×    |

## 功能说明

- 算子功能：

  `MixedQuantSparseFlashMla`算子旨在完成量化和稀疏场景下的MLA（Multi-head Latent Attention）注意力计算，支持SWA（Sliding Window Attention）、CSA（Compressed Sparse Attention）、HCA（Heavily Compressed Attention）三类Attention计算场景。与`SparseFlashMla`的区别在于，本算子支持KV的per-token-group量化输入。该算子的三种典型场景：

  - **SWA（Sliding Window Attention）**：仅传入`ori_kv`，对原始KV做滑动窗口注意力。
  - **CSA（Compressed Sparse Attention）**：同时传入`ori_kv`、`cmp_kv`和`cmp_sparse_indices`，对原始KV窗口和topK选择出的压缩KV共同做注意力。
  - **HCA（Heavily Compressed Attention）**：同时传入`ori_kv`和`cmp_kv`，对原始KV窗口和连续压缩KV段共同做注意力。

  调用时需要使用`MixedQuantSparseFlashMlaMetadata`生成的任务列表`metadata`，在主算子执行前生成，当前版本主算子必须传入该`metadata`。典型调用流程如下：

  1. 根据调用场景准备`q`、`ori_kv`、`cmp_kv`等对应输入。
  2. 调用`MixedQuantSparseFlashMlaMetadata`生成`metadata`，作为`MixedQuantSparseFlashMla`的入参。
  3. 调用`MixedQuantSparseFlashMla`，将上一步得到的`metadata`传入主算子，生成计算结果。

- 计算公式：

  `MixedQuantSparseFlashMla`采用MLA对KV共享输入的稀疏注意力进行计算，其原理是对输入的KV进行选择性压缩与量化处理，再将Query与拼接后的KV计算结果通过Softmax得到注意力权重。

  MLA的计算公式一般定义如下，其中$\tilde{K}=\tilde{V}$为基于入参控制的实际参与计算的KV，由`ori_kv`的滑动窗口部分和`cmp_kv`的压缩部分共同组成，实际参与计算的KV范围由`cmp_ratio`、`ori_mask_mode`、`cmp_mask_mode`、`ori_win_left`、`ori_win_right`以及`cmp_sparse_indices`决定。

    $$
    O = \text{softmax}(Q@\tilde{K}^T \cdot \text{softmax\_scale})@\tilde{V}
    $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px">
  <colgroup>
  <col style="width: 220px">
  <col style="width: 150px">
  <col style="width: 700px">
  <col style="width: 180px">
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
      <td>q</td>
      <td>输入</td>
      <td>表示对应公式中的Q。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ori_kv</td>
      <td>可选输入</td>
      <td>表示对应公式中K和V的一部分，为原始不经压缩的量化KV，Key和Value共享同一份数据。由nope、rope、scale、padding拼接而成，详见quant_mode。</td>
      <td>详见quant_mode</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_kv</td>
      <td>可选输入</td>
      <td>表示对应公式中K和V的一部分，为经过压缩的量化KV，Key和Value共享同一份数据。由nope、rope、scale、padding拼接而成，详见quant_mode。</td>
      <td>详见quant_mode</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ori_sparse_indices</td>
      <td>可选输入</td>
      <td>表示原始KV topK索引，无效位置填-1。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_sparse_indices</td>
      <td>可选输入</td>
      <td>表示压缩KV topK索引，无效位置填-1。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ori_block_table</td>
      <td>可选输入</td>
      <td>表示PageAttention场景下ori_kv使用的block映射表。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_block_table</td>
      <td>可选输入</td>
      <td>表示PageAttention场景下cmp_kv使用的block映射表。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_q</td>
      <td>可选输入</td>
      <td>表示TND布局下不同batch中q的累积序列长度。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_ori_kv</td>
      <td>可选输入</td>
      <td>表示TND布局下不同batch中ori_kv的累积序列长度。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_cmp_kv</td>
      <td>可选输入</td>
      <td>表示TND布局下不同batch中cmp_kv的累积序列长度。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_q</td>
      <td>可选输入</td>
      <td>表示不同batch中q实际参与计算的token数。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_ori_kv</td>
      <td>可选输入</td>
      <td>表示不同batch中ori_kv实际参与计算的token数。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_cmp_kv</td>
      <td>可选输入</td>
      <td>表示不同batch中cmp_kv实际参与计算的token数。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_residual_kv</td>
      <td>可选输入</td>
      <td>表示压缩KV余数，用于恢复cmp侧mask使用的压缩前KV长度。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ori_topk_length</td>
      <td>可选输入</td>
      <td>表示ori_sparse_indices实际参与计算的长度。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_topk_length</td>
      <td>可选输入</td>
      <td>表示cmp_sparse_indices实际参与计算的长度。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sinks</td>
      <td>可选输入</td>
      <td>表示各注意力头设置独立可学习虚拟偏移项，用于维持长文本推理时的稳定性。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>metadata</td>
      <td>可选输入</td>
      <td>表示MixedQuantSparseFlashMlaMetadata生成的分核信息。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>quant_mode</td>
      <td>属性</td>
      <td>表示量化模式。量化模式1表示K、V nope为per-token-group量化，K、V依次由rope（64，bfloat16）、nope（448，FLOAT8_E4M3FN）、scale（7，bfloat16）、pad（18B）拼接而成；量化模式2表示K、V nope为per-token-group量化，K、V依次由nope（448，FLOAT8_E4M3FN）、rope（64，bfloat16）、scale（7，FLOAT8_E8M0）、pad（1B）拼接而成。当前仅支持1和2，量化模式2仅支持layout_kv为PA_BBND。各量化模式均支持使用UINT8、FLOAT8_E4M3FN作为单字节存储视图，底层字节内容保持不变。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rope_head_dim</td>
      <td>可选属性</td>
      <td>表示rope头的维度，仅支持64。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmax_scale</td>
      <td>可选属性</td>
      <td>表示对应公式中的softmax_scale，默认值为1.0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_ratio</td>
      <td>可选属性</td>
      <td>表示cmp_kv相对于压缩前KV长度的压缩倍率，用于恢复cmp侧mask使用的压缩前KV长度；仅传入ori_kv时不参与压缩KV计算。支持1到128。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_mask_mode</td>
      <td>可选属性</td>
      <td>表示q和ori_kv计算的mask模式。<br/>0: No mask。<br/>3: rightDownCausal模式。<br/>4: sliding window模式。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_mask_mode</td>
      <td>可选属性</td>
      <td>表示q和cmp_kv计算的mask模式。<br/>0: No mask。<br/>3: rightDownCausal模式。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_win_left</td>
      <td>可选属性</td>
      <td>表示q和ori_kv计算中q对历史token计算的数量，-1表示无穷大，即全部参与运算。默认值为-1。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_win_right</td>
      <td>可选属性</td>
      <td>表示q和ori_kv计算中q对未来token计算的数量，-1表示无穷大，即全部参与运算。默认值为-1。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_q</td>
      <td>可选属性</td>
      <td>表示输入q的数据排布格式，支持"BSND"和"TND"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_kv</td>
      <td>可选属性</td>
      <td>表示输入ori_kv和cmp_kv的数据排布格式，支持"BSND"、"TND"和"PA_BBND"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topk_value_mode</td>
      <td>可选属性</td>
      <td>表示topK索引取值模式，默认值为1。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>return_softmax_lse</td>
      <td>可选属性</td>
      <td>表示是否返回softmax的lse结果，默认值为False。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attn_out</td>
      <td>输出</td>
      <td>表示对应公式中的输出O。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>softmax_lse</td>
      <td>可选输出</td>
      <td>表示对query乘key的结果先取max得到softmax_max，query乘key的结果减去softmax_max后取exp再取sum得到softmax_sum，最后对softmax_sum取log再加上softmax_max得到的结果。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持aclgraph模式。
- 该接口支持batch一致性。
- 该接口当前支持三种计算场景：SWA（Sliding Window Attention）场景仅传入`ori_kv`；CSA（Compressed Sparse Attention）场景传入`ori_kv`、`cmp_kv`及`cmp_sparse_indices`；HCA（Heavily Compressed Attention）场景传入`ori_kv`及`cmp_kv`。

### 常见字段释义

|    命名    |                            含义                            |
| :---------: | :---------------------------------------------------------: |
|      b      |      输入样本batch大小                |
|     q_s     |      输入q的序列长度      |
|    ori_kv_s    |  输入ori_kv的序列长度  |
|    cmp_kv_s    |  输入cmp_kv的序列长度  |
|     q_n     |        输入q的头数        |
|    kv_n    |    输入ori_kv/cmp_kv的头数    |
|      q_d      |          输入q的注意力头的维度         |
|      kv_d      |          输入ori_kv/cmp_kv的注意力头的维度         |
|     q_t     |          输入q所有batch序列长度的累加和          |
|     ori_kv_t    |          输入ori_kv所有batch序列长度的累加和          |
|     cmp_kv_t    |          输入cmp_kv所有batch序列长度的累加和          |
|      ori_kv_k      |           输入ori_sparse_indices中topK选出的token个数         |
|      cmp_kv_k      |           输入cmp_sparse_indices中topK选出的token个数         |
|      ori_kv_s_max      |           输入ori_kv的最大序列长度         |
|      cmp_kv_s_max      |           输入cmp_kv的最大序列长度         |
|      ori_kv_block_size      |           输入ori_kv在PagedAttention场景下的block大小         |
|      cmp_kv_block_size      |           输入cmp_kv在PagedAttention场景下的block大小         |
|      ori_kv_block_nums      |           输入ori_kv在PagedAttention场景下的block数量         |
|      cmp_kv_block_nums      |           输入cmp_kv在PagedAttention场景下的block数量         |

- 通用规格约束如下：
  - kv_n仅支持1，q_d仅支持512。其中，`ori_kv`和`cmp_kv`的kv_d由nope、rope、scale、padding拼接而成，详见`quant_mode`。
  - `cmp_ratio`表示`cmp_kv`相对于压缩前KV长度的压缩倍率；仅传入`ori_kv`时，`cmp_ratio`不参与压缩KV计算，需保持默认值1；支持1到128。
  - `ori_mask_mode`支持0、3和4，`cmp_mask_mode`支持0和3，`ori_win_left`和`ori_win_right`支持-1或非负数，-1表示对应方向不受限，只有`ori_mask_mode`为4时，`ori_win_left`和`ori_win_right`可以>=0。
  - `rope_head_dim`仅支持64。
  - `layout_q`和`layout_kv`组合仅支持"BSND"/"BSND"、"TND"/"TND"、"BSND"/"PA_BBND"、"TND"/"PA_BBND"；非PA_BBND场景下`layout_q`和`layout_kv`必须一致；PA_BBND场景下`block_size`支持1到1024。
- 当`layout_q`为TND时，功能使用限制如下：
  - `q`的shape需要为[q_t, q_n, q_d]。
  - `ori_sparse_indices`的shape需要为[q_t, kv_n, ori_kv_k]。
  - `cmp_sparse_indices`的shape需要为[q_t, kv_n, cmp_kv_k]。
  - `cu_seqlens_q`必须传入，输入维度为b+1，每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须>=前一个元素的值且首元素必须为0。
- 当`layout_q`为BSND时，功能使用限制如下：
  - `q`的shape需要为[b, q_s, q_n, q_d]。
  - `cmp_sparse_indices`的shape需要为[b, q_s, kv_n, ori_kv_k]。
  - `cmp_sparse_indices`的shape需要为[b, q_s, kv_n, cmp_kv_k]。
- PageAttention场景下，功能使用限制如下：
  - `ori_kv`和`cmp_kv`的shape分别为[ori_kv_block_nums, ori_kv_block_size, kv_n, kv_d]和[cmp_kv_block_nums, cmp_kv_block_size, kv_n, kv_d]，其中ori_kv_block_nums和cmp_kv_block_nums为PagedAttention场景下的block数量，ori_kv_block_size和cmp_kv_block_size为一个block的token数，取值为1到1024。
  - `ori_block_table`和`cmp_block_table`的shape为2维，其中第一维长度为b，第二维长度不小于所有batch中最大的ori_kv_s和cmp_kv_s对应的block数量，即ori_kv_s_max / ori_kv_block_size和cmp_kv_s_max / cmp_kv_block_size向上取整。
- `metadata`为算子实际需要使用的分核结果，目前该参数必传，shape大小固定为[1024]。
- `layout_kv`支持输入"BSND"、"TND"和"PA_BBND"，需满足上述`layout_q`和`layout_kv`组合约束。
  - 当输入为PA_BBND时，`seqused_ori_kv`和`ori_block_table`必须传入；当输入为BSND时，`seqused_ori_kv`可用于表达每个batch的`ori_kv`有效长度；当输入为TND时，`ori_kv`有效长度由`cu_seqlens_ori_kv`表达。
  - 当输入为BSND时，`ori_kv`和`cmp_kv`的layout都必须为BSND，ori_kv的shape为[b, ori_kv_s, kv_n, kv_d]，cmp_kv的shape为[b, cmp_kv_s, kv_n, kv_d]。
  - 当输入为TND时，`cu_seqlens_ori_kv`必须传入；若存在`cmp_kv`，`cu_seqlens_cmp_kv`也必须传入。
- `return_softmax_lse`为False时返回占位Tensor；为True时返回softmax的log-sum-exp结果。
- 除`ori_topk_length`和`cmp_topk_length`等预留输入可不传或传入空Tensor外，其余已传入Tensor不支持为空。
- `seqused_cmp_kv`为所有`layout_kv`下的可选输入，显式传入时用于覆盖cmp侧逻辑有效长度；未传时由`cmp_kv` shape、`cu_seqlens_cmp_kv`或PA block table相关语义推导。
- `cmp_residual_kv`为算子的可选入参；传入后用于按`cmp_len * cmp_ratio + residual`恢复cmp侧mask使用的压缩前KV长度，其中`cmp_len`优先来自显式传入的`seqused_cmp_kv`。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| -------- | ------------------------------------------------------------- | ------------------------------------------------------------- |
| aclnn API | [test_aclnn_mixed_quant_sparse_flash_mla](./examples/test_aclnn_mixed_quant_sparse_flash_mla.cpp) | 通过[aclnnMixedQuantSparseFlashMla](./docs/aclnnMixedQuantSparseFlashMla.md)调用MixedQuantSparseFlashMla算子 |
| PyTorch API | [mixed_quant_sparse_flash_mla](../../attention/mixed_quant_sparse_flash_mla/docs/torchapi_mixed_quant_sparse_flash_mla.md) | 通过`cann_ops_transformer.mixed_quant_sparse_flash_mla`调用MixedQuantSparseFlashMla算子 |
