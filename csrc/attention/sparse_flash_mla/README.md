# SparseFlashMla

开发者学习入口：[算子设计与开发指南](./docs/design.md)，包含 tiling、Metadata 分核、内存分配、计算流水、架构差异与验证建议。

倍率扩展：[A2/A3 cmp_ratio=1/2 适配说明](./docs/ratio2_a2a3.md)。

## 产品支持情况

| 产品                                                          | 是否支持 |
| :------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>                         |     √    |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>         |     √    |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>         |     √    |
|<term>Atlas 200I/500 A2推理系列产品</term>                      |     ×    |
|<term>Atlas 推理系列产品</term>                                 |     ×    |
|<term>Atlas 训练系列产品</term>                                 |     ×    |

## 功能说明

- 算子功能：

  `SparseFlashMla`算子旨在完成以下公式描述的Attention计算，支持SWA（Sliding Window Attention）、CSA（Compressed Sparse Attention）、HCA（Heavily Compressed Attention）三类Attention计算场景。调用时需要使用`SparseFlashMlaMetadata`生成的任务列表`metadata`。

  典型调用流程如下：

  1. 准备`q`、`ori_kv`、`cmp_kv`、序列长度、`block table`、`sinks`等输入。
  2. 调用`SparseFlashMlaMetadata`生成`metadata`。
  3. 调用`SparseFlashMla`，将上一步得到的`metadata`传入主算子。

- 计算公式：

    $$
    O = \text{softmax}(Q@\tilde{K}^T \cdot \text{softmax\_scale})@\tilde{V}
    $$

    其中$\tilde{K}=\tilde{V}$为基于ori_kv、cmp_kv以及cmp_ratio等入参控制的实际参与计算的 $KV$。

## 参数说明
>
> **说明：**<br>
> 参数维度含义：B表示Batch Size，Q_S、ORI\_KV\_S和CMP\_KV\_S分别表示query、oriKv和cmpKv的Sequence Length，Q_N和KV_N分别表示query和key/value的Head Num，Q_T、ORI_KV_T和CMP_KV_T分别表示query、oriKv和cmpKv的Total Tokens。

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
      <td>对应公式中的Q。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ori_kv</td>
      <td>可选输入</td>
      <td>对应公式中K和V的一部分，表示原始不经压缩的KV。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_kv</td>
      <td>可选输入</td>
      <td>对应公式中K和V的一部分，表示经过压缩的KV。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ori_sparse_indices</td>
      <td>可选输入</td>
      <td>SWA稀疏ori_kv场景表示从ori_kv中离散取数的逻辑索引，-1表示无效或填充slot，shape为(T1, N2, K)或(B, S1, N2, K)。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_sparse_indices</td>
      <td>可选输入</td>
      <td>表示从cmp_kv中离散取数的索引。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ori_block_table</td>
      <td>可选输入</td>
      <td>表示PageAttention中ori_kv使用的block映射表。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_block_table</td>
      <td>可选输入</td>
      <td>表示PageAttention中cmp_kv使用的block映射表。</td>
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
      <td>SWA稀疏ori_kv场景表示不同q token对应的ori_kv关键稀疏token的实际个数，shape为(T1, N2)或(B, S1, N2)。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_topk_length</td>
      <td>可选输入</td>
      <td>预留输入，当前版本不支持传入非空Tensor。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sinks</td>
      <td>可选输入</td>
      <td>表示attention sinks输入。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>metadata</td>
      <td>可选输入</td>
      <td>配套metadata前置接口生成的任务切分结果。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>softmax_scale</td>
      <td>可选属性</td>
      <td>对应公式中的softmax_scale。</td>
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
      <td>表示q和ori_kv计算的mask模式。<br/>0: No Mask。<br/>3: RightDownCausal模式。<br/>4: Band模式。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_mask_mode</td>
      <td>可选属性</td>
      <td>表示q和cmp_kv计算的mask模式。<br/>0: No Mask。<br/>3: RightDownCausal模式。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_win_left</td>
      <td>可选属性</td>
      <td>表示q和ori_kv计算中q对过去token计算的数量，支持-1或非负数，其中-1表示窗口不受限。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_win_right</td>
      <td>可选属性</td>
      <td>表示q和ori_kv计算中q对未来token计算的数量，支持-1或非负数，其中-1表示窗口不受限。</td>
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
      <td>表示TopK索引取值模式。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>return_softmax_lse</td>
      <td>可选属性</td>
      <td>表示是否返回softmax_lse。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attn_out</td>
      <td>输出</td>
      <td>对应公式中的输出O。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>softmax_lse</td>
      <td>可选输出</td>
      <td>返回softmax的log-sum-exp结果。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口支持aclgraph模式。
- 该接口支持batch一致性。
- 该接口当前支持四种计算场景：SWA（Sliding Window Attention）场景仅传入`ori_kv`；SWA稀疏ori_kv场景传入`ori_kv`、`ori_sparse_indices`及`ori_topk_length`；CSA（Compressed Sparse Attention）场景传入`ori_kv`、`cmp_kv`及`cmp_sparse_indices`；HCA（Heavily Compressed Attention）场景传入`ori_kv`及`cmp_kv`。
- 通用规格约束如下：
    - KV\_N仅支持1，D仅支持512。其中，`ori_kv`和`cmp_kv`的D_kv由nope(448)和rope(64)拼接而成。
    - `cmp_ratio`表示`cmp_kv`相对于压缩前KV长度的压缩倍率；仅传入`ori_kv`时，`cmp_ratio`不参与压缩KV计算，取值为0；传入`cmp_kv`时支持1到128。
    - `ori_mask_mode`、`cmp_mask_mode`、`ori_win_left`和`ori_win_right`的取值随产品型号而变化，详见“产品型号约束”。
    - PageAttention的block_size支持1到1024。
    - `layout_q`和`layout_kv`组合仅支持"BSND"/"BSND"、"TND"/"TND"、"BSND"/"PA_BBND"、"TND"/"PA_BBND"；非PA_BBND场景下`layout_q`和`layout_kv`必须一致。
    - SWA稀疏ori_kv场景下，`ori_topk_length`必须传入，配套Metadata接口的`ori_topk`为`ori_sparse_indices`最后一维K，且`ori_topk_length`的元素取值应在[0, K]范围内；其他场景`ori_topk_length`传入nullptr或空Tensor。
- 产品型号约束如下：
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：Q\_N支持1、2、4、8、16、32、64、128，KV\_N只支持1；cmp_ratio在SWA场景传入0，CSA支持传入1、2或4，HCA支持传入128；block_size取值为16的倍数，最大支持1024；SWA稀疏ori_kv场景支持`ori_sparse_indices`和`ori_topk_length`，`ori_mask_mode`为0，`ori_win_left`和`ori_win_right`为非负数；非SWA稀疏ori_kv场景的`ori_mask_mode`为4、`ori_win_left`为127、`ori_win_right`为0，`cmp_sparse_indices`的最后一维K2当前支持512或1024，`cmp_mask_mode`仅支持3。
    - <term>Ascend 950PR/Ascend 950DT</term>：Q\_N支持1-128，KV\_N只支持1。`ori_mask_mode`支持0、3、4，`cmp_mask_mode`支持0、3；`ori_win_left`和`ori_win_right`支持-1或非负数，-1表示对应方向不受限。只有`ori_mask_mode`为4时，`ori_win_left`和`ori_win_right`可以>=0。

- 当`layout_q`为TND时，功能使用限制如下：
    - `q`的shape需要为[Q\_T, Q\_N, D]。
    - SWA稀疏ori_kv场景下，`ori_sparse_indices`的shape为[Q\_T, KV\_N, K]，`ori_topk_length`的shape为[Q\_T, KV\_N]。
    - `cmp_sparse_indices`的shape需要为[Q\_T, KV\_N, K2]，其中K2为对`cmp_kv`一次离散选取的token数。
    - `cu_seqlens_q`必须传入，shape为[B+1,]，第一个数固定为0，即前缀0。后面的每个元素表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。

- 当`layout_q`为BSND时，功能使用限制如下：
    - `q`的shape需要为[B, Q\_S, Q\_N, D]。
    - SWA稀疏ori_kv场景下，`ori_sparse_indices`的shape为[B, Q\_S, KV\_N, K]，`ori_topk_length`的shape为[B, Q\_S, KV\_N]。
    - `cmp_sparse_indices`的shape需要为[B, Q\_S, KV\_N, K2]，其中K2为对`cmp_kv`一次离散选取的token数。

- PageAttention场景下，功能使用限制如下：
    - `ori_kv`和`cmp_kv`的shape分别为[ori\_block\_num, ori\_block\_size, KV\_N, D]和[cmp\_block\_num, cmp\_block\_size, KV\_N, D]，其中ori\_block\_num和cmp\_block\_num为PageAttention时block总数，ori\_block\_size和cmp\_block\_size为一个block的token数，ori\_block\_size和cmp\_block\_size取值支持1到1024，KV_N仅支持1。
    - `ori_block_table`和`cmp_block_table`的shape为2维，其中第一维长度为B，第二维长度不小于所有batch中最大的S2和S3对应的block数量，即ORI\_KV\_S\_max / block\_size和CMP\_KV\_S\_max / block\_size向上取整。
- `metadata`为算子实际需要使用的分核结果，目前该参数必传，shape大小固定为[1024]。
- `layout_kv`支持输入"BSND"、"TND"和"PA_BBND"，需满足上述`layout_q`和`layout_kv`组合约束。
    - 当输入为PA_BBND时，`seqused_ori_kv`和`ori_block_table`必须传入；当输入为BSND时，`seqused_ori_kv`可用于表达每个batch的`ori_kv`有效长度，且有效长度不超过key/value中的KV_S的大小且不小于0；当输入为TND时，`ori_kv`总长度由`cu_seqlens_ori_kv`表达，若`seqused_ori_kv`被传入，则`ori_kv`的有效长度由`seqused_ori_kv`表达，否则有效长度与总长度相同。
    - 当输入为BSND时，`ori_kv`和`cmp_kv`的layout都必须为BSND，ori_kv的shape为[B, ORI\_KV\_S, KV\_N, D]，cmp_kv的shape为[B, CMP\_KV\_S, KV\_N, D]。
    - 当输入为TND时，`cu_seqlens_ori_kv`必须传入；若存在`cmp_kv`，`cu_seqlens_cmp_kv`也必须传入。
- `return_softmax_lse`为False时返回占位Tensor；为True时返回softmax的log-sum-exp结果。
- SWA稀疏ori_kv场景仅支持SWA模板，仅传入`ori_kv`，必须同时传入`ori_sparse_indices`和`ori_topk_length`，`ori_win_left`和`ori_win_right`仅在`ori_mask_mode`为4时取非负数；配套Metadata接口的`ori_topk`为K，该场景不传入`cmp_kv`。
- `ori_topk_length`表示每个q token和KV head的实际有效索引条目数，取值应在[0, K]范围内。`ori_sparse_indices`的[0, ori_topk_length)区间为左对齐的有效索引条目，[ori_topk_length, K)区间为无效或填充条目，建议填-1。
- 除`cmp_topk_length`等预留输入可不传或传入空Tensor外，其余已传入Tensor不支持为空。
- `seqused_cmp_kv`为所有`layout_kv`下的可选输入，显式传入时用于覆盖cmp侧逻辑有效长度；未传时由`cmp_kv` shape、`cu_seqlens_cmp_kv`或PA block table相关语义推导。
- `cmp_residual_kv`为主接口和metadata前置接口的可选入参；传入后用于按`cmp_len * cmp_ratio + residual`恢复cmp侧mask使用的压缩前KV长度，其中`cmp_len`优先来自显式传入的`seqused_cmp_kv`。
- `q`、`ori_kv`、`cmp_kv`数据排布格式支持从多种维度解读，B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Hidden-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示hidden层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
- Q\_S表示q shape中的S，ORI\_KV\_S表示ori_kv shape中的S，CMP\_KV\_S表示cmp_kv shape中的S；Q\_N表示num\_q\_heads，KV\_N表示num\_ori_kv\_heads和num\_cmp_kv\_heads；Q\_T表示q shape中的输入样本序列长度的累加和。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnnSparseFlashMla](./examples/test_aclnn_sparse_flash_mla.cpp) | 通过[aclnnSparseFlashMla](./docs/aclnnSparseFlashMla.md)调用SparseFlashMla算子 |
| PyTorch API | [sparse_flash_mla](../../torch_extension/cann_ops_transformer/docs/zh/sparse_flash_mla.md) | 通过`cann_ops_transformer.sparse_flash_mla`调用SparseFlashMla算子 |
