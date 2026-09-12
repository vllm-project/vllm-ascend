# SparseFlashMlaMetadata

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>                        | √  |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>        | √  |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>        | √  |
|<term>Atlas 200I/500 A2推理系列产品</term>                    | ×  |
|<term>Atlas 推理系列产品</term>                                | ×  |
|<term>Atlas 训练系列产品</term>                                | ×  |

## 功能说明

- 算子功能：`SparseFlashMlaMetadata`是`SparseFlashMla`算子的前置算子，用于后续Attention计算生成负载均衡的任务划分方案。本算子不执行实际的Attention计算，而是根据输入参数在AI CPU计算出每个AI Core应处理的Attention计算起止范围，从而最大化计算资源的利用率，避免各Core间负载不均衡的问题。

- 场景简称：SWA（Sliding Window Attention）、CSA（Compressed Sparse Attention）、HCA（Heavily Compressed Attention）。

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
      <td>num_heads_q</td>
      <td>属性</td>
      <td>表示`q`的头数，支持[1, 128]。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_heads_kv</td>
      <td>属性</td>
      <td>表示`ori_kv`和`cmp_kv`的头数，仅支持1。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>head_dim</td>
      <td>属性</td>
      <td>注意力头的维度，仅支持512。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cu_seqlens_q</td>
      <td>可选输入</td>
      <td>表示TND布局下不同batch中`q`的累积序列长度，shape为(B+1, )。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_ori_kv</td>
      <td>可选输入</td>
      <td>表示TND布局下不同batch中`ori_kv`的累积序列长度，shape为(B+1, )。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_cmp_kv</td>
      <td>可选输入</td>
      <td>表示TND布局下不同batch中`cmp_kv`的累积序列长度，shape为(B+1, )。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_q</td>
      <td>可选输入</td>
      <td>表示不同batch中`q`实际参与计算的token数，shape为(B, )。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_ori_kv</td>
      <td>可选输入</td>
      <td>表示不同batch中`ori_kv`实际参与计算的token数，shape为(B, )。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_cmp_kv</td>
      <td>可选输入</td>
      <td>表示不同batch中`cmp_kv`实际参与计算的token数，shape为(B, )。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_residual_kv</td>
      <td>可选输入</td>
      <td>表示压缩KV余数，用于恢复cmp侧mask使用的压缩前KV长度，shape为(B, )。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ori_topk_length</td>
      <td>可选输入</td>
      <td>SWA稀疏ori_kv场景表示不同q token对应的ori_kv部分关键稀疏token的个数，必须传入，shape为(B, S1, N2)或(T1, N2)。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_topk_length</td>
      <td>可选输入</td>
      <td>表示不同q token对应的cmp_kv部分关键稀疏token的个数，shape为(B, S1, N2)或(T1, N2)。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>batch_size</td>
      <td>可选属性</td>
      <td>表示输入样本批量大小；传入0时表示由接口推导，默认值为0。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_q</td>
      <td>可选属性</td>
      <td>表示所有batch中`q`的最大有效token数；传入0时表示由接口推导，默认值为0。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_ori_kv</td>
      <td>可选属性</td>
      <td>表示所有batch中`ori_kv`的最大有效token数；传入0时表示由接口推导，默认值为0。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_cmp_kv</td>
      <td>可选属性</td>
      <td>表示所有batch中`cmp_kv`的最大有效token数；传入0时表示由接口推导，默认值为0。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_topk</td>
      <td>可选属性</td>
      <td>表示从`ori_kv`中筛选出的关键稀疏token个数；SWA稀疏ori_kv场景为主算子`ori_sparse_indices`最后一维K且必须大于0，其他场景默认值为0。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_topk</td>
      <td>可选属性</td>
      <td>表示从`cmp_kv`中筛选出的关键稀疏token个数，默认值为0。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_ratio</td>
      <td>可选属性</td>
      <td>表示`cmp_kv`相对于压缩前KV长度的压缩倍率，用于恢复cmp侧mask使用的压缩前KV长度；仅传入`ori_kv`时不参与压缩KV计算。传入`cmp_kv`时支持[1, 128]，默认值为0。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_mask_mode</td>
      <td>可选属性</td>
      <td>表示`q`和`ori_kv`计算的mask模式，默认值为0。<br/>0: No Mask。<br/>3: RightDownCausal模式。<br/>4: Band模式。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_mask_mode</td>
      <td>可选属性</td>
      <td>表示`q`和`cmp_kv`计算的mask模式，默认值为0。<br/>0: No Mask。<br/>3: RightDownCausal模式。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_win_left</td>
      <td>可选属性</td>
      <td>表示`q`和`ori_kv`计算中`q`对过去token计算的数量，支持-1或非负数，其中-1表示窗口不受限，默认值为-1。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_win_right</td>
      <td>可选属性</td>
      <td>表示`q`和`ori_kv`计算中`q`对未来token计算的数量，支持-1或非负数，其中-1表示窗口不受限，默认值为-1。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_q</td>
      <td>可选属性</td>
      <td>表示输入`q`的数据排布格式，支持"BSND"和"TND"，默认值为"BSND"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_kv</td>
      <td>可选属性</td>
      <td>表示输入`ori_kv`和`cmp_kv`的数据排布格式，支持"BSND"、"TND"和"PA_BBND"，默认值为"BSND"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>has_ori_kv</td>
      <td>可选属性</td>
      <td>表示`SparseFlashMla`主算子是否传入`ori_kv`，默认值为true。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>has_cmp_kv</td>
      <td>可选属性</td>
      <td>表示`SparseFlashMla`主算子是否传入`cmp_kv`，默认值为true。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadata</td>
      <td>输出</td>
      <td>表示`SparseFlashMla`主算子使用的任务切分结果，shape固定为(1024, )。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>
<ul>
  <li><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> ：num_heads_q/num_heads_kv仅支持1、2、4、8、16、32、64、128，不支持seqused_q、cmp_topk_length；SWA稀疏ori_kv场景支持ori_topk_length、ori_topk大于0及ori_mask_mode为0，ori_win_left和ori_win_right支持非负数；其他SWA场景ori_topk为0、ori_mask_mode为4、ori_win_left为127、ori_win_right为0；cmp_topk仅支持0、512、1024，cmp_mask_mode仅支持3，cmp_ratio在SWA支持0、CSA支持1、2或4、HCA支持128。</li>
  <li><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> ：num_heads_q/num_heads_kv仅支持1、2、4、8、16、32、64、128，不支持seqused_q、cmp_topk_length；SWA稀疏ori_kv场景支持ori_topk_length、ori_topk大于0及ori_mask_mode为0，ori_win_left和ori_win_right支持非负数；其他SWA场景ori_topk为0、ori_mask_mode为4、ori_win_left为127、ori_win_right为0；cmp_topk仅支持0、512、1024，cmp_mask_mode仅支持3，cmp_ratio在SWA支持0、CSA支持1、2或4、HCA支持128。</li>
</ul>

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口支持aclgraph模式。
- 通用规格约束如下：
    - B（Batch）表示输入样本批量大小，q、ori_kv、cmp_kv为配套的SparseFlashMla算子的入参，S1表示layout_q=BSND时，q shape中的S轴的大小，T1表示layout_q=TND时，q shape中的T轴的大小，S2表示layout_kv=BSND时，ori_kv shape中的S轴的大小，S3表示layout_kv=BSND时，cmp_kv shape中的S轴的大小，N2表示ori_kv、cmp_kv shape中的N轴的大小。
    - 参数`cu_seqlens_q`、`cu_seqlens_ori_kv`及`cu_seqlens_cmp_kv`要求其值为当前Batch与前序Batch有效token数的累加值，第一个元素固定为0，后一个元素的值必须大于等于前一个元素的值。
    - 参数`seqused_q`、`seqused_ori_kv`、`seqused_cmp_kv`要求其值表示每个Batch中的有效token数。
    - `layout_q`和`layout_kv`组合仅支持"BSND"/"BSND"、"TND"/"TND"、"BSND"/"PA_BBND"、"TND"/"PA_BBND"；非PA_BBND场景下`layout_q`和`layout_kv`必须一致。
    - 参数`cmp_residual_kv`需满足`cmp_residual_kv`[i] < `cmp_ratio`。
- Ascend 950PR/Ascend 950DT约束：
    - has_ori_kv为true时，ori_topk大于0认为ori_kv部分是稀疏的，ori_topk为0则认为ori_kv部分是非稀疏的。
    - has_cmp_kv为true时，cmp_topk大于0认为cmp_kv部分是稀疏的，cmp_topk为0则认为cmp_kv部分是非稀疏的。
    - has_ori_kv为true，ori_topk不为0且ori_mask_mode为0时，ori_topk_length必须传入，此时取ori_mask_mode规则与ori_topk_length元素的最小值作为当前q token对应的ori_kv的有效seqlen，其他ori_kv稀疏场景取ori_mask_mode规则与ori_topk的最小值作为当前q token对应的ori_kv的有效seqlen。
    - has_cmp_kv为true，cmp_topk不为0且cmp_mask_mode为0时，cmp_topk_length必须传入，此时取cmp_mask_mode规则与cmp_topk_length元素的最小值作为当前q token对应的cmp_kv的有效seqlen，其他cmp_kv稀疏场景取cmp_mask_mode规则与cmp_topk的最小值作为当前q token对应的cmp_kv的有效seqlen。
    - layout_q=BSND场景
        - max_seqlen_q必须传入S1的值。
    - layout_kv=BSND场景
        - has_ori_kv为true时，max_seqlen_ori_kv必须传入S2的值。
        - has_cmp_kv为true时，max_seqlen_cmp_kv必须传入S3的值。
    - layout_q=TND场景
        - cu_seqlens_q必须传入。
    - layout_kv=TND场景
        - has_ori_kv为true时，cu_seqlens_ori_kv必须传入。
        - has_cmp_kv为true时，cu_seqlens_cmp_kv必须传入。
    - layout_kv=PA_BBND场景
        - has_ori_kv为true，ori_topk不为0且ori_mask_mode为0时（ori_topk_length必传场景），seqused_ori_kv可选传入，其他场景seqused_ori_kv必须传入。
        - has_cmp_kv为true，cmp_topk不为0且cmp_mask_mode为0时（cmp_topk_length必传场景），seqused_cmp_kv可选传入，其他场景seqused_cmp_kv必须传入。
    - Batch取值规则
        - layout_q为BSND时，优先通过seqused_q的shape推导batch，seqused_q未传入则通过batch_size获取batch数。
        - layout_q为TND时，优先通过seqused_q的shape推导batch，seqused_q未传入则通过cu_seqlens_q的shape推导batch。
    - q Seqlen取值规则
        - layout_q为BSND时，优先通过seqused_q中的元素获取seqlen，seqused_q未传入则通过max_seqlen_q获取seqlen。
        - layout_q为TND时，优先通过seqused_q中的元素获取seqlen，seqused_q未传入则通过cu_seqlens_q中的元素获取seqlen。
    - ori_kv Seqlen取值规则
        - layout_kv为BSND时，优先通过seqused_ori_kv中的元素获取seqlen，seqused_ori_kv未传入则通过max_seqlen_ori_kv获取seqlen。
        - layout_kv为TND时，优先通过seqused_ori_kv中的元素获取seqlen，seqused_ori_kv未传入则通过cu_seqlens_ori_kv中的元素获取seqlen。
        - layout_kv为PA_BBND时，优先通过seqused_ori_kv中的元素获取seqlen，seqused_ori_kv未传入则通过ori_topk_length获取seqlen。
    - cmp_kv Seqlen取值规则
        - layout_kv为BSND时，优先通过seqused_cmp_kv中的元素获取seqlen，seqused_cmp_kv未传入则通过max_seqlen_cmp_kv获取seqlen。
        - layout_kv为TND时，优先通过seqused_cmp_kv中的元素获取seqlen，seqused_cmp_kv未传入则通过cu_seqlens_cmp_kv中的元素获取seqlen。
        - layout_kv为PA_BBND时，优先通过seqused_cmp_kv中的元素获取seqlen，seqused_cmp_kv未传入则通过cmp_topk_length获取seqlen。
- Atlas A3 训练系列产品/Atlas A3 推理系列产品约束：
    - SWA稀疏ori_kv场景下，仅支持SWA模板，`has_ori_kv`为true、`has_cmp_kv`为false、`ori_topk`大于0、`ori_mask_mode`为0，`ori_win_left`和`ori_win_right`为非负数，且必须传入`ori_topk_length`。`ori_topk`应与配套主算子`ori_sparse_indices`最后一维K保持一致；`ori_topk_length`表示每个q token和KV head的左对齐有效索引条目数，取值应在[0, K]范围内；Metadata仅使用`ori_topk_length`生成任务切分。配套主算子在PA_BBND场景仍要求传入`seqused_ori_kv`。
    - `cmp_ratio`表示`cmp_kv`相对于压缩前KV长度的压缩倍率；仅传入`ori_kv`时传0且不参与压缩KV计算。CSA场景传1、2或4，HCA场景传128。
    - `cmp_topk`在CSA场景支持512或1024，SWA、HCA场景传0。
- Atlas A2 训练系列产品/Atlas A2 推理系列产品约束：
    - SWA稀疏ori_kv场景下，仅支持SWA模板，`has_ori_kv`为true、`has_cmp_kv`为false、`ori_topk`大于0、`ori_mask_mode`为0，`ori_win_left`和`ori_win_right`为非负数，且必须传入`ori_topk_length`。`ori_topk`应与配套主算子`ori_sparse_indices`最后一维K保持一致；`ori_topk_length`表示每个q token和KV head的左对齐有效索引条目数，取值应在[0, K]范围内；Metadata仅使用`ori_topk_length`生成任务切分。配套主算子在PA_BBND场景仍要求传入`seqused_ori_kv`。
    - `cmp_ratio`表示`cmp_kv`相对于压缩前KV长度的压缩倍率；仅传入`ori_kv`时传0且不参与压缩KV计算。CSA场景传1、2或4，HCA场景传128。
    - `cmp_topk`在CSA场景支持512或1024，SWA、HCA场景传0。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_sparse_flash_mla_metadata](./examples/test_aclnn_sparse_flash_mla_metadata.cpp) | 通过[aclnnSparseFlashMlaMetadata](./docs/aclnnSparseFlashMlaMetadata.md)调用SparseFlashMlaMetadata算子 |
| PyTorch API | [test_torch_sparse_flash_mla_metadata](./examples/test_torch_sparse_flash_mla_metadata.py) | 通过[sparse_flash_mla_metadata](../../torch_extension/cann_ops_transformer/docs/zh/sparse_flash_mla.md)接口生成SparseFlashMla主算子使用的metadata |
