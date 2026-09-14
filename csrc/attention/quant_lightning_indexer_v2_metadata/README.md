# QuantLightningIndexerV2Metadata

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：`QuantLightningIndexerV2Metadata`是`QuantLightningIndexerV2`算子的前置算子，用于生成负载均衡的任务划分方案。本算子不执行实际的LightningIndexer计算，而是根据输入参数在AI CPU计算出每个AI Core应处理的计算起止范围，从而最大化计算资源的利用率，避免各Core间负载不均衡的问题。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 180px">
  <col style="width: 100px">
  <col style="width: 700px">
  <col style="width: 90px">
  <col style="width: 80px">
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
      <td>cu_seqlens_q</td>
      <td>可选输入</td>
      <td>表示不同Batch中q的有效Sequence Length，shape为(B+1, )，仅layout_q为TND场景下必传，第一个值固定为0。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_k</td>
      <td>可选输入</td>
      <td>表示不同Batch中k的有效Sequence Length，shape为(B+1, )，仅layout_k为TND场景下必传，第一个值固定为0。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_q</td>
      <td>可选输入</td>
      <td>表示不同Batch中q实际参与运算的Sequence Length，shape为(B, )。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_k</td>
      <td>可选输入</td>
      <td>表示不同Batch中k实际参与运算的Sequence Length，shape为(B, )。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cmp_residual_k</td>
      <td>可选输入</td>
      <td>表示不同Batch中k压缩后Sequence Length的余数，配合cmp_ratio实现mask和负载计算，shape为(B, )。cmp_ratio不为1且mask_mode为3场景下必传。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_heads_q</td>
      <td>属性</td>
      <td>表示q的head个数，当前支持[1, 64]。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_heads_k</td>
      <td>属性</td>
      <td>表示k的head个数，当前仅支持1。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>head_dim</td>
      <td>属性</td>
      <td>表示注意力头的维度，当前仅支持128。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topk</td>
      <td>属性</td>
      <td>表示从q中筛选出的关键稀疏token的个数，当前仅支持[1, 8192]。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_mode</td>
      <td>属性</td>
      <td>表示量化模式，当前支持1/2/3/4/5。1表示qk: fp8(e4m3) per-token-head, scale: fp32；2表示qk: int8 per-token-head, scale: fp16, w: fp16；3表示qk: mxfp8(e4m3), scale: fp8(e8m0)；4表示qk: hif8 per-tensor, scale: fp32；5表示qk: mxfp4(e2m1), scale: fp8(e8m0)。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batch_size</td>
      <td>可选属性</td>
      <td>表示Batch数量，默认值为0。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_q</td>
      <td>可选属性</td>
      <td>表示q的最长Sequence Length，-1表示任意可能长度，默认值为-1。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_k</td>
      <td>可选属性</td>
      <td>表示k的最长Sequence Length，-1表示任意可能长度，默认值为-1。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_q</td>
      <td>可选属性</td>
      <td>表示q的排列格式，支持BSND、TND，默认值为BSND。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_k</td>
      <td>可选属性</td>
      <td>表示k的排列格式，支持BSND、TND、PA_BBND，默认值为BSND。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mask_mode</td>
      <td>可选属性</td>
      <td>表示sparse模式，0表示No mask，3表示rightDownCausal模式，默认值为0。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_ratio</td>
      <td>可选属性</td>
      <td>表示k的压缩率，取值范围[1, 128]，默认值为1，表示无压缩。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadata</td>
      <td>输出</td>
      <td>表示负载均衡结果输出，shape固定为(1024, )</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

  <ul>
    <li><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> ：num_heads_q仅支持64，不支持quant_mode = 1/3/4/5，topk仅支持[1, 2048]，不支持layout_k = BSND/TND，不支持cmp_ratio在[1，128]任意取值，仅支持cmp_ratio = 1/2/4/8/16/32/64/128。</li>
    <li><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> ：num_heads_q仅支持64，不支持quant_mode = 1/3/4/5，topk仅支持[1, 2048]，不支持layout_k = BSND/TND，不支持cmp_ratio在[1，128]任意取值，仅支持cmp_ratio = 1/2/4/8/16/32/64/128。</li>
  </ul>

## 约束说明

- QuantLightningIndexerV2Metadata算子需要与QuantLightningIndexerV2算子配套使用。
- B（Batch）表示输入样本批量大小，q、k为配套的QuantLightningIndexerV2算子的入参，S1表示layout_q=BSND时，q shape中的S轴的大小，S2表示layout_k=BSND时，k shape中的S轴的大小。
- 参数cu_seqlens_q、cu_seqlens_k要求其值为当前Batch与前序Batch有效token数的累加值，第一个元素固定为0，后一个元素的值必须大于等于前一个元素的值。
- 参数seqused_q、seqused_k要求其值表示每个Batch中的有效token数。
- 参数cmp_residual_k需满足cmp_residual_k[i] < cmp_ratio。
- mask_mode所表示的mask模式的详细介绍见[sparse_mode参数说明](../../docs/zh/context/sparse_mode_introduction.md)。
- 非PA场景layout_q、layout_k须相同。
- layout_q=BSND场景
    - max_seqlen_q必须传入S1的值。
- layout_k=BSND场景
    - max_seqlen_k必须传入S2的值。
- layout_q=TND场景
    - cu_seqlens_q必须传入。
- layout_k=TND场景
    - cu_seqlens_k必须传入。
- layout_k=PA_BBND场景
    - seqused_k必须传入。
- Batch取值规则
    - layout_q为BSND时，优先通过seqused_q的shape推导batch，seqused_q未传入则通过batch_size获取batch数。
    - layout_q为TND时，优先通过seqused_q的shape推导batch，seqused_q未传入则通过cu_seqlens_q的shape推导batch。
- q Seqlen取值规则
    - layout_q为BSND时，优先通过seqused_q中的元素获取seqlen，seqused_q未传入则通过max_seqlen_q获取seqlen。
    - layout_q为TND时，优先通过seqused_q中的元素获取seqlen，seqused_q未传入则通过cu_seqlens_q中的元素获取seqlen。
- k Seqlen取值规则
    - layout_k为BSND时，优先通过seqused_k中的元素获取seqlen，seqused_k未传入则通过max_seqlen_k获取seqlen。
    - layout_k为TND时，优先通过seqused_k中的元素获取seqlen，seqused_k未传入则通过cu_seqlens_k中的元素获取seqlen。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_quant_lightning_indexer_v2_metadata](./examples/test_aclnn_quant_lightning_indexer_v2_metadata.cpp) | 通过[aclnnQuantLightningIndexerV2Metadata](./docs/aclnnQuantLightningIndexerV2Metadata.md)接口调用QuantLightningIndexerV2Metadata算子 |
| PyTorch API | [test_torch_quant_lightning_indexer_v2_metadata](./examples/test_torch_quant_lightning_indexer_v2_metadata.py) | 通过[quant_lightning_indexer_metadata](../../torch_extension/cann_ops_transformer/docs/zh/quant_lightning_indexer.md)接口调用QuantLightningIndexerV2Metadata算子 |
