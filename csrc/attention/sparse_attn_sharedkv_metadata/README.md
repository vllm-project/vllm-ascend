# SparseAttnSharedkvMetadata

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 推理系列产品</term>   | √  |
|<term>Atlas A3 推理系列产品</term>   | √  |

## 功能说明

- API功能：`SparseAttnSharedkvMetadata`算子旨在生成一个任务列表，包含每个AIcore的Attention计算任务的起止点的Batch、Head、以及 Q 和 K 的分块的索引，供后续`SparseAttnSharedkv`算子使用。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px">
  <colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
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
      <td>Q的多头数，目前仅支持64。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_heads_kv</td>
      <td>属性</td>
      <td>K和V的多头数，目前仅支持1。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>head_dim</td>
      <td>属性</td>
      <td>注意力头的维度，目前仅支持512。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cu_seqlens_q</td>
      <td>可选输入</td>
      <td>当layout_query为TND时，表示不同Batch中q的有效token数，维度为B+1，大小为参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_ori_kv</td>
      <td>可选输入</td>
      <td>当layout_kv为TND时，表示不同Batch中ori_kv的有效token数，维度为B+1，大小为参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和。目前layout_kv仅支持PA_ND，故设置此参数无效。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens_cmp_kv</td>
      <td>可选输入</td>
      <td>当layout_kv为TND时，表示不同Batch中cmp_kv的有效token数，维度为B+1，大小为参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和。目前layout_kv仅支持PA_ND，故设置此参数无效。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_q</td>
      <td>可选输入</td>
      <td>表示不同Batch中q实际参与运算的token数，维度为B。目前暂不支持指定该参数。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused_kv</td>
      <td>可选输入</td>
      <td>表示不同Batch中ori_kv实际参与运算的token数，维度为B。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>batch_size</td>
      <td>可选属性</td>
      <td>输入样本批量大小。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_q</td>
      <td>可选属性</td>
      <td>表示所有batch中`q`的最大有效token数。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_kv</td>
      <td>可选属性</td>
      <td>表示所有batch中`ori_kv`的最大有效token数。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_topk</td>
      <td>可选属性</td>
      <td>表示通过QLI算法从`ori_kv`中筛选出的关键稀疏token的个数。目前暂不支持指定该参数。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_topk</td>
      <td>可选属性</td>
      <td>表示通过QLI算法从`cmp_kv`中筛选出的关键稀疏token的个数，目前仅支持512。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_ratio</td>
      <td>可选属性</td>
      <td>表示对`ori_kv`的压缩率，数据范围支持4/128，</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_mask_mode</td>
      <td>可选属性</td>
      <td>表示q和ori_kv计算的mask模式，仅支持输入默认值4，代表band模式的mask。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_mask_mode</td>
      <td>可选属性</td>
      <td>表示q和cmp_kv计算的mask模式，仅支持输入默认值3，代表rightDownCausal模式的mask。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_win_left</td>
      <td>可选属性</td>
      <td>表示q和ori_kv计算中q对过去token计算的数量，仅支持默认值127。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ori_win_right</td>
      <td>可选属性</td>
      <td>表示q和ori_kv计算中q对未来token计算的数量，仅支持默认值0。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_q</td>
      <td>可选属性</td>
      <td>用于标识输入q的数据排布格式,默认值为BSND，目前支持传入BSND和TND。</td>
      <td>STRING</td>
    </tr>
    <tr>
      <td>layout_kv</td>
      <td>可选属性</td>
      <td>用于标识输入ori_kv和cmp_kv的数据排布格式，目前仅支持传入默认值PA_ND。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>has_ori_kv</td>
      <td>可选属性</td>
      <td>是否传入ori_kv。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>has_cmp_kv</td>
      <td>可选属性</td>
      <td>是否传入cmp_kv。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>device</td>
      <td>可选属性</td>
      <td>用于获取设备信息。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadata</td>
      <td>输出</td>
      <td>每个cube核上FlashAttention计算任务的Batch、Head、以及 Q 和 K 的分块的索引，以及每个vector核上FlashDecode的规约任务索引。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持aclgraph模式。
- Tensor不能全传None。
