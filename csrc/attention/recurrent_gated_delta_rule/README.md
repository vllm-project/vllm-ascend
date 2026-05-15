# RecurrentGatedDeltaRule

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：完成变步长的Recurrent Gated Delta Rule计算。

- 计算公式：

  Recurrent Gated Delta Rule（循环门控Delta规则，RGDR）是一种应用于循环神经网络的算子，也被应用于一种线性注意力机制中。
  在每个时间步 $t$，网络根据当前的输入 $q_t$、$k_t$、$v_t$ 和上一个隐藏状态 $S_{t-1}$，计算当前的输出 $o_t$ 和新的隐藏状态 $S_t$。
  在这个过程中，门控单元会决定有多少新信息存入隐藏状态，以及有多少旧信息需要被遗忘。

  $$
  S_t := S_{t-1}(\alpha_t Diag(\alpha_{kt})(I - \beta_t k_t k_t^T)) + \beta_t v_t k_t^T = \alpha_t Diag(\alpha_{kt})S_{t-1} + \beta_t (v_t - \alpha_t Diag(\alpha_{kt})S_{t-1}k_t)k_t^T
  $$

  $$
  o_t := \frac{S_t q_t}{\sqrt{d_k}}
  $$

  其中，$S_{t-1},S_t \in R^{d_v \times d_k}$，$q_t, k_t \in R^{d_k}$，$v_t \in R^{d_v}$，$\alpha_t \in R$，$\alpha_k \in R^{d_k}$，$\beta_t \in R$，$o \in R^{d_v}$。

## 参数说明

<table style="undefined;table-layout: fixed; width: 900px"><colgroup>
<col style="width: 180px">
<col style="width: 120px">
<col style="width: 200px">
<col style="width: 300px">
<col style="width: 100px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td>query</td>
    <td>输入</td>
    <td>公式中的q。</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>key</td>
    <td>输入</td>
    <td>公式中的输入k。</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>value</td>
    <td>输入</td>
    <td>公式中的输入v。</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>beta</td>
    <td>输入</td>
    <td>公式中的β。</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>state</td>
    <td>输入&输出</td>
    <td>状态矩阵，公式中的输入S。</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>g</td>
    <td>输入</td>
    <td>衰减系数，公式中的α=e^g</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gk</td>
    <td>输入</td>
    <td>衰减系数，公式中的αk=e^gk</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>out</td>
    <td>输出</td>
    <td>公式中的o。</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## 约束说明

- 输入tensor的shape大小需满足一定约束，具体见[aclnnRecurrentGatedDeltaRule](./docs/aclnnRecurrentGatedDeltaRule.md)。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_recurrent_gated_delta_rul.cpp](./examples/test_aclnn_recurrent_gated_delta_rule.cpp) | 通过[aclnnRecurrentGatedDeltaRule](./docs/aclnnRecurrentGatedDeltaRule.md)调用aclnnRecurrentGatedDeltaRule算子 |
