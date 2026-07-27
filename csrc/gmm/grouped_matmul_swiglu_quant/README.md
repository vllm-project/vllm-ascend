# GroupedMatmulSwigluQuant

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Kirin X90 处理器系列产品</term> | √ |
| <term>Kirin 9030 处理器系列产品</term> | √ |

## 功能说明

- 接口功能：融合GroupedMatmul 、dquant、swiglu和quant，详细解释见计算公式。
- 计算公式：

    - **定义**：

        - **⋅** 表示矩阵乘法。
        - **⊙** 表示逐元素乘法。
        - $\left \lfloor x\right \rceil$ 表示将x四舍五入到最近的整数。
        - $\mathbb{Z_8} = \{ x \in \mathbb{Z} | −128≤x≤127 \}$
        - $\mathbb{Z_{32}} = \{ x \in \mathbb{Z} | -2147483648≤x≤2147483647 \}$
    - **输入**：

        - $X∈\mathbb{Z_8}^{M \times K}$：输入矩阵（左矩阵），M是总token 数，K是特征维度。
        - $W∈\mathbb{Z_8}^{E \times K \times N}$：分组权重矩阵（右矩阵），E是专家个数，K是特征维度，N是输出维度。
        - $bias∈\mathbb{Z_{32}}^{E  \times N}$：矩阵乘计算的偏移值，E是专家个数，N是输出维度。
        - $offset∈\mathbb{R}^{E  \times N}$：per-channel非对称反量化的偏移，E是专家个数，N是输出维度。
        - $w\_scale∈\mathbb{R}^{E \times N}$：分组权重矩阵（右矩阵）的逐通道缩放因子，E是专家个数，N是输出维度。
        - $x\_scale∈\mathbb{R}^{M}$：输入矩阵（左矩阵）的逐 token缩放因子，M是总token 数。
        - $groupList∈\mathbb{N}^{E}$：前缀和的分组索引列表。
    - **输出**：

        - $Q∈\mathbb{Z_8}^{M \times N / 2}$：量化后的输出矩阵。
        - $Q\_scale∈\mathbb{R}^{M}$：量化缩放因子。
        - $Q\_offset∈\mathbb{R}^{M}$：量化偏移因子。
    - **计算过程**

        - 1.根据groupList[i]确定当前分组的 token ，$i \in [0,Len(groupList)]$。

      >例子：假设groupList=[3,4,4,6]，从0开始计数。
      >
      >第0个右矩阵`W[0,:,:]`，对应索引位置[0,3)的token`x[0:3]`（共3-0=3个token），对应`x_scale[0:3]`、`w_scale[0]`、`bias[0]`、`offset[0]`、`Q[0:3]`、`Q_scale[0:3]`、`Q_offset[0:3]`；
      >
      >第1个右矩阵`W[1,:,:]`，对应索引位置[3,4)的token`x[3:4]`（共4-3=1个token），对应`x_scale[3:4]`、`w_scale[1]`、`bias[1]`、`offset[1]`、`Q[3:4]`、`Q_scale[3:4]`、`Q_offset[3:4]`；
      >
      >第2个右矩阵`W[2,:,:]`，对应索引位置[4,4)的token`x[4:4]`（共4-4=0个token），对应`x_scale[4:4]`、`w_scale[2]`、`bias[2]`、`offset[2]`、`Q[4:4]`、`Q_scale[4:4]`、`Q_offset[4:4]`；
      >
      >第3个右矩阵`W[3,:,:]`，对应索引位置[4,6)的token`x[4:6]`（共6-4=2个token），对应`x_scale[4:6]`、`w_scale[3]`、`bias[3]`、`offset[3]`、`Q[4:6]`、`Q_scale[4:6]`、`Q_offset[4:6]`；
      >
      >请注意：groupList中未指定的部分将不会参与更新。
      >例如groupList=[12,14,18]，X的shape为[30，:]。
      >
      >则第一个输出Q的shape为[30，:]，其中Q[18:，：]的部分不会进行更新和初始化，其中数据为显存空间申请时的原数据。
      >
      >同理，第二个输出Q的shape为[30]，其中Q\_scale[18:]的部分不会进行更新或初始化，其中数据为显存空间申请时的原数据。
      >
      >即输出的Q[:groupList[-1],:]和Q\_scale[:groupList[-1]]为有效数据部分。

        - 2.根据分组确定的入参进行如下计算：

      $C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i\ BroadCast} \odot w\_scale_{i\ BroadCast}$

      $C_{i,act}, gate_{i} = split(C_{i})$

      $S_{i}=Swish(C_{i,act})\odot gate_{i}$  &nbsp;&nbsp;其中$Swish(x)=\frac{x}{1+e^{-x}}$

      >注：当前版本不支持$bias_{i}$、$offset_{i}$，未来版本将支持的计算公式如下：
      >$C_{i} =(X_{i}\cdot W_{i} + bias_{i\ BroadCast})\odot x\_scale_{i\ BroadCast} \odot w\_scale_{i\ BroadCast}+offset_{i\ BroadCast}$

        - 3.确定量化方式

            - 当量化方式为对称量化时：

        $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$

        $Q_{i} = \left \lfloor \frac{S_{i}}{Q\_scale_{i}}\right \rceil $

            - 当量化方式为非对称量化时：(暂不支持)

        $Q\_scale_{i} = \frac{max(S_{i})-min(S_{i})}{255}$

        $Q\_offset_{i} = -128 - \left \lfloor \frac{min(S_{i})}{Q\_scale_{i}}\right \rceil$

        $Q_{i} = \left \lfloor \frac{S_{i}}{ Q\_scale_{i} } + Q\_offset_{i}\right \rceil $

## 参数说明

<table style="table-layout: auto; width: 100%">
<thead>
<tr>
<th style="white-space: nowrap">参数名</th>
<th style="white-space: nowrap">输入/输出/属性</th>
<th style="white-space: nowrap">描述</th>
<th style="white-space: nowrap">数据类型</th>
<th style="white-space: nowrap">数据格式</th>
</tr>
</thead>
<tbody>
<tr>
<td style="white-space: nowrap">x</td>
<td style="white-space: nowrap">输入</td>
<td style="white-space: nowrap">左矩阵，公式中的X。</td>
<td style="white-space: nowrap">INT8</td>
<td style="white-space: nowrap">ND</td>
</tr>
<tr>
<td style="white-space: nowrap">weight</td>
<td style="white-space: nowrap">输入</td>
<td style="white-space: nowrap">权重矩阵，公式中的W。</td>
<td style="white-space: nowrap">INT8</td>
<td style="white-space: nowrap">ND / NZ</td>
</tr>
<tr>
<td style="white-space: nowrap">bias</td>
<td style="white-space: nowrap">输入</td>
<td style="white-space: nowrap">矩阵乘计算的偏移值，公式中的bias。</td>
<td style="white-space: nowrap">INT32</td>
<td style="white-space: nowrap">ND</td>
</tr>
<tr>
<td style="white-space: nowrap">offset</td>
<td style="white-space: nowrap">输入</td>
<td style="white-space: nowrap">per-channel非对称反量化的偏移，公式中的offset。</td>
<td style="white-space: nowrap">FLOAT32</td>
<td style="white-space: nowrap">ND</td>
</tr>
<tr>
<td style="white-space: nowrap">weightScale</td>
<td style="white-space: nowrap">输入</td>
<td style="white-space: nowrap">右矩阵的量化因子，公式中的w_scale。</td>
<td style="white-space: nowrap">FLOAT、FLOAT16、BFLOAT16</td>
<td style="white-space: nowrap">ND</td>
</tr>
<tr>
<td style="white-space: nowrap">xScale</td>
<td style="white-space: nowrap">输入</td>
<td style="white-space: nowrap">左矩阵的量化因子，公式中的x_scale。</td>
<td style="white-space: nowrap">FLOAT32</td>
<td style="white-space: nowrap">ND</td>
</tr>
<tr>
<td style="white-space: nowrap">groupList</td>
<td style="white-space: nowrap">输入</td>
<td style="white-space: nowrap">指示每个分组参与计算的Token个数，公式中的groupList。</td>
<td style="white-space: nowrap">INT64</td>
<td style="white-space: nowrap">ND</td>
</tr>
<tr>
<td style="white-space: nowrap">output</td>
<td style="white-space: nowrap">输出</td>
<td style="white-space: nowrap">输出的量化因子，公式中的Q。</td>
<td style="white-space: nowrap">FLOAT</td>
<td style="white-space: nowrap">ND</td>
</tr>
<tr>
<td style="white-space: nowrap">outputScale</td>
<td style="white-space: nowrap">输出</td>
<td style="white-space: nowrap">输出的量化因子，公式中的Q_scale。</td>
<td style="white-space: nowrap">FLOAT</td>
<td style="white-space: nowrap">ND </td>
</tr>
<tr>
<td style="white-space: nowrap">outputOffset</td>
<td style="white-space: nowrap">输出</td>
<td style="white-space: nowrap">输出的非对称量化的偏移，公式中的Q_offset。</td>
<td style="white-space: nowrap">FLOAT</td>
<td style="white-space: nowrap">ND</td>
</tr>
</tbody>
</table>

- Kirin X90/Kirin 9030 处理器系列产品: 不支持BFLOAT16。

## 约束说明

- N轴长度不能超过10240。
- K轴长度不能超过65536。

## 调用说明

| 调用方式      | 调用样例                 | 说明                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_grouped_matmul_swiglu_quant](examples/test_aclnn_grouped_matmul_swiglu_quant.cpp) | 通过接口方式调用[GroupedMatmulSwigluQuant](docs/aclnnGroupedMatmulSwigluQuant.md)算子。 |
