# chunk_scaled_dot_kkt_fwd

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

- 算子功能：`chunk_scaled_dot_kkt_fwd` 计算 Gated Delta Rule 中 chunk 内的 WY 下三角变换矩阵 $A$，即 $\beta\odot(K@K^T)$，并可选地融合门控 $g$。该矩阵随后会送入 `solve_tril` 求解单位下三角逆矩阵。

- 计算公式：

  对于每个 chunk 内的位置 $i,j$（$i,j\in[0,\text{BT})$），给定 key $K\in\R^{B\times T\times H_g\times d}$、缩放因子 $\beta\in\R^{B\times T\times H}$、累积门控 $g\in\R^{B\times T\times H}$：

  $$
  A[i, j, h] = \beta[i, h]\cdot\left(\sum_{d} K[i, h_g, d]\cdot K[j, h_g, d]\right)\cdot \exp\big(g[i, h]-g[j, h]\big),\quad i>j
  $$

   - 当 $i\le j$ 时 $A[i,j,h]=0$（严格下三角，对角线及上三角置 0）。
   - $h_g = h\ //\ (H/H_g)$，把 query head 折回 kv-head，支持 GQA/MQA。
   - 当 `g_cumsum=None` 时不应用门控项。
   - 门控使用 `safe_exp`，对指数为正的位置输出 0 以防止溢出。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 |
|---|---|---|---|
| k | 输入 | key 张量，shape 为 `(B, T, Hg, K)`。 | FLOAT16、BFLOAT16、FLOAT32 |
| beta | 输入 | 缩放因子，shape 为 `(B, T, H)`。 | FLOAT16、BFLOAT16、FLOAT32 |
| g_cumsum | 输入 | 累积门控（`chunk_local_cumsum` 的输出），shape 为 `(B, T, H)`。默认 `None` 表示无门控。 | FLOAT16、BFLOAT16、FLOAT32 |
| cu_seqlens | 输入 | 变长序列累积长度，shape 为 `(N+1,)`。默认 `None`。 | INT32、INT64 |
| chunk_indices | 输入 | 变长模式下的分块索引，shape 为 `(NT, 2)`。默认 `None` 时自动生成。 | INT32、INT64 |
| chunk_size | 属性 | chunk 大小，默认 64。 | INT |
| output_dtype | 属性 | 输出数据类型，默认 `torch.float32`。 | - |
| A | 输出 | chunk 内下三角变换矩阵，shape 为 `(B, T, H, BT)`，其中 `BT=chunk_size`。 | 由 `output_dtype` 决定 |

## 约束说明

- 该接口支持图模式。
- `H` 必须能被 `Hg` 整除（GQA 约束）。
- `A` 的最后一个维度 `BT` 等于 `chunk_size`。
- 输入 `k`、`beta`、`g_cumsum` 的前三个维度 `(B, T)` 需一致。
- `g_cumsum` 应为 `chunk_local_cumsum` 的输出（log 空间累积门控），通常为单调递减序列。
- 变长模式下各序列之间严格隔离，不会互相影响。
- chunk 不足 `BT` 的尾部位置，超出有效长度的行/列对应的 `A` 元素未定义，调用方不应读取。

## 调用示例

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">调用方式</th>
    <th class="tg-0pky">样例代码</th>
    <th class="tg-0pky">说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="6">Python接口</td>
    <td class="tg-0pky">
    <a href="../../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_chunk_scaled_dot_kkt_fwd.py">test_chunk_scaled_dot_kkt_fwd
    </a>
    </td>
    <td class="tg-lboi" rowspan="6">
    通过
    <a href="./chunk_scaled_dot_kkt.py">chunk_scaled_dot_kkt_fwd
    </a>
    接口方式调用算子
    </td>
  </tr>
</tbody></table>
