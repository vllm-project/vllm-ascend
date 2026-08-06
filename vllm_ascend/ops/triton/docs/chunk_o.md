# chunk_fwd_o

## 功能说明

- 算子功能：`chunk_fwd_o` 计算 Gated Delta Rule 前向中每个 chunk 的输出 $o$，融合了 chunk 间隐状态贡献（$q@h$）与 chunk 内注意力贡献（$A_{\text{intra}}@v$），并可选地融合门控 $g$。

- 计算公式：

  对于每个 chunk 内的位置 $i$，给定 query $Q\in\R^{B\times T\times H_g\times d}$、key $K\in\R^{B\times T\times H_g\times d}$、value $V\in\R^{B\times T\times H\times d_v}$、chunk 间隐状态 $h\in\R^{(\text{B}\cdot\text{NT})\times H\times d\times d_v}$、累积门控 $g\in\R^{B\times T\times H}$、缩放因子 $s$：

  $$
  A_{\text{intra}}[i, j, h] = \mathbb{1}_{i\ge j}\cdot\left(\sum_{d} Q[i, h_g, d]\cdot K[j, h_g, d]\right)\cdot \exp\big(g[i, h]-g[j, h]\big)
  $$

  $$
  o[i, h, d_v] = s\cdot\Big(\exp(g[i, h])\cdot\sum_{d} Q[i, h_g, d]\cdot h[c, h, d, d_v]\Big) + s\cdot\sum_{j} A_{\text{intra}}[i, j, h]\cdot V[j, h, d_v]
  $$

    - $h_g = h\ //\ (H/H_g)$，把 query head 折回 kv-head，支持 GQA/MQA。
    - $c$ 为当前 chunk 的索引，$h$ 为该 chunk 之前累积的隐状态。
    - $A_{\text{intra}}$ 为 chunk 内下三角（含对角线）的 query-key 相关性矩阵。
    - 当 `g=None` 时不应用门控项；当 `scale=None` 时默认 $s=d^{-1/2}$。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 |
|---|---|---|---|
| q | 输入 | query 张量，shape 为 `(B, T, Hg, K)`。 | FLOAT16、BFLOAT16、FLOAT32 |
| k | 输入 | key 张量，shape 为 `(B, T, Hg, K)`。 | FLOAT16、BFLOAT16、FLOAT32 |
| v | 输入 | value 张量，shape 为 `(B, T, H, V)`。 | FLOAT16、BFLOAT16、FLOAT32 |
| h | 输入 | chunk 间隐状态，shape 为 `(B*NT, H, K, V)`，`NT` 为每个序列的 chunk 数。 | FLOAT16、BFLOAT16、FLOAT32 |
| g | 输入 | 累积门控（`chunk_local_cumsum` 的输出），shape 为 `(B, T, H)`。默认 `None`。 | FLOAT16、BFLOAT16、FLOAT32 |
| scale | 属性 | 缩放因子，默认 `K^{-1/2}`。 | FLOAT |
| cu_seqlens | 输入 | 变长序列累积长度，shape 为 `(N+1,)`。默认 `None`。 | INT32、INT64 |
| chunk_size | 属性 | chunk 大小，默认 64。 | INT |
| chunk_offsets | 输入 | 变长模式下的 chunk 偏移，shape 为 `(N+1,)`。默认 `None` 时自动生成。 | INT32、INT64 |
| o | 输出 | chunk 输出，shape 为 `(B, T, H, V)`。 | 与 `v` 一致 |

## 约束说明

- 该接口支持图模式。
- `H` 必须能被 `Hg` 整除（GQA 约束）。
- `h` 的第一维必须等于 `B * NT`，其中 `NT = ceil(T / chunk_size)`；变长模式下需配合 `chunk_offsets`。
- `g` 应为 `chunk_local_cumsum` 的输出（log 空间累积门控）。
- 输入 `q`、`k`、`v`、`g` 的 `(B, T)` 维度需一致。
- 输入张量需在最后一维连续。
- `chunk_size` 当前默认且建议为 64。

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
    <a href="../../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_chunk_fwd_o.py">test_chunk_fwd_o
    </a>
    </td>
    <td class="tg-lboi" rowspan="6">
    通过
    <a href="./chunk_o.py">chunk_fwd_o
    </a>
    接口方式调用算子
    </td>
  </tr>
</tbody></table>
