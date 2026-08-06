# recompute_w_u_fwd

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

- 算子功能：`recompute_w_u_fwd` 在 Gated Delta Rule 的 chunk 扫描中，根据下三角变换矩阵 $A$（已求解逆矩阵）、key/value、缩放因子 $\beta$ 和累积门控 $g$，重计算 WY 表示中的中间变量 $w$ 与 $u$。$u$ 即后续计算使用的新 value。

- 计算公式：

  对于每个 chunk 内的位置 $i$，给定 $K\in\R^{B\times T\times H_g\times d}$、$V\in\R^{B\times T\times H\times d_v}$、$\beta\in\R^{B\times T\times H}$、$A\in\R^{B\times T\times H\times \text{BT}}$、累积门控 $g\in\R^{B\times T\times H}$：

  $$
  w[i, h, d] = \sum_{j} A[i, j, h]\cdot K[j, h_g, d]\cdot \beta[j, h]\cdot \exp\big(g[j, h]\big)
  $$

  $$
  u[i, h, d_v] = \sum_{j} A[i, j, h]\cdot V[j, h, d_v]\cdot \beta[j, h]
  $$

   - $h_g = h\ //\ (H/H_g)$，把 query head 折回 kv-head，支持 GQA/MQA。
   - $A$ 为 `solve_tril` 求解后的单位下三角逆矩阵，shape 最后维 `BT` 为 chunk 大小。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 |
|---|---|---|---|
| k | 输入 | key 张量，shape 为 `(B, T, Hg, K)`。 | FLOAT16、BFLOAT16、FLOAT32 |
| v | 输入 | value 张量，shape 为 `(B, T, H, V)`。 | FLOAT16、BFLOAT16、FLOAT32 |
| beta | 输入 | 缩放因子，shape 为 `(B, T, H)`。 | FLOAT16、BFLOAT16、FLOAT32 |
| g_cumsum | 输入 | 累积门控（`chunk_local_cumsum` 的输出），shape 为 `(B, T, H)`。 | FLOAT16、BFLOAT16、FLOAT32 |
| A | 输入 | 下三角变换矩阵（`solve_tril` 的输出），shape 为 `(B, T, H, BT)`。 | FLOAT16、BFLOAT16、FLOAT32 |
| cu_seqlens | 输入 | 变长序列累积长度，shape 为 `(N+1,)`。默认 `None`。 | INT32、INT64 |
| chunk_indices | 输入 | 变长模式下的分块索引，shape 为 `(NT, 2)`。默认 `None` 时自动生成。 | INT32、INT64 |
| w | 输出 | 加权 key，shape 为 `(B, T, H, K)`。 | 与 `k` 一致 |
| u | 输出 | 加权 value，shape 为 `(B, T, H, V)`。 | 与 `v` 一致 |

## 约束说明

- 该接口支持图模式。
- `H` 必须能被 `Hg` 整除（GQA 约束）。
- `A` 的最后一个维度 `BT` 必须等于 `chunk_size`（取自 `A.shape[-1]`）。
- `k` 和 `v` 的前三个维度 `(B, T, H)` 需一致（`k` 的 head 维为 `Hg`）。
- 非变长（`cu_seqlens=None`）场景当前仅支持 `B=1`；多 batch 场景请使用变长输入（`B=1` 配合 `cu_seqlens`）。
- `g_cumsum` 应为 `chunk_local_cumsum` 的输出（log 空间累积门控）。
- 输入张量需在最后一维连续。
- 变长模式下各序列之间严格隔离，不会互相影响。

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
    <a href="../../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_recompute_w_u_fwd.py">test_recompute_w_u_fwd
    </a>
    </td>
    <td class="tg-lboi" rowspan="6">
    通过
    <a href="./wy_fast.py">recompute_w_u_fwd
    </a>
    接口方式调用算子
    </td>
  </tr>
</tbody></table>
