# chunk_local_cumsum

## 功能说明

- 算子功能：`chunk_local_cumsum` 对输入张量沿序列维度（T）按 `chunk_size` 分块后，在每个 chunk 内独立执行累积求和（cumsum），可选支持反向累积、缩放因子以及变长序列。该算子是 flash-linear-attention 系列（如 Gated Delta Rule）中用于对门控信号 `g` 进行 chunk 级局部 cumsum 的核心组件。

- 计算公式：

  对于输入 $g\in\R^{B\times T\times H}$（`head_first=False` 时），按 `chunk_size` 将 $T$ 切分为若干 chunk，对每个 chunk 内部沿 $T$ 维执行累积求和：

  $$
  \text{out}[b, t, h] = \sum_{j=\text{chunk\_start}}^{t} g[b, j, h], \quad t\in[\text{chunk\_start},\ \text{chunk\_end})
  $$

    - 当 `reverse=True` 时，在每个 chunk 内沿 $T$ 维反向累积：

    $$
    \text{out}[b, t, h] = \sum_{j=t}^{\text{chunk\_end}-1} g[b, j, h]
    $$

    - 当 `scale` 不为 `None` 时，对结果再做缩放：$\text{out} \leftarrow \text{scale}\cdot \text{out}$。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 |
|---|---|---|---|
| g | 输入 | 输入张量，shape 为 `(B, T, H)`（`head_first=False`）或 `(B, H, T)`（`head_first=True`）。不支持空 tensor 和非连续。 | FLOAT32 |
| chunk_size | 属性 | 分块大小，必须是 2 的幂次。 | INT |
| reverse | 属性 | 是否在每个 chunk 内反向执行 cumsum，默认 `False`。 | BOOL |
| scale | 属性 | 输出缩放因子，默认 `None` 表示不缩放。 | FLOAT |
| cu_seqlens | 输入 | 变长序列的累积长度，shape 为 `(N+1,)`。提供时 batch size 必须为 1。默认 `None`。 | INT32 |
| head_first | 属性 | 是否 head 维度在前，默认 `False`。 | BOOL |
| output_dtype | 属性 | 输出数据类型，默认 `torch.float`。 | - |
| output | 输出 | 分块局部 cumsum 结果，shape 与输入一致。 | 由 `output_dtype` 决定，默认 FLOAT32 |

## 约束说明

- 该接口支持图模式。
- `chunk_size` 必须是 2 的幂次，否则抛出 `AssertionError`。
- 当前算子仅支持 3 维输入 `(B, T, H)`，不支持 4 维输入。
- 使用 `cu_seqlens` 时，batch size 必须为 1，否则抛出 `AssertionError`。
- `output_dtype` 默认为 `torch.float`，在反向传播和 context parallel 场景下可有效防止中间结果溢出。
- 输入张量需在最后一维连续。

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
    <a href="../../../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_chunk_local_cumsum.py">test_chunk_local_cumsum
    </a>
    </td>
    <td class="tg-lboi" rowspan="6">
    通过
    <a href="./cumsum.py">chunk_local_cumsum
    </a>
    接口方式调用算子
    </td>
  </tr>
</tbody></table>
