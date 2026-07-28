# SituMxQuant 算子

## 功能说明

SituMxQuant 算子将 Situ 激活函数与动态 MX (Microscaling) 量化融合为一个算子。

计算流水线：**Situ 激活 → MxQuant**

### Situ 激活

```python
d = last_dim(x) // 2
gate = x[..., :d]
up   = x[..., d:]
situ_a = beta * tanh(gate / beta) * sigmoid(gate)
if linear_beta > 0:
    up = linear_beta * tanh(up / linear_beta)
situOut = situ_a * up
```

### MxQuant (OCP 算法)

```text

shared_exp = floor(log2(max(|V_i|))) - emax
mxscale = 2^shared_exp  (E8M0 格式)
y = cast_to_fp8(V_i / mxscale)
```

## 输入输出

| 参数 | 输入/输出 | 数据类型 | Shape | 说明 |
|------|-----------|----------|-------|------|
| x | 输入 | BF16 | [N..., 2H] | bfloat16，最后一维为偶数 |
| y | 输出 | FP8_E4M3FN / FP8_E5M2 | [N..., H] | FP8 量化输出 |
| mxscale | 输出 | FP8_E8M0 | [N..., ceil(H/64), 2] | MX scale (E8M0) |

## 属性

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| beta | Float | 1.0 | Situ 的 beta 参数，必须 > 0 |
| linear_beta | Float | 0.0 | Situ 的 linear_beta 参数，≤0 时不启用 |
| activate_left | Bool | false | gate 在前半 (true) 还是后半 (false) |
| axis | Int | -1 | 量化轴，当前仅支持 -1 |
| dst_type | Int | 36 | 输出类型: 36=FP8_E4M3FN, 35=FP8_E5M2 |

## 支持平台

- Ascend950 (arch35)

## 约束

- 输入 x 最后一维必须为偶数
- 输入 x 支持 1-7 维
- 仅支持 axis=-1（尾轴量化）
- 仅支持 OCP 量化算法
- 仅支持 bfloat16 输入
- 仅支持 FP8 输出（E4M3FN / E5M2）
- beta 必须 > 0
