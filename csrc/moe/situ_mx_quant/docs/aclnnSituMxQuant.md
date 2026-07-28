# aclnnSituMxQuant

## 功能说明

SituMxQuant 算子执行 Situ 激活，随后进行动态 MX (Microscaling) 量化。

计算公式：

```text
situ_a = beta * tanh(gate / beta) * sigmoid(gate)
situOut = situ_a * up  (+ optional linear_beta * tanh(up / linear_beta) on up)
shared_exp = floor(log2(max(|situOut_i|))) - emax
mxscale = 2^shared_exp  (E8M0)
y = cast_to_fp8(situOut / mxscale)
```

## 接口定义

```cpp
aclnnStatus aclnnSituMxQuant(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
```

## 算子原型

```cpp
REG_OP(SituMxQuant)
    .INPUT(x, TensorType({DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2}))
    .OUTPUT(mxscale, TensorType({DT_FLOAT8_E8M0}))
    .ATTR(beta, Float, 1.0f)
    .ATTR(linear_beta, Float, 0.0f)
    .ATTR(activate_left, Bool, false)
    .ATTR(axis, Int, -1)
    .ATTR(dst_type, Int, 36)
    .OP_END_FACTORY_REG(SituMxQuant)
```

## 参数说明

| 参数 | 输入/输出 | 类型 | 说明 |
|------|-----------|------|------|
| x | 输入 | bfloat16 | 输入张量，shape 为 [N..., 2H]，最后一维必须为偶数 |
| y | 输出 | float8_e4m3fn / float8_e5m2 | 量化输出，shape 为 [N..., H] |
| mxscale | 输出 | float8_e8m0 | MX scale，shape 为 [N..., ceil(H/64), 2] |

## 属性说明

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| beta | Float | 1.0 | Situ 激活的 beta 参数，必须 > 0 |
| linear_beta | Float | 0.0 | Situ 激活的 linear_beta 参数，≤0 时不启用 up 的 tanh 变换 |
| activate_left | Bool | false | true: gate 在前半部分；false: gate 在后半部分 |
| axis | Int | -1 | 量化轴，当前仅支持 -1 |
| dst_type | Int | 36 | 输出数据类型: 36=FP8_E4M3FN, 35=FP8_E5M2 |

## 约束条件

- 输入 x 的最后一维必须能被 2 整除
- 输入 x 支持 1-7 维张量
- axis 必须为 -1（尾轴量化）
- beta 必须 > 0
- dst_type 必须为 36 (FP8_E4M3FN) 或 35 (FP8_E5M2)
- 仅支持 Ascend950 平台

## mxscale Shape 计算

- `H = x.shape[-1] / 2`
- `scaleNum = ceil(H / 64)` (偶对齐的 32 元素 block 数)
- `mxscale.shape = x.shape[:-1] + [scaleNum, 2]`

## 调用示例

```cpp
// 1. 创建 op executor
aclOpExecutor* executor;
aclnnSituMxQuantGetWorkspaceSize(x, y, mxscale, beta, linearBeta, activateLeft, axis, dstType, &workspaceSize, &executor);
void* workspace = nullptr;
if (workspaceSize > 0) {
    aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
}
// 2. 执行算子
aclnnSituMxQuant(workspace, workspaceSize, executor, stream);
```
