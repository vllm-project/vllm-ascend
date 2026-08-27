# 二分调试法详细指南

## 原理

二分调试法按照算子的数学公式，将计算过程拆分成多个阶段，逐步验证每个阶段的中间结果。

**为什么作为保底手段**：
- 需要修改代码添加 printf 输出，较耗时
- 应先尝试误差分析、检查常见陷阱等快速方法
- 但当问题难以定位时，这是最可靠的手段

## 何时使用

满足以下任一条件时，立即切换到二分调试：
1. **快速方法尝试超过 7 次**仍未定位问题
2. **已经尝试过所有前面手段**（误差分析、Printf 特定位置、常见陷阱排查）

> **重要原则**：不要盲目试错超过 7 次，二分调试能更快定位问题。

## 实施步骤

### 基本流程

```
数学公式分解
    │
    ├─ 第1层：最外层运算
    │   └─ 添加 printf，验证中间结果
    │
    ├─ 第2层：向内一层
    │   └─ 添加 printf，验证中间结果
    │
    ├─ ...继续分解...
    │
    └─ 发现首个差异步骤 → 问题定位
```

### 示例1：sinh(x) 调试

数学公式：`sinh(x) = (exp(x) - exp(-x)) / 2`

```cpp
// 完整二分拆解代码

// 第1步：验证 exp(x)
half exp_x = Exp(input);
printf("Step1 - exp(%.6f) = %.6f\n",
       static_cast<float>(input),
       static_cast<float>(exp_x));

// 第2步：验证 exp(-x)
half exp_neg_x = Exp(-input);
printf("Step2 - exp(%.6f) = %.6f\n",
       static_cast<float>(-input),
       static_cast<float>(exp_neg_x));

// 第3步：验证分子减法
half numerator = exp_x - exp_neg_x;
printf("Step3 - numerator = %.6f - %.6f = %.6f\n",
       static_cast<float>(exp_x),
       static_cast<float>(exp_neg_x),
       static_cast<float>(numerator));

// 第4步：验证最终除法
half result = numerator / 2.0h;
printf("Step4 - result = %.6f / 2 = %.6f\n",
       static_cast<float>(numerator),
       static_cast<float>(result));
```

### 示例2：Softmax 调试

数学公式：`softmax(x_i) = exp(x_i - max(x)) / sum(exp(x - max(x)))`

```cpp
// 第0步：打印输入（前3个元素）
printf("Input samples: [%.6f, %.6f, %.6f]\n",
       static_cast<float>(input[0]),
       static_cast<float>(input[1]),
       static_cast<float>(input[2]));

// 第1步：验证 ReduceMax
half max_val = ReduceMax(input);
printf("Step1 - max_val = %.6f\n", static_cast<float>(max_val));

// 第2步：验证广播后的 Sub（前3个元素）
for (int i = 0; i < 3 && i < size; ++i) {
    half shifted = input[i] - max_val;
    printf("Step2 - shifted[%d] = %.6f - %.6f = %.6f\n",
           i,
           static_cast<float>(input[i]),
           static_cast<float>(max_val),
           static_cast<float>(shifted));
}

// 第3步：验证 Exp（前3个元素）
LocalTensor<half> exp_vals;
// ... 分配内存 ...
for (int i = 0; i < 3 && i < size; ++i) {
    half shifted = input[i] - max_val;
    half exp_val = Exp(shifted);
    exp_vals[i] = exp_val;
    printf("Step3 - exp(%.6f) = %.6f\n",
           static_cast<float>(shifted),
           static_cast<float>(exp_val));
}

// 第4步：验证 ReduceSum
half exp_sum = ReduceSum(exp_vals);
printf("Step4 - exp_sum = %.6f\n", static_cast<float>(exp_sum));

// 第5步：验证归一化（前3个元素）
for (int i = 0; i < 3 && i < size; ++i) {
    half output_val = exp_vals[i] / exp_sum;
    output[i] = output_val;
    printf("Step5 - output[%d] = %.6f / %.6f = %.6f\n",
           i,
           static_cast<float>(exp_vals[i]),
           static_cast<float>(exp_sum),
           static_cast<float>(output_val));
}

// 验证：输出之和应该接近1
half output_sum = ReduceSum(output);
printf("Verification - output_sum = %.6f (expected: 1.0)\n",
       static_cast<float>(output_sum));
```

### 示例3：ReduceSum 调试

```cpp
// 第1步：打印输入（前N个元素）
printf("Input samples: ");
for (int i = 0; i < min(5, size); ++i) {
    printf("%.6f ", static_cast<float>(input[i]));
}
printf("...\n");

// 第2步：验证累加过程（分段打印）
float sum_fp32 = 0.0f;
for (int i = 0; i < size; ++i) {
    float val = static_cast<float>(input[i]);
    sum_fp32 += val;

    // 每100个元素打印一次
    if ((i + 1) % 100 == 0 || i == size - 1) {
        printf("Step2 - accumulated %d elements: sum = %.6f\n",
               i + 1, sum_fp32);
    }
}

// 第3步：验证最终输出
half output = static_cast<half>(sum_fp32);
printf("Step3 - final output = %.6f\n", static_cast<float>(output));
```

## 调试技巧

### 1. 从外向内验证

从数学公式最外层开始，逐层向内验证：
- 最外层通常是最容易验证的
- 一旦发现某层有问题，聚焦该层内部

### 2. 对比参考值

```cpp
// 使用 CPU/Python 计算参考值
// Python: numpy.exp(x)
half exp_ref = /* 从外部获取的参考值 */;
half exp_npu = Exp(input);

printf("exp comparison: NPU=%.6f, Ref=%.6f, Diff=%.2e\n",
       static_cast<float>(exp_npu),
       static_cast<float>(exp_ref),
       static_cast<float>(abs(exp_npu - exp_ref)));
```

### 3. 选择性打印

避免输出爆炸，只打印关键信息：

```cpp
// 只打印前N个元素
const int PRINT_N = 3;
for (int i = 0; i < PRINT_N && i < size; ++i) {
    printf("arr[%d] = %.6f\n", i, static_cast<float>(arr[i]));
}

// 条件打印：只打印大误差位置
for (int i = 0; i < size; ++i) {
    if (abs(output[i] - expected[i]) > threshold) {
        printf("Mismatch @%d: got %.6f, exp %.6f\n",
               i, output[i], expected[i]);
    }
}

// 采样打印：每隔N个打印一个
for (int i = 0; i < size; i += 100) {
    printf("arr[%d] = %.6f\n", i, static_cast<float>(arr[i]));
}
```

### 4. 数组边界检查

```cpp
// 打印数组边界，检查是否有越界
printf("Array bounds: [0] = %.6f, [size-1] = %.6f\n",
       static_cast<float>(arr[0]),
       static_cast<float>(arr[size - 1]));

// 打印数组长度
printf("Array size: %d\n", size);
```

### 5. 数学性质验证

利用算子的数学性质验证：

```cpp
// Softmax: 输出之和应该等于1
half output_sum = ReduceSum(output);
printf("Softmax check: sum(output) = %.6f (expected: 1.0)\n",
       static_cast<float>(output_sum));

// ReLU: 输出不应该有负值
bool has_negative = false;
for (int i = 0; i < size; ++i) {
    if (output[i] < 0.0h) {
        has_negative = true;
        break;
    }
}
printf("ReLU check: has_negative = %s\n", has_negative ? "true" : "false");

// 对称性：sin(-x) = -sin(x)
half sin_x = Sin(x);
half sin_neg_x = Sin(-x);
half symmetry_sum = sin_x + sin_neg_x;
printf("Symmetry check: sin(%.2f) + sin(%.2f) = %.6f (expected: 0)\n",
       static_cast<float>(x),
       static_cast<float>(-x),
       static_cast<float>(symmetry_sum));
```

## 二分调试决策树

```
开始二分调试
    │
    ├─ 选择数学公式的第一个中间步骤
    │   └─ 添加 printf，输出中间结果
    │
    ├─ 对比参考值（CPU/Python 计算）
    │   │
    │   ├─ 一致 → 此步骤正确，继续下一步
    │   └─ 不一致 → 问题在此步骤，深入分析
    │       │
    │       ├─ 检查 API 调用是否正确
    │       ├─ 检查数据类型是否正确
    │       ├─ 检查是否溢出/下溢
    │       └─ 检查是否满足硬件约束
    │
    └─ 重复直到找到问题根源
```

## 常见问题定位

| 步骤输出异常 | 可能原因 | 检查方向 |
|------------|---------|---------|
| exp() 结果为 Inf | 输入过大 | 是否先减 max |
| ReduceMax 结果错误 | 维度/对齐问题 | 检查硬件约束 |
| 输出之和不为1 | 归一化问题 | 检查 Div 操作 |
| 减法结果误差大 | 减法抵消 | 重排公式 |
| 累加结果误差大 | 精度不足 | 用 FP32 累加器 |
