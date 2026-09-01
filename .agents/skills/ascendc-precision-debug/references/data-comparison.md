# 数据对比法

## 概述

数据对比法是通过系统地构造测试用例，对比不同条件下的输出结果来定位精度问题的方法。

## 最小可复现测试

### 测试顺序原则

```
优先级顺序：
1. 32字节对齐 + FP32  → 排除对齐和精度问题
2. 非对齐测试          → 检查对齐处理
3. FP16 精度测试      → 验证 FP16 精度
```

### 为什么这个顺序？

| 测试类型 | 目的 | 通过说明 |
|---------|------|---------|
| 对齐 + FP32 | 排除对齐和精度问题 | 算法逻辑正确 |
| 非对齐 | 检查对齐处理 | 需要处理非对齐输入 |
| FP16 | 验证 FP16 精度 | FP16 精度是否足够 |

### 测试用例构造

#### 1. 最小对齐测试（优先）

```python
import numpy as np

# 32字节对齐 + FP32（排除对齐和精度问题）
# FP32: 32字节 = 8个元素
test_input_fp32 = np.random.rand(8, 8, 8).astype(np.float32)  # 尾轴8=8*4字节=32字节对齐

# 或使用简单值便于验证
test_input_fp32 = np.ones((8, 8, 8), dtype=np.float32)

# 或使用已知输入验证输出
test_input_fp32 = np.zeros((8, 8, 8), dtype=np.float32)
test_input_fp32[0, 0, 0] = 1.0  # 单个非零值
```

#### 2. 非对齐测试

```python
# 非对齐输入，测试是否需要特殊处理
test_input_unaligned = np.random.rand(8, 8, 9).astype(np.float32)  # 尾轴9，非32字节对齐

# 或
test_input_unaligned = np.random.rand(8, 8, 17).astype(np.float32)  # 尾轴17
```

#### 3. FP16 精度测试

```python
# FP16 测试（保持对齐）
# FP16: 32字节 = 16个元素
test_input_fp16 = np.random.rand(8, 8, 16).astype(np.float16)  # 尾轴16=16*2字节=32字节对齐

# 对比 FP16 和 FP32 结果
result_fp32 = run_operator(test_input_fp32.astype(np.float32))
result_fp16 = run_operator(test_input_fp16.astype(np.float16))

# 分析精度差异
error = np.abs(result_fp32 - result_fp16)
print(f"FP16 vs FP32: max error = {error.max():.2e}")
```

## 对齐计算参考

### 32字节对齐规则

| 数据类型 | 每元素字节数 | 32字节对齐元素数 | 示例形状 |
|---------|-------------|----------------|---------|
| FP16 | 2 字节 | 16 个元素 | (..., 16), (..., 32), (..., 48) |
| FP32 | 4 字节 | 8 个元素 | (..., 8), (..., 16), (..., 24) |
| INT8 | 1 字节 | 32 个元素 | (..., 32), (..., 64), (..., 96) |

### 检查是否对齐

```python
def is_32byte_aligned(shape, dtype):
    """检查形状尾轴是否32字节对齐"""
    element_size = np.dtype(dtype).itemsize
    last_dim = shape[-1]
    return (last_dim * element_size) % 32 == 0

# 示例
print(is_32byte_aligned((8, 16), np.float16))  # True: 16*2=32字节
print(is_32byte_aligned((8, 8), np.float32))   # True: 8*4=32字节
print(is_32byte_aligned((8, 17), np.float16))  # False: 17*2=34字节
```

### 生成对齐测试数据

```python
def generate_aligned_test(shape, dtype):
    """生成32字节对齐的测试数据"""
    element_size = np.dtype(dtype).itemsize
    aligned_size = 32 // element_size

    # 调整尾轴为对齐大小的倍数
    adjusted_shape = list(shape)
    adjusted_shape[-1] = ((shape[-1] + aligned_size - 1) // aligned_size) * aligned_size

    return np.random.rand(*adjusted_shape).astype(dtype)

# 示例
test_data = generate_aligned_test((8, 10), np.float32)  # 尾轴调整为16（8的倍数）
print(f"Adjusted shape: {test_data.shape}")  # (8, 16)
```

## 边界值测试

### 标准边界值集合

```python
boundary_cases = {
    "零值": 0.0,
    "极小值": 1e-10,
    "小值": 1e-6,
    "正常值": 1.0,
    "大值": 1e6,
    "极大值": 1e10,
    "负值": -1.0,
    "FP16饱和": 65504.0,     # FP16 最大值
    "FP16负饱和": -65504.0,  # FP16 最小值
}

# 生成边界测试数据
for name, value in boundary_cases.items():
    test_input = np.full((8, 8), value, dtype=np.float32)
    result = run_operator(test_input)
    print(f"{name}: input={value}, output={result[0, 0]}")
```

### 特殊值测试

```python
# 特殊浮点值
special_values = {
    "正无穷": np.inf,
    "负无穷": -np.inf,
    "NaN": np.nan,
}

# 注意：Ascend C 可能不支持 Inf/NaN，需要特殊处理
```

## 中间结果对比

### 逐步对比方法

```python
# 假设算子分为多个步骤
def step1_exp(x):
    return np.exp(x)

def step2_minus(exp_x, exp_neg_x):
    return exp_x - exp_neg_x

def step3_divide(numerator):
    return numerator / 2.0

# 每个步骤单独验证
x = 1.5
exp_x = step1_exp(x)
exp_neg_x = step1_exp(-x)
numerator = step2_minus(exp_x, exp_neg_x)
result = step3_divide(numerator)

print(f"Step 1 - exp({x}) = {exp_x}")
print(f"Step 1 - exp({-x}) = {exp_neg_x}")
print(f"Step 2 - {exp_x} - {exp_neg_x} = {numerator}")
print(f"Step 3 - {numerator} / 2 = {result}")
```

### CPU vs NPU 对比

```python
# CPU 参考（使用 NumPy）
def softmax_cpu(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

# NPU 结果（从算子获取）
npu_output = run_operator_on_npu(input_data)

# 对比
cpu_output = softmax_cpu(input_data)
error = np.abs(npu_output - cpu_output)

print(f"Max error: {error.max():.2e}")
print(f"Mean error: {error.mean():.2e}")

# 找出最大误差位置
max_error_idx = error.argmax()
print(f"Worst case @ {max_error_idx}:")
print(f"  CPU: {cpu_output.flatten()[max_error_idx]}")
print(f"  NPU: {npu_output.flatten()[max_error_idx]}")
```

## 类型转换测试

### 逐级降精度测试

```python
# 从 FP32 开始，逐步降精度到 FP16
input_data = np.random.rand(8, 16).astype(np.float64)  # 最高精度
expected = softmax_cpu(input_data)

for dtype in [np.float32, np.float16]:
    result = run_operator(input_data.astype(dtype))
    error = np.abs(result - expected)

    print(f"dtype={dtype.__name__}:")
    print(f"  Max error: {error.max():.2e}")
    print(f"  Mean error: {error.mean():.2e}")
```

### 混合精度测试

```python
# 测试不同中间精度的效果
# 需要修改算子代码以支持不同的累加器精度

# 测试1：全 FP16
result_all_fp16 = run_operator_all_fp16(input_data)

# 测试2：累加器 FP32
result_fp32_accum = run_operator_fp32_accum(input_data)

# 对比
print(f"All FP16: max error = {np.abs(result_all_fp16 - expected).max():.2e}")
print(f"FP32 Accum: max error = {np.abs(result_fp32_accum - expected).max():.2e}")
```

## 规模测试

### 不同规模测试

```python
# 测试不同规模下的精度
test_shapes = [
    (8, 8),      # 小规模
    (16, 16),    # 中小规模
    (32, 32),    # 中等规模
    (64, 64),    # 中大规模
    (128, 128),  # 大规模
    (256, 256),  # 超大规模
]

for shape in test_shapes:
    test_input = np.random.rand(*shape).astype(np.float32)
    result = run_operator(test_input)
    expected = reference_implementation(test_input)
    error = np.abs(result - expected)

    print(f"Shape {shape}: max error = {error.max():.2e}")
```

### 硬件约束边界测试

```python
# 测试 Reduce 操作的最小元素数约束
reduce_sizes = [1, 2, 4, 8, 16, 32, 64]

for size in reduce_sizes:
    test_input = np.random.rand(8, size).astype(np.float32)
    result = run_operator(test_input)
    expected = reference_implementation(test_input)
    error = np.abs(result - expected)

    status = "PASS" if error.max() < 1e-5 else "FAIL"
    print(f"Reduce size {size}: {status}, max error = {error.max():.2e}")
```

## 测试脚本模板

```python
import numpy as np

def compare_results(output, expected, rtol=1e-5, atol=1e-6):
    """对比结果并打印详细误差信息"""
    abs_error = np.abs(output - expected)
    rel_error = abs_error / (np.abs(expected) + atol)

    print(f"Max abs error: {abs_error.max():.2e}")
    print(f"Mean abs error: {abs_error.mean():.2e}")
    print(f"Max rel error: {rel_error.max():.2e}")
    print(f"Mean rel error: {rel_error.mean():.2e}")

    # 通过率
    pass_mask = np.logical_or(abs_error < atol, rel_error < rtol)
    pass_rate = pass_mask.sum() / pass_mask.size * 100
    print(f"Pass rate: {pass_rate:.2f}%")

    # 最差样本
    worst_idx = abs_error.argmax()
    print(f"Worst case @ {np.unravel_index(worst_idx, output.shape)}:")
    print(f"  Output: {output.flatten()[worst_idx]:.6f}")
    print(f"  Expected: {expected.flatten()[worst_idx]:.6f}")
    print(f"  Abs error: {abs_error.flatten()[worst_idx]:.2e}")

    return pass_rate > 99.0  # 99% 通过为合格
```
