# 工具和命令参考

## 误差分析命令

### 快速误差统计

```bash
# 最大误差和平均误差
python3 -c "import numpy as np; pred=np.load('output.npy'); truth=np.load('expected.npy'); \
  print(f'Max: {abs(pred-truth).max():.2e}, Mean: {abs(pred-truth).mean():.2e}')"

# 找出最差样本
python3 -c "import numpy as np; pred=np.load('output.npy'); truth=np.load('expected.npy'); \
  err=abs(pred-truth); idx=err.argmax(); \
  print(f'Worst@{idx}: pred={pred.flat[idx]}, truth={truth.flat[idx]}')"

# 完整误差统计（包括相对误差和分位数）
python3 -c "import numpy as np; pred=np.load('output.npy'); truth=np.load('expected.npy'); \
  err=abs(pred-truth); rel_err=err/(abs(truth)+1e-9); \
  print(f'Max abs: {err.max():.2e}, Max rel: {rel_err.max():.2e}, 95th: {np.percentile(rel_err, 95):.2e}')"
```

### 详细误差分析脚本

```python
# error_analysis.py
import numpy as np
import sys

def analyze_error(pred_file, truth_file, rtol=1e-5, atol=1e-6):
    pred = np.load(pred_file)
    truth = np.load(truth_file)

    abs_error = np.abs(pred - truth)
    rel_error = abs_error / (np.abs(truth) + atol)

    print("=" * 60)
    print("误差分析报告")
    print("=" * 60)
    print(f"预测文件: {pred_file}")
    print(f"真值文件: {truth_file}")
    print()

    # 绝对误差统计
    print("绝对误差统计:")
    print(f"  最大值: {abs_error.max():.6e}")
    print(f"  平均值: {abs_error.mean():.6e}")
    print(f"  中位数: {np.median(abs_error):.6e}")
    print(f"  标准差: {abs_error.std():.6e}")
    print()

    # 相对误差统计
    print("相对误差统计:")
    print(f"  最大值: {rel_error.max():.6e}")
    print(f"  平均值: {rel_error.mean():.6e}")
    print(f"  中位数: {np.median(rel_error):.6e}")
    print(f"  95分位: {np.percentile(rel_error, 95):.6e}")
    print(f"  99分位: {np.percentile(rel_error, 99):.6e}")
    print()

    # 通过率
    pass_mask = np.logical_or(abs_error < atol, rel_error < rtol)
    pass_rate = pass_mask.sum() / pass_mask.size * 100
    print(f"通过率: {pass_rate:.2f}%")
    print()

    # 误差分布
    print("误差分布:")
    for threshold in [1e-3, 1e-4, 1e-5, 1e-6]:
        count = (abs_error > threshold).sum()
        rate = count / abs_error.size * 100
        print(f"  误差 > {threshold:.0e}: {count} ({rate:.2f}%)")
    print()

    # 最差样本
    worst_idx = abs_error.argmax()
    worst_pos = np.unravel_index(worst_idx, pred.shape)
    print(f"最差样本 @ {worst_pos}:")
    print(f"  预测值: {pred[worst_pos]:.6f}")
    print(f"  真值: {truth[worst_pos]:.6f}")
    print(f"  绝对误差: {abs_error[worst_pos]:.6e}")
    print(f"  相对误差: {rel_error[worst_pos]:.6e}")

    return pass_rate > 99.0

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python3 error_analysis.py <output.npy> <expected.npy>")
        sys.exit(1)

    success = analyze_error(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
```

## Printf 格式化参考

### 基本格式

| 格式符 | 类型 | 说明 | 示例 |
|-------|------|------|------|
| `%f` | float | 小数形式 | `3.141593` |
| `%.6f` | float | 6位小数 | `3.141593` |
| `%.2e` | float | 科学计数法 | `3.14e+00` |
| `%d` | int | 整数 | `42` |
| `%u` | unsigned | 无符号整数 | `42` |
| `%x` | hex | 十六进制 | `0x2a` |
| `%c` | char | 字符 | `A` |
| `%s` | string | 字符串 | `hello` |

### Ascend C Printf 示例

```cpp
#include "kernel_printf.h"

// 基础打印
printf("Value: %f\n", value);

// 指定小数位数
printf("Value: %.6f\n", value);     // 6位小数
printf("Value: %.2f\n", value);     // 2位小数

// 科学计数法
printf("Large: %.2e\n", large_value);

// 多个值
printf("x=%.6f, y=%.6f\n", x, y);

// 整数
printf("Index: %d\n", index);
printf("Size: %d x %d\n", height, width);

// 字符串
printf("Status: %s\n", "OK");

// 调试信息
printf("[DEBUG] Line %d: value=%.6f\n", __LINE__, value);

// FP16 需要转换
half h = 3.14h;
printf("Half: %.6f\n", static_cast<float>(h));
```

## NPU 运行命令

### 基本测试运行

```bash
# 进入 Docker 容器运行
./env_setup.sh "cd ops/my_operator/build && ./my_operator"

# 带参数运行
./env_setup.sh "cd ops/my_operator/build && ./my_operator 16 16 8 fp32"

# FP16 测试
./env_setup.sh "cd ops/my_operator/build && ./my_operator 16 16 8 fp16"
```

### 批量测试脚本

```bash
#!/bin/bash
# batch_test.sh

# 测试不同的输入规模
shapes=("8:8:8" "16:16:8" "32:16:8" "64:32:8")
dtypes=("fp32" "fp16")

for shape in "${shapes[@]}"; do
    for dtype in "${dtypes[@]}"; do
        IFS=':' read -r M N K <<< "$shape"
        echo "Testing: M=$M, N=$N, K=$K, dtype=$dtype"

        ./env_setup.sh "cd ops/my_operator/build && ./my_operator $M $N $K $dtype"

        if [ $? -eq 0 ]; then
            echo "  PASS"
        else
            echo "  FAIL"
        fi
    done
done
```

## 数据生成脚本

### 生成对齐测试数据

```python
# gen_aligned_data.py
import numpy as np
import sys

def generate_aligned(shape, dtype, output_path):
    """
    生成32字节对齐的测试数据

    shape: tuple, 数据形状
    dtype: str, 数据类型 (fp16, fp32, int8)
    output_path: str, 输出文件路径
    """
    np_type = {
        'fp16': np.float16,
        'fp32': np.float32,
        'int8': np.int8,
    }[dtype]

    # 检查并调整对齐
    element_size = np.dtype(np_type).itemsize
    aligned_size = 32 // element_size

    adjusted_shape = list(shape)
    adjusted_shape[-1] = ((shape[-1] + aligned_size - 1) // aligned_size) * aligned_size

    # 生成随机数据
    data = np.random.rand(*adjusted_shape).astype(np_type)

    # 保存
    np.save(output_path, data)
    print(f"生成数据: {output_path}")
    print(f"  原始形状: {shape}")
    print(f"  调整形状: {tuple(adjusted_shape)}")
    print(f"  数据类型: {dtype}")
    print(f"  数据范围: [{data.min():.6f}, {data.max():.6f}]")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("用法: python3 gen_aligned_data.py <M> <N> <K> <dtype>")
        sys.exit(1)

    M, N, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    dtype = sys.argv[4]

    generate_aligned((M, N, K), dtype, f"input_{M}_{N}_{K}_{dtype}.npy")
```

### 生成边界值测试数据

```python
# gen_boundary_data.py
import numpy as np

def generate_boundary_tests(dtype):
    """生成边界值测试数据"""
    np_type = np.float16 if dtype == 'fp16' else np.float32

    cases = {
        "zero": 0.0,
        "tiny": 1e-10,
        "small": 1e-6,
        "normal": 1.0,
        "large": 1e6,
        "huge": 1e10,
        "negative": -1.0,
    }

    if dtype == 'fp16':
        cases["saturation"] = 65504.0
        cases["negative_saturation"] = -65504.0

    for name, value in cases.items():
        data = np.full((8, 16), value, dtype=np_type)
        np.save(f"boundary_{name}_{dtype}.npy", data)
        print(f"生成: boundary_{name}_{dtype}.npy (value={value})")

if __name__ == "__main__":
    generate_boundary_tests("fp32")
    generate_boundary_tests("fp16")
```

## 结果验证脚本

```python
# verify_result.py
import numpy as np
import sys

def verify_result(output_file, expected_file, rtol=1e-5, atol=1e-6):
    """验证算子输出结果"""
    output = np.load(output_file)
    expected = np.load(expected_file)

    # 检查形状
    if output.shape != expected.shape:
        print(f"形状不匹配: output={output.shape}, expected={expected.shape}")
        return False

    # 计算误差
    abs_error = np.abs(output - expected)
    rel_error = abs_error / (np.abs(expected) + atol)

    max_abs_error = abs_error.max()
    max_rel_error = rel_error.max()

    # 打印结果
    print("=" * 60)
    print("验证结果")
    print("=" * 60)
    print(f"最大绝对误差: {max_abs_error:.6e}")
    print(f"最大相对误差: {max_rel_error:.6e}")

    # 判断通过
    pass_mask = np.logical_or(abs_error < atol, rel_error < rtol)
    pass_count = pass_mask.sum()
    total_count = pass_mask.size
    pass_rate = pass_count / total_count * 100

    print(f"通过率: {pass_count}/{total_count} ({pass_rate:.2f}%)")

    if pass_rate >= 99.0:
        print("验证: PASS")
        return True
    else:
        print("验证: FAIL")

        # 打印失败样本
        fail_indices = np.where(~pass_mask)
        if len(fail_indices[0]) > 0:
            print("\n失败样本（前10个）:")
            fail_count = min(10, len(fail_indices[0]))
            for i in range(fail_count):
                idx = tuple(dim[i] for dim in fail_indices)
                print(f"  @{idx}: output={output[idx]:.6f}, "
                      f"expected={expected[idx]:.6f}, "
                      f"abs_err={abs_error[idx]:.2e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python3 verify_result.py <output.npy> <expected.npy> [rtol] [atol]")
        sys.exit(1)

    rtol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5
    atol = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-6

    success = verify_result(sys.argv[1], sys.argv[2], rtol, atol)
    sys.exit(0 if success else 1)
```

## 推荐容差配置

### 默认容差

| 场景 | rtol | atol | 说明 |
|-----|------|------|------|
| FP16 | 1e-3 | 1e-4 | 浮点精度有限 |
| FP32 | 1e-5 | 1e-6 | 标准精度 |
| INT8 | - | 0 | 必须精确匹配 |
| Softmax (FP16) | 1e-3 | 1e-4 | 概率输出 |
| Softmax (FP32) | 1e-5 | 1e-6 | 概率输出 |
| Reduce (FP16) | 5e-3 | 1e-4 | 累加误差较大 |
| Reduce (FP32) | 1e-5 | 1e-6 | 标准精度 |

### 使用示例

```bash
# FP16 验证（宽松容差）
python3 verify_result.py output.npy expected.npy 1e-3 1e-4

# FP32 验证（标准容差）
python3 verify_result.py output.npy expected.npy 1e-5 1e-6

# Reduce 算子验证（更宽松）
python3 verify_result.py output.npy expected.npy 5e-3 1e-4
```
