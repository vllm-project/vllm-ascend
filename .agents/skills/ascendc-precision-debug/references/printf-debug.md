# Printf 调试法

## 概述

Printf 调试是最直接、最快速的调试方法，通过在关键位置打印变量值来定位问题。

## Ascend C Printf 基础

### 引入头文件

```cpp
#include "kernel_printf.h"
```

### 基本语法

```cpp
// 打印单个值
printf("Value: %f\n", value);

// 打印多个值
printf("x = %.6f, y = %.6f\n", x, y);

// 打印整数
printf("Index: %d\n", index);

// 科学计数法
printf("Value: %e\n", large_value);
```

### 格式化选项

| 格式 | 用途 | 示例 |
|-----|------|------|
| `%f` | 浮点数（小数形式） | `3.141593` |
| `%.6f` | 浮点数（6位小数） | `3.141593` |
| `%.2e` | 科学计数法（2位小数） | `3.14e+00` |
| `%d` | 整数 | `42` |
| `%s` | 字符串 | `hello` |
| `\n` | 换行 | - |

## Printf 调试技巧

### 1. 选择性打印

避免输出爆炸，只打印可疑位置：

```cpp
// 只打印前 N 个元素
const int PRINT_N = 3;
for (int i = 0; i < PRINT_N && i < size; ++i) {
    printf("arr[%d] = %.6f\n", i, static_cast<float>(arr[i]));
}

// 条件打印：只打印误差大的位置
half threshold = 1e-3h;
for (int i = 0; i < size; ++i) {
    if (abs(output[i] - expected[i]) > threshold) {
        printf("Mismatch @%d: got %.6f, exp %.6f, diff=%.2e\n",
               i,
               static_cast<float>(output[i]),
               static_cast<float>(expected[i]),
               static_cast<float>(abs(output[i] - expected[i])));
    }
}

// 采样打印：每隔 N 个打印一个
for (int i = 0; i < size; i += 100) {
    printf("arr[%d] = %.6f\n", i, static_cast<float>(arr[i]));
}
```

### 2. FP16 打印注意事项

FP16 需要转换为 float 才能正确打印：

```cpp
half value = 3.14h;

// 错误：直接打印可能不准确
printf("Value: %f\n", value);

// 正确：先转换为 float
printf("Value: %.6f\n", static_cast<float>(value));
```

### 3. 关键位置标记

```cpp
printf("[DEBUG] Entering function\n");
// ... 代码 ...
printf("[DEBUG] Before reduce\n");
// ... 代码 ...
printf("[DEBUG] After reduce\n");
// ... 代码 ...
printf("[DEBUG] Exiting function\n");

// 使用行号自动标记
printf("[DEBUG] Line %d: value=%.6f\n", __LINE__, value);
```

### 4. 对比打印

```cpp
// 并排打印期望值和实际值
for (int i = 0; i < size; i += 10) {
    printf("[%d] got=%.6f, exp=%.6f, diff=%.2e\n",
           i,
           static_cast<float>(output[i]),
           static_cast<float>(expected[i]),
           static_cast<float>(abs(output[i] - expected[i])));
}
```

### 5. 数组边界检查

```cpp
// 打印数组边界，检查是否有越界
printf("Array [0] = %.6f\n", static_cast<float>(arr[0]));
printf("Array [size-1] = %.6f\n", static_cast<float>(arr[size-1]));

// 打印数组长度
printf("Array size: %d\n", size);
```

### 6. 统计信息打印

```cpp
// 打印最小值、最大值
half min_val = input[0];
half max_val = input[0];
float sum = 0.0f;

for (int i = 0; i < size; ++i) {
    min_val = min(min_val, input[i]);
    max_val = max(max_val, input[i]);
    sum += static_cast<float>(input[i]);
}

printf("Array stats: min=%.6f, max=%.6f, mean=%.6f\n",
       static_cast<float>(min_val),
       static_cast<float>(max_val),
       sum / static_cast<float>(size));

// 检查是否有 Inf/NaN
bool has_inf = false;
bool has_nan = false;
for (int i = 0; i < size; ++i) {
    float val = static_cast<float>(input[i]);
    if (isinf(val)) has_inf = true;
    if (isnan(val)) has_nan = true;
}
printf("Array checks: has_inf=%d, has_nan=%d\n", has_inf, has_nan);
```

## 高级用法

### 1. 条件编译调试开关

```cpp
// 定义调试开关
#define DEBUG_PRECISION 1

#if DEBUG_PRECISION
    #define DEBUG_PRINT(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
    #define DEBUG_PRINT(fmt, ...)
#endif

// 使用
DEBUG_PRINT("Debug info: value=%.6f\n", value);
```

### 2. 分段打印

```cpp
// 在长循环中分段打印
for (int i = 0; i < size; ++i) {
    // ... 计算 ...

    // 每 1000 次迭代打印一次进度
    if ((i + 1) % 1000 == 0) {
        printf("Progress: %d/%d (%.1f%%)\n",
               i + 1, size, (i + 1) * 100.0f / size);
    }
}
```

### 3. 函数入口/出口追踪

```cpp
half Compute(half x) {
    printf("[ENTER] Compute(%.6f)\n", static_cast<float>(x));

    // ... 计算 ...

    printf("[EXIT] Compute() -> %.6f\n", static_cast<float>(result));
    return result;
}
```

### 4. 断言式打印

```cpp
// 打印并验证条件
bool condition = /* ... */;
printf("[ASSERT] condition=%s (expected: true)\n",
       condition ? "true" : "false");

// 打印并验证值
half expected = 1.0h;
half actual = /* ... */;
printf("[VERIFY] expected=%.6f, actual=%.6f, match=%s\n",
       static_cast<float>(expected),
       static_cast<float>(actual),
       (abs(actual - expected) < 1e-6h) ? "true" : "false");
```

## 常见 Printf 模式

### 模式1：数值范围检查

```cpp
void CheckValueRange(const char* name, half* arr, int size) {
    half min_val = arr[0];
    half max_val = arr[0];
    float sum = 0.0f;
    int inf_count = 0;
    int nan_count = 0;

    for (int i = 0; i < size; ++i) {
        min_val = min(min_val, arr[i]);
        max_val = max(max_val, arr[i]);
        sum += static_cast<float>(arr[i]);

        float val = static_cast<float>(arr[i]);
        if (isinf(val)) inf_count++;
        if (isnan(val)) nan_count++;
    }

    printf("[%s] min=%.6f, max=%.6f, mean=%.6f, inf=%d, nan=%d\n",
           name,
           static_cast<float>(min_val),
           static_cast<float>(max_val),
           sum / static_cast<float>(size),
           inf_count,
           nan_count);
}
```

### 模式2：步骤追踪

```cpp
void TraceSteps(const char* step, half value) {
    printf("[STEP] %s: value=%.6f\n", step, static_cast<float>(value));
}

// 使用
TraceSteps("initial", input);
TraceSteps("after_exp", exp_result);
TraceSteps("after_sum", sum_result);
TraceSteps("final", output);
```

### 模式3：错误定位

```cpp
void LocateErrors(half* output, half* expected, int size) {
    int error_count = 0;
    half max_error = 0.0h;
    int max_error_idx = -1;

    for (int i = 0; i < size; ++i) {
        half error = abs(output[i] - expected[i]);
        if (error > 1e-3h) {
            error_count++;
            if (error > max_error) {
                max_error = error;
                max_error_idx = i;
            }
        }
    }

    printf("[ERRORS] count=%d, max_error=%.2e @%d\n",
           error_count,
           static_cast<float>(max_error),
           max_error_idx);
}
```

## Printf 性能注意

1. **生产代码移除**：Printf 会影响性能，调试完成后应移除
2. **避免过多输出**：过多 Printf 会输出爆炸，影响调试
3. **使用条件编译**：通过宏定义控制调试输出
4. **选择性打印**：只打印关键信息，避免全量打印
