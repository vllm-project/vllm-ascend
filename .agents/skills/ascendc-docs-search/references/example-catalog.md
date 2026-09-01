# 示例代码目录

基于 `$ASC_DEVKIT_DIR/examples/` 的完整示例代码索引，共 587 个代码文件。

---

## 目录结构概览

```
examples/
├── 00_introduction/      # 基础示例（入门必读）
├── 01_utilities/         # 调试工具示例
├── 02_features/          # 特性展示
├── 03_libraries/         # API 类库使用
└── 04_best_practices/    # 性能优化实践
```

---

## 00_introduction - 基础示例

入门级示例，适合初学者了解 Ascend C 编程基础。

| 示例名称 | 路径 | 用途 |
|---------|------|------|
| Hello World | `00_helloworld/` | CPU/NPU 版本入门示例 |
| 加法算子 | `01_add/` | 各类加法实现 |
| 矩阵乘法 | `02_matmul/` | 矩阵乘法基础实现 |
| MatMul+LeakyReLU | `03_matmulleakyrelu/` | 组合算子示例 |
| 简单算子 | `04_simple_operator/` | 多种简单算子 |

### 01_add - 加法算子详解

| 子示例 | 路径 | 说明 |
|-------|------|------|
| 内存分配器加法 | `basic_api_memory_allocator_add/` | 基础 API + 内存分配器 |
| TQueue 加法 | `basic_api_tque_add/` | 基础 API + TQueue |
| C API 异步 | `c_api_async_add/` | C API 异步调用 |
| C API 同步 | `c_api_sync_add/` | C API 同步调用 |
| 微 API | `micro_api_add/` | 微 API 实现 |

### 04_simple_operator - 简单算子集

| 子示例 | 说明 |
|-------|------|
| `add_dynamic/` | 动态形状加法 |
| `broadcast/` | 广播操作 |
| `sub/` | 减法算子 |
| `pure_simt_gather/` | SIMT gather 操作 |
| `tmp_buffer/` | 临时缓冲区使用 |

---

## 01_utilities - 调试工具

| 示例名称 | 路径 | 用途 |
|---------|------|------|
| Printf | `00_printf/printf.asc` | 打印调试方法 |
| Assert | `01_assert/assert.asc` | 断言使用示例 |
| DumpTensor | `02_dumptensor/` | 张量转储调试 |
| CPU Debug | `03_cpudebug/` | CPU 调试方法 |

**重点**：`printf.asc` 是调试代码的必读参考。

---

## 02_features - 特性展示

展示 Ascend C 的各种高级特性。

| 特性类别 | 路径 | 说明 |
|---------|------|------|
| 框架启动 | `00_framework_launch/` | ACLNN/ACLOP 调用 |
| 三重尖括号 | `01_triple_chevron_notation/` | Kernel 启动语法 |
| C API | `02_c_api/` | C API 使用 |
| SIMT | `03_simt/` | SIMT 编程模型 |
| 微 API | `04_micro_api/` | 微 API 特性 |
| 静态张量 | `06_static_tensor_programming/` | 静态张量编程 |
| 数据移动 | `07_data_movement/` | 高效数据搬运 |
| **Tiling** | `08_tiling/` | Tiling 优化方法 |
| 非对齐访问 | `09_unalign/` | 非对齐数据处理 |
| 内存管理 | `10_memory_management/` | 内存池、分配器 |
| 同步控制 | `11_synchronous_control/` | 多核同步 |
| 系统变量 | `12_system_variable_access/` | 系统变量访问 |
| 原子操作 | `13_atomic_operations/` | 原子操作使用 |
| Cube 组 | `14_cube_group_management/` | Cube 单元管理 |
| 工具函数 | `15_utility_function/` | 辅助工具 |
| 标量计算 | `16_scalar_computation/` | 标量运算 |

### 00_framework_launch - 框架启动

| 子示例 | 说明 |
|-------|------|
| `aclnn_invocation/` | ACLNN 接口调用 |
| `aclop_invocation/` | ACLOP 接口调用 |
| `custom_op/` | 自定义算子 |
| `tiling_sink_programming/` | Tiling 汇编编程 |
| `tiling_template_programming/` | Tiling 模板编程 |

---

## 03_libraries - API 类库

封装好的高级 API 库，可直接调用。

| 库名称 | 路径 | 说明 |
|-------|------|------|
| 数学库 | `00_math/` | 三角函数、位运算、类型转换等 |
| 激活函数 | `01_activation/` | ReLU、Sigmoid、Softmax 等 |
| 归一化 | `02_normalization/` | BatchNorm、LayerNorm 等 |
| 池化 | `03_pooling/` | MaxPool、AvgPool 等 |
| 卷积 | `04_convolution/` | Conv2D 等 |
| 矩阵乘法 | `05_matmul/` | MatMul 高级实现 |
| 归约 | `06_reduction/` | ReduceSum、ReduceMax 等 |
| 排序 | `07_sort/` | 排序操作 |
| 量化 | `08_quantization/` | 量化相关 |
| 变换 | `09_transform/` | 数据变换 |

### 00_math - 数学库详解

三角函数：`acos`、`acosh`、`asin`、`atanh` 等
位运算：`bitwiseand`、`bitwisenot`、`bitwiseor`、`bitwisexor`
类型转换：`cast`
复合运算：`addcdiv`、`addsub` 等

---

## 04_best_practices - 性能优化实践

**高性能算子开发必读**，包含双缓冲、流水线等核心优化技术。

| 优化主题 | 路径 | 说明 |
|---------|------|------|
| **双缓冲加法** | `00_add_doublebuffer/` | **高性能模板**，双缓冲+流水线 |
| 存储体冲突 | `01_bank_conflict/` | Bank 冲突优化 |
| AI CPU Tiling | `02_aicpu_device_tiling/` | AI CPU 设备 Tiling |
| L2 缓存旁路 | `03_l2_cache_bypass/` | L2 Cache 优化 |
| 地址冲突 | `05_mata_address_conflict/` | 地址冲突优化 |
| 分组矩阵乘法 | `06_grouped_matmul/` | Grouped MatMul |
| 兼容性案例 | `10_compatibility_cases/` | 兼容性处理 |
| 模式转换 | `11_pattern_transformation/` | 数据模式转换 |
| 高性能 VF | `12_high_performance_vf/` | 矢量函数优化 |
| 数据拷贝优化 | `13_optimize_datacopy/` | DataCopy 优化 |

### 00_add_doublebuffer - 高性能模板

**位置**：`$ASC_DEVKIT_DIR/examples/04_best_practices/00_add_doublebuffer/`

这是实现高性能算子的**核心参考模板**，展示：
- 双缓冲技术
- 流水线并行
- 数据搬运与计算重叠
- 内存分配器使用

---

## 使用建议

1. **学习路径**：
   ```
   00_introduction/01_add/  →  基础入门
   01_utilities/00_printf/  →  调试方法
   04_best_practices/00_add_doublebuffer/  →  高性能模板
   ```

2. **开发新算子时**：
   - 第一步：参考 `00_add_doublebuffer/` 的高性能模板
   - 第二步：查阅 `03_libraries/` 中是否有可直接使用的库函数
   - 第三步：参考 `02_features/` 了解特定 API 的使用方法

3. **遇到问题时**：
   - 调试：使用 `01_utilities/00_printf/` 的方法
   - 性能问题：参考 `04_best_practices/` 中的优化示例
   - API 使用：查阅 `02_features/` 中的对应特性示例

---

## 统计信息

- 代码文件总数：587 个（.asc、.cpp、.c、.h、.hpp）
- README 文件数：351 个

---

## 相关资源

- [API 文档索引](api-index.md)
- [环境兼容性表](compatibility.md)
