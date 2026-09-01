# SIMT 精度调试技巧

## SIMT 精度调试

### Host 侧（Tiling）

tiling 侧可以直接使用 `std::cout` 打印：

```cpp
std::cout << "perCoreElements: " << perCoreElements << std::endl;
```

### Device 侧

#### SIMT VF 内部

使用 `AscendC::Simt::printf` 打印变量值：

```cpp
AscendC::Simt::printf("simt value %ld", value);
```

#### SIMT VF 外部

使用 `PRINTF` 宏打印变量值：

```cpp
PRINTF("simt value %ld", value);
```

### 调试步骤

1. 检查 Tiling 参数是否正确传递，是否符合预期
2. 在 VF 内关键位置打印变量值，验证计算逻辑
3. 对比 GM 输入数据和输出数据，定位精度问题

### 头文件

```cpp
#include "simt_api/asc_simt.h"       // printf
```

---

## DCache 一致性

### 问题

Scalar 单元访问 Global Memory（SIMT_VF 外部通过 GlobalTensor 或者 GM 地址单点访问），首先会访问每个核内的 Data Cache，因此存在 Data Cache 与 Global Memory 的 Cache 一致性问题。

### 解决方案

用户通过 Scalar 单元写 Global Memory 的数据后，需要使用 `DataCacheCleanAndInvalid` 刷新 Cache，保证 Cache 的一致性。

### 函数原型

```cpp
template <typename T, CacheLine entireType, DcciDst dcciDst>
__aicore__ inline void DataCacheCleanAndInvalid(const GlobalTensor<T>& dst);
```

### 使用示例

```cpp
AscendC::DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(offsetGlobal[0]);
// offsetGlobal 为 uint64_t 类型的 GlobalTensor
```

### 触发场景

| 场景 | 是否需要刷新 | 原因 |
|------|------------|------|
| SIMT VF 内直接读写 GM | 否 | VF 内走线程路径，不经过 Scalar DCache |
| VF 外通过 GlobalTensor 单点写 GM | 是 | 写操作经过 DCache，需刷新保证一致性 |
| VF 外通过 GM 指针单点写 GM | 是 | 同上 |
| 多核之间通过 workspace 交换数据 | 是 | 写入后其他核读取需保证可见 |