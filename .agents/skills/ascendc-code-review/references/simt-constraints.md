# 禁止事项

## 参数传递

- 参数传递数组和结构体
- VF 参数 size 超过 28x32bit

## 启动方式

- 使用 `cce_parallel` 启动 SIMT 函数

## UB 使用

- 在 SIMT 函数内部直接使用 UB（需在 tiling 侧设置 `SetLocalMemory`）

## TilingData

- 禁止在 kernel 代码中对 tilingdata 进行赋值
- 禁止将 VF 线程数（threadNum/blockDimX/blockDimY 等）放入 TilingData 结构体
  - VF 线程数是编译期常量，不属于 tiling 参数

## Dim3 参数

- 禁止在 `Simt::Dim3(...)` 中使用从 tilingData 读取的变量
  - Dim3 参数必须是 `constexpr` 变量或数字字面量

## 函数调用

- `__simt_vf__` 内调用 `__simd_callee__` 函数
- `__simd_vf__` 内调用 `__simt_callee__` 函数

## Kernel 入口参数

- kernel 入口函数的入参包含定义的 attr
- 重复的 Input/Output 未在入口参数中对 Output 添加 `_out` 后缀（inplace 算子场景）

---

# 必须遵守规则

## VF_CALL 启动

- 必须使用 `AscendC::Simt::VF_CALL` 启动 SIMT 函数
- 禁止使用 `cce_parallel` 启动 SIMT 函数

## 编译期线程数

- 线程数必须定义为 `constexpr` 常量（如 `constexpr uint32_t THREAD_NUM = 512;`）
- `LAUNCH_BOUND(THREAD_NUM)` 与 `Simt::Dim3(THREAD_NUM)` 必须使用同一个常量

## 函数修饰符

- `AscendC::Simt::VF_CALL` 所调用的函数声明和定义中必须带有 `__simt_vf__` 修饰符
- `__simt_vf__` 修饰的函数内所调用的自定义子函数必须带有 `__simt_callee__` 修饰

## GM 地址使用

- 纯 SIMT 算子应该直接使用传入的 `GM_ADDR` 参数
- 避免申请 GlobalTensor 后 GetPhyAddr

## TilingData

- 禁止在 kernel 代码中对 tilingdata 进行赋值、修改 tilingdata 的内容

## 版权声明

- 代码文件头部 License 声明中时间修改为实际年份

---

# 编译期线程数约束

## 核心规则

SIMT VF 的线程数**必须在编译期确定**，禁止从 tiling 数据动态获取。

## 三处一致

以下三处必须使用同一个 `constexpr` 常量：

1. `constexpr uint32_t THREAD_NUM = 512;`
2. `LAUNCH_BOUND(THREAD_NUM)`
3. `Simt::Dim3(THREAD_NUM)`

## 正确写法

```cpp
constexpr uint32_t THREAD_NUM = 512;

// VF 函数声明
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void OpComputeSimt(...);

// VF 调用
Simt::VF_CALL<OpComputeSimt<T>>(Simt::Dim3(THREAD_NUM), args...);
```

## 错误写法（严禁）

```cpp
// 线程数从 tiling 数据获取（运行时变量）
int32_t threadNum = static_cast<int32_t>(tilingData_->threadNum);
Simt::VF_CALL<OpComputeSimt<T>>(Simt::Dim3(threadNum), args...);
```

## 为什么

- 硬件在编译期需要知道线程数以分配寄存器和调度资源
- 运行时变量无法用于寄存器分配决策
- 动态线程数会导致编译错误或运行时异常

## 数据量小时的调整

如果数据量较小导致线程空转过多，应通过调整**核数**（`SetBlockDim`）来适配，而不是减少线程数。

---

# VF 参数大小限制

## 约束

SIMT VF 入口函数的参数 size 控制在 **28x32bit** 以内。

## 优化原则

1. VF 参数能用模板参数传递，**尽可能使用模板参数传递**
2. VF 参数能在 SIMT VF 内部计算得到的，**就不要作为参数传递**
3. 如果参数 size 超过 28x32bit：
   - 将 VF 内部频繁使用的参数作为入参
   - 其余参数存到 UB 的 buffer 中
   - 将 buffer 地址（`__ubuf__` 指针）作为入参

## 参数类型占用

| 类型 | 占用 (32bit 单位) |
|------|-----------------|
| int32_t / uint32_t / float | 1 |
| int64_t / uint64_t / double | 2 |
| `__gm__ T*` / `__ubuf__ T*` | 2 (64bit 地址) |
| half / bfloat16_t | 1 |
| bool | 1 |

## 示例

```cpp
// 好的做法：常用参数直接传，不常用的放 UB
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void OpSimt(
    uint64_t count,        // 2 units
    __gm__ T* x,           // 2 units
    __gm__ T* y,           // 2 units
    __ubuf__ int64_t* ub   // 2 units - UB 指针传其余参数
) {
    int64_t dimSize = ub[0];      // 从 UB 读取
    int64_t stride = ub[1];
    // ...
}
```

---

# __simt_callee__ 调用约束

## 规则

`__simt_vf__` 修饰的函数内所调用的自定义子函数**必须**带有 `__simt_callee__` 修饰符，表示该函数可在 SIMT VF 内使用。

## 函数调用约束矩阵

| 调用方 | 被调用方 | 是否允许 |
|--------|---------|---------|
| `__simt_vf__` | `__simt_callee__` | 允许 |
| `__simt_vf__` | `constexpr` 函数 | 允许 |
| `__simt_vf__` | `__simd_callee__` | 禁止 |
| `__simt_vf__` | 普通 `__aicore__` 函数 | 禁止 |
| `__simd_vf__` | `__simd_callee__` | 允许 |
| `__simd_vf__` | `constexpr` 函数 | 允许 |
| `__simd_vf__` | `__simt_callee__` | 禁止 |

## 正确写法

```cpp
// SIMT callee 子函数
__simt_callee__ inline int64_t CalcOffset(int64_t base, int64_t stride, int64_t idx) {
    return base + stride * idx;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void OpSimt(
    uint64_t count, __gm__ T* x, __gm__ T* y) {
    for (...) {
        int64_t offset = CalcOffset(0, 1, i);  // 调用 __simt_callee__
        y[i] = x[offset];
    }
}
```

## 错误写法

```cpp
// 缺少 __simt_callee__ 修饰
__aicore__ inline int64_t CalcOffset(int64_t base, int64_t stride, int64_t idx) {
    return base + stride * idx;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void OpSimt(...) {
    int64_t offset = CalcOffset(0, 1, i);  // 编译错误
}
```

---

# UB 与 DCache 约束

## 256KB 总量分区

| 区域 | 大小 | 说明 |
|------|------|------|
| 静态内存 | 编译期确定 | `__ubuf__` 数组声明 |
| 动态内存 | tiling 侧 `SetLocalMemory` 设置 | TBuf/LocalTensor 申请 |
| 预留空间 | 固定 8KB | 编译器预留，不可使用 |
| Data Cache | 256KB - 静态 - 动态 - 8KB | SIMT 专有 DCache |

## DCache 最低要求

- **DCache 必须 >= 32KB**
- 若 DCache < 32KB，编译校验报错
- 计算：可用 UB = 256KB - 8KB - 32KB = **216KB**

## Tiling 侧设置

```cpp
constexpr uint64_t DCACHE_SIZE = 128 * 1024;  // 定义为 128KB
uint64_t ubsize = 256 * 1024;
context->SetLocalMemorySize(ubsize - DCACHE_SIZE);
```

## 内存超用后果

| 问题 | 表现 | 原因 |
|------|------|------|
| DCache 不足 | 编译校验报错 | DCache < 32KB |
| 动态内存不足 | 运行时 UB 访问越界 | SetLocalMemorySize 设置过大 |
| 静态 + 动态超限 | 编译或运行时错误 | 静态 + 动态 > 216KB |