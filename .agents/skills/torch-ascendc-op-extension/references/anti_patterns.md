# Stream 同步反模式大全

> 来源：`torchair/examples/_kernel_extension_aclgraph/torch_library/csrc/add_custom.asc`
>
> NPU 的 taskqueue 是设备端任务队列。"清 queue"指等待队列中已有任务完成后再执行当前任务；"入 queue"指将当前任务放入队列中按顺序执行。自定义算子的内核启动方式不当会导致乱序或死锁。

---

## 架构区分

| 架构 | Host 文件 | Kernel 调用方式 | 推荐模式 |
|------|----------|----------------|---------|
| **单文件 .asc**（torchair 风格） | `.asc` 内含 host 代码 | `<<<>>>` 内核启动符 | 方式1（传 NPUStream）或方式3（OpCommand） |
| **多文件 CMake**（本 SKILL） | `xxx_torch.cpp`（C++ 编译） | 函数调用（无 `<<<>>>`） | **只能用方式2**（`stream(true)`） |

---

## 方式1（推荐）：直接传入 NPUStream 对象

**适用**：单文件 `.asc` 架构，使用 `<<<>>>` 启动符。

```cpp
at::Tensor ascendc_add(const at::Tensor &x, const at::Tensor &y)
{
    auto npu_stream = c10_npu::getCurrentNPUStream();
    at::Tensor z = at::empty_like(x);
    uint32_t blockDim = 8;
    uint32_t totalLength = x.numel();
    auto xGm = (uint8_t *)(x.mutable_data_ptr());
    auto yGm = (uint8_t *)(y.mutable_data_ptr());
    auto zGm = (uint8_t *)(z.mutable_data_ptr());
    add_custom<<<blockDim, nullptr, npu_stream>>>(xGm, yGm, zGm, totalLength);
    return z;
}
```

**原理**：`<<<>>>` 接收 NPUStream 对象时，内部在内核启动前自动清 queue。

---

## 方式2（正确）：`stream(true)` 清 queue

**适用**：单文件 `.asc` 和多文件 CMake 架构。**多文件 CMake 架构的唯一推荐方式**。

```cpp
at::Tensor ascendc_add2(const at::Tensor &x, const at::Tensor &y)
{
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);
    at::Tensor z = at::empty_like(x);
    uint32_t blockDim = 8;
    uint32_t totalLength = x.numel();
    auto xGm = (uint8_t *)(x.mutable_data_ptr());
    auto yGm = (uint8_t *)(y.mutable_data_ptr());
    auto zGm = (uint8_t *)(z.mutable_data_ptr());
    add_custom<<<blockDim, nullptr, acl_stream>>>(xGm, yGm, zGm, totalLength);
    return z;
}
```

**多文件 CMake 架构等价写法**（`xxx_torch.cpp` 中无法使用 `<<<>>>`）：

```cpp
auto aclStream = c10_npu::getCurrentNPUStream().stream(true);
xxx_kernel(blockNum, nullptr, aclStream,
    reinterpret_cast<uint8_t*>(x1.mutable_data_ptr()),
    reinterpret_cast<uint8_t*>(x2.mutable_data_ptr()),
    reinterpret_cast<uint8_t*>(y.mutable_data_ptr()),
    reinterpret_cast<uint8_t*>(tilingTensor.mutable_data_ptr()));
```

**原理**：`stream(true)` 在返回 aclrtStream 前会清 queue（等待之前任务完成），与方式1等价。

---

## 方式3（推荐）：`stream(false)` + OpCommand 入 queue

**适用**：单文件 `.asc` 架构，使用 `<<<>>>` 启动符 + OpCommand 管理。

```cpp
#include "torch_npu/csrc/framework/OpCommand.h"

at::Tensor ascendc_add3(const at::Tensor &x, const at::Tensor &y)
{
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    at::Tensor z = at::empty_like(x);
    uint32_t blockDim = 8;
    uint32_t totalLength = x.numel();
    auto xGm = (uint8_t *)(x.mutable_data_ptr());
    auto yGm = (uint8_t *)(y.mutable_data_ptr());
    auto zGm = (uint8_t *)(z.mutable_data_ptr());

    auto acl_call = [=]() -> int {
        add_custom<<<blockDim, nullptr, acl_stream>>>(xGm, yGm, zGm, totalLength);
        return 0;
    };

    at_npu::native::OpCommand::RunOpApiV2("ascendc_add", acl_call);
    return z;
}
```

**原理**：`stream(false)` 返回 aclrtStream 但不清 queue，由 OpCommand 负责入 queue/出 queue，保证执行顺序。

---

## 方式4（反模式）：`stream(false)` 直接启动 — 乱序

**严重程度**：高。结果随机错误，难以复现和定位。

```cpp
// ❌ 反模式：stream(false) 直接启动，不清 queue 也不入 queue
at::Tensor ascendc_add4(const at::Tensor &x, const at::Tensor &y)
{
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);  // 不清 queue
    at::Tensor z = at::empty_like(x);
    uint32_t blockDim = 8;
    uint32_t totalLength = x.numel();
    auto xGm = (uint8_t *)(x.mutable_data_ptr());
    auto yGm = (uint8_t *)(y.mutable_data_ptr());
    auto zGm = (uint8_t *)(z.mutable_data_ptr());
    add_custom<<<blockDim, nullptr, acl_stream>>>(xGm, yGm, zGm, totalLength);
    return z;
}
```

**问题**：`stream(false)` 不清 queue，kernel 启动后可能先于之前提交的 NPU 操作执行。例如在 `x *= 2` 之后调用自定义算子，`add_custom` 可能先于乘法执行。

**多文件 CMake 架构同样错误**：
```cpp
// ❌ 同样错误：stream(false) + 函数调用
auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
xxx_kernel(blockNum, nullptr, aclStream, ...);  // 乱序风险
```

**修复**：改用 `stream(true)` 或配合 OpCommand。

---

## 方式5（反模式）：NPUStream + OpCommand — 死锁

**严重程度**：高。程序挂死，必须重启。

```cpp
// ❌ 反模式：lambda 内传入 NPUStream 对象 + OpCommand = 死锁
at::Tensor ascendc_add5(const at::Tensor &x, const at::Tensor &y)
{
    auto npu_stream = c10_npu::getCurrentNPUStream();  // NPUStream 对象
    at::Tensor z = at::empty_like(x);
    uint32_t blockDim = 8;
    uint32_t totalLength = x.numel();
    auto xGm = (uint8_t *)(x.mutable_data_ptr());
    auto yGm = (uint8_t *)(y.mutable_data_ptr());
    auto zGm = (uint8_t *)(z.mutable_data_ptr());

    auto acl_call = [=]() -> int {
        // lambda 内 <<<>>> 传入 npu_stream，内部会清 queue
        // 但 lambda 本身在 queue 中等待执行 → 循环等待 → 死锁
        add_custom<<<blockDim, nullptr, npu_stream>>>(xGm, yGm, zGm, totalLength);
        return 0;
    };

    at_npu::native::OpCommand::RunOpApiV2("ascendc_add", acl_call);
    return z;
}
```

**死锁流程**：
1. `OpCommand::RunOpApiV2` 将 lambda 入 queue
2. lambda 执行时，`<<<..., npu_stream>>>` 要求先清 queue
3. 但 queue 中有 OpCommand 放入的任务，等待清 queue 才能完成
4. 循环等待 → 死锁

**修复**：lambda 内改用 `aclrtStream`（通过 `stream(true)` 或 `stream(false)` 获取），不要传 NPUStream 对象。

---

## 方式6（正确）：`stream(true)` + OpCommand 入 queue

**适用**：单文件 `.asc` 架构，需要 OpCommand 管理的场景。

```cpp
at::Tensor ascendc_add6(const at::Tensor &x, const at::Tensor &y)
{
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);  // 先清 queue
    at::Tensor z = at::empty_like(x);
    uint32_t blockDim = 8;
    uint32_t totalLength = x.numel();
    auto xGm = (uint8_t *)(x.mutable_data_ptr());
    auto yGm = (uint8_t *)(y.mutable_data_ptr());
    auto zGm = (uint8_t *)(z.mutable_data_ptr());

    auto acl_call = [=]() -> int {
        add_custom<<<blockDim, nullptr, acl_stream>>>(xGm, yGm, zGm, totalLength);
        return 0;
    };

    at_npu::native::OpCommand::RunOpApiV2("ascendc_add", acl_call);
    return z;
}
```

**原理**：`stream(true)` 先清 queue 确保之前的任务完成，lambda 内用 `aclrtStream`（非 NPUStream 对象），不会触发二次清 queue。

---

## 方式7（反模式）：`stream(true)` + `zeros_like` — 乱序

**严重程度**：中。`zeros_like` 触发 NPU 操作入 queue，但 kernel 不入 queue，可能乱序。

```cpp
// ❌ 反模式：stream(true) 清 queue 后，zeros_like 入 queue 但 kernel 不入 queue
at::Tensor ascendc_add7(const at::Tensor &x, const at::Tensor &y)
{
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);
    at::Tensor z = at::zeros_like(x);  // zeros_like 触发 NPU 初始化操作，入 queue
    uint32_t blockDim = 8;
    uint32_t totalLength = x.numel();
    auto xGm = (uint8_t *)(x.mutable_data_ptr());
    auto yGm = (uint8_t *)(y.mutable_data_ptr());
    auto zGm = (uint8_t *)(z.mutable_data_ptr());
    add_custom<<<blockDim, nullptr, acl_stream>>>(xGm, yGm, zGm, totalLength);  // 不入 queue
    return z;
}
```

**问题**：`stream(true)` 清 queue 后，`zeros_like` 是一个 NPU 操作会入 queue，但 `add_custom` 不通过 OpCommand 入 queue。两者可能乱序：`add_custom` 在 `zeros_like` 完成前就开始执行，覆盖未初始化的数据。

**修复**：使用 `empty_like` 代替 `zeros_like`（`empty_like` 不触发 NPU 操作）。

```cpp
// ✅ 修复：empty_like 不触发 NPU 操作，无乱序风险
at::Tensor z = at::empty_like(x);
```

---

## 方式8（正确）：`stream()` 无参数

**适用**：与方式2等价，`stream()` 无参数时在返回 aclrtStream 前等待 queue 内操作完成。

```cpp
at::Tensor ascendc_add8(const at::Tensor &x, const at::Tensor &y)
{
    auto acl_stream = c10_npu::getCurrentNPUStream().stream();  // 无参数，等价 stream(true)
    at::Tensor z = at::empty_like(x);
    uint32_t blockDim = 8;
    uint32_t totalLength = x.numel();
    auto xGm = (uint8_t *)(x.mutable_data_ptr());
    auto yGm = (uint8_t *)(y.mutable_data_ptr());
    auto zGm = (uint8_t *)(z.mutable_data_ptr());
    add_custom<<<blockDim, nullptr, acl_stream>>>(xGm, yGm, zGm, totalLength);
    return z;
}
```

---

## 速查表

| # | 方式 | 适用架构 | 判定 | 关键代码 |
|---|------|---------|------|---------|
| 1 | 传 NPUStream 对象 | 单文件 .asc | **推荐** | `npu_stream = getCurrentNPUStream()` → `<<<..., npu_stream>>>` |
| 2 | `stream(true)` 清 queue | 两种均可 | **正确** | `stream(true)` → `<<<..., acl_stream>>>` 或函数调用 |
| 3 | `stream(false)` + OpCommand | 单文件 .asc | **推荐** | `stream(false)` → lambda + `OpCommand::RunOpApiV2` |
| 4 | `stream(false)` 直接启动 | — | **反模式** | 乱序风险 |
| 5 | NPUStream + OpCommand | — | **反模式** | 死锁 |
| 6 | `stream(true)` + OpCommand | 单文件 .asc | **正确** | 先清 queue 再入 queue |
| 7 | `stream(true)` + `zeros_like` | — | **反模式** | 乱序（用 `empty_like` 替代） |
| 8 | `stream()` 无参数 | 两种均可 | **正确** | 等价方式2 |

**多文件 CMake 架构**（本 SKILL 的目标架构）：**只能用方式2**，即 `stream(true)` + 函数调用。
