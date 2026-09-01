# Ascend C Kernel 直调进阶

> 基础用法和修改指南见 `add_custom/op_host/add_custom.asc` 中的注释。本文档提供深入理解。

## 内存层次与数据流

```
┌─────────────────────────────────────┐
│        Global Memory (GM)           │  大容量，高延迟
│        (HBM/DDR)                    │
├─────────────────────────────────────┤
│        Local Memory (UB)            │  小容量，低延迟
│        (Unified Buffer)             │  用于计算
├─────────────────────────────────────┤
│        Registers                    │  最快，最少
└─────────────────────────────────────┘
```

数据流：Global Memory → Local Memory (CopyIn) → Compute → Local Memory → Global Memory (CopyOut)

## 性能优化

### Double Buffer

通过重叠内存搬运和计算提高性能：

```cpp
// 使用两个 Event ID 交替执行
AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

for (uint32_t i = 0; i < loopCount; i++) {
    int32_t eventID = (i % 2 == 0 ? EVENT_ID0 : EVENT_ID1);
    
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventID);
    // CopyIn + Compute + CopyOut
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventID);
}
```

```
时间轴:  |------|------|------|------|------|
Buffer A: [MTE2] [COMP] [MTE3] [MTE2] [COMP] ...
Buffer B:        [MTE2] [COMP] [MTE3] [MTE2] ...
```

### 同步机制

| 同步方式 | 适用场景 |
|----------|----------|
| `PipeBarrier<PIPE_*>` | 简单场景，自动等待流水线 |
| `SetFlag/WaitFlag` | Double Buffer，精确控制 |

### 内存分配策略

| 方式 | 特点 | 适用场景 |
|------|------|----------|
| `LocalMemAllocator` | 简单易用，线性分配 | Bank 冲突不敏感 |
| `TQue` 手动管理 | 精确控制 Bank | 性能敏感场景 |

```cpp
// 手动分配示例（高级）
AscendC::TPipe pipe;
AscendC::TQue<AscendC::TPosition::VECIN, 1> queue;  // 模板 depth=1 即可
pipe.InitBuffer(queue, 2, TILE_LENGTH * sizeof(float));  // num=2 开启 Double Buffer
auto xLocal = queue.AllocTensor<float>();
```

## 常见问题

### 如何选择 TILE_LENGTH？

考虑 Local Memory 大小和数据对齐要求：
- float32: 4096 或 8192
- float16: 8192 或 16384
- Double Buffer 需要额外空间

### 如何支持多输入多输出？

修改核函数签名和 KernelCall：

```cpp
// 核函数
__global__ __aicore__ void kernel_custom(
    GM_ADDR input0, GM_ADDR input1,
    GM_ADDR output0, GM_ADDR output1,
    GM_ADDR tiling
);

// main 函数中
std::vector<ArgInfo> inputsInfo = {
    {"./input/input0.bin", size0},
    {"./input/input1.bin", size1}
};
std::vector<ArgInfo> outputsInfo = {
    {"./output/output0.bin", size0},
    {"./output/output1.bin", size1}
};
```

### 如何处理不对齐的数据？

在 Tiling 中添加边界参数：

```cpp
struct TilingData {
    uint32_t totalLength;      // 总长度
    uint32_t alignedLength;    // 对齐长度
    uint32_t lastTileLength;   // 最后一个 tile 的长度
};
```

## 参考资源

- [Ascend C 编程指南](https://www.hiascend.com/document)
- [Ascend C 示例代码](https://gitcode.com/cann/asc-devkit/tree/master/examples)
