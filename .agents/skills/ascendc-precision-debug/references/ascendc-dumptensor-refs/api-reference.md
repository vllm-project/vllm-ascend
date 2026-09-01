# DumpTensor API Reference

## API Signature

```cpp
DumpTensor(const LocalTensor<T> &tensor, uint32_t desc, uint32_t dumpSize)
```

## Parameters

| Parameter  | Type              | Description                                  |
|------------|-------------------|----------------------------------------------|
| `tensor`   | `LocalTensor<T>&` | The tensor to dump                           |
| `desc`     | `uint32_t`        | Unique identifier (use systematic numbering) |
| `dumpSize` | `uint32_t`        | Number of elements to dump                   |

## Desc Numbering Convention

| Range   | Stage                |
|---------|----------------------|
| 100-199 | Input tensors        |
| 200-299 | Intermediate results |
| 300-399 | Output tensors       |

Increment by 10 within each range.

## Best Practices

```cpp
// Control dump size - dump subset for large tensors
uint32_t dumpSize = std::min(tileLength, 32u);
DumpTensor(outputLocal, 300, dumpSize);

// Avoid dumping entire large tensors
DumpTensor(outputLocal, 300, 8192);  // ❌ Too much
DumpTensor(outputLocal, 300, 32);    // ✅ Better
```

## Important Notes

- DumpTensor outputs to system log
- Remove dumps in production code (performance overhead)
- Start with 32-64 elements, increase only if needed

---

## 调用约束

| 约束             | 说明                                                                       |
|------------------|----------------------------------------------------------------------------|
| 调用上下文       | 仅在 kernel 函数内对 `LocalTensor` 调用；`GlobalTensor` 不能直接 dump      |
| 同步要求         | 数据依赖搬运/计算完成后才能 dump，详见父文档「使用陷阱 §1」                |
| dumpSize 上限    | 不能超过 tensor 实际元素数，超出会越界                                     |
| dtype 支持       | 常用 `half / float / int32_t / bfloat16_t` 均支持，特殊 dtype 以 CANN 版本为准 |
| 性能影响         | 显著时序开销，可能改变流水线行为，定位完成必须移除                         |

## desc 编号扩展约定

基础三段式（100/200/300）够覆盖单核简单算子。复杂算子建议扩展：

```
desc = base + blockOffset + stageOffset + iterOffset

base       : 100=输入, 200=中间, 300=输出
blockOffset: GetBlockIdx() * 1000     // 多核分离
stageOffset: 0/10/20...               // 同段内多个插桩点
iterOffset : tileIdx                   // 同一 tile 多次迭代区分
```

示例：core 1 上、第 2 个 tile、Compute 内第 2 个插桩点：
```cpp
uint32_t desc = 200 + GetBlockIdx() * 1000 + 20 + tileIdx;
DumpTensor(midLocal, desc, 32);
```

## 与 PRINTF / printf 的关系

| API           | 调用位置          | 用途                                    |
|---------------|-------------------|----------------------------------------|
| `DumpTensor`  | NPU kernel 内     | 批量看 LocalTensor 元素值              |
| `PRINTF`      | NPU kernel 内     | 看 scalar、控制流、tile 形参           |
| `printf`      | Host / CPU 仿真   | 看 host 侧 buffer、CPU golden 输出     |
| `AscendC::Simt::printf` | SIMT VF 内 | SIMT 算子内单核打印（见父 SKILL SIMT 节）|

调试 tensor 数据用 DumpTensor，调试控制流（tileNum、blockIdx、循环计数）配合 PRINTF。

## 常见踩坑

- **dumpSize 写死太大**：日志被淹没且 dump 自身耗时遮蔽 bug → 默认 32，定位收敛后再加
- **多 tile 循环只看到最后一次**：每个 tile 单独编号，否则后面的 dump 覆盖前面观察
- **修改后 dump 完全一致**：先怀疑 kernel cache，`rm -rf build/ $HOME/atc_data/kernel_cache/`
- **NaN 出现位置不确定**：从输入往后逐段加 dump，找出 NaN 第一次出现的 desc
