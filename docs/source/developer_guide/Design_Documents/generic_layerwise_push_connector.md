# Layerwise Push Connector

## 范围

`LayerwisePushConnector` 仅保留 P 主动写入的控制流程，不提供 pull 模式。

普通 HBM 目标支持 MemFabric 和 Mooncake。MemFabric 默认使用异步 WRITE；
Mooncake 仅支持同步 WRITE，需要显式配置 `push_write_mode: "sync"`。
Sparse decode offload 仅支持 MemFabric：D 侧将共享 CPU KV 池的 GVA 和 block 区段
注册到本 rank 的传输引擎，P 侧即可直接写入。Mooncake 无法使用该内存池，仍在
缓存注册前明确拒绝。Sparse decode offload 继续要求 Decode PP=1。

## 配置

在 P/D 两端均替换 connector 名称，保留各自原来的 `kv_role`、端口和拓扑配置：

```json
{
  "kv_connector": "LayerwisePushConnector",
  "kv_connector_extra_config": {
    "transfer_backend": "memfabric"
  }
}
```

使用 Mooncake 时配置为：

```json
{
  "kv_connector": "LayerwisePushConnector",
  "kv_connector_extra_config": {
    "transfer_backend": "mooncake",
    "push_write_mode": "sync"
  }
}
```

## 数据流

1. D 注册本地 HBM；P/D 交换 layout 和 PP/TP 归属，D 返回传输 session。
2. D scheduler 的目标 block 信息到达 D worker 后，通过控制通道返回 P。
   目标信息按请求缓存，不逐层重复请求。某个请求尚未就绪，不阻塞其他连接。
3. P 的计算线程记录事件并入队。控制线程等待源事件和目标信息就绪，再派发 WRITE。
4. P 汇集同一目标、同一层的请求 block，使用 NumPy 批量计算地址并合并连续区间。
   固定组件布局匹配结果按来源和层缓存。
5. WRITE 成功后，P 释放本次传输对源 slot 的占用，不发送逐层控制消息。
6. 每个 P 来源向对应 D endpoint 写完请求的最终 chunk、所有相关层后，发送一次
   `REQUEST_DONE`；D 汇总所有预期 PP/TP 来源的终态后进入 decode，不回复 ACK。
   P 请求 block 的释放只等待所有目标的本地 WRITE 成功、完成通知发出及 scheduler
   结束请求，不等待 D 确认。目标 block 信息也在本地终态处理时清理。

## 有限并发

一个 P worker 共用一个 transfer engine。MemFabric 异步模式使用一条专用 NPU stream，
通过同 stream 上记录的 event 判断源 buffer 何时可复用；Mooncake 同步模式使用两个固定
WRITE 线程，每个线程初始化自己的 NPU device context。同一 D endpoint 最多执行一个
WRITE batch，不同 endpoint 可以并发。控制线程独占 ZeroMQ socket，不在线程间共享
socket 操作。

提交队列有界，达到上限时施加背压；源 slot 从入队前就受到保护。
计算复用 slot 时等待所有相关写入完成，不能仅等待最快的 D。
同步后台完成通过 socket 唤醒控制线程；NPU 源事件或异步 WRITE event 未就绪时使用
短周期查询。layout 握手有固定超时，但不会阻塞其他 endpoint 的控制消息。
源 slot 的释放不依赖 D 的消息处理速度；异常仍通过 `WRITE_FAILED` 通知 D 停止。

后端内部 stream/队列、冷连接和实际链路仍可能限制并发收益；两个同步 WRITE 线程
也不保证底层带宽翻倍。

## 正确性边界

- 沿用 chunk 增量范围、prefix 命中跳过和各 KV group 的映射。
- 普通 HBM 路径沿用不等 TP 的复制组件分工及不等 PP 的全局层路由。
- 同一 endpoint 的任务和完成通知保序；最后一个来源完成前 D 不进入 decode。
- D 取消请求时仍保留目标 block，直到全部预期来源结束写入。
- P 已派发部分 chunk 后提前取消，目前会明确报错停止，而不是释放仍可能在用的源 block。
  实验版尚未实现部分 prefill 的取消排空协议；压力对比应避免 P 侧抢占/提前取消。
- WRITE 失败或超时会停止 worker，不作为可安全回收 block 的普通请求失败。
  超时不证明 DMA 已停止；实验版不做自动重试或静默本地重算。
- 多机、MTP 和具体后端组合仍需在目标 NPU 环境验证，CPU mock 测试不等于实机支持认证。

## 验证

单元测试保留 scheduler、批量地址规划、TP/PP、prefix/chunk 与取消生命周期覆盖，
并新增真实 ZeroMQ 控制通道测试：慢 D 不阻塞另一个 D、同 D 保序、目标延迟发布、
源 slot 完成门、WRITE 失败、以及 PP/chunk/层复用组合。

这些测试使用模拟 WRITE 和 NPU 事件。实机验证已完成：A3 Mini 跨机 P TP8 → D TP4
的普通/sparse、同步/异步组合均通过；61 层全模型 P TP16/PP1 → D TP8/PP2 的普通
HBM sync/async，以及 P TP16/PP1 → D TP8/PP1 + experts prefetch 的 sparse sync
均通过。首请求、短序列串行/并发和 17175-token chunked prefill 均返回 HTTP 200，
P/D 未出现 `Traceback`、`WRITE_FAILED`、`EngineDead` 或 `TransformAddr`。
