# DumpTensor 视角下的错误模式

> 本文聚焦「在 dump 输出里看到什么样的异常 → 该往哪个方向查」。父 SKILL 的「症状-原因速查表」更全面，本文只列与 DumpTensor 插桩诊断强相关的模式。

## DumpTensor 专属诊断模式

| Dump 现象                                       | 最可能的根因                          | 下一步                                                         |
|-------------------------------------------------|---------------------------------------|----------------------------------------------------------------|
| desc=100（输入）就异常                          | DataCopy 未完成 / 未 DeQue 就 dump    | 把 DumpTensor 移到 DeQue 之后；或临时 PipeBarrier 验证        |
| desc=100 正确，desc=200（中间）异常             | Compute 阶段问题                      | 在 Compute 内细分插桩，二分定位是哪个 API/参数                |
| desc=200 正确，desc=300（输出）全 0 / 全旧值    | CopyOut 没生效 / 队列写错 (VECIN)     | 检查输出队列类型、EnQue/DeQue、DataCopyPad 对齐               |
| 单核 dump 正确，多核合并后乱                    | desc 未含 blockIdx，多核日志交错      | `desc = base + GetBlockIdx() * 1000`                          |
| Dump 显示输入与 CPU golden 不一致               | host 侧数据准备 / DataCopy stride 错  | 比对 host buffer，检查 DataCopy 参数                          |
| 每隔 N 个值一处异常                             | stride / 对齐问题                     | 检查 DataCopy stride、是否需要 DataCopyPad                    |
| dump 出现 NaN / Inf                             | 除零、exp 溢出、未初始化              | 在出现 NaN 的最早 desc 上游加 dump 缩小范围                   |
| 改代码后 dump 完全没变                          | 二进制未更新 / kernel cache 命中      | `rm -rf build/ $HOME/atc_data/kernel_cache/` 重编             |

## 通用错误模式速查

仅当 dump 不足以定位、需要回到通用思路时使用：

| Pattern                    | Root Cause                    | Fix                                            |
|----------------------------|-------------------------------|------------------------------------------------|
| All values off by constant | Missing bias/scale            | Check Adds/Muls operations                     |
| Every Nth value wrong      | Stride/alignment issue        | Verify DataCopy stride, check vector alignment |
| NaN or Inf values          | Division by zero/overflow     | Check denominators, verify input ranges        |
| First/last values wrong    | Boundary/padding issue        | Check tile alignment, edge case handling       |
| Errors accumulate          | Uninitialized vars/queue sync | Check init, verify EnQue/DeQue                 |
| Random sporadic errors     | Race condition/queue depth    | Increase BUFFER_NUM, check sync                |
| Output all zeros           | Missing compute/wrong queue   | Verify Compute called, check queue             |
| Output matches input       | Computation not applied       | Verify operation executed                      |

更完整的精度陷阱见父 SKILL 的 [common-traps.md](../common-traps.md)。

## 诊断工作流

1. 从 desc 最小值（输入侧）开始看，找到**第一个**与 CPU golden 不一致的阶段
2. 对照上方「DumpTensor 专属诊断模式」表匹配根因
3. 在该阶段上游再插一层 dump，验证假设
4. 修复后必须清编译缓存重跑，确认 dump 数值变化
