# AttnResFwd

## 功能说明

`AttnResFwd` 是 Kimi K3 的注意力残差融合算子。单个 Torch 入口
`torch.ops._C_ascend.attn_res_fwd` 将可选的 BF16 加法、残差 bank 混合、
可选的输出 RMSNorm，以及 bank 写回放在一次调用中。源码注册覆盖
A2、A3 和 A5，小 batch 协作路径同时接入三类平台。已在 Ascend910B3 上完成
CANN 9.3.0 编译和小 shape 精度、性能验证。A5 使用独立 RegBase 协作实现，
已完成 ascend950 交叉编译；A3、A5 及大 H 分段路径仍需对应硬件验证。

令 `p` 为 `prefix_sum`（有 `addend` 时先做 BF16 加法），`v_i` 为
`block_residual` 中前 `num_valid_blocks` 行，并令最后一行 `v_N=p`。
启用 `mix` 时，每行先按 `norm_eps` 做 RMSNorm，再与 `norm_weight`、
`proj_weight` 相乘并沿隐藏维求和，所得分数经 softmax 后对原始 `v_i`
加权求和并舍入到 BF16。`mix=False` 时直接使用 `p`。传入
`output_norm_weight` 时，对混合结果再做一次 RMSNorm。

## 接口与参数

```python
output, raw_prefix, materialized = torch.ops._C_ascend.attn_res_fwd(
    prefix_sum, addend, block_residual, proj_weight, norm_weight,
    norm_eps, num_valid_blocks, output_norm_weight=None,
    output_norm_eps=1e-5, block_write_idx=-1,
    return_materialized=False, mix=True, optimize_prefill=False,
)
```

| 参数 | 形状与约束 | 作用 |
| --- | --- | --- |
| `prefix_sum` | `[T,H]`，BF16，连续 | 当前层的残差前缀。 |
| `addend` | `None` 或 `[T,H]`，BF16，连续 | 可选的前层输出；加法结果以 BF16 供后续计算。 |
| `block_residual` | `[T,S,H]`，BF16；每行连续，token 间可有 stride | 残差 bank；仅前 `num_valid_blocks` 行参与混合。 |
| `proj_weight`、`norm_weight` | 分别为 `[1,H]`、`[H]`，BF16，连续 | 混合分数的投影和 RMSNorm 权重。 |
| `norm_eps`、`num_valid_blocks` | `norm_eps` 为 FP32 可表示的有限正数；`0 ≤ num_valid_blocks ≤ S` | 混合归一化参数和有效 bank 行数。 |
| `output_norm_weight`、`output_norm_eps` | `None` 或 `[H]`，BF16，连续；传入权重时 epsilon 为 FP32 可表示的有限正数 | 可选的输出 RMSNorm。 |
| `block_write_idx` | `-1` 或 `[num_valid_blocks,S)` 中的行号 | 将 `raw_prefix` 写入尚未参与本次混合的 bank 行；`-1` 表示不写回。 |
| `return_materialized` | `bool` | 为真时单独保留归一化前的混合结果。 |
| `mix` | `bool` | 为假时跳过 bank 混合，用于前层或只做加法的路径。 |
| `optimize_prefill` | `bool` | 请求 prefill 策略；满足输出归一化、每核多 token 和 UB 容量条件时复用缓存权重。 |

所有 Tensor 必须位于同一设备。`output`、`raw_prefix` 和
`materialized` 均为 `[T,H]`、BF16。无 `addend` 时 `raw_prefix`
复用输入 `prefix_sum`；`return_materialized=False` 时
`materialized` 复用 `output`。写回只修改指定的 bank 行，不修改
`prefix_sum` 或 `addend`。`T=0` 可走空输出路径。
启用 bank 写回时，`block_residual` 不得与其他只读输入共享存储；Torch 入口会在
提交 kernel 前拒绝这类别名。关闭写回时仍允许只读输入共享 bank 存储。
底层 ACLNN 调用方同样需要保证写回 bank 与其他输入不重叠。

## 实现与验证

A2/A3 小 batch 使用 `AttnResFwdCooperative::ProcessRowParallel()`，
A5 使用 `AttnResFwdCooperativeRegBase::ProcessRowParallel()`，分核逻辑相同。设 token 数为 T、
有效历史行数为 N、候选行数 R=N+1、可用 AIV 核数为 C：

1. 打分任务按 `(token, candidate)` 分配，任务数为 T*R，启动
   `min(C,T*R)` 个核；任务多于核数时按核号循环领取。每个任务读取完整 H，
   计算一行的 RMSNorm 和投影点积，向独立的 32B workspace 槽写一个分数。
2. 所有核执行一次 `SyncAll<true>()`，保证各候选分数和融合 prefix 已写完。
3. 每个 token 的 prefix 打分核兼任输出核，读取该 token 的 R 个分数，
   做 softmax，遍历全部候选行并生成完整 H 的加权输出。
4. 混合结果先舍入 BF16。启用输出 RMSNorm 时，由该输出核对完整 H 计算
   平方和、invRms 和 gamma 乘法，然后写出结果，不需要再次核间同步。

完整行快路径不切 H，也不使用第二个 kernel。一次就绪同步仍是全核同步；没有改成跨核
轮询或异步通知。每核只处理一个打分任务时，输出核复用本核 UB 中的原始
BF16 prefix；循环领取多个任务时从 GM 读取，避免复用已被覆盖的数据。
打分任务优先处理 prefix，但写入分数仍保持“历史行在前、prefix 在后”的顺序。

完整 H 输出使用两份 BF16 输入 buffer，计算当前残差行时预取下一行；队列深度仍为 1。
Host 将新增输入 buffer 计入 UB 预算，H 分段路径仍使用单缓冲。
对齐后的 FP32 行大小为 512B 整数倍时，在 `rowBuf_` 末尾预留 256B，
错开后续 buffer 地址以减少 UB bank/group 冲突；padding 不参与计算。
输出循环外一次设置 Counter mask，Cast 和累加使用已有的无重复 mask 设置接口，
保留必要的 Vector 依赖屏障，退出循环后恢复 mask。
BF16 Copy 使用多 repeat 批量调用，归约只初始化未对齐的尾块。

在 40 个 AIV、N=8、H=7168 下，T=1 启动 9 核，T=4 启动 36 核；
打分结束后分别由 1 核、4 核完成输出。用户 workspace 分别为 288B、1152B，
仅保存分数，不含系统空间。Host 日志显示 `rowParallel=1`、
`scoreSplits=1`、`outputSplits=1`、`residentGroup=0`。
这项策略以减少同步为目的，是否降低延迟以目标硬件实测为准。

协作路径要求 T 不超过 C 的二分之一、H 至少为 512、R 在 2 到 128 之间。
只有完整 H 的队列、FP32 工作区、元数据和安全余量能放入 UB，才选择上述路径。
不满足协作选择条件时使用原有按 token 分核的 RESIDENT/RELOAD。
完整行缓冲超过 UB 时保留 H 分段回退：先汇总各段平方和得到整行 invRms，
再汇总各段分数做 softmax，输出按 H 分段；输出 RMSNorm 需要再汇总平方和。
分段任务超过核数时按核号步进循环处理。旧 UB 驻留核组实现已删除。
H=7168 不是固定分界，分段数由 UB 预算选择；超过所有受支持分段预算时明确报错。

Host 在 A2/A3/A5 上选择协作 tiling key。A5 的打分沿用原有 RegBase RMSNorm
及投影归约，输出使用寄存器内 BF16 转换和独立乘加；softmax 对有效候选行
执行 masked max/sum 归约，覆盖不足一个寄存器和跨两个寄存器的候选数量。
A5 的 metadata 按 64 个 FP32 元素补齐，Host 计入对应 UB 开销；A2/A3 的
256B padding 和外置 Counter mask 调优不直接套用到 A5。
其他场景保留原来的按 token 分核路径，Host 根据 H、有效行数和 UB 容量选择
RESIDENT 或 RELOAD；prefill 缓存策略和公开 Torch 入口不变。
协作 kernel 只使用 AIV，以 `MIX_AIV_1_0` 类型和 batch 调度模式启用硬件同步。
所有启动核都参加同样的同步，不依赖输出核数量。

A2/A3 的原 RESIDENT/RELOAD 路径保留按 token 分核，并修正了标量 Brcb
临时区不足 256B、非对齐 H 归约尾部未初始化的问题；较多候选行的广播分批执行，
避免 repeat 字段溢出，softmax 填充值也覆盖极大负分数。Host 也禁止 bank 写回
槽覆盖有效历史行，与 Torch 入口的约束一致。
Host 还会在整数收窄前检查 shape、候选行数、stride 和写槽范围，拒绝会导致
`N+1`、对齐或 buffer 字节长度溢出的元数据；平台信息缺失和属性指针缺失时返回失败。
投影必须严格为 `[1,H]`，epsilon 必须为 FP32 可表示的有限值。
Torch、ACLNN 和 Tiling 共用 bank 尺寸/stride 范围校验；零 token 大 view
也必须在尺寸乘法前通过检查，空输出快速返回前同样检查所有 tensor 的 BF16 dtype。
原生 ACLNN 入口在空输出快速返回前同样校验 bank 属性，拒绝非法有效行数和写槽。

单卡精度用例位于
`tests/e2e/nightly/single_node/ops/singlecard_ops/test_attn_res_fwd.py`，
使用独立 CPU 参考值覆盖 Kimi K3 的 H=7168、decode/prefill、
可选加法、输出归一化和写回。该用例不代表整网 TP/EP/DSpark 吞吐验收。
协作路径的补充用例位于同目录的 `test_attn_res_fwd_cooperative.py`，
覆盖小 batch、非对齐 H、带 token stride 的 bank、独立输出归一化 epsilon
以及输出别名，并补充大 batch 原路径回归用例。无需 NPU 的检查位于
`tests/ut/ops/test_attn_res_fwd_parallel_plan.py`，覆盖 Host 任务规划，以及通过
CPU API 替身直接运行生产 kernel 的分核、地址和数值逻辑。替身将搬运和流水
视为同步操作，不模拟真实硬件归约指令的加法树，也不代替 CANN 编译和上板验证。

A2/A3 版本在 Ascend910B3、CANN 9.3.0 上完成编译和安装；重新编译生产 Torch
入口后，43 个 NPU 用例通过，4 个大 H 用例按测试范围排除。
A5 接入后的 CPU 回归为 5 项测试通过，分别覆盖 25 个 A2/A3 协作场景和
25 个 A5 控制流场景（使用数学替身，不模拟寄存器指令），并保留原路径回归。
A5 在 Ascend950PR_9579（arch3510）完成原生编译、安装及实机验证：
CANN 9.3.0 weekly.20261014.01、torch/torch_npu 2.10.0，46 个 NPU 用例通过，
4 个大 H 用例按测试范围排除。原有 A5 路径补充超过 128 行的通用 softmax，
并修正广播 repeat 溢出及极大负分数填充值，覆盖 2040 行广播边界。
A3 和大 H 的 NPU 验证仍需补充。

### Ascend910B3 性能对照

同一卡、相同 BF16 输入，两版均带 `-g`；Add 和输出 RMSNorm 开启。
原始计算代码使用提交 `86add7adfd1194c225fd8880699d5be7a248c3ce`，
仅增加一行 ascend910b 平台注册。四轮交换顺序，每组预热 30 次、采集 100 次，
每种 shape/版本共 400 次 kernel 采样，1600 次测量输出逐次参考校验通过。

| Shape `[T,N,H]` | 原始版均值（μs） | 优化版均值（μs） | 延迟降低 |
| --- | --- | --- | --- |
| `[1,8,7168]` | 18.495028 | 9.055510 | 51.04% |
| `[4,8,7168]` | 18.965053 | 13.692963 | 27.80% |

本表仅表示目标 kernel 耗时，不代表整网吞吐。性能采集后的检视修复集中于 Host
元数据校验和 Torch 别名检查，A2/A3 计算 kernel 源码保持不变。
小 shape NPU 精度覆盖原有路径、非对齐 H 和输出别名；大 H 仅有 CPU 协作控制流回归，
不能据此认定 H 分段的异步硬件行为已通过验证。

### Ascend950 性能对照

在 Ascend950PR_9579 的同一卡上，使用相同 BF16 输入、上述 CANN weekly 环境，
Add 和输出 RMSNorm 开启，三版均带 `-g`。最原始基线为提交
`86add7adfd1194c225fd8880699d5be7a248c3ce`，仅增加编译调试选项，
不需要修改平台注册。本轮前基线为 `30aaaa296`，已包含行并行协作优化。
四轮交换顺序，每组预热 30 次、采集 100 次，每种 shape/版本共 400 次采样；
2400 次测量输出逐次 CPU 参考校验及 prefix 精确比较通过。
原生 `kernel_details.csv` 随结果归档，调优阶段样本未混入下表。

| Shape `[T,N,H]` | 最原始版（μs） | 本轮前（μs） | 最终版（μs） | 本轮降低 | 相对最原始降低 |
| --- | --- | --- | --- | --- | --- |
| `[1,8,7168]` | 12.846988 | 7.839843 | 7.548183 | 3.72% | 41.25% |
| `[4,8,7168]` | 13.189648 | 9.103173 | 8.798895 | 3.34% | 33.29% |

本轮保留输出 RMSNorm 的 RegBase 融合：混合结果先舍入 BF16，再把转换、
平方和及 invRms 合并执行，减少 FP32 行重读及中间标量往返。
B=1 四轮本轮收益为 1.96%～6.07%，B=4 为 2.16%～4.63%。
另一个寄存器内 proj 与 norm 权重融合候选两种 shape 均约慢 0.18%，已回退。
原始版分别使用 1、4 核，最终版打分使用 9、36 核，输出分别使用 1、4 核。
分核及一次就绪同步保持不变；H 分段路径和 A2/A3 kernel 不受本轮融合影响。
最终版 NPU 精度 46 项通过、4 个大 H 用例排除，CPU 回归 5 项通过。
以上只比较目标 kernel 延迟，不代表整网收益，也不构成大 H 分段路径硬件验收。

### B=16 协作选择

Host 将协作阈值从 `T <= C/4` 放宽为 `T <= C/2`，继续检查候选行数、
H 和 UB 容量，选用已有的协作 tiling key；不为 B=16 新增专用 key。
对于 N=8、H=7168，40 AIV 的 A2 启动 40 个打分核，56 AIV 的 A5 启动 56 个。
144 个打分任务按核号循环领取；一次就绪同步后，由 16 个输出核完成完整 H。
超过新阈值的大 batch 仍使用原有按 token 分核路径。

A2 四轮对照：原始版 22.103840 μs，旧阈值版 21.384055 μs，
新阈值协作版 19.112613 μs；相对旧阈值降低 10.62%。
旧阈值版及新阈值版分别通过 46 个 NPU 用例，4 个大 H 用例排除。
A5 新阈值包完成编译、安装及实机验证，46 个 NPU 用例通过、4 个大 H 用例排除。
B=16 四轮对照：原始版 13.740678 μs，旧阈值版 13.739058 μs，
新阈值版 11.958008 μs；相对旧阈值降低 12.96%，相对原始版降低 12.97%。
Host 日志与原始 CSV 确认新阈值使用 56 核及 `rowParallel=1`，旧阈值使用 16 核。
此前 A5 B=16 约 13.758 μs 为旧阈值数据，不能当作新阈值性能结果。
