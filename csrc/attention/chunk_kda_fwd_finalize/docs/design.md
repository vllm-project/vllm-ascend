# ChunkKdaFwdFinalize 设计

## 阶段边界

Prepare 已生成按 HV 展开的 `qg_scaled` 和 `Aqk`，FwdH 已生成
`v_new` 和每个 chunk 的前态 `h`。Finalize 只消费这四项并写出
`attn_out`，不读取原始 `q/k/v/g/beta`，不输出反向检查点或
`h/final_state`。公开约束见[算子 README](../README.md#输入输出)。

## 数学与精度

对每个 batch、value head、序列及其第 `c` 个 chunk，设有效长度为
`M<=64`，物理 `h_c` 按 `state_v_first` 解释为逻辑 `[K,V]`：

```text
P_c = fp32(qg_scaled_c[M,128]) @ fp32(h_c[128,128])
R_c = fp32(Aqk_c[M,M]) @ fp32(v_new_c[M,128])
attn_out_c = bf16(P_c + R_c)
```

两个矩阵乘均以 BF16 输入、FP32 累加，只有最终和落盘时转 BF16。
`qg_scaled` 和 `Aqk` 在 Prepare 边界已经舍入且已应用 scale，
本阶段不能重新缩放，也不能从未舍入的原始 q/k 重算。

`Aqk` 的物理列数为 64，但尾 chunk 只读前 M 列；`v_new` 同样只读
有效 token。不同序列不会共享 chunk，`h` 的 chunk 轴按规范的
sequence-major 顺序映射，不按 tensor 的 T 维简单除以 64。

## 调度与存储

各 chunk 不存在跨 chunk 数据依赖，先按 chunk 分核；仅当 chunk
任务少于可用核时增加完整 value head 的分区。输出写回地址按
`(batch,chunk,head)` 分区，不与其他 AIC 重叠。

A5 dense 的主体按完整 chunk 均分；最后不足一轮核数的 chunk，再按
4 个完整 head 的组均摊。AIC/AIV 使用相同映射和组内消费顺序。例如
128 个 chunk、96 个 head、28 核时，先每核分 4 个完整 chunk，余下
16 个 chunk 的 384 个 head 组按每核 13/14 组分配，避免原来每核
4/5 个 chunk 的负载差。每个 `(chunk,head)` 只由一个核组处理，
不增加计算、workspace 或同步 flag。变长路径保持原有任务划分。

Host 根据平台和每核工作量选择 `USE_AIV_INPUT_MOVER` 编译期模板参数：

- `false`：A2/A3 始终使用，A5 在 Host 已划分 head 分区，或每核完整
  chunk 数不足阈值时使用（dense 为 4、变长为 8）。模板组合指定为
  AIC-only，四个输入由 AIC MTE2 直接从 GM
  搬到 L1。
- `true`：仅 A5 使用，要求 Host 未划分 head 分区，且每核至少有
  4 个 dense chunk 或 8 个变长 chunk。dense 尾部负载均衡由 Kernel
  进一步划分完整 head 组。模板组合指定为 MIX AIC 1:2，两个 AIV 负责 GM 到
  UB 再到 L1 的格式转换，AIC 只消费已经就绪的 L1 操作数。

TilingKey 由 `GET_TPL_TILING_KEY(USE_AIV_INPUT_MOVER)` 生成；Kernel 全局
入口直接以同名模板参数实例化，不做数值 key 的运行时
二次分派。两种模板组合通过 `ASCENDC_TPL_KERNEL_TYPE_SEL` 分别
声明核类型，因此 AIC-only 路径不会启动空闲 AIV。

`USE_AIV_INPUT_MOVER=false` 只有一个 Cube stage：

| Stage | 核 | 本轮计算 | 阶段结果 |
| --- | --- | --- | --- |
| C0 | Cube | MTE2 搬入四个输入；第一次 MMAD 以 `initC=true` 计算 `qg_scaled@h`；第二次以 `initC=false` 将 `Aqk@v_new` 直接累加到同一 L0C | 一次 Fixpipe 以 `F322BF16` 转换并写出 `attn_out` |

`USE_AIV_INPUT_MOVER=true` 按一个 head 的两组依赖拆成下列流水。
V0 发布 Q/H 后即可继续
搬 Aqk/V，C1 不必等待四个输入全部到达；同理，C3 消费 L1 后立即
发布 free，AIV 不等待 Fixpipe 才复用该 head 槽。

| Stage | 核 | 本轮计算 | 阶段结果 |
| --- | --- | --- | --- |
| V0 | Vector | AIV MTE2 将 `qg_scaled/h` 从 GM 搬入 UB，MTE3 跳搬到 L1 并完成 ND2NZ | 发布 Q/H ready |
| C1 | Cube | 等待 Q/H ready，MTE1 搬入 L0A/L0B，MMAD 以 `initC=true` 计算 `qg_scaled@h` | 第一项保留在 L0C |
| V2 | Vector | AIV MTE2 将 `Aqk/v_new` 从 GM 搬入 UB，MTE3 跳搬到 L1 并完成 ND2NZ | 发布 Aqk/V ready |
| C3 | Cube | 等待 Aqk/V ready，第二次 MMAD 以 `initC=false` 累加到同一 L0C；发布 L1 free；Fixpipe 转换并写出 | `attn_out` |

每个 head 的 L1 操作数地址固定：`qg_scaled` 16 KiB、`h` 32 KiB、
`Aqk` 8 KiB、`v_new` 16 KiB，共 72 KiB；四个 head 为 288 KiB，
小于每 AIC 的 512 KiB。C0 读完对应 head 的操作数后才能复用其
L1 槽。`h` 的 32 KiB 是 BF16 `[128,128]`，不能和两个 FP32
计算结果的空间混同。

两项矩阵乘的 L0A 区域分别为 `qg_scaled` 16 KiB 和 `Aqk` 8 KiB，
共 24 KiB；
L0B 分别为 `h` 32 KiB 和 `v_new` 16 KiB，共 48 KiB。单个 head
只产生一份 FP32 `[64,128]` 结果；两块 32 KiB L0C 是相邻 head 的
MMAD/Fixpipe ping-pong 槽，不是两项乘积的独立保存区。第二次 MMAD
直接读取第一次的 L0C 累加结果，随后每个 head 只提交一次 Fixpipe。
Arch35 以 PIPE_MTE1/PIPE_M/PIPE_FIX 的 Mutex 约束对应槽位，Arch22
使用对应的 HardEvent。

A2/A3 的尾块不足 16 行时，L1 左操作数按每个 K 分形只填零无效的
M 行，MMAD 的物理 M 补到 16；Fixpipe 与输出仍只写有效行。
填零和输入搬运都由 MTE2 完成，现有 MTE2 到 MTE1 的事件覆盖两者，
不会读取未初始化的 L0A 行，也不会改变有效行的计算语义。

`USE_AIV_INPUT_MOVER=true` 的 UB 采用两个静态槽，基址为 0 和
128 KiB。每槽固定分配
`qg_scaled` 18 KiB、`h` 36 KiB、`Aqk` 10 KiB、`v_new` 18 KiB，
共 82 KiB；第二槽末端为 210 KiB，小于 A5 的 248 KiB UB。128 列
BF16 数据的 UB 行 pitch 为 9 个 32B datablock，64 列数据为 5 个，
避免连续物理行反复命中同一组 bank。两槽语义固定，不做 UB 内位置
移动；Q/H 与 Aqk/V 分别使用 Mutex 0/1 和 2/3 保护同槽 MTE2/MTE3
并发访问。一次性 GM 输入关闭 L2 cache，避免污染后续可复用数据。

`USE_AIV_INPUT_MOVER=true` 时，AIV0 负责组内 local head 0/2，AIV1
负责 1/3。mode 0x4 下
Q/H ready 在 AIV 侧使用 flag 0/1，AIC 侧映射为 0/16/1/17；Aqk/V
ready 使用 2/3，映射为 2/18/3/19；L1 free 使用 4/5，映射为
4/20/5/21。每个 ready 都由对应 AIC 消费，每个在途 L1 槽都在复用
前等待 free，禁止连续无消费地设置同一个 flag。A2/A3 不编译 A5
Vector mover，也不参与上述跨核同步。

两条路径都不把中间结果写入 GM relay，Host 只申请平台库 API
workspace。Fixpipe 将 L0C FP32 结果直接转换为 BF16：BNSD/NTD
输出的 `dstStride=128`，BSND/TND 输出的 `dstStride=HV*128`，因此
四种 layout 都不需要额外 AIV scatter。两个 L0C slot 的复用由 AIC
内 M/FIX 生命周期保护。

## layout 与元数据

四个输入始终 head-major。`qg_scaled/Aqk` 为 rank-4 时允许
`BSND/BNSD` 输出，为 rank-3 时允许 `TND/NTD` 输出；packed 模式的
FwdH 主路径的 `v_new/h` 仍保留 rank-4/rank-5 首维 1，不能按
`qg_scaled` 的 rank 自动删掉；独立调用也允许 rank-3 `v_new`。
输入的物理形状不随 `output_layout` 变化。
`state_v_first=true` 时，读取 `h` 时交换末两维语义，避免变更公开
输入的物理存储。

变长 `cu_seqlens` 严格递增且覆盖全部 T token。若提供
`chunk_indices`，它必须与 `cu_seqlens` 同时存在，且恰为所有
`(sequence_id, local_chunk_id)` 的规范 sequence-major 枚举。输出
始终保持输入 token 的相对顺序；`BSND/BNSD` 交换 rank-4 的
token/head 轴，`TND/NTD` 交换 rank-3 的 token/head 轴。

## 兼容边界

独立算子不接收 HK；GVA 复用发生在 Prepare，Finalize 只能验证
已展开为 HV 的数据。`h/final_state` 的公开保留策略属于 FwdH 或
完整 forward，Finalize 必须实际收到内部 `h`，即使调用者不请求
公开 intermediate states。稳定 `fla_npu.ops.ascendc` 入口通过 ctypes
直接调用 aclnn，不注册 legacy `torch.ops.npu` 接口。算子私有 ATK
直调只验证设备实现，不能替代稳定 Python 入口验证。
