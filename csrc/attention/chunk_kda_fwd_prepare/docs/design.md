# ChunkKdaFwdPrepare 设计

## 1. 边界

`ChunkKdaFwdPrepare` 是独立的 Ascend C 算子，完成：

```text
Q/K L2 norm + gate cumsum + prepare + post-WU
```

算子返回固定的 13 槽位合同。`gk/Aqk/w/u/kg/qg_scaled` 是 Prepare 与 FwdH/Finalize
之间的正向必需输出；其余七项是可按反向策略裁剪的检查点。chunk 间状态递推由 FwdH
完成，最终 attention 输出由 Finalize 完成，二者不属于本算子。

A5 对每个 chunk/value head 仅读取一次的原始 V 和 gate 禁用 L2 cache；
`HK=HV` 时 Q/K 也禁用 L2，GVA 场景保留 Q/K 的默认缓存策略供多个 value head 复用。
beta、A_log、dt_bias、workspace 和后续阶段会重读的 gk 保持原缓存策略。
纯推理 `OutputMode::None` 的 q/k 归一化仍计算 rstd 并在寄存器内使用，
但不将仅供反向保存的 q_rstd/k_rstd 写入 UB；其他输出模式不变。

Shape 符号沿用 [KDA 模型符号表](../../README.md#核心符号)，公开输入输出与已知限制统一见
[算子 README](../README.md#输入输出)。

## 2. 数学语义

对长度 `M<=64`、K/V 维为 128 的每个 chunk，先生成：

```text
q_rstd[i] = rsqrt(sum_d(q[i,d]^2) + epsilon)
k_rstd[i] = rsqrt(sum_d(k[i,d]^2) + epsilon)
q_hat = cast_bf16(q * q_rstd)
k_hat = cast_bf16(k * k_rstd)
```

关闭 L2 norm 时，`q_hat/k_hat` 等于输入，两个 rstd 输出为 FP32 1。

gate 输入先变成自然对数域 step：

```text
x[i,d] = g_raw[i,d] + optional_dt_bias[d]
delta_g[i,d] = g_raw[i,d]                              # 预计算 step
               -exp(a_log[h]) * softplus(x[i,d])       # Softplus
               lower_bound/(1+exp(-exp(a_log[h])*x))   # SafeSigmoid
```

`use_exp2=true` 时在 cumsum 前乘 `1/ln(2)`，累计量保存于 log2 域；否则保存于自然对数域。
后续统一以 `E(x)` 表示对应域的指数函数。

每个 16 行子块选择中点参考行，S 固定为 4：

```text
Gref[s] = G[floor((16*s + min(M,16*(s+1)))/2)]
Qplus[i] = q_hat[i] * E(G[i] - Gref[s(i)])
Kplus[i] = k_hat[i] * E(G[i] - Gref[s(i)])
Kminus[s,j] = k_hat[j] * E(Gref[s] - G[j])
```

每个 `Kminus[s]` 只保存该 query band 可访问的 key 前缀，四段分别为 16、32、48、64 行。
`Gref` 始终只是 `[4,128]` FP32，不物化 broadcast 矩阵。

C2 得到两个因果 score 矩阵后，V3 施加 mask 和 beta，完成两个 32x32 叶子逆并构造
`Aqk/Akk`。C4/C5 计算跨叶子修正，V6 生成：

```text
qg = cast_bf16(q_hat * E(G))
qg_scaled = cast_bf16(fp32(qg) * scale)
kg = cast_bf16(k_hat * E(Glast-G))
K_positive = cast_bf16(k_hat * E(G))
K_beta_g = cast_bf16(fp32(K_positive) * beta_eff)
V_beta = cast_bf16(fp32(v) * beta_eff)
```

C7 最后计算 `W=Akk@K_beta_g` 和 `U=Akk@V_beta`。BF16 操作数均使用 FP32 累加。

### 2.1 输出生命周期

三种公开输出策略如下：

```text
none:
    gk, Aqk, w, u, kg, qg_scaled

recompute:
    none 的六项
    + Akk, q_hat, k_hat, q_rstd, k_rstd, beta_eff

save:
    recompute 的十二项 + qg
```

策略只影响最终的公开 GM store。V0 仍生成 norm、gate、beta 状态；V3 仍把 `Akk` 写入
workspace 并供 C4/C5/C7 使用；V6 仍生成 UB 中的 `qg`，再基于其 BF16 舍入值计算
`qg_scaled`。因此三档的公式、UB/L1/workspace 布局、VF 调用数和同步协议一致。

算子 IR 和 kernel ABI 固定保留 13 个 `REQUIRED` 输出槽位。L2 未请求的槽位传
`nullptr`，L0 用不会被当前编译实例写入的合法 descriptor 占位，防止 launcher 压缩参数。

`output_final_state` 与 `return_intermediate_states` 属于完整 forward，分别控制用户可见的
`final_state/h`，和本节的反向策略正交；Prepare 不接收这两个属性。

## 3. 八阶段流水

| Stage | 核 | 内容 |
| --- | --- | --- |
| V0 | Vector | norm、beta、gate、cumsum，生成并写出 Q/K 保存量与 `gk/beta_eff` |
| V1 | Vector | 一次 VF 生成 S=4 的 `Qplus/Kplus/Kminus` |
| C2 | Cube | 四个 stacked band MMAD，同时生成 raw Aqk/raw Akk |
| V3 | Vector | causal mask、beta、叶子逆，生成 Aqk/B/X0/X1/negX1/Akk |
| C4 | Cube | `M>32` 时计算 `T=B@X0` |
| C5 | Cube | `M>32` 时计算 `q10=negX1@T` |
| V6 | Vector | 生成 qg/qg_scaled/kg/K_beta_g/V_beta |
| C7 | Cube | 计算 W/U |

每个 Vector Stage 只调用一次 VF；VF 循环体中只有 `if constexpr` 模式分支。每个 Stage
只属于 Vector 或 Cube 一类，Cube 不读取同一 Stage 新生成的数据。

## 4. C2 分块

每个 band 把 Qplus 和 Kplus 沿行堆叠为 `[32,128]`：

```text
[Qplus_s] @ Kminus_s^T = [rawAqk_s]
[Kplus_s]                [rawAkk_s]
```

四次 MMAD API 的 N 依次为 16、32、48、64，每次同时覆盖两个数学乘积。因此 full chunk
是 4 次 MMAD 提交、8 个数学矩阵乘积，不是 16 次 `16x16` 独立提交。

## 5. 分核与 GVA

Prepare 不存在 chunk 间依赖，Host 首先仅按 chunk 分核。只有 chunk work item 少于可用 AIC
时才增加 head partition。head partition 的边界按 `group_size=HV/HK` 对齐，禁止拆开共享同一
Q/K head 的完整 GVA value-head 组。

A5 dense 在全局只有一个 head partition、但 chunk 数不能整除已用核时，继续保持完整轮次只按
chunk 分核。例如 128 个 chunk、28 个核时，每核先处理 4 个完整 chunk，共 112 个；剩余 16 个
chunk 已不足以再覆盖全部核，因此只把这部分按 4 个 value head 的硬件处理组展开并均摊。仅当
`group_size=HV/HK` 能整除 4 时启用该路径，保证尾部边界仍不拆开 GVA 组。AIC 和两个 AIV
使用完全相同的主体、尾部映射顺序，尾部不同核写入互不重叠的 `(chunk, head)` 区域。变长、
A2/A3、已经启用全局 head partition 或 GVA 边界不满足条件的场景仍使用原调度。

每个 AIC workgroup 一轮最多处理 4 个 value head。A5 的 AIV0 处理 local head 0/1，AIV1
处理 2/3；A2/A3 的两个 AIV 按 pair wave 处理 0/2 和 1/3。不同 value head 即使映射到同一
Q/K head，也只允许 GVA 组首 owner 写 `q_hat/k_hat/q_rstd/k_rstd`，避免 GM 重叠写。

当前实现仍按 value-head slot 搬运和归一化 Q/K。多个 value head 映射到同一 Q/K head 时，
会从 GM 重读对应 Q/K；这样无需引入跨 AIV 的 UB 共享、额外 VF 调用或 UB 搬位，同时保持
静态双缓冲和既定 Stage 合同。后续若消除该重读，必须同时证明跨 AIV 共享与成对同步不会
破坏当前生命周期。

## 6. 静态内存

内核不使用动态 UB 队列，所有 UB/L1/workspace 地址在编译期固定。地址可在不同 Stage 换义，
但活跃生命周期不重叠前不会复用，也不在 UB/L1 内移动数据。

- A5 UB 为两个 112 KiB 计算槽和两个 12 KiB 状态槽，总计 248 KiB。
- A2/A3 使用 184 KiB 可用 UB，按两个私有区和一个共享区规划，尾部 8 KiB 保留。
- L1 为 512 KiB，四个 head lane 保存 C2 payload，其余区域保存 X0、negX1、T 和 Akk。
- Arch35 每个 workspace slot 为 `0x1A400` Byte，保存 Qhat、Khat、betaEff
  和 72 KiB payload。
- Arch22 每个 slot 追加 20 KiB AIC 独占 relay，总计 `0x1F400` Byte；C2
  写四段 raw score，V3 消费后 C4 在同址前 4 KiB 写 `T`。host 根据目标
  平台选择 slot 大小，A5 不预留 Arch22 的追加区。

V1 payload 由 16 KiB Qplus、16 KiB Kplus 和 40 KiB Kminus prefix 组成，共 72 KiB。

## 7. 同步

A5 使用 `AscendC::Mutex` 约束 MTE2、V、MTE3、MTE1、M 和 Fixpipe 对静态本地地址的
生命周期；A2/A3 使用配对的 HardEvent。核间数据通过 ready/free 双向握手传递：

A5 的 V1/V3 使用公开 `asc_vf_call` 启动 VF。目标 CANN 9.1 规定该接口返回时 VF 内部
Reg 矢量指令已经完成，因此编译器会在调用边界生成 V 到 Scalar 的 join；旧
`AscendC::VF_CALL` 包装生成同一条 join，当前版本没有可替换的 SIMD 异步入口。Mutex
仍负责 MTE2、V、MTE3 对静态 UB 槽位的生命周期交接，不能用 VF 返回语义替代。

```text
AIV: wait free -> 写 workspace/UB -> set ready
AIC: wait ready -> 消费 -> set free
```

A5 在 AIC 视角使用 ready `0/1/16/17` 和 free `4/5/20/21`；A2/A3 mode2 pair wave使用
ready `0/1` 和 free `2/3`。同一 flag ID 在上一代 set/wait 闭环后才复用，不依赖核启动顺序。

C4 对全部 active head 先提交一轮，再提交 C5，使后续 head 的独立 C4 能进入流水。C4 把
B/X0/negX1/Akk 搬入每 head 独立 L1 后即可归还 workspace payload，V6 可与 C4/C5 排空重叠。

## 8. 模板轴

TilingKey 编译期选择 gate dtype、beta dtype、norm 模式、beta 模式、gate 模式、
`use_exp2`、`safe_gate` 和三档 `OUTPUT_MODE`。`q/k/v` 不进入 dtype 模板轴，固定为 BF16。架构通过
`__CCE_AICORE__` 编译宏选择，kernel 内没有运行时 Arch22/Arch35 分支。

输出搬出在模板实例中使用 `if constexpr` 消除；设备侧 tiling 不保存运行时
`outputMask`，Stage 和 VF 循环都不执行运行时输出判断。
