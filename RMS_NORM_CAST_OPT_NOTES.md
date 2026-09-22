# rms_norm_cast 优化记录（910B_perf）

## Baseline（2026-09-22，未改动）

NPUGraph benchmark（`benchmarks/rms_norm_cast.py`，µs/次，hidden=7168）：

> **fused** = 被测融合算子 `torch.ops._C_ascend.npu_rms_norm_cast`，一次 kernel 调用直接产出
> 两份输出：低精度 `y`（bf16/fp16）+ 宽化 `y_fp32`（fp32）。
> **ref（unfused）** = 参照基线 `torch_npu.npu_rms_norm`（只产出 y）+ `y.float()`（Cast 出 fp32），
> 两个算子串联得到同样的两份输出，代表"不融合时的现有做法"。
> fused 少一次 y 的中间读写（少 ~20% 数据搬运），理论上应始终更快；大 shape 下反而更慢，
> 正是本次要优化的点。

| tokens | bf16 fused | bf16 ref | fp16 fused | fp16 ref |
|---|---|---|---|---|
| 1 | 2.91 | 3.66 | 2.80 | 3.60 |
| 4 | 3.06 | 4.65 | 2.90 | 4.64 |
| 16 | 3.82 | 6.27 | 3.69 | 6.32 |
| 64 | 7.11 | 9.44 | 6.80 | 9.43 |
| 128 | 10.60 | 11.69 | 9.92 | 11.60 |
| 512 | 25.72 | 22.95 | 23.25 | 22.75 |
| 1024 | 47.56 | 39.02 | 42.39 | 37.87 |
| 2048 | 102.07 | 77.37 | 93.61 | 76.85 |
| 4096 | 222.39 | 197.39 | 209.27 | 202.74 |

> ref = unfused `torch_npu.npu_rms_norm` + `.float()`（同样的两份输出）。
> fused 搬运数据比 ref 少 20%（117MB vs 147MB @2048）但在大 shape 更慢 → kernel 有明确优化空间。

msprof op（PipeUtilization，2048×7168 bf16，Block Dim 40）：
- RmsNormCast: 103.0µs，MTE2/MTE3 active 时带宽利用率 < 80%
- torch_npu.profiler 口径（同 shape）：90.7µs = vec 65.6% + mte2 10.9% + mte3 19.1% + scalar 6.5%
  → **vec 管线是瓶颈**

## 诊断结论（bf16 路径每行 ~9 遍 vec pass，2 遍可省）

1. `Muls(work, inv_num_col_, num_col)` 全向量乘 1/N 在 reduce 前 —— 可改为 reduce 后对 1 个标量做
2. `Cast(gamma→fp32)` 在 bf16 分支每行重复 —— 可每核提出来做一次

## 轮次计划

- Round 1: 省掉上述 2 遍 vec pass（纯减 vec 工作，无流水改动）✅
- Round 2: 行间双缓冲流水 + rstd 向量广播（Gather+stride-0 Mul 消标量往返）✅
  （首次尝试两个 bug 的根因与"退出标志必须清零"契约见详设 v2 与轮次记录）
- Round 3: 视 Round 2 profile 决定 —— vec 已 91%，主攻**减 vec pass 数**
  （bf16 9 遍：候选=平方与 reduce 树的融合指令、两次输出 Cast 的合并可行性）
  与小 shape（1-16）回退（+0.1~0.25µs，序言固定开销）
- Round 4: 视 profile 收尾（候选池：4096 写带宽结构性上限确认、
  DataCopyPad→对齐 DataCopy、小 shape 单核串行链缩短）
  原则：任何 shape 不回退；以 msprof 数据决定取舍。

## 测量方法一致性验证（NPUGraph benchmark vs msprof op）

| tokens (bf16) | NPUGraph | msprof op | 差值 |
|---|---|---|---|
| 512 | 25.9µs | 30.2µs | +4.3µs |
| 2048 | 101.2µs | 104.3µs | +3.1µs |

结论：msprof op 采集自带 ~3-4µs 固定开销，相对量级与 NPUGraph 一致。
**每轮对比用 NPUGraph 数字，pipe 归因用 msprof op。**

## 优化详设（2026-09-22，基于代码走读 + profile）

### 现状：bf16 路径每行 11 个 vec pass（fp16 路径 10 个）

| # | 操作 | 备注 |
|---|---|---|
| 1 | Cast x→fp32 | 必须 |
| 2 | Mul work=x² | 必须 |
| 3 | Muls work×inv_num_col（全 7168 向量） | **Round1 可省** |
| 4 | ReduceSumCustom（Add 树 + WholeReduceSum） | 必须 |
| 5 | 标量链 Adds(eps)→Sqrt→Div（1 元素） | 代价可忽略 |
| 6 | V_S→S_V 标量往返取 rstd | 每行一次同步 |
| 7 | Muls x_fp32×rstd | 必须 |
| 8 | Cast gamma→fp32（**每行重复**） | **Round1 可省（bf16 独有）** |
| 9 | Mul x_fp32×gamma_fp32 | 必须 |
| 10 | Cast RINT → x_local | 必须 |
| 11 | Cast widen → x_fp32 | 契约要求 |

vec busy ≈ 59.5µs ÷ 11 遍 ≈ 5.4µs/遍（2048×7168，profiler 口径 90.7µs，vec 65.6%）。
另外每行 MTE2→V→MTE3 完全串行，vec 空转 ≈ 31µs（ Round 3 目标）。

### Round 1：省 2 遍 pass（纯减 vec 工作，无流水改动，无数值域变化）

- **1a inv 折叠**：删除 reduce 前的全向量 `Muls(work, inv_num_col_, num_col_)`，
  在 reduce 后对 1 个标量做 `Muls(work, work, inv_num_col_, 1)`。
  数学等价（mean = sum×inv），仅 fp32 舍入路径不同（~1 ulp，远小于输出容差）。
- **1b gamma 预转换（bf16）**：新增 `gamma_fp32_buf_`，每核 Process() 里 Cast 一次，
  行循环内直接 `Mul(x_fp32, x_fp32, gamma_fp32_local)`。
  tiling `BYTES_PER_COLUMN` 12→16（两个 b16 + 三个 fp32 buffer）。
  7168×16B = 112KB < UB−1KB（910B UB=192KB）✓；fp16 路径不受影响。
- 预期：11→9 遍 ≈ -18% vec → 2048 tokens ~102→~89µs。

### Round 2：行间双缓冲流水（首次实现已回退，待修复后重试）

**硬件约束（重要发现，2026-09-22）**：本 CANN 版本（9.1.0，dav_c220 = 910B）的
基础向量算子对 bfloat16 **没有**元素级支持：

- `Muls<bf16>`：intrinsic 仅 half/float/int16/int32（`dav_c100/kernel_operator_vec_binary_scalar_impl.h`）
- `Mul<bf16>`：`MulImpl` 断言 `SupportType<T, half, float, int16_t, int32_t>`（`dav_c220/kernel_operator_vec_binary_impl.h`）
- `Duplicate<bf16>`：`CheckDuplicateSupportedType` 同样不含 bf16

因此 bf16 数据的所有元素级数学只能在 fp32 域做（这正是原实现的写法），
原计划的"bf16 域单舍入重构"（对齐 fp16 路径）在本硬件上**不可实现**，
Round 2 改为原 Round 3 的双缓冲流水方案。

**首次实现结果**：编译通过，但精度测试失败 —— 小 shape（4 tokens）输出 NaN，
大 shape 挂死（`vector core timeout`，AICore ~95% 空转）。定位到两个 bug 后
**代码已回退到 Round 1**（`git checkout`，并重编译安装 Round 1 版本，精度复验通过）：

1. **序言预取槽位错位**（→ NaN）：序言把第 `begin+k` 行加载到 `x_buf_[k]`，
   而行处理按奇偶槽位 `x_buf_[begin&1]` 取数 —— begin 为奇数的核心读到
   未初始化数据，产生 NaN。应为 `x_buf_[(row_begin_+k)&1]`。
2. **事件 ID 分配模型**（→ 挂死）：`TPipe::FetchEventID` 是按方向从占用位图
   找"第一个空闲 ID"（`kernel_tpipe_impl.h:489`，`sff0(eventOccupy)`），
   **连续两次 fetch 之间没有 Set/Wait 时返回同一个 ID**。乒乓设计要求每个
   方向同时 2 个在途标志，此约束不成立 → 事件配对塌缩 → 死锁。
   （小 shape 没挂死而只出 NaN，正是因为同 ID 折叠后 wait 全部立即通过。）

**重试方案要点**（Round 2 二次尝试时按此执行）：

- 事件编排必须满足"每个方向严格单在途标志"（set→wait→set→wait 顺序复用），
  乒乓改为"提前发起、延迟等待"，不依赖双 ID；或先实测确认 eventOccupy 的
  置位/释放时机（大概率在 SetFlag/WaitFlag 执行时）再决定能否用双 ID。
- 序言预取用行的奇偶槽位；循环内预取 row+2（row+1 已由序言/上轮加载）。
- UB 预算 24B/col（bf16，含 gamma_fp32）+256B reduce ≈ 168.3KB ✓；
  out_fp32 槽兼作该行 reduce 累加器（平方项 reduce 后即死，widen 前重写同槽）。
- tiling BYTES_PER_COLUMN 按 dtype 分档 20（fp16）/24（bf16）；7168 ✓，
  最大 hidden 收缩到 ~8.1K/~9.8K（唯一调用方 DeepSeek v4/v41 hidden=7168 无影响）。
- 预期：消除 ~25µs 行间串行空转（2048 tokens），91→~65-70µs。

### Round 2 重试详设 v2（2026-09-22，CANN 9.1.0 源码级确认后定稿）

针对上述两个 bug 做了 CANN 源码级根因确认（`asc/impl/basic_api/` 下），
三个关键事实推翻/修正了首次尝试的假设：

**事实 1 —— 事件 ID 的占用模型**（`kernel_tpipe_impl.h`）：
- `FetchEventID` 只读 `sff0(eventOccupy)` 找第一个空闲位，**不置位**；
- `AllocEventID` 找空闲位并**置位**，`ReleaseEventID` 清位；
- `TQue::EnQue` = AllocEventID+SetFlag，`DeQue` = WaitFlag+ReleaseEventID
  （`kernel_tquebind_impl.h:203/343`）——框架自身就是靠 Alloc/Release
  持有多个在途 ID 实现深度 2 队列，每方向上限 `QUE_MAX_EVENT=8`。
- 首次尝试全用 FetchEventID：两次 fetch 永远返回同一位 → 双在途塌缩。
  **修正：一律用 AllocEventID/SetFlag/WaitFlag/ReleaseEventID 四件套。**

**事实 2 —— 单 ID 交替 set/wait 也不安全**：即使发射序严格 set→wait→set→wait，
两个 pipe 执行速度不同（MTE2 0.19µs/行 vs V 0.96µs/行），快 pipe 可以在慢 pipe
消费标志前连续两次 SetFlag，二值标志塌缩 → 过度同步（碰巧正确）或尾部死锁。
独立 ID 从根上消除该问题。

**事实 3 —— WaitFlag 挂在 dstPipe 上执行**（`kernel_event.h:318` `wait_flag(srcPipe, dstPipe)`）：
- MTE2_V/V_MTE2/V_MTE3/MTE3_MTE2/MTE3_V 五个方向的 wait 全部不阻塞发射线程，
  发射线程可以自由跑到指令队列满（背压）为止；
- **唯一阻塞发射线程的是 V_S**（GetValue 标量往返，dst=S=发射线程所在 pipe）。
  Round 1 代码里它就是每行的同步点。

**rstd 标量往返的消除**（新发现，替代 GetValue 方案）：
- `Gather(rstd8, work, zero_offsets, 0, 8)`：vgather 在 c220 支持 float，
  dst[i]=src[offset[i]]，一次把 work[0] 复制成 8 份（zero_offsets 每核
  Duplicate 一次 uint32×8）；
- `Mul(x_fp32, x_fp32, rstd8, mask, repeats, {dstBlk=1, src0Blk=1, src1Blk=0,
  dstRep=8, src0Rep=8, src1Rep=0})`：src1 钉在 32B 的 rstd8 上做块广播
  （antiquant_c220/batchnorm_v220 同款 stride-0 范式）；
- 数值与 `Muls(x_fp32, x_fp32, rstd, num_col)` **位级一致**（同一个 fp32 rstd
  参与同样的 fp32 乘法），精度测试零风险；
- 依赖变成纯 V→V（Brcb/Gather/Mul 全在 V pipe，PipeBarrier 即可），每行省掉
  V_S/S_V 两次跨 pipe 往返 + 发射线程停等。

**缓冲区布局**（y 仍写回 x 槽，省一个 y 缓冲且 MTE3_V 不上关键路径）：

| 缓冲 | 数量 | 大小 | 复用链（事件方向） |
|---|---|---|---|
| x[2] | 2×A×2B | 输入行 + 兼作 y 输出 | MTE2→V→(V_MTE2+MTE3_MTE2)→MTE2 |
| x_fp32[2] | 2×A×4B | 宽化 x + 兼作 y_fp32 输出 | V→MTE3→(MTE3_V)→V |
| gamma / gamma_fp32 | A×2B / A×4B | 每核一次 | —（gamma_fp32 仅 bf16） |
| work | A×4B | 平方 + rstd 标量链 | V 内部独占 |
| reduce+rstd8+off | 320B | reduce 树/gather 目的/偏移 | V 内部独占 |

- BYTES_PER_COLUMN 按 dtype 分档：bf16 22（含 gamma_fp32 4）/ fp16 18；
  @7168 bf16 共 158KB < 191KB ✓；最大 hidden 收缩到 ~8.8K/~10.8K
  （比首次尝试方案的 24/20 更省，因 y 复用 x 槽 + gamma_fp16 不分配）。
- 槽位一律用**核内局部行号** i 的奇偶（i&1），彻底避开首次尝试的绝对奇偶错位 bug。

**事件编排**（每方向 ≤1 个在途 ID，全用 AllocEventID 四件套）：

```
序言: alloc/set(MTE2_V, e_load)  # load 局部行0 → x[0]
循环 i:
  wait(MTE2_V, e_load)→release                    # V 等输入 i 就绪
  if i+1<n:                                        # 预取行 i+1 → x[(i+1)&1]
    if i>=1: wait(V_MTE2)→release; wait(MTE3_MTE2)→release   # 槽位上个主人是行 i-1
    alloc/set(MTE2_V, e_load')
  if i>=2: wait(MTE3_V)→release                    # x_fp32[i&1] 上个主人是行 i-2 的落盘
  V 链（Cast/Mul/Reduce/标量链/Gather/广播Mul/·gamma/Rint/Cast widen）
  alloc/set(V_MTE2, e_vm2)                         # x[i&1] 交给 load i+2（在 widen 后）
  alloc/set(V_MTE3, e_v3); wait(V_MTE3)→release    # MTE3 等本行 V 链完成
  DataCopy y ← x[i&1]
  alloc/set(MTE3_MTE2, e_m32)                      # x[i&1] 落盘完可被 load i+2 覆盖
  DataCopy y_fp32 ← x_fp32[i&1]
  alloc/set(MTE3_V, e_m3v)                         # x_fp32[i&1] 落盘完可被行 i+2 重写
```

无环性：所有依赖都指向更老的行（i ← i-1/i-2），发射线程永不阻塞 → 无死锁。
稳态关键路径 = 纯 V 链（0.96µs/行）；MTE2 0.19、MTE3 0.34µs/行全部被覆盖。
预期 2048 tokens：80.1µs（R1 profiler 口径）→ ~52-58µs，NPUGraph 91.2 → ~60-66µs。

### Round 3：视 Round 2 重试后的 profile 决定

### Round 4：视 profile 收尾（候选池）

- ReduceSumCustom 树形结构优化（Add 链 112 次 repeat）
- DataCopyPad vs 对齐 DataCopy 路径
- 小 shape（decode 1-16 行）单核串行链缩短
- 原则：任何 shape 不回退；以 msprof 数据决定取舍。

### 风险与回退

- 每轮独立提交、独立可验证；精度门槛 = `test_rms_norm_cast.py` 8 用例全过（零回归）。
- tiling 12→16B/col 使支持的最大 hidden 从 ~16.3K 收缩到 ~12.2K（当前唯一调用方
  DeepSeek v4/v41 hidden=7168，无影响）。

## 轮次结果记录

（每轮：精度结果 / NPUGraph 数据 / msprof 数据 / 与上轮对比）

### Round 1（2026-09-22）：inv 折叠到标量 + bf16 gamma 每核预转换 — ✅ 完成

改动：删除 reduce 前的全向量 Muls（改为 reduce 后 1 元素 Muls）；bf16 路径 gamma
每核 Cast 一次（新增 gamma_fp32_buf_，tiling BYTES_PER_COLUMN 12→16）。

- **精度**：`test_rms_norm_cast.py` 8/8 通过，零回归。
- **NPUGraph benchmark（µs，vs baseline）**：

| tokens | bf16 base→R1 | Δ | fp16 base→R1 | Δ |
|---|---|---|---|---|
| 512 | 25.7→23.2 | **-9.7%** | 23.2→22.3 | -4.1% |
| 1024 | 47.3→42.1 | **-11.0%** | 42.4→40.1 | -5.3% |
| 2048 | 102.1→91.2 | **-10.7%** | 93.6→87.9 | **-6.1%** |
| 4096 | 222.4→208.2 | **-6.4%** | 209.3→203.6 | -2.7% |

小 shape（1-128）无回退（1 token: 2.91→2.71）。
fp16 提升小于 bf16：fp16 路径只吃到 inv 折叠（无 gamma cast 可省）。

- **pipe 归因**（torch_npu.profiler，2048×7168 bf16，同口径 vs baseline）：
  duration 90.7→80.1µs；vec busy 59.5→48.9µs（65.6%→61.0%，恰好 2 遍 pass）；
  mte2/mte3 绝对时间不变（9.9/17.3µs）→ 仍为 vec 瓶颈但空转扩大（~31µs），
  Round 3（双缓冲流水）收益空间明确。
- **教训**：首轮构建因 build 树 src_copy 不重编（见"构建陷阱"）空跑一轮测的是旧 kernel；
  修复后实测生效。

### Round 2 首次尝试（2026-09-22）：行间双缓冲流水 — ⏸️ 已回退，待重试

目标：吃掉 profile 中 ~25µs 的行间串行空转（行首等 load、行尾等 store）。

过程：按奇偶槽位乒乓重写 ProcessRow（x 双缓冲 + out_fp32 双缓冲 + 单 y 缓冲 +
手工 MTE2_V/V_MTE2/MTE3_V/V_MTE3 事件），两个 tiling key 编译通过、包构建成功。

结果：**精度测试失败，未进入性能测量**——
- 4 tokens（rows_per_core=1）：跑完但输出 NaN；
- pytest 大 shape 用例：挂死 `rtDeviceSynchronizeWithTimeout ... vector core timeout`，
  AICore ~95% 空转（典型 wait_flag 死循环）。

根因（详见上方"详设 Round 2"）：
1. 序言预取把行加载到 `x_buf_[k]` 而非奇偶槽位 `x_buf_[(begin+k)&1]` → 错行/未初始化数据 → NaN；
2. `FetchEventID` 连续 fetch（中间无 Set/Wait）返回同一 ID → 双在途标志设计塌缩 → 死锁。

处置：`git checkout` 回退 kernel + tiling 到 Round 1 提交（09fdb4f03），重编译安装后
精度复验通过；修复方案（单在途事件编排 + 槽位修正 + row+2 预取）已写入详设，待重试。

### Round 2 二次实现（2026-09-22）：流水 + rstd 向量广播 — ✅ 完成

按详设 v2 实现，相对首次尝试的三个关键差异：
1. **事件一律 AllocEventID/SetFlag/WaitFlag/ReleaseEventID 四件套**（框架 TQue 同款），
   不再用 FetchEventID；MTE3_V 方向因隔 2 行才消费，用 `fp32_stored_evt_[2]` 双 ID。
2. **消除每行 V_S/GetValue/S_V 标量往返**：`Gather(rstd8, work, zero_off, 0, 8)` 把
   work[0] 复制成 8 份，再 `Mul(x_fp32, x_fp32, rstd8, mask, reps, {src1BlkStride=0,
   src1RepStride=0})` 块广播——rstd 全程留在 UB，发射线程不再停等，数值与 Muls 位级一致。
3. **槽位一律用核内局部行号奇偶**（i&1），序言与循环天然对齐，根除首次尝试的错位 bug。

实现过程中发现并修复的**第三个 bug（新知识，重要）**：
- 现象：本 kernel 正常完成（sync 返回），但**同进程后续第一个算子挂死**
  （首测 pytest 全挂、sanity 里 `torch_npu.npu_rms_norm` 挂死；单独跑参照正常）。
- 根因：kernel 退出时留下了未消费的 SetFlag（尾部 2 行的 V_MTE2/MTE3_MTE2/MTE3_V），
  污染同核下一个 kernel 的事件标志状态。CANN 的 `TPipe::DestroyWithoutPipeAll` 专门在
  退出前 wait 所有挂起 free-buf 事件，证明"退出时事件必须清零"是框架级契约。
- 修复：**发射条件与消费条件严格一致的条件下才 SetFlag**（`i+2 < local_rows` 才发
  V_MTE2/MTE3_MTE2/MTE3_V——它们的消费者（load i+2 / 行 i+2 的 step1）恰好同条件存在）。
  退出时零悬挂标志、零占用 ID，无需 drain。

UB/tiling：y 复用 x 槽（MTE3 不上 V 关键路径），BYTES_PER_COLUMN 分档
bf16 22 / fp16 18 + 固定 320B；@7168 bf16 共 158KB；最大 hidden ~8.8K/~10.8K。

- **精度**：`test_rms_norm_cast.py` **8/8 通过**（12.8s），零回归。
  （bf16 各 shape 对 ref max_err ≤ 1.56e-2；fp16 ≤ 3.9e-3 = 大值处恰 1 个 fp16 ulp，
  均在 assert_close 容差内。注意自写 sanity 不要用平坦阈值，会比真实测试严。）
- **NPUGraph benchmark（µs，vs Round 1）**：

| tokens | bf16 R1→R2 | Δ | fp16 R1→R2 | Δ |
|---|---|---|---|---|
| 1 | 2.71→2.95 | +8.8% | 2.66→2.84 | +6.8% |
| 4 | 3.06→3.12 | +2.0% | 2.94→3.03 | +3.1% |
| 16 | 3.82→4.08 | +6.8% | 3.69→3.93 | +6.5% |
| 64 | 7.11→6.84 | -3.8% | 6.80→6.54 | -3.8% |
| 128 | 10.60→8.83 | **-16.7%** | 9.92→8.55 | **-13.8%** |
| 512 | 23.2→17.5 | **-24.4%** | 22.3→16.9 | **-24.2%** |
| 1024 | 42.1→29.9 | **-29.0%** | 40.1→28.7 | **-28.4%** |
| 2048 | 91.2→57.4 | **-37.1%** | 87.9→54.5 | **-38.0%** |
| 4096 | 208.2→178.3 | **-14.4%** | 203.6→173.7 | **-14.7%** |

- 累计 vs 未优化基线：2048 bf16 102.1→57.4（**-43.8%**）、fp16 93.6→54.5（-41.8%）；
  大 shape 已稳定快于 unfused 参照（77.4µs @2048）约 26%。
- 小 shape（1-16 tokens）+0.1~0.25µs：每核固定的 Gather 偏移 Duplicate + 序言开销，
  1 行/核时流水收益吃不到（Round 4 候选：小 shape 专用路径）。
- 4096 档 -14%（低于线性外推）：疑似输出写带宽（每字节输入写 3 字节输出）饱和，
  需 msprof MTE3 数据确认（见下）。
- **pipe 归因（msprof op，2048×7168 bf16，40 核均值）**：

| 指标 | Round 1 | Round 2 |
|---|---|---|
| duration | 80.1µs（torch_npu.profiler 口径）/ 91.2µs（NPUGraph） | **53.7µs**（msprof task）/ 57.4µs（NPUGraph） |
| vec busy / ratio | 48.9µs / 61.0% | **46.7µs / 90.9%** |
| mte2 busy / ratio | 9.9µs / — | 20.5µs / 39.9%（含 MTE2 上执行的 wait_flag 等） |
| mte3 busy / ratio | 17.3µs / — | 17.4µs / 33.8% |
| scalar_vector_stall | — | 39.5µs（发射线程背压，符合预期） |

  → **流水目标达成**：MTE2/MTE3 时间不变但全部移出关键路径，vec 占比 61%→91%。
  剩余 ~4.6µs/核 的 vec 空闲（9%）≈ 序言 gamma 串行 + 尾行 store 排空 + 首行预取深度 1。
  Round 3 的唯一大杠杆 = **减 vec pass 数**（现 bf16 9 遍）；4096 档为写带宽饱和
  （235MB/178µs ≈ 1.3TB/s），属结构性，优化空间在减少 y_fp32 输出量（契约不允许）或接受。


## 每轮工作流（精度是硬门槛）


```bash
# 0. Round N：改 csrc/moe/rms_norm_cast/op_kernel/*.cpp|*.h

# 1. 编译（CANN 9.1.0 下零补丁直接过）
# 1a. 强制刷新目标算子的 build 树（规避 src_copy 不重编，见下方"构建陷阱"）
rm -rf csrc/build/binary/ascend910b/src/rms_norm_cast \
       csrc/build/binary/ascend910b/bin/rms_norm_cast \
       csrc/build/binary/ascend910b/gen/rms_norm_cast_ascend910b_*.done
# 1b. 全量构建（build_aclnn.sh 内部调 build.sh --pkg 并安装到 repo 内 vendor 目录）
bash csrc/build_aclnn.sh $(pwd) ascend910b
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}
pip install -e . --no-build-isolation

# 2. 精度测试（必须全过，不允许精度回归）
pytest tests/e2e/nightly/single_node/ops/singlecard_ops/test_rms_norm_cast.py -q

# 3. 性能记分板（NPUGraph replay）
python benchmarks/rms_norm_cast.py

# 4. pipe 归因（msprof op；脚本入仓 benchmarks/rms_norm_cast_msprof.py）
msprof op --kernel-name=RmsNormCast --output=./profiling --aic-metrics=PipeUtilization \
  python benchmarks/rms_norm_cast_msprof.py 2048

# 5. 记录数据到本文档并提交
```

> **构建陷阱（必读）**：`csrc/build` 是增量保留的，kernel 源拷贝规则（`func.cmake` 的
> `src_copy`）只以 `.done` 标记为 OUTPUT、不依赖源文件 —— **改了 op_kernel/*.h|*.cpp 后
> 不会自动重编**。每轮编译前必须先删目标算子的 build 树拷贝与标记：
>
> ```bash
> rm -rf csrc/build/binary/ascend910b/src/rms_norm_cast \
>        csrc/build/binary/ascend910b/bin/rms_norm_cast
> ```
>
> 编译后自检：`csrc/build/binary/ascend910b/src/rms_norm_cast/op_kernel/rms_norm_cast.h`
> 应与仓库源文件一致（size/md5），且安装产物 .o 的 mtime 晚于源文件 mtime。
> （Round 1 曾因此空跑一轮：装的是旧 .o，测出来"零提升"。）

> msprof op 注意事项：
> - 只保留 `--kernel-name` / `--output`（不加 --launch-count/--warm-up），`--aic-metrics=PipeUtilization` 按需选；
> - 被测命令作为位置参数直接跟在选项后面（不要用 `--application=python3 script.py`，参数传不进去）；
> - shape 用位置参数传给 runner（环境变量穿不透 msprof 启动层）。
