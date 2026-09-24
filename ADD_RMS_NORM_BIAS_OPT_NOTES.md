# add_rms_norm_bias 优化记录（add-rms-norm-bias-perf）

> 910B3 (dav_c220, 40 AIV, UB 192KB/核), CANN 9.1.0。
> 方法论同族见 `rms-norm-cast-perf` 分支的 RMS_NORM_CAST_OPT_NOTES.md（事件四件套、
> 退出标志清零契约、Gather+stride-0 广播范式均沿用）。

## 第 0 步：基线（2026-09-24，未改动代码，commit 7dcbbe56a）

### 0.1 精度基线 — ✅ 全绿

`pytest tests/e2e/nightly/single_node/ops/singlecard_ops/test_add_rms_norm_bias.py -q`
→ **126 passed**（7 rows × 6 cols × 3 dtype，numpy golden）。

测试 shape 与变体覆盖映射（见 0.4）：col=8/16/128 → MERGE_N；col=3000 rows≥41 →
MULTI_N(bf16)/NORMAL(fp16/fp32)；col=7168 rows≥41 → NORMAL(bf16)/MULTI_N(fp16)；
rows≤40 → SINGLE_N；col=15000 → SPLIT_D。**全部 5 个变体都被 126 用例覆盖。**

契约（bf16 golden，kernel 双舍入结构与之一致）：
x = bf16(fp32(x1)+fp32(x2))；rstd = 1/sqrt(mean(x²)+eps)（fp32）；
y = bf16( fp32(bf16(x_norm)) × fp32(gamma) + fp32(beta) )。

### 0.2 性能基线（NPUGraph，benchmarks/add_rms_norm_bias.py，hidden=7168，µs）

fused = `torch.ops._C_ascend.npu_add_rms_norm_bias`（1 kernel 出 y/rstd/x 三份输出）；
ref = `torch_npu.npu_add_rms_norm`（3 输出元组 y/rstd/residual）+ `y+beta`
（2 kernel 同样三份输出）。

| tokens | bf16 fused | bf16 ref | fp16 fused | fp16 ref |
|---|---|---|---|---|
| 1 | 3.36 | 4.19 | 2.60 | 3.31 |
| 4 | 3.70 | 6.44 | 2.93 | 5.84 |
| 16 | 5.32 | 7.79 | 4.55 | 7.11 |
| 64 | 10.10 | 13.85 | 8.65 | 12.73 |
| 128 | 13.98 | 18.89 | 11.42 | 15.71 |
| 512 | 33.16 | 43.82 | 21.50 | 27.11 |
| 1024 | 59.04 | 62.94 | 35.36 | 42.70 |
| 2048 | 119.41 | 127.67 | 62.95 | 73.30 |
| 4096 | 250.89 | 320.96 | 171.68 | 275.87 |

信号：
- fused 比 ref 全档快（本算子无 rms_norm_cast 那种"大 shape 反超"），但 **bf16
  在 1024/2048 档优势收窄到 6-6.5%** —— bf16 kernel 明显劣于自身硬件能力。
- **bf16 fused @2048 = 119.4µs vs fp16 fused = 62.9µs（1.9×）**：搬运字节数完全相同
  （同元素宽度），差距全部来自 kernel 实现（bf16 pass 数更多 + 无流水）。
  bf16 的现实目标 ≈ fp16 fused + bf16 额外 cast 开销 ≈ 70-80µs @2048。
- fp16 档 fused 对 ref 优势稳定（≥14%）。

> 修复记录：入仓 benchmark 的 ref 解包 bug（`npu_add_rms_norm` 返回 3 元组
> (y, rstd, residual)，原代码按 2 元组解包直接 ValueError）——本轮已修。

### 0.3 pipe 基线（msprof op PipeUtilization，40 核均值）

采集：`env -u ASCEND_RT_VISIBLE_DEVICES msprof op --warm-up=10 --launch-count=1
--output=<dir> --kernel-name=AddRmsNormBias --aic-metrics=PipeUtilization
python benchmarks/add_rms_norm_bias_msprof.py <tokens>`

**@2048×7168 bf16（key 30，NORMAL，40 核 × 52 行）——生产 prefill 路径：**

| 指标 | 值 | 占墙钟(124.4µs) |
|---|---|---|
| aiv active | 120.2µs | 97% |
| vec busy | 87.9µs | **73%**（1.69µs/行） |
| mte2 busy | 27.0µs | 22%（~52GB/核低带宽=短拷贝 DataCopyPad） |
| mte3 busy | 18.7µs | 15% |
| scalar busy | 13.1µs | 11% |
| **scalar_wait** | **106.0µs** | **85%** —— 发射线程几乎全程停等 |

→ 双瓶颈结构：vec 是最大占用者但未饱和（27% 空转），scalar_wait 85% 说明
发射线程被每行的 V_S/S_V 往返 + 单缓冲队列串行点钉死。
**消除标量往返 + 减 pass 是第一杠杆，行间流水是第二杠杆。**

**@1×7168 bf16（key 33，SINGLE_N，1 核）——decode 路径：**

wall 5.1µs；vec 1.79µs(40%) / mte2 1.38µs(27%) / mte3 0.50µs(10%) /
scalar 1.49µs / scalar_wait 1.24µs(25%)。发射+延迟混合瓶颈，优化优先级低于 prefill。

### 0.4 变体映射（代码走读 + 实测 tiling key 实证）

> **重要修正**：任务简报中"生产 shape (hidden=7168 bf16) 走 SPLIT_D"不成立。
> `DetermineModeParameters` 的 SPLIT_D 条件是 `numCol > ubFactor`，而初始
> ubFactor（with beta）bf16=11264 / fp32=9216（无 beta 12288/10240）——
> 7168 远小于该值。实测（ASCEND_GLOBAL_LOG_LEVEL=1 采集 tiling 日志）：

| shape (hidden=7168) | tiling key | 变体 | kernel 文件 |
|---|---|---|---|
| bf16, rows≤40 | 33 | SINGLE_N | add_rms_norm_bias_single_n.h |
| bf16, rows≥41 | **30** | **NORMAL** | add_rms_norm_bias.h |
| fp16, rows≤40 | 13 | SINGLE_N | 同上 |
| fp16, rows≥41 | 14 | MULTI_N | add_rms_norm_bias_multi_n.h |
| col=15000（测试） | *1 | SPLIT_D | add_rms_norm_bias_split_d.h |
| col=8/16/128（测试） | *2 | MERGE_N | add_rms_norm_bias_merge_n.h |
| col=3000 bf16 rows≥41 | 34 | MULTI_N（isPerformance=1） | add_rms_norm_bias_multi_n.h |

**DeepSeek 生产热路径 = key 30（NORMAL，bf16）**：每层 decoder 必调
（`AscendRMSNorm.forward_oot`，residual 分支）。SPLIT_D 仅 numCol>11264 才触发。

NORMAL 变体结构（bf16 路径，每行全串行，单缓冲队列 BUFFER_NUM=1）：

```
每行: CopyIn(MTE2: x1→queue, x2→sqxBuf上半; V: cast×2, add, rint-cast=x输出,
            MTE3: x落盘)
      → Compute(V: mul平方, muls×avgFactor[全宽], reduce树, 标量链(4×1elem),
                V_S往返→GetValue→S_V往返→SetValue, muls×rstd[全宽],
                rint-cast, 重新widen, cast gamma[全宽], mul gamma, cast beta[全宽],
                add beta, rint-cast=y输出)
      → CopyOutY(MTE3: y落盘)
```

bf16 每行全宽 V pass 清单（15 遍）：

| # | pass | 备注 |
|---|---|---|
| 1 | Cast x1→fp32 | 必须（c220 无 bf16 元级运算） |
| 2 | Cast x2→fp32 | 必须 |
| 3 | Add x1+x2 (fp32) | 必须 |
| 4 | Cast RINT → x 输出 | 契约 |
| 5 | Mul sqx=x² | 必须 |
| 6 | Muls sqx×avgFactor（全宽） | **可折叠到 reduce 后 1 元素** |
| 7 | ReduceSumCustom（Add 树） | 必须 |
| 8 | 标量链 Adds/Sqrt/Dup(ONE)/Div | 1 元素，可忽略 |
| 9 | **V_S→GetValue→S_V→SetValue 标量往返** | **每行 2 次跨 pipe 同步，可消除** |
| 10 | Muls x_fp32×rstd（全宽） | 必须（可改 stride-0 广播 Mul，位级一致） |
| 11 | Cast RINT → y 中间 | 契约 |
| 12 | Cast 重新 widen | bf16 域乘法必需 |
| 13 | Cast gamma→fp32（**每行重复**） | **gamma 每核只载一次，cast 却每行做** |
| 14 | Mul ×gamma | 必须 |
| 15 | Cast beta→fp32（**每行重复**） | **同 13** |
| 16 | Add +beta | 必须 |
| 17 | Cast RINT → y 输出 | 契约 |

（13/15 与 rms_norm_cast 优化前同款异味；9 与 split_d 262-267 行同款——
该异味在 NORMAL/SPLIT_D/SINGLE_N 三个变体中都存在。）

fp16 NORMAL（key 10，仅非对齐中宽 col 触发）：每行 9 遍全宽 pass，同 9 号异味。
fp32 NORMAL（key 20/21）：gamma/beta 原生 fp32 无 cast，同 9 号异味。
MULTI_N（fp16 key14 / bf16 key34，双缓冲 inQueueX/outQueueY）：
**已实现 Gather+stride-0 广播**（rstd 向量域广播）且 gamma/beta cast 每 batch 一次
→ 是本仓内部的"正确答案"参考实现；bf16 未用它只因 tiling 把 bf16 排除在
MULTI_N 之外（`dataType == FP16 || isPerformance==1`）。

### 0.5 UB 预算盘点（bf16 NORMAL @7168）

当前 tiling 给 NORMAL 的 ub_factor=11264（未按 numCol 收紧，缓冲全按 11264 分配）：

| 缓冲 | 大小 |
|---|---|
| inQueueX/gamma/beta/outQueueY（bf16×11264） | 4×22.5KB = 90KB |
| xFp32Buf/sqxBuf（fp32×11264） | 2×45KB = 90KB |
| outQueueRstd/reduceFp32Buf | 0.5KB |
| **合计** | **≈180.5KB**（UB 191KB，余量 ~10KB） |

**tiling 把 ub_factor 收紧到 numColAlign=7168 后**：合计 ≈115KB，余量 ~76KB
→ 可容纳 gamma_fp32+beta_fp32 常驻（57.4KB），甚至 x 双缓冲。
（注意 x2 的 bf16 stash 在 sqxBuf 上半部，依赖 numCol ≤ ubFactor 不变量，
收紧后仍成立。）

### 0.6 "为什么慢"量化清单（bf16 NORMAL @2048）

1. **每行 2 次跨 pipe 标量往返**（V_S/S_V）→ scalar_wait 85%，发射线程无法
   预取下一行指令。
2. **3 遍冗余全宽 pass**（avgFactor 折叠、gamma/beta cast 上提）≈ vec 1.69µs/行
   中 ~0.34µs/行（20%）。
3. **行间零流水**：单缓冲队列使 MTE2 装载 i+1 行必须等 i 行 V 消费完，
   vec 空转 27%。
4. 落盘 x 需要先 rint-cast 一遍（bf16 域加法不可用所致），属算法必要开销，
   非冗余。

## 轮次计划

- Round 1: NORMAL 变体消标量往返 + 冗余 pass（tiling ub_factor 收紧为前置）— 纯
  等价变换（avgFactor 折叠仅 fp32 舍入路径 ~1ulp 差异，rms_norm_cast R1 先例）
- Round 2: NORMAL 行间流水（双缓冲/事件四件套，rms_norm_cast R2 方法论）
- Round 3: 视 profile 决定（SINGLE_N decode 路径 / fp16 MULTI_N / 其余变体对齐）
- Round 4: 结案，候选池全部评估（含负结果）

每轮闭环（构建陷阱与口径见 skill 与 rms_norm_cast 笔记）：
清 build 树 → build_aclnn.sh → pip install -e → md5 自检 →
pytest 126 全绿（硬门槛）→ `python benchmarks/add_rms_norm_bias.py` 记分板 →
msprof 归因 → 记录 → 提交。

流水图留档：凡行间流水轮次，前后各采 `--aic-metrics=InstrTimeline` trace.json
存 traces/（描述性文件名 + README 采集条件），并行度 = 三线忙碌和/墙钟。

## 轮次结果记录

（每轮：精度 / NPUGraph / msprof / 与上轮对比 / 教训）

### Round 1（2026-09-24）：NORMAL 消标量往返 + 冗余 pass —— 详设

对象：key 30（bf16 NORMAL，DeepSeek 生产热路径）；tiling 前置 + kernel 4 项。

1. **tiling（op_host/add_rms_norm_bias_tiling.cpp）**：NORMAL 落选其余分支后
   `ubFactor = numColAlign`。原来 hidden=7168+beta 按 11264 列分配缓冲（~180.5KB，
   接近 191KB 上限），收紧后 ~115KB，腾出 ~76KB。不变量校验：kernel 全部行内
   逻辑以 numCol 为界、x2 的 bf16 stash 位于 sqxBuf 上半部（依赖 numCol ≤
   ubFactor，收紧后仍成立；写读重叠逐元素分析 j=(i+ubFactor/2)/2 < i 仅当
   i > ubFactor，安全）。MULTI_N 的 rowFactor==0 回退 NORMAL 情形同表达式覆盖。
2. **gamma/beta cast 上提（bf16 独有）**：新增常驻 gammaFp32Buf/betaFp32Buf
   （ubFactor×4B ×2 = 57.4KB @7168），Process() 序言每核 Cast 一次；
   行循环内直接 Mul/Add fp32 缓冲。省 2 遍全宽 pass/行。
3. **avgFactor 折叠**：删除 reduce 前全宽 `Muls(sqx, ×avgFactor)`，改为 reduce 后
   1 元素 `Muls(sqx, sqx, avgFactor, 1)`。省 1 遍全宽 pass/行。舍入路径：
   mean = rawsum×(1/N) 与 golden 的逐元素 pre-mul 相比有 ~1ulp fp32 差异
   （rms_norm_cast R1 同款，容差内）。
4. **rstd 广播（全 dtype）**：`MulByRstd(dst, src, sqx, numCol)` helper：
   `Gather(rstd8, sqx, zeroOff×8, 0, 8)` 复制成 8 lane 块 →
   `Mul(dst, src, rstd8, 64, reps, {1,1,0, 8,8,0})` stride-0 块广播（MULTI_N
   同款、c220 官方先例同款）。**位级一致**：同一个 fp32 rstd 参与同样的 fp32
   乘法，只是操作数不再走 S pipe。每行消掉 S_V 同步 + 标量操作数编码；
   V_S/GetValue/SetValue 仅为 rstd 的 GM 输出累积保留（outQueueRstd 契约路径
   不动，零风险）。

预期：bf16 每行 15 遍全宽 pass → 12 遍（-20% vec 工作）；小 shape 增加一次
序言开销（每核固定）。UB 预算 bf16 @7168：115 + 57.4 ≈ 172.4KB < 191KB ✓；
fp32 @7168：~144KB ✓；fp16 NORMAL 中宽 col 更小 ✓。

风险点：
- Gather/Mul 广播是 MULTI_N 验证过的范式（fp16 生产路径在用），rstd8/zeroOff
  均 32B 对齐（独立 TBuf）。
- tiling ubFactor 收紧影响所有 NORMAL 实例（3 dtype × 各 shape），126 用例
  中 NORMAL 覆盖 = col 3000(fp16/fp32 rows≥41)、7168(bf16/fp32 rows≥41)。
- SINGLE_N/MERGE_N/SPLIT_D 本轮不动（SPLIT_D ubFactor 自算，不受影响）。

#### Round 1 结果 — ✅ 完成（2026-09-24）

- **精度**：`test_add_rms_norm_bias.py` **126/126 通过**（107s），device 日志无
  [ERROR] 行，零回退。
- **NPUGraph（µs，vs 基线）**：

| tokens | bf16 base→R1 | Δ | fp16 base→R1（未动参照） |
|---|---|---|---|
| 64 | 10.10→9.73 | -3.7% | 8.65→8.74（+1%，噪声） |
| 128 | 13.98→13.43 | -3.9% | 11.42→12.62（噪声带 ±1.2µs） |
| 512 | 33.16→30.84 | **-7.0%** | 21.50→21.53（0%） |
| 1024 | 59.04→54.20 | **-8.2%** | 35.36→35.28（0%） |
| 2048 | 119.41→109.34 | **-8.4%** | 62.95→62.80（0%） |
| 4096 | 250.89→231.14 | **-7.9%** | 171.68→166.63（-3%，噪声带） |

- **msprof@2048（40 核均值）**：wall 124.4→**114.7µs**（-7.8%，与 NPUGraph 一致）；
  vec busy 87.9→**77.8µs**（-10.1µs = 3 遍 pass 实际 ~0.19µs/行，低于 0.34 估算，
  因被删 pass 里 gamma/beta cast 部分与 MTE3 x 落盘重叠）；mte2 27.0→27.8、
  mte3 18.7→18.7、scalar_wait 106→96.4µs（结构未动，符合"纯减 vec 工作"定位）。
- **tiling 验证**：probe 实测 key 30 ub_factor 11264→**7168**；key 13/33（SINGLE_N）
  11264 不变、key 14（MULTI_N）7168 不变 ✓。
- **噪声口径发现**：未动代码的 fp16 路径在 128 档出现 +1.2µs（+10%）波动 →
  中小档噪声带 ~±1µs；判断收益以 ≥512 档为准（该段波动 <1%）。
- 结论：3 遍 pass 消除兑现 ~8%；剩余大头是行间串行（vec 空转 30%、
  scalar_wait 84%）→ Round 2 行间流水。

### Round 2（2026-09-24）：NORMAL bf16 行间流水 — ✅ 完成

对象：key 30 的 bf16 路径（生产热路径）。fp16/fp32 NORMAL 维持 Round-1 串行
实现（非生产 shape；生产 fp16 走 MULTI_N 双缓冲）。

**实现**（`add_rms_norm_bias.h`，方法论沿用 rms_norm_cast R2 已验证模板）：
- 槽位：x1[2]/x2[2] 双缓冲（局部行号奇偶索引），x1[slot] 兼作 x_out，
  x2[slot] 兼作 y 输出（省 y 缓冲，MTE3 不上 V 关键路径）；xf/sqx 单缓冲
  （仅 V 触碰，V 有序天然安全）。
- 事件：AllocEventID 四件套，三方向各单在途（MTE2_V 装载→V、V_MTE2 槽位
  释放→装载、MTE3_MTE2 落盘→装载）+ 每行一对 V_MTE3；wait 全部落在消费
  pipe 上（发射线程零阻塞）；发射条件与消费条件严格一致（`i+2 < rowWork`），
  退出零悬挂标志。
- gamma/beta bf16 暂存借道 row-0 的 x 槽（一个 V_MTE2 标志守卫槽位释放），
  不占常驻 UB。
- **rstd 全程 V 域**：reduce 后 1 元素 `Adds` 进 8-lane/行 累积缓冲（32B 对齐
  向量写）→ 每 rowFactor 行用 Gather 压缩（iota 字节偏移）→ 单次 DataCopyPad
  落盘。**kernel 内零 V_S/S_V**（Round 1 保留的 GetValue/SetValue 一并消除）。
- 序言外提 one8（Div 的 1.0 广播）。

UB 预算 @7168：x1[2]+x2[2]+xf+sqx+γfp32+βfp32 = 6×28.7KB + 固定 ~4.8KB
= **177KB < 191KB** ✓。

- **精度**：`test_add_rms_norm_bias.py` **126/126 一次通过**（110s），零挂死
  零 NaN，device 日志无 [ERROR]。
- **NPUGraph（µs，vs Round 1）**：

| tokens | bf16 R1→R2 | Δ vs R1 | 累计 vs 基线 | 备注 |
|---|---|---|---|---|
| 16 | 5.34→5.32* | 0%（SINGLE_N 未动） | 0% | *首轮 5.67 为噪声尖峰 |
| 64 | 9.73→10.11 | ≈持平（噪声带内） | ≈0% | 2 行/核，流水不回本 |
| 128 | 13.43→13.26 | -1.3%（噪声带内） | -5.1% | 4 行/核 |
| 512 | 30.84→28.43 | **-7.8%** | **-14.3%** | 13 行/核 |
| 1024 | 54.20→47.93 | **-11.6%** | **-18.8%** | 26 行/核 |
| 2048 | 109.34→86.67 | **-20.7%** | **-27.4%** | 52 行/核 |
| 4096 | 231.14→180.60 | **-21.9%** | **-28.0%** | 103 行/核 |

（fp16 金丝雀同轮波动 ±1.5%，中档判断以此为准；64 tokens=2 行/核时流水
收益与序言固定开销相抵，属预期，无回退。）

- **msprof@2048（40 核均值，孤立口径）**：

| 指标 | 基线 | R1 | R2 |
|---|---|---|---|
| wall | 124.4µs | 114.7µs | **86.4µs**（-30.5% vs 基线） |
| vec busy | 87.9µs (73%) | 77.8µs (70%) | **74.8µs (89.5%)** |
| mte2 busy | 27.0µs | 27.8µs | 26.0µs（全部藏于 V 之下） |
| mte3 busy | 18.7µs | 18.7µs | 13.5µs（同上） |
| **scalar_wait** | **106.0µs (85%)** | 96.4µs (84%) | **1.86µs (2%)** |

→ 流水目标达成：发射线程完全解放（scalar_wait 归零），三 pipe 重叠度
（26.0+13.5+74.8)/86.4 = 1.32 > 1。剩余 vec 空闲 10.5% ≈ ramp/tail/序言。
vec busy 74.8µs ≈ 12 pass × 0.113µs/行 × 52 行（模型吻合）→ 大 shape 唯一
剩余杠杆 = 减 pass 数（bf16 12 遍 vs fp16 MULTI_N ~6 遍）。
- **流水图留档**：本机 msprof 不支持 `--aic-metrics=InstrTimeline`（报
  "Unexpected argument"），对应指标名为 `TimelineDetail`；前后 trace 见
  traces/ 目录（README 注明采集条件与该差异）。

#### Round 2 流水图前后对比（traces/，simulator 指令级，core18，2048×7168 bf16）

| 指标 | R1（流水前） | R2（流水后） |
|---|---|---|
| wall | 101.4µs | **80.7µs（-20%）** |
| VECTOR 忙碌 | 98.2µs（96.9%） | 78.1µs（96.8%） |
| MTE2 忙碌 | 51.5µs（**50.8%**） | 76.0µs（**94.2%**） |
| MTE3 忙碌 | 80.6µs（79.5%） | 77.4µs（95.9%） |
| SCALAR 忙碌 | 91.3µs（**90.0%**） | 2.0µs（**2.5%**） |
| 三线并行度 | 2.27 | **2.87** |

解读：流水前 SCALAR 占 90%（串行编排全压在发射线程上）、MTE2 只有 51%
（装载被 V 串行化）；流水后 SCALAR 归零、MTE2 94%——装载/落盘完全与 V
重叠，墙钟 -20%。（simulator 绝对时长与真机 msprof 有差，结构性结论一致；
真机口径见上表 PipeUtilization 三轮对比。）

### Round 3（2026-09-24）：广度轮 —— 序言重叠 + MULTI_N 折叠 + SINGLE_N 消标量往返 — ✅ 完成

三个独立小改动，一轮闭环验证：

1. **NORMAL bf16 序言重排**（add_rms_norm_bias.h）：gamma/beta 的 MTE2 装载
   提前到 iota 预计算（64 次 Duplicate，~1.6µs V 发射）之前，两者重叠 →
   64-128 档（行数少、序言占比高）受益。
2. **MULTI_N avgFactor 折叠**（add_rms_norm_bias_multi_n.h，fp16+bf16 共用）：
   删除全宽 `Muls(sqx, ×avgFactor)`（calc_row_num×numColAlign），改为 reduce 后
   对 rstdLocal（8 lane/行，≤512 元素）做 Muls。fp16 生产路径（key 14）主收益。
3. **SINGLE_N 消 V_S/S_V + avgFactor 折叠**（add_rms_norm_bias_single_n.h，
   fp16/fp32/bf16 三路径）：V_S/GetValue/S_V → tmpLocal 尾部划出的 zeroOff +
   Gather + stride-0 广播 Mul（位级一致）；tmpLocal 是 reduce 工作区，此后已死。
   同时折叠 avgFactor 全宽 Muls（同 2）。decode 路径（≤40 行，1-4 tokens
   生产 decode 档）主收益。

- **精度**：`test_add_rms_norm_bias.py` **126/126 通过**（127s），零回退。
- **NPUGraph（µs，vs Round 2）**：

| tokens | bf16 R2→R3 | fp16 R2→R3 |
|---|---|---|
| 1 | 3.43→**3.34**（-2.6%） | 2.63→2.60 |
| 4 | 3.84→**3.69**（-3.9%） | 2.89→2.86 |
| 16 | 5.32→5.36（噪声） | 4.53→4.50 |
| 64 | 10.11→10.08（持平） | 8.74→8.75（持平） |
| 512 | 28.43→**27.73**（-2.5%） | 21.53→**20.60**（-4.3%） |
| 1024 | 47.93→**47.07**（-1.8%） | 35.28→**33.43**（-5.3%） |
| 2048 | 86.67→**85.93**（-0.9%） | 62.80→**59.55**（-5.2%） |
| 4096 | 180.60→180.26（持平） | 166.63→171.98（噪声带） |

各档收益与预期方向一致：bf16 decode 档吃 SINGLE_N 改动，fp16 大档吃
MULTI_N 折叠，bf16 大档持平（流水后序言已摊薄）。

### Round 4 结案（2026-09-24）：候选池全部评估完毕，优化收官

**候选池逐项结论（含负结果，避免未来重复推导）**：

1. **bf16 NORMAL 再减 pass**：13 遍全宽 pass 全部承载语义——x1/x2 widen
   ×2（c220 无 bf16 元级加法）、x_out RINT（契约）、平方、reduce 树、×rstd
   广播、y-mid RINT（契约：y = bf16(x_norm)·γ+β 的中间舍入）、re-widen
   （γ/β 必须在 fp32 域运算）、×γ、+β、y RINT。c220 无点积 reduce、无 bf16
   元级运算 → **算法层下限，负结果**。
2. **rstd×γ 折叠**（消 y-mid RINT + re-widen 两遍）：会改变数值路径
   （bf16(x·rstd)·γ ≠ bf16(x·rstd·γ)），违反契约。**死路**。
3. **R=2 双行批处理**：需 x1[2]×2 行 + x2[2]×2 行，+57KB → ~234KB > 191KB。
   **被 UB 阻断**。
4. **预取深度 2（x 三槽）**：真机 mte2 busy 仅占墙钟 30%，装载不是约束；
   加深预取只治 ramp/tail（~10%）。rms_norm_cast R4 同结论。**无收益**。
5. **SPLIT_D 变体优化**（其 V_S/GetValue 异味更重，按 (行,chunk) 重复）：
   SPLIT_D 仅 numCol > 11264 触发，当前无生产 shape（DeepSeek hidden=7168；
   测试 col=15000 覆盖其正确性）。优化无用户价值，风险/收益不匹配。
   **留档不做**（若未来出现超宽 hidden 模型，按 Round 1/2 方法移植）。
6. **4096 档背靠背 HBM 争抢**：NPUGraph 177-180µs vs 孤立 ~172µs，前核
   写出排空与下一 kernel 加载争抢（系统级效应），参照算子同样受影响
   （ref@4096 320-325µs）。**kernel 层无解，不追**。
7. **SINGLE_N 深度优化**：装载已按数据流最大重叠、V_S 已消，剩余为
   延迟链，上限 ~0.2µs。**收益不成比例，留档**。

**最终记分板**（NPUGraph µs，hidden=7168，基线 → 三轮后终值；终值取
Round 3 两次稳定运行）：

| tokens | bf16 | Δ | fp16 | Δ |
|---|---|---|---|---|
| 1 | 3.36→3.35 | -0.3% | 2.60→2.57 | -1.2% |
| 4 | 3.70→3.68 | -0.6% | 2.93→2.86 | -2.2% |
| 16 | 5.32→5.34 | +0.3%（噪声） | 4.55→4.50 | -1.1% |
| 64 | 10.10→10.10 | 0% | 8.65→8.87 | +2.5%（噪声带） |
| 128 | 13.98→13.03 | **-6.8%** | 11.42→11.86 | +3.9%（噪声带*） |
| 512 | 33.16→27.66 | **-16.6%** | 21.50→20.61 | **-4.2%** |
| 1024 | 59.04→47.08 | **-20.3%** | 35.36→33.47 | **-5.3%** |
| 2048 | 119.41→85.93 | **-28.1%** | 62.95→59.53 | **-5.4%** |
| 4096 | 250.89→178.77 | **-28.8%** | 171.68→177.07 | +3.1%（噪声带） |

\* fp16 @128 基线读数偏低（未动代码的 R2 复测 12.62、R3 12.01，同日同
条件配对 -4.8% 才是该档真值；基线日的 11.42 与 4096 档 ±3-5% 波动同源）。

vs unfused ref：bf16 @2048 快 **30%**（85.9 vs 122.5-124.3），@1024 快 25%；
fp16 @2048 快 19%。基线时 bf16 @1024/2048 仅快 6-6.5% —— 差距已拉开。

**最终态（2048×7168 bf16，40 核均值）**：wall 124.4→86.4µs（msprof 孤立），
vec busy 89.5%（结构上限 = ramp/tail/序言 ~10%）、scalar_wait 2%、
MTE2/MTE3 全隐藏（指令级三线并行度 2.27→2.87）；精度 126/126 零回退
（每轮闭环验证，共 3 次全量回归全绿）。

**方法论文档**：traces/README.md（流水图采集条件 + InstrTimeline→
TimelineDetail 差异）、本文件各轮详设/结果/负结果。
