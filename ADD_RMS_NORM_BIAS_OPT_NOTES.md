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
