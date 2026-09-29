# dequant_swiglu_quant 优化记录（dequant-swiglu-quant-perf）

> 910B3 (dav_c220, 40 AIV, UB 192KB/核), CANN 9.1.0。
> 方法论沿用 rms_norm_cast（910B_perf 分支）与 add_rms_norm_bias
> （add-rms-norm-bias-perf 分支）两役：事件四件套、退出标志清零契约、
> Gather+stride-0 广播、严格断言先行、未动路径作噪声金丝雀。

## 第 0 步：Phase A 审计与测量（2026-09-29，零代码改动）

### 0.1 算子契约与生产画像

反量化 + SwiGLU + 动态量化三合一（W8A8 int32 GEMM 输出 → int8 下一层输入）：

```
f     = fp32(x_int32) × weight_scale[2H] × activation_scale[row]   # 反量化
gate  = f 左半（activate_left）/ up = f 右半
swiglu_mode=1（SwiGluGate）: gate/up 各自 clamp(limit) → up += β →
             silu(α·gate) × up                       # gpt-oss 风格,α/β/limit 可为 0
swiglu_mode=0（普通）   : silu(gate) × up
y     = int8( swiglu / (max|swiglu|/127) )           # 动态量化（逐行 scale）
scale = max|swiglu|/127 (fp32, 每行)
```

数值链（CastFloatToInt8）：fp32 → Cast RINT → int32 → SetDeqScale(1.0) Cast
ROUND → half → Cast TRUNC → int8。half 中间步对 |v|≤127 的值无损。

### 0.2 变体分派矩阵（静态走读 + 设备实证）

tiling 模板按优先级注册：DskTiling(0, `dequant_swiglu_quant_tiling.cpp`)
→ DequantSwigluQuantTiling(1, `..._tiling_base.cpp`) → arch35 两类(1000/2000,
910_93 专用)。**DskTiling 的 IsCapable = group_index 存在，或 (x=int32 且
swiglu_mode=1)**——两个生产调用点全部命中它，即**旧基类
`DequantSwigluQuantBase`（dequant_swiglu_quant.h, 817 行）+ cut_group 变体**：

| 调用方 | 配置 | tiling key（设备实证） | blockDim | kernel |
|---|---|---|---|---|
| **shared_experts.py:396**（DeepSeek 每层，prefill+decode，shared-expert 流） | x=int32 [T,2H], 无 group, mode=1, clamp=0(V3)/7.0(V3.2), dynamic, 无 bias/smooth | **200000000** | 36 | DequantSwigluQuantBase<float,float,float,int32_t> |
| w8a8_dynamic.py:456（MC2 decode 回退路径：fusion 关闭时） | x=int32, group=int64 每组行数, mode=0/1, dynamic | **100000000**（普通）/ **110000000**（cut：组数≥32 且均摊≤16 行） | ≤36 / 40 | Base / DequantSwigluQuantGroup（cut_group.h） |
| （无生产调用方）bf16/fp16 x 直入、int32+无 group+mode0 | 10000-30013 | — | 10 个变体 hpp（dynamic/static × base/bf16/bias_float/bias_int32/performance）+ apt.cpp(arch35) | **非生产路径，只保回归** |

设备实证方式：msprof `OpBasicInfo.csv` 的 kernel 名后缀即 tiling key
（`DequantSwigluQuant_5aec..._200000000` / `..._110000000`）。本机
DumpTilingInfo 的 OP_LOGI 不可见（装好的 tiling 库未输出到 plog），以 msprof
为 key 判据。编译产物核对：int32 编译单元含 35 个 key（10004-10011, 30001-30008,
30013, 1e8/1.1e8/2e8 系全量），fp16/bf16 单元各 4/7 个。

**注意事项（探针踩坑，已留档）**：group 路径要求 weight_scale 为 2D
[groupNum, 2H]；group_index 语义是**每组行数**（cumsum_group_list(..., 1) 的
dst_list_type=1），不是 cumsum——传 cumsum 会导致 kernel 按递增值当行数累加
出界，aivec 崩溃（MTE 非法 GM 地址），表象像 kernel bug 实为调用契约。

DeepSeek V3 生产 shape：shared experts 2H=4096/TP 度（TP1=4096, TP8=512）；
MoE w1 输出恒 2H=4096（moe_intermediate=2048，EP 不切 intermediate）。
swiglu_limit：V3 无（0.0），V3.2 类 dst-quant 7.0。

### 0.3 精度基线 — ✅ 全绿

`pytest tests/e2e/nightly/single_node/ops/singlecard_ops/test_dequant_swiglu_quant.py`
→ **5/5 通过**（`--noconftest`；conftest 因环境 vllm 0.29 vs 分支期望 0.30 的
`vllm.v1.attention.backends.mla.index_group` 缺失而 ImportError，与本算子无关，
op 级测试自包含。GitHub 恢复后需补 fetch vllm v0.30.0 (ced6857a) 修正环境）。

覆盖缺口（Phase B 严格测试要补）：现有 5 用例全部是 config A（无 group、
mode=1、clamp=0、x=[4608,2048] 等）；**group 路径、mode=0、clamp>0、bias、
smooth scale、static 模式均无覆盖**。

构建自检：build 树三份 kernel 源 md5 与仓内一致（安装二进制=基线）。

### 0.4 性能基线（NPUGraph，`benchmarks/dequant_swiglu_quant.py`，µs）

**Config A（shared experts 主路径，mode=1，clamp=0）：**

| tokens | 2H=4096 | ref | 2H=2048 | 2H=1024 | 2H=512 |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.33 | 8.50 | 3.13 | 3.13 | 3.12 |
| 4 | 3.95 | 19.07 | 4.15 | 3.37 | 3.17 |
| 16 | 4.58 | 31.16 | 4.12 | 4.12 | 4.43 |
| 64 | 9.22 | 51.82 | 5.80 | 5.08 | 4.82 |
| 128 | 11.44 | 55.69 | 8.34 | 6.03 | 5.21 |
| 512 | 20.98 | 87.22 | 14.15 | 11.21 | 7.91 |
| 1024 | 32.96 | 108.34 | 20.19 | 14.31 | 11.24 |
| 2048 | **57.06** | 153.09 | 32.27 | 20.32 | 14.32 |
| 4096 | **106.01** | 352.66 | 56.42 | 32.96 | 20.64 |

ref = eager 反量化+SwiGLU+`npu_dynamic_quant`（bf16 舍入后量化，与 kernel
契约一致）。fused 全档快 2.2-8.1×，无 rms_norm_cast 那种"反超"。

**Config B（MC2 group 路径，mode=0，2H=4096）：**

| 配置 | fused µs | ref µs | key |
|---|---:|---:|---|
| g256 / T512（decode 型，均摊 2 行/组） | 35.67 | 76.54 | 110000000 (cut, 40 核) |
| g256 / T2048 | 61.37 | 148.53 | 110000000 |
| g64 / T64 | 16.67 | 35.65 | 100000000 |
| g64 / T512 | 23.91 | 80.93 | 100000000 |
| g64 / T2048（均摊 32 行/组 > 16） | **93.01** | 160.83 | 100000000 (**不 cut → 负载不均**) |

### 0.5 pipe 基线（msprof op PipeUtilization）

**@2048×4096 config A（key 200000000，36 核均值；NPUGraph 57.1 / msprof 59.1 一致）：**

| 指标 | 值 | 占墙钟 59.1µs |
|---|---:|---:|
| aiv active | 55.6µs | 94% |
| **vec busy** | **47.3µs** | **80%**（占 aiv 85%） |
| mte2 busy | 15.7µs | 28%（active BW 57.5GB/s/核） |
| mte3 busy | 5.7µs | 10% |
| scalar busy | 16.5µs | 28% |
| **scalar_vector_stall** | **42.5µs** | **72%** —— 发射线程被 V 串行链顶住 |
| 有效带宽 | 37.75MB ÷ 59.1µs | **639 GB/s ≈ 名义 1.6TB/s 的 40%** |

→ **vec 吞吐瓶颈**（非带宽墙）：20 遍全宽 pass × 单缓冲 tile 串行。
理论地板：37.75MB ÷ 1.6TB/s = 23.6µs（@可实现 1.36TB/s = 27.8µs），
当前 57-59µs，**带宽侧留有 ~2× 空间**。

**@512×4096 g256 config B（key 110000000，40 核均值，墙钟 36.4µs）：**

| 指标 | 值 | 占墙钟 |
|---|---:|---:|
| scalar busy | 23.8µs | **65%（占 aiv 80%）——发射/参数装载瓶颈** |
| vec busy | 10.7µs | 29%（占 aiv 37%） |
| mte2 busy | 7.9µs | 22% |

→ 每组 2 行的小 tile：每组付出整套 param 装载（weight scale 16KB MTE2）+
队列同步 + 单 tile 计算；scalar 指令占满。vec 大量空转。

### 0.6 Pass 清单（config A，每行，H=2048；mode=1，clamp=0，无 bias/smooth）

| # | 操作 | 宽度(元素/行) | 判定 |
|---|---|---|---|
| 1 | CopyReshape weight_scale [1,2H]→[p,2H] | 4096 | **每 tile 重复，组内不变 → 可上提每核一次** |
| 2 | Cast x int32→fp32 | 4096 | 必须（反量化语义） |
| 3 | Mul ×weight_scale | 4096 | 必须 |
| 4 | CopyReshape act_scale [p]→[p,2H] | 4096 | 逐行值广播，repeat 结构不匹配 stride-0 Mul，保留 |
| 5 | Mul ×act_scale | 4096 | 必须 |
| 6 | Copy act 半边（去交错） | 2048 | **可用 strided 视图(repStride 2H/8)直读原位，消除** |
| 7 | Copy gate 半边（去交错） | 2048 | **同 6** |
| 8 | Adds glu_bias（**β=0 时也在跑**） | 2048 | **β=0 时数值恒等（±0 差异不影响 int8），可跳过** |
| 9 | Muls −gluAlpha | 2048 | 必须 |
| 10 | Exp | 2048 | 必须 |
| 11 | Adds 1.0 | 2048 | 必须 |
| 12 | Div | 2048 | 必须 |
| 13 | Mul gate×silu | 2048 | 必须 |
| 14 | Abs | 2048 | 必须 |
| 15 | ComputeReduceMax（逐行 Max 树） | ~2048 | 必须（可减发射次数） |
| 16 | WholeReduceMax / Muls(1/127) / Brcb | 窄 | 必须 |
| 17 | Copy scale 广播 [p,8]→[p,H] | 2048 | 逐行值广播，保留 |
| 18 | Div ÷scale | 2048 | 必须 |
| 19 | Cast fp32→int32 (RINT) | 2048 | 必须 |
| 20 | Cast int32→half (ROUND, deq 1.0) | 2048 | **待验证：int32→int8 直转是否等价**（half 步对 ≤127 无损） |
| 21 | Cast half→int8 (TRUNC) | 2048 | 必须 |
| | **合计** | **≈45,056 元素操作/行** | 可省 ≈10,240（#1/6/7/8，**23%**，上界） |

外加每 tile 固定开销：44 处 PipeBarrier、4 个 TQue 的 Alloc/EnQue/DeQue/Free
（xActQueue **DB_BUFFER=1 单缓冲**）、SetMaskCount/SetVectorMask/ResetMask ×6。

### 0.7 UB 预算（config A @2H=4096, mode=1, ubFactorDimx=2）

| 缓冲 | 大小 |
|---|---|
| xActQueue 单缓冲（x int32 [2,4096] + act scale [2,8]） | 65.7KB |
| weightScaleQueue [1,4096] fp32 | 16.4KB |
| inScaleQueue [1,2048] fp32（dynamic） | 8.2KB |
| outQueue（y int8 [2,2048] + scale [2] + 32B） | 4.1KB |
| tmpBuf1 [2,4096] fp32（act+gate 工作区） | 32.8KB |
| scaleBuf | 0.25KB |
| **tmpBuf2（死分配，全仓无任何使用）** | **20.5KB** |
| **合计** | **≈148KB / 191KB**（余 43KB） |

tiling 公式按 db=2 预留 xActQueue 预算但代码只建 1 缓冲——**双缓冲的空间
早已在公式里**，删掉死分配 tmpBuf2 后余量更足。

### 0.8 "为什么慢"量化清单（config A @2048×4096）

1. **~23% 的 vec 元素操作可等价消除**（#1 上提 / #6,7 去交错消除 / #8 跳过）
   ——这是 vec busy 47.3µs 中 ~10µs。
2. **tile 间零流水**：xActQueue 单缓冲，每核 29 个 tile（ubFactorDimx=2）
   串行走 MTE2→V→MTE3；scalar_vector_stall 72% 即发射线程逐 tile 停等。
   vec 只占墙钟 80%，20% 是 tile 边界串行 —— 双缓冲/事件四件套可回收大部分。
3. **36 核上限**（PERFORMANCE_CORE_NUM=36，40 核机器闲 4 核）≈ -10% 吞吐。
4. **尾部不均**：2048 行 = 57×35+53，最后核少 4 行（~7% 尾部空转，与 3 叠加）。
5. （config B 独有）**非 cut group 路径负载不均**：每组都从 core 0 重新分派，
   g64/T2048 = 93µs vs g256/T2048 = 61µs 同工作量 —— 1.5× 差距实证。
   cut 条件（组数≥32 且均摊≤16 行）之外的 prefill 型 group 形态全部受害。
6. （config B 独有）每组 param 重复装载 + 队列同步，scalar 80%。

### 0.9 Go/No-Go 判定 — **GO**

门槛是"profile 显示 ≥10% 可兑现空间"，实测远超：

| 杠杆 | 预期（保守） | 风险 |
|---|---|---|
| R1 等价 pass 消除（#1/6/7/8） | vec -15~20% → 墙钟 **-8~12%** | 低（#6/7 需逐位验证；#8 β=0 有 ±0 边界证明） |
| R2 tile 间双缓冲流水（事件四件套） | **-10~15%**（回收 20% 串行空转的大半） | 中（沿用已验证模板：局部行号奇偶、退出零悬挂） |
| 36→40 核 + 尾部均衡 | 大档 **-7~10%** | 低（tiling 一行改动；需验证为何有 36 上限） |
| 死分配 tmpBuf2 删除 | UB +20KB（为 R2 铺路） | 零 |
| group 路径分派修复（0.8#5） | 受害 shape **-30%+** | 低（分派逻辑改动，kernel 不动） |

主路径（shared experts）三项叠加保守估计 **-20~30%**（57→40-46µs @2048），
且不触碰 HBM 墙（当前仅 40% 带宽利用率）。精度路径：#1/6/7/8 全部为位级
等价或舍入恒等变换，不改数值链。

### 环境备注（本役）

- 宿主 loadavg ~33-45（<100，数据可信）。
- GitHub 断连中：vllm v0.30.0 (ced6857a) fetch 失败，环境停在 v0.29.0
  → conftest ImportError，op 级测试用 `--noconftest`（自包含不受影响）；
  恢复后补齐。
- msprof：kernel-name 需用通配 `DequantSwigluQuant*`（符号带 hash+key 后缀），
  OpBasicInfo.csv 的 kernel 名即 tiling key 实证来源。

## 轮次计划（Phase B/C）

- **R0（前置）**：严格精度测试 `test_dequant_swiglu_quant_strict.py`
  （参照 add-rms-norm-bias 的 strict：y=1 atol（int8 量化逐位）或 ulp 级、
  scale fp32 1e-5、按 dtype/极值/边界 shape/拒绝路径分级；补 group 路径、
  mode=0、clamp>0、bias、smooth、static 的覆盖）——**先有裁判再动手**。
- **R1**：等价 pass 消除（#1 ws 上提 / #6,7 strided 视图 / #8 β=0 跳过 /
  tmpBuf2 删除）。
- **R2**：tile 间双缓冲流水 + 40 核/尾部均衡（视 R1 后 profile）。
- **R3**：group 路径分派修复 + config B 专项（param 装载合并/每组开销）。
- **R4**：结案，候选池全部评估（含负结果）。

每轮闭环：清 build 树 → build_aclnn.sh → pip install -e → md5 自检 →
精度全绿（原 5 + strict）→ 记分板 → msprof 归因 → 记录 → 提交。

## 轮次结果记录

（每轮：精度 / NPUGraph / msprof / 与上轮对比 / 教训）

### R0（2026-09-29）：严格测试先行 — ✅ 完成

`test_dequant_swiglu_quant_strict.py`（40 用例，不改原文件）：golden 按 kernel 精确
fp32 数值链（(ws×x)×act → SwiGLU(α/β/clamp) → 行max×1/127 → 除 → RINT → int8，
**无 bf16 预舍入**——原测试 golden 把 swiglu 转 bf16 再量化，atol=1 恰好把
per-row/per-group scale 类缺陷全部吸收）。y=atol 1/rtol 0（c220 向量 Exp 与 CPU
exp 的 ~1-2ulp 差在 RINT 边界翻转一格），scale=rtol 1e-5。补齐 group 路径
（ragged/空组/cut/mode1+group/>16 行组）、mode0、clamp>0、α/β、activate_left
翻转、act=None、2D ws、tile 尾、36 核边界、极值、全负行、9 条拒绝路径、
零行钉死（scale==0, y==0，与 CANN npu_dynamic_quant 一致）。

基线：**strict 40/40 + 原 5/5 全绿**。教训两条：
1. 模块级 `enable_custom_op()` 忘调用 → 全部用例 AttributeError，拒绝路径用例
   反而"通过"（任何异常都算拒绝）——拒绝路径必须与正常路径分开验证；
2. 给 op 传 CPU 张量（group_index 忘 .npu()）→ 设备端 "scalar instruction
   accesses an invalid GM address" 崩溃，表象像 kernel bug 实为 host 指针当 GM。

### 侦察期假警报（留档避免重蹈）

- **"act scale 被乘两次"**：uniform scale 探针 s_out 随 s² 变化（136×/13399×）
  报警——实际 **SwiGLU 本身双线性**（silu(s·gate)×(s·up)∝s²），136=100×
  sigmoid 饱和因子，属正确行为。golden 对比 5 个 shape 全部 maxdiff=1 证实。
  strided CopyRepeatParams（srcStride=0 + [p,8] scale 布局）语义可疑但实证逐行
  正确——**不动该路径**。
- **int32→int8 直转 cast**：c220 classic API 无此类型对
  （`CastIntrinsicsImpl` 无匹配重载，编译期失败；仅 MicroAPI 有
  `castTraitU32toU8Even`，见 swiglu_group_quant）。RINT 后值域 [-127,127] 使
  half 中转数学上无损，但无法跳过——**负结果，3-cast 链是 classic API 下限**。

### R1（2026-09-29）：等价 pass 消除 — ✅ 完成

对象：`DequantSwigluQuantBase`（两生产路径共用）。四项等价变换 + 一项门槛修正：

1. **死分配 tmpBuf2 删除**（swigluMode=1 时分配 5B/输出元素却全仓无引用）：
   mode1 @2H=4096 腾出 20.5KB UB（为 R2 双缓冲铺路），零风险。
2. **weight scale 行乘**：[1,2H] 行均匀 → 逐行 `Mul(x_row, x_row, ws, 2H)`，
   消除每 tile 的 [p,2H] 广播拷贝（位级一致：同操作数同乘法）。
3. **去交错拷贝消除**：SwiGluGate（mode1）/ComputeSwiGLU（mode0）不再把
   act/gate 半边拷到连续缓冲，clamp/β/silu 直接在 x 缓冲的半区上原地进行，
   仅最终 Mul 物化 [p,H] 结果供量化段（res=[0,pH), denom=[pH,2pH)，位级一致）。
4. **β=0 跳过**：`Adds(gate, β)` 在 gluBias==0 时跳过（x+0.0 恒等，-0.0→+0.0
   翻转不影响 int8 乘积）。
5. **门槛修正（R1b）**：初版门槛 `proDimsx ≤ 8` 使 2H=2048（p=5）回退
   +3.5~8.6%、cut-group decode（每组单 tile）回退 +5.8%——per-row 形态用少量
   额外指令换向量工作量，只在 **vec-bound** 形状赚回。修正为
   `rowPath_ = (UbFactorDimx ≤ 4) && (ubDimxLoop ≥ 2)`（单 tile 核 = decode 型
   issue-bound，回旧全宽路径）。

- **精度**：原 5 + strict 40 = **45/45 两轮全绿**（门槛修正前后各一轮）。
- **NPUGraph（最终版，µs；仅采信跨测量时段稳定的读数）**：

| shape | base | R1 | Δ |
|---|---:|---:|---:|
| 2H=4096 @1024 | 32.96 | 31.86 | -3.3% |
| 2H=4096 @2048 | 57.06 | 53.33 | **-6.6%** |
| 2H=4096 @4096 | 106.01 | 98.46 | **-7.1%** |
| 2H=2048 @4096（门槛修正后回旧路径） | 56.42 | 56.35 | -0.1%（修正前 +3.5% 已消除） |

大档收益随 tokens 单调增大（-3.3→-7.1%），与 vec-bound 摊销模型一致；
三轮独立测量时段（loadavg 45/80/92）读数稳定复现。

- **测量环境警告（2026-09-29 下午）**：共享宿主 loadavg 从 45 持续升至 96+，
  同一 shape 连续三次测量 37.7→41.3→43.9µs 单调漂移 +17%——小/中档与
  group 档读数不可判（old-path 金丝雀形状代码与基线逐分支等价也出现
  +6~14% 散布）。**小档与 group 档的 R1 判定留待环境安静后复测**；R2 开工
  前先复测全表作为 R2 的对照基线。
- **教训（门槛）**：per-row 等价变换不是免费午餐——它用少量指令数换向量
  工作量，只在 vec-bound 形状赚回。**门槛判据要用运行时形态（ubDimxLoop）
  而非静态 tile 参数**：单 tile 核（decode 型小 group）是 issue-bound，
  +3~5 条指令直接上关键路径（cut-group decode 实测 +5.8%）。
- **负结果**：int32→int8 直转（见上）；初版门槛 proDimsx≤8（2H=2048 p=5
  回退 +3.5~8.6%，已修正）。
