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

- Round 1: 省掉上述 2 遍 vec pass（纯减 vec 工作，无流水改动）
- Round 2: bf16 路径单舍入重构（对齐 fp16 路径：x*rstd/gamma 全在 bf16 域做，最后 widen 一次）
  → 契约 `y_fp32 == y.float()`（atol=0）保持，容差 vs npu_rms_norm 2e-2 内
- Round 3: 行间双缓冲流水（MTE2 预取下一行 / MTE3 异步落盘，消除 pipe 空转）
- Round 4: 视 profile 决定（DataCopyPad→DataCopy 对齐路径 / reduce 结构 / 收尾全量回归）

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

### Round 2：bf16 单舍入重构（再省 1 遍 pass，有明确定义的数值变化）

- bf16 分支改为与 fp16 路径同构（在低精度域做 scale/mul）：
  `Muls(x_local, x_local, rstd)`（bf16）→ `Mul(x_local, gamma_local)`（bf16）→ widen。
- 移除 `gamma_fp32_buf_`，tiling 回到 12B/col。
- 数值影响：y 由 bf16(x·rstd·γ)（单舍入）变为 bf16(bf16(x·rstd)·γ)（双舍入），
  多 ≤0.5 ulp（~0.4% 相对），golden 容差 2e-2 覆盖；fp16 路径现状本就是双舍入（行为对齐）。
  `y_fp32 == y.float()` 精确契约不变（仍从最终 bf16 y widen）。
- 预期：9→8 遍 → 2048 tokens ~89→~83µs。

### Round 3：行间双缓冲流水（消除 pipe 空转）

- 现状：每行 MTE2→V→MTE3 串行，等 load、等 store。
- 方案：x 载入改 `TQue<VECIN, depth=2>` 双缓冲，输出对 (y, y_fp32) 经
  `TQue<VECOUT, depth=2>` 异步落盘；EnQue/DeQue 自动管理 MTE2_V/V_MTE3/MTE3_MTE2 事件，
  行 i+1 的 MTE2 与行 i 的 V/MTE3 重叠。
- UB 预算（Round 2 后无 gamma_fp32）：x×2 28K + gamma 14K + x_fp32×2 56K + work 28K
  + reduce 0.25K ≈ 126KB ✓
- 预期：逼近 max(vec, mte) 带宽下限，2048 tokens ~83→~72-76µs（累计约 -28%）。

### Round 4：视 profile 收尾（候选池）

- ReduceSumCustom 树形结构优化（Add 链 112 次 repeat）
- DataCopyPad vs 对齐 DataCopy 路径
- 小 shape（decode 1-16 行）单核串行链缩短
- 原则：任何 shape 不回退；以 msprof 数据决定取舍。

### 风险与回退

- 每轮独立提交、独立可验证；精度门槛 = `test_rms_norm_cast.py` 8 用例全过（零回归）。
- Round 2 的双舍入变化是唯一数值行为改动，单独成轮便于归因/回退。
- tiling 12→16B/col 使支持的最大 hidden 从 ~16.3K 收缩到 ~12.2K（当前唯一调用方
  DeepSeek v4/v41 hidden=7168，无影响）；Round 2 后恢复。

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
