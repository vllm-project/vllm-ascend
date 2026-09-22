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

## 每轮工作流（精度是硬门槛）

```bash
# 0. Round N：改 csrc/moe/rms_norm_cast/op_kernel/*.cpp|*.h

# 1. 编译（CANN 9.1.0 下零补丁直接过）
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

> msprof op 注意事项：
> - 只保留 `--kernel-name` / `--output`（不加 --launch-count/--warm-up），`--aic-metrics=PipeUtilization` 按需选；
> - 被测命令作为位置参数直接跟在选项后面（不要用 `--application=python3 script.py`，参数传不进去）；
> - shape 用位置参数传给 runner（环境变量穿不透 msprof 启动层）。
