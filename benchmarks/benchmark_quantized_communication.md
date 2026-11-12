# BF16 / INT8 / MXFP8 同批量化通信 benchmark

[benchmark_quantized_communication.py](benchmark_quantized_communication.py) 在同一次 `torchrun`、同一组 rank、同一个默认 HCCL 通信域和同一份每 rank 输入上测量所有方案，避免用不同批次的 BF16 基线比较两条量化路线。它复用现有独立 benchmark 算法，不导入 vLLM、不新增内核，也不修改运行时路由。

默认对照如下，共 11 个方案：

| 方案 | 含义 |
| --- | --- |
| `bf16` | 普通 BF16 SUM AllReduce |
| `int8_full` | 整张量共享逐通道 scale，INT8 SUM，BF16 反量化与写回 |
| `int8_serial_chunks_2/4` | 同一份整张量 scale，2/4 块依次量化、SUM、反量化写回 |
| `int8_pipeline_chunks_2/4` | 同样的分块与 scale，两块预取，异步 SUM 与计算重叠 |
| `mxfp8_full` | 整块 MXFP8 AllToAll、BF16 反量化/局部 SUM、BF16 AllGather |
| `mxfp8_serial_chunks_2/4` | 2/4 块串行执行完整 MXFP8 路径 |
| `mxfp8_pipeline_chunks_2/4` | 相同分块，两块预取及显式异步 Work 依赖 |

INT8 的每次调用只有一次整张量 amax/MAX，`max_q=127//TP`；所有块复用 scale。MXFP8 的数据和每 32 元素尺度按现有字节布局传输。两条路线均反量化为 BF16，流水使用 `lookahead=2`。所有变体原地更新独立工作张量，每次调用前从相同原始输入重置；这是算法 benchmark 口径，运行时独立返回值和显存可能不同。INT8 流水仅作为 benchmark 对照，不恢复已撤回的运行时接入。

## 环境与正确性

默认两条路线一起测，目标环境为 A5，需要匹配的 PyTorch / torch_npu、`npu_quantize`、`npu_anti_quant`、`npu_dynamic_mx_quant` 和 `npu_anti_mx_quant`。无需 ops-transformer 的原生 QuantAllReduce。某个算子不可用时直接报错，不静默替换实现。A3 可用 `--backends int8` 单独测 INT8；没有 MXFP8 的目标环境不能用来比较两条路线。

先从仓库根目录运行小 shape 正确性检查：

```bash
export HCCL_NPU_SOCKET_PORT_RANGE="61000-61050"
PYTHON=python3

"$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=2 \
  benchmarks/benchmark_quantized_communication.py \
  --tokens 136 --hidden-sizes 5120 --chunks 2 4 --check-only \
  --output results/quantized_communication_check_tp2.json
```

136 行适用于 TP2/4/8，可覆盖 MXFP8 在 TP4/TP8 下的不均匀尾块。MXFP8 要求总行数被 TP 整除，并至少包含 `chunks` 个 TP 行单位；H 必须是 64 的倍数。INT8 没有按 TP 整除行数的要求，可用 `--backends int8 --tokens 129` 独立检查奇数行。

脚本自动执行零值、正负同号边界、抵消、rank 常数、跨 rank/通道/块离群值，以及 MXFP8 的 rank/token/block/chunk 字节路由检查。每个正式 shape 还执行：

- 统一对原始 BF16 输入做 FP32 SUM，报告各方案相对 L2 和最大绝对误差。
- INT8 校验量化安全范围、分块与整块量化字节一致、INT8 SUM 与 INT32 SUM 精确一致，再检查 BF16 输出。
- MXFP8 独立解码 E4M3/E8M0 字节，检查本地反量化、同 payload 通信和完整路径，确认分块量化字节与整块一致。
- 报告两条路线各变体相对自身整块输出的最大绝对差异。参考计算中的 FP32 和 CPU 解码只用于精度检查，不进入性能计时。

`relative_l2_vs_fp32` 是比例，乘 100% 才是百分数；它不是最大单元素误差。`same_payload_max_abs` 的零值表示相对量化后参考的通信检查一致，不表示相对原始输入无量化误差。检查通过不等于真实模型质量已经验收。

## 同批性能对照

```bash
for tp in 2 4 8; do
  "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node="$tp" \
    benchmarks/benchmark_quantized_communication.py \
    --tokens 8192 16384 32768 --hidden-sizes 5120 --chunks 2 4 \
    --warmup 10 --iterations 55 --block-iterations 5 --rounds 4 \
    --output "results/quantized_communication_tp${tp}.json"
done
```

每种 TP 的所有方案在同一次进程运行内完成；不同 TP 仍为独立运行。固定卡选择、rank 映射、HCCL 设置和机器负载，记录 CANN/驱动版本及拓扑。脚本记录外部环境变量、命令、torch/torch_npu 版本、主机名、设备名与全部复用脚本的 SHA256，不自动修改外部配置，也不推断没有记录的软件版本或拓扑。

默认每轮每方案 55 个样本，每连续测 5 次就切换到下一方案，即 11 个 cycle。每个 cycle 旋转方案顺序，下一个完整旋转组反向测量；所有 rank 使用相同顺序。默认每轮每个方案恰好在每个顺序位置出现一次；4 轮共 220 个样本/方案。自定义样本数仍保证各方案样本数一致，但不保证每轮位置完全均衡，实际顺序保存在 JSON 的 `schedule` 中。

如仅测两块，可改为 `--chunks 2 --iterations 35`：7 个方案，每轮 7 个 cycle。仅测 INT8 时为 6 个方案，可使用 `--backends int8 --iterations 30`。完整位置均衡要求每轮 cycle 数为方案数的整数倍，且各 cycle 的样本数相同；减小 `--block-iterations` 可增加切换频率。

## 计时与结果

所有方案共用 `benchmark_int8_all_reduce._measure`：

- NPU Event 覆盖完整路径，包括 INT8 的 amax/MAX、量化、通信，以及全部异步等待和 BF16 写回；MXFP8 包括量化、打包、AllToAll、反量化/SUM 和最终 AllGather 等待。
- 输入重置位于 Event 区间之外；每个样本取最慢 rank。不同方案之间通过同一测量器同步隔开，不让两个方案的未完成通信互相重叠。
- `mean_ms`、`median_ms`、`p95_ms`、`p99_ms`、`stddev_ms` 和 `cv_pct` 描述 Event 样本；CV 为标准差/均值的百分比，不是置信区间。
- `wall_ms_including_reset` 包含输入重置和主机提交，为各块墙钟均值按样本数加权；它不是 Event mean，也不是模型 TTFT。
- `extra_peak_allocated_mib` 是各块中最大的 PyTorch allocator 增量峰值，排除块开始时已有的工作输入，不包含 reserved 或 HCCL 外部/持久资源，也不代表整机 HBM。

输出会直接列出 mean、p50/p95/p99、CV、相对 BF16、自身整块、同块数串行的加速比和额外显存。大于 1 的加速比表示更快；耗时下降使用 `100*(1-候选耗时/基线耗时)`，与加速百分比区分。

跨路线对照同时列出：

1. 整块 INT8 与整块 MXFP8。
2. 相同块数的串行/流水 INT8 与 MXFP8。
3. 各自最低 p50 的候选，明确标出实际方案；不把串行控制组纳入最佳候选。该选择是本批已测最小值，不是自动运行时选型或稳定性保证。

JSON 保留原始 `samples_ms`、每块时间与样本数、真实测试顺序、每轮 mean/median、每轮配对的相对 BF16 耗时变化，以及两条路线每轮 mean 的配对变化。百分比为负表示劣化。面对 1%–2% 差异，应查看各轮是否一致、波动是否超过收益，并重复整轮测试；同批次减少跨批漂移，不保证消除波动。p99 为本批样本的描述统计，小样本不能建立稳定的尾延迟结论。

已有 JSON 可在 CPU 上重新打印，不导入 torch_npu 或初始化通信：

```bash
"$PYTHON" benchmarks/benchmark_quantized_communication.py \
  --report-json results/quantized_communication_tp2.json
```

这是 Eager 微基准，不能直接换算成模型 TPS/TTFT。历史跨批结果仍保留在[分块流水报告](quantized_communication_pipeline_report.md)，新的同批数据应按独立实验记录，不覆盖旧值。
