# INT8 共享 scale 分块流水 benchmark

运行时状态：2026-11-12 按用户反馈撤回 INT8 流水接入，实际运行测试未观察到收益。本脚本和历史微基准数据保留用于实验；服务中的 INT8 恢复整块路径，MXFP8 流水保留。INT8 启动配置需删除 `pipeline_chunks` 和 `pipeline_min_numel`，流水参数仅用于 MXFP8 后端。

脚本测试 Eager 下的 INT8 分块流水，不修改运行时或原有 INT8/MXFP8 benchmark。默认使用 `npu_quantize`、`npu_anti_quant`，输入、反量化输出和最终结果均为 BF16。

## 对照组与依赖

| 名称 | 路径 |
| --- | --- |
| `bf16` | 原生 BF16 AllReduce |
| `int8_full` | 原 INT8 benchmark 的整张量路径 |
| `serial_chunks_K` | 共用整张量 scale，K 块串行量化、INT8 AllReduce、反量化写回 |
| `pipeline_chunks_K` | 相同分块与 scale，通过异步 INT8 AllReduce 和两块预取重叠计算/通信 |

每次调用先计算整张量逐通道 amax，做一次 FP32 MAX AllReduce，再计算共享 scale。所有块复用这一份 scale，保留 `max_q=127//TP`：TP2/4/8 分别为 63/31/15，保证 INT8 SUM 不溢出。这一版本没有改为每块独立 scale，不改变量化粒度。

```text
全量 amax → MAX AllReduce → scale
→ Q0 / AR0(async) → Q1 / AR1(async)
→ wait(AR0) / D0 → Q2 / AR2(async)
→ wait(AR1) / D1 → … → wait(最后一个 AR) / D最后一块
```

Q 表示量化，AR 表示 INT8 AllReduce，D 表示 BF16 反量化和写回。目标是让 AR0 与 Q1、AR1 与 D0/Q2 重叠。当前计算流与 ProcessGroupHCCL 内部通信流通过 `Work.wait()` 建立依赖；只使用同一个默认 HCCL 组，不创建额外通信域或用户流。通信本身保持相同顺序，是否产生实际重叠由 profiler 验证。

每次调用仍包含全量 amax 和 MAX 同步；K 块有 K 次 INT8 SUM，增加了通信启动和 Python 调度次数。流水能否抵消这些开销需要实测，不能从 MXFP8 的结果直接推算。INT8 AllReduce 无需按 TP 切分 owner，token 数不要求被 TP 整除，只要求每块至少一行。

## 先验证正确性

从仓库根目录运行，使用已有的 PyTorch/torch_npu 环境，无需 ops-transformer 量化通信算子或 C++ 编译。

```bash
export HCCL_NPU_SOCKET_PORT_RANGE="61000-61050"
PYTHON=python3

"$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=2 \
  benchmarks/benchmark_int8_pipeline.py \
  --tokens 129 --hidden-sizes 5120 --chunks 2 4 --check-only \
  --output results/int8_pipeline_check_tp2.json
```

正确性检查在性能采样之前执行，包含零值、正负同号边界、跨 rank 抵消、不同 rank 的常数，以及跨通道/块的离群值和不均匀尾块。每个实际 shape 还会检查随机输入：

- 分块量化字节与整张量量化一致，scale 均来自整张量。
- 输入量化值位于整数安全范围，实际分块 INT8 SUM 与同一 payload 的 INT32 SUM 精确一致。
- 完整路径与独立 `INT32 SUM → FP32 乘 scale → BF16` 参考对比；原生反量化允许原 benchmark 的 BF16 舍入容差。
- 输出相对原始 FP32 SUM 的相对 L2、最大绝对误差，以及相对整块 INT8 输出的最大绝对误差。相对 L2 在表格中为比例，例如 `0.03` 代表 3%。

失败直接终止，不自动回退 BF16 或切换量化后端。

## 性能测试

```bash
for tp in 2 4 8; do
  "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node="$tp" \
    benchmarks/benchmark_int8_pipeline.py \
    --tokens 8192 16384 32768 --hidden-sizes 5120 --chunks 2 4 \
    --warmup 10 --iterations 50 --rounds 4 \
    --output "results/int8_pipeline_tp${tp}.json"
done
```

复用 MXFP8 测试的设备选择、HCCL 配置与拓扑，并顺序运行各 TP。脚本记录外部 HCCL/阻塞配置，不自动修改。`--quant-backend torch`、`--dequant-backend torch` 和 `--amax-method abs` 可显式选择原 benchmark 的替代实现；不同后端应分开报告。

主计时包含全量 amax、MAX 通信、scale、量化、全部 INT8 SUM 等待、反量化和 BF16 写回。输入复位排除在 NPU event 区间之外，每个 sample 取最慢 rank，报告 p50/p95、各轮中位数和原始样本。另报包含输入复位与主机提交的 `wall/reset`。

`vs BF16`、`vs full`、`vs chunk serial` 分别使用对应基线的 p50 除以当前 p50，大于 1 表示更快。流水相对串行变快不一定超过整块 INT8，必须同时查看三个基线。

额外峰值显存仅统计 PyTorch allocator，不包括 reserved 和 HCCL 外部分配。预取最多保留两块的量化张量，反量化后及时释放引用；实际峰值还取决于设备流执行与 allocator 回收时机，不能只按块数推算。

## 单独捕获 profiler

在性能命令上添加以下参数即可在性能采样完成后另外导出 trace：

```bash
--profile-dir results/int8_pipeline_traces --profile-steps 2
```

每个 shape/rank 分目录保存；各方案用 variant 名称标记。捕获时每组末尾同步以划清范围，这些额外运行不进入性能样本。检查设备计算与通信时间轴的实际重叠，尤其区分前置 amax/MAX 与后半段量化/SUM/反量化。

```bash
# 已有 JSON 重新打印表格：只需 PyTorch，不初始化 NPU。
"$PYTHON" benchmarks/benchmark_int8_pipeline.py \
  --report-json results/int8_pipeline_tp2.json

# CPU 多 rank 模拟：不导入 vLLM/torch_npu，不需要 pytest。
"$PYTHON" tests/ut/distributed/device_communicators/test_int8_pipeline_benchmark.py
```

CPU 模拟验证共享 scale、整数求和、异步提交/等待顺序、张量生命周期、尾块与连续调用；实际算子精度、重叠和收益仍须在 A5 上验证，模型 TTFT/精度另行测试。

当前方案、INT8 / MXFP8 已测性能对照和后续待补数据见 [分块流水方案与性能记录](quantized_communication_pipeline_report.md)。
