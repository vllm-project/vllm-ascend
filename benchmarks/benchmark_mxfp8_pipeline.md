# A5 MXFP8 分块流水测试

本脚本验证不同 token 块之间能否重叠量化、反量化/归约与通信。只测试 Eager，反量化和本地 SUM 输出均为 BF16，最终 AllGather 也使用 BF16。它不修改 vLLM 运行时配置或原有 MXFP8 benchmark。

## 对照组

| 名称 | 路径 |
| --- | --- |
| `bf16` | 原生 BF16 AllReduce |
| `mxfp8_full` | 原脚本的不分块 MXFP8 AllToAll → BF16 反量化/SUM → BF16 AllGather |
| `serial_chunks_K` | 相同流程分为 K 块，逐块串行执行 |
| `pipeline_chunks_K` | 相同分块，使用异步通信和两块预取 |

默认测试 K=2、4。按行分块，不改变 hidden 维每 32 个元素的 MX scale 分组。各块行数必须是 TP 的整数倍；无法完全均分时，将多出的 TP 行单位分配给前面的块。脚本另外检查分块量化的原始字节与整块量化一致。

流水使用同一个默认 HCCL 通信组。计算提交到当前计算流，通信交给 ProcessGroupHCCL 内部通信流，依赖由 `Work.wait()` 建立。K=4 的主机提交顺序如下；Q 表示量化/打包，A 表示 AllToAll，D 表示反量化/本地 SUM，G 表示 AllGather：

```text
Q0 A0 → Q1 A1 → wait(A0) D0 G0 → Q2 A2
              → wait(A1) D1 G1 → Q3 A3
              → wait(A2) D2 G2 → wait(A3) D3 G3 → wait(all G)
```

设计允许 Q1 与 A0、后续 Q/D 与前面的通信发生重叠。同一组的通信仍保持顺序。实际能否重叠、能隐藏多少时间，需要看设备 trace；AIV 通信与量化/反量化可能争用 Vector 核和内存带宽。增加分块也会增加通信启动和 Python 调度开销。

## 先验证正确性

在仓库根目录执行，复用现有可运行 MXFP8 benchmark 的 Python/NPU 环境。无需 `ops-transformer` 原生量化通信算子，只需 `npu_dynamic_mx_quant`、`npu_anti_mx_quant` 和普通 HCCL。

```bash
export HCCL_NPU_SOCKET_PORT_RANGE="61000-61050"
PYTHON=python3

"$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=2 \
  benchmarks/benchmark_mxfp8_pipeline.py \
  --tokens 128 --hidden-sizes 5120 --chunks 2 4 \
  --check-only --output results/mxfp8_pipeline_check_tp2.json
```

正确性检查包括：零值、各 rank 常数、正负抵消、不同 rank/token/block 的编码与 scale、不能均分的尾块，以及随机输入的独立 CPU 字节解码参考。分别检查本地反量化、同一量化 payload 的通信和完整量化通信路径。报告保留相对 L2、最大绝对误差、相同 payload 的通信最大误差。FP32 SUM 作为误差参考，候选路径仍输出 BF16。

## 性能测试

```bash
for tp in 2 4 8; do
  "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node="$tp" \
    benchmarks/benchmark_mxfp8_pipeline.py \
    --tokens 8192 16384 32768 --hidden-sizes 5120 --chunks 2 4 \
    --warmup 10 --iterations 50 --rounds 4 \
    --output "results/mxfp8_pipeline_tp${tp}.json"
done
```

先以实际业务的通信 shape 测试，例如 `[16384,5120]`。测试不同 TP 时确保可见设备数量和设备拓扑符合预期。脚本记录 HCCL 模式等环境配置，不自动切换；不同模式对比应分别启动独立进程，保持其它条件一致。开启阻塞执行/等待的环境设置可能压缩重叠机会，应结合报告中的环境值和 trace 判断。

计时沿用原 benchmark 的 NPU event：排除输入复位，包含量化、打包、通信、反量化、SUM、全部最终 AllGather 的等待，每个 sample 取最慢 rank。各组轮换执行顺序，保留 p50/p95、各轮中位数和原始样本，另报包含输入复位的主机墙钟时间。profiler 在性能采样完成后单独执行，不计入这些样本。

判断结果时同时看三项：

- `vs chunk serial`：流水相对相同分块串行的变化，隔离分块本身的影响。
- `vs full`：流水是否超过原不分块 MXFP8 完整路径。
- `vs BF16`：是否仍有实际通信优化收益。大于 1 表示更快。

`extra peak MiB` 是单次测量期间的 PyTorch allocator 额外峰值，包含流水中同时在途的张量，不包含 allocator reserved 或 HCCL 外部分配。发送/接收队列最多保留两块；所有 AllGather 输入和输出视图保留到最终 join，因此额外局部 SUM 结果总量约为完整 BF16 张量的 `1/TP`。这不表示额外开销随调用次数累加。

## 查看重叠

如果只想一次采集“非量化、整块 MXFP8、MXFP8 分块流水”的对照时间轴，可使用独立的 [一键采集脚本](profile_mxfp8_pipeline.md)，支持在同一 trace 中比较三个方案，以及一次顺序采集多个 TP。

```bash
"$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=2 \
  benchmarks/benchmark_mxfp8_pipeline.py \
  --tokens 16384 --hidden-sizes 5120 --chunks 2 4 \
  --warmup 10 --iterations 30 --rounds 4 \
  --profile-dir results/mxfp8_pipeline_traces --profile-steps 2 \
  --output results/mxfp8_pipeline_profile_tp2.json
```

每个 shape/rank 分别导出 torch_npu profiler trace。用 Perfetto 或对应的 Ascend 分析工具检查计算流与 HCCL 流的时间轴，观察实际 kernel 的重叠和总时长，不能仅从 CPU 标签重叠推断加速。脚本用 variant 名称标记各组；每组末尾同步以划清捕获范围，此同步只存在于额外的 profiler 运行。

```bash
# 从 JSON 重新打印表格，不初始化 NPU；仍需安装 PyTorch。
"$PYTHON" benchmarks/benchmark_mxfp8_pipeline.py \
  --report-json results/mxfp8_pipeline_tp2.json

# CPU 模拟测试：多 rank 路由、顺序、缓冲区生命周期、尾块与连续调用。
"$PYTHON" \
  tests/ut/distributed/device_communicators/test_mxfp8_pipeline_benchmark.py
```

分块方案、当前运行时状态和已回传的 TP2/4/8 A5 微基准数据见 [分块流水方案与性能记录](quantized_communication_pipeline_report.md)。各 shape 的收益不同，模型 TTFT、真实模型精度和图模式需独立验收。
