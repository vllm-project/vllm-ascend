# 一次采集 BF16、MXFP8 与 MXFP8 分块流水

`profile_mxfp8_pipeline.py` 在每个 shape/rank 的同一个 profiler 捕获中，依次执行以下方案，每轮重复相同顺序：

| 时间轴标签 | 方案 |
| --- | --- |
| `variant/bf16_all_reduce` | 非量化 BF16 AllReduce |
| `variant/mxfp8_full` | 整块 MXFP8 AllToAll → BF16 反量化/SUM → BF16 AllGather |
| `variant/mxfp8_pipeline_chunks_2` | 同一 MXFP8 路径，2 块流水 |
| `variant/mxfp8_pipeline_chunks_4` | 加 `--chunks 2 4` 时，同时采集 4 块流水 |

复用已实现的 benchmark 算法，只增加 profiler 标签。只采集 Eager；MXFP8 反量化、本地 SUM 输出和最终 AllGather 都使用 BF16。

## 一条命令采集

在仓库根目录、已能运行 MXFP8 benchmark 的 NPU 环境执行：

```bash
export HCCL_NPU_SOCKET_PORT_RANGE="61000-61050"

# 自动启动 torchrun，采集 TP4 下三个方案的流水。
python3 benchmarks/profile_mxfp8_pipeline.py \
  --tp 4 --tokens 16384 --hidden-sizes 5120 --chunks 2 \
  --warmup 10 --steps 3 \
  --output-dir results/mxfp8_compare_tp4
```

`--steps 3` 表示同一次捕获中重复三轮“非量化 → 整块量化 → 量化流水”。这三种方案顺序执行，便于比较；每个流水方案内部保留原有计算/通信重叠。

无需另开一次性能 benchmark。脚本先在捕获外检查量化方案精度、预热，再采集 CPU/NPU 时间轴。各方案使用同一份各 rank 随机 BF16 源数据，每次调用前恢复输入。

## 一次采集多个 TP、shape 和分块数

```bash
# 确保可见设备至少有 8 张；TP2、TP4、TP8 顺序启动独立 torchrun。
python3 benchmarks/profile_mxfp8_pipeline.py \
  --tp 2 4 8 --tokens 8192 16384 32768 --hidden-sizes 5120 \
  --chunks 2 4 --warmup 10 --steps 3 \
  --output-dir results/mxfp8_compare_all
```

继承当前的设备可见性、HCCL 模式及其它环境设置，不自动切换。不同 TP 默认使用可见设备中的前 TP 张卡。若要比较其它设备组合，先设置 `ASCEND_RT_VISIBLE_DEVICES`，分别执行命令。

也支持手动启动 torchrun，此时 `--tp` 可省略；若指定，必须与进程数一致：

```bash
python3 -m torch.distributed.run --standalone --nproc-per-node=4 \
  benchmarks/profile_mxfp8_pipeline.py \
  --tokens 16384 --chunks 2 4 \
  --output-dir results/mxfp8_compare_manual
```

普通 Python 启动且未指定 `--output-dir` 时，会生成带时间戳和随机后缀的目录；手动 torchrun 默认使用 `results/mxfp8_compare`。已有 shape/rank 目录会报错，避免混入旧 trace；重复采集请换新的输出目录。

## 输出与查看

```text
results/mxfp8_compare_tp4/
  tp4/
    manifest.json
    16384x5120/
      rank0/
      rank1/
      rank2/
      rank3/
```

每个 rank 目录中由 torch_npu profiler 同步导出并分析数据，通常包含 profiler 子目录及其中的 `ASCEND_PROFILER_OUTPUT/trace_view.json`、kernel/算子统计 CSV。具体文件集取决于 torch_npu/CANN 版本。`manifest.json` 保存设备、版本、配置、环境、精度和各 rank 的目录。

用 MindStudio Insight 打开对应 profiler 目录，或用 Perfetto 打开导出的 `trace_view.json`。查找 `variant/` 标签定位三种方案，展开设备时间轴查看计算与 HCCL stream。额外阶段标签包括：

- `mxfp8/quantize/rows_0_8192`：量化的输入行范围，便于区分不同块。
- `mxfp8/pack_payload`：FP8 数据与 scale 打包。
- `mxfp8/all_to_all`：整块方案的 AllToAll 提交。
- `mxfp8/dequantize`、`mxfp8/dequantize_and_sum`：反量化及流水方案的本地归约。
- `mxfp8/dequantize_sum_and_gather`：整块方案的反量化、归约和 AllGather。

这些 CPU 标签用于定位提交顺序，实际重叠应看关联的 NPU kernel/HCCL 时间轴。流水方案中的 AllToAll、AllGather 和 Work 等待由 profiler 的底层算子/通信事件展示。

输入复位单独标为 `input_reset/方案名`。每种方案末尾同步，确保输出完成并分开三种方案；块之间不增加同步。精度参考计算、预热和方案之间的 rank barrier 均在捕获外。

采集使用与现有通信 benchmark 相同的 vLLM 风格配置：CPU/NPU、Level1、Text、PipeUtilization，并同步导出分析。默认关闭内存和调用栈；可按需加 `--profile-memory`、`--profile-with-stack`。后者使用 torch_npu 的 `with_modules`，与当前 vLLM profiler 一致。

profiler 标签、采集和组间同步会引入额外开销。延迟收益继续以 `benchmark_mxfp8_pipeline.py` 的非 profiler 测量为准；本脚本用于判断计算与通信是否实际重叠，以及空隙/等待发生在哪里。

CPU 测试：

```bash
python3 tests/ut/distributed/device_communicators/test_mxfp8_profile.py
```
