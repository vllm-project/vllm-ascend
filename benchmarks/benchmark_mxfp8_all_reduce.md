# A5 MXFP8 AllToAll/AllGather 性能单测

该脚本对普通 HCCL 集合通信组合方案做独立正确性和性能测试：

```text
BF16 [tokens, hidden]
  -> 每个 rank 独立动态量化为 E4M3FN 数据和 E8M0 块尺度
  -> 按行将数据和尺度打包并执行 AllToAll
  -> shard owner 反量化为 BF16（默认）或 FP32，按来源 rank 求和
  -> 转为 BF16，并通过 AllGather 恢复完整结果
```

脚本只依赖 `torch` 和 `torch_npu`，不导入 vLLM，也不需要
`cann_ops_transformer`、原生 QuantAllReduce 或本地 C++ 扩展。测试结果只代表这条
AllToAll/AllGather 组合路径，不能用原生 MXFP8 QuantAllReduce 的结果代替。

`--dequant-dtype bf16|fp32` 同时选择反量化张量和局部求和的输出类型：

- `bf16`：新的默认值，`npu_anti_mx_quant(dst_type=15)`，局部 SUM 输出 BF16，
  随后执行 BF16 AllGather。性能路径不会显式生成完整的 FP32 反量化副本。
- `fp32`：保留原路径，`npu_anti_mx_quant(dst_type=6)`，局部 SUM 输出 FP32，
  再转 BF16 执行 AllGather。

这里只规定 SUM 的输入和输出类型，内部累加精度与 workspace 由后端决定。
两种模式的 MXFP8 AllToAll payload 和 BF16 AllGather 类型相同。

## 先验证正确性

```bash
PYTHON=/usr/local/python3.12.16/bin/python3
export HCCL_NPU_SOCKET_PORT_RANGE="61000-61050"

"$PYTHON" -m torch.distributed.run --standalone --nproc-per-node 2 \
    benchmarks/benchmark_mxfp8_all_reduce.py \
    --tokens 128 --hidden-sizes 5120 \
    --dequant-dtype bf16 \
    --check-only --output mxfp8_alltoall_tp2_check.json
```

脚本依次检查：

1. 全零、不同 rank 常数和正负抵消输入；
2. 不依赖动态量化结果的 rank/token/block 不同字节尺度用例；
3. `npu_anti_mx_quant` 相对独立 CPU E4M3FN/E8M0 字节解码的结果；
4. 固定量化 payload 经过 AllToAll、所选类型的局部求和和 BF16 AllGather 后的结果；
5. 从 BF16 动态量化开始的完整路径结果。

任何检查失败都会终止，不会回退到 BF16 AllReduce。通信结果以独立字节解码后的
FP32 SUM 为参考；整体 relative L2 和最大绝对误差以原始 BF16 输入的 FP32 SUM 为参考。
切换到 BF16 模式不会把独立解码和通信参考也改成 BF16。局部反量化检查按所选输出
类型比较，通信和完整路径仍对照独立 FP32 SUM 后的 BF16 结果，并检查所有输出有限。

## 性能冒烟测试

```bash
"$PYTHON" -m torch.distributed.run --standalone --nproc-per-node 2 \
    benchmarks/benchmark_mxfp8_all_reduce.py \
    --tokens 512 --hidden-sizes 2048 \
    --dequant-dtype bf16 \
    --warmup 5 --iterations 20 --rounds 3 --breakdown \
    --output mxfp8_alltoall_tp2_smoke.json
```

完整路径使用 NPU Event 计时。每次迭代都恢复相同输入，恢复操作位于 event 区间外；
BF16 和候选路径按轮交替，单个样本取所有 rank 中的最大耗时，再计算中位数和 p95。
JSON 同时保存每轮中位数、全部样本、包含输入恢复的 wall time，以及 PyTorch allocator
可见的额外峰值内存。

性能路径在打包完成后释放原始量化张量和尺度，并在 AllToAll 提交后释放发送缓冲，
以缩短临时张量的生命周期。额外峰值仍包含接收缓冲、拆包产生的连续副本、所选类型的
完整反量化结果以及分片求和结果；这些是当前组合算法的实际成本。以 `[16384,5120]`
为例，反量化张量从 FP32 的 320 MiB 降至 BF16 的 160 MiB，整体峰值还取决于后端
workspace、异步生命周期和其他缓冲，不能按此推断整体峰值减半。

报告和 JSON 会记录 `dequant_dtype`、`sum_output_dtype`，本地反量化检查保存在
`accuracy.dequantization`。旧 JSON 的 `accuracy.dequant_fp32` 仍可离线渲染。

`--breakdown` 独立测量五个阶段：

1. `dynamic_mx_quant`
2. `pack_payload`
3. `all_to_all`
4. `dequantize_and_sum`
5. `bf16_all_gather`

阶段测量包含额外 event 标记开销，各阶段中位数不能直接相加作为完整路径耗时。
判断收益应以完整路径的 `speedup = BF16 median / AllToAll median` 为准。

## 扫描 TP 和通信量

建议先在相同设备和环境变量下，对你正在分析的 H=5120、TP=2 分别测试两种模式：

```bash
for dtype in bf16 fp32; do
    "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node 2 \
        benchmarks/benchmark_mxfp8_all_reduce.py \
        --tokens 2048 4096 8192 16384 --hidden-sizes 5120 \
        --dequant-dtype "$dtype" \
        --warmup 10 --iterations 50 --rounds 4 --breakdown \
        --output "mxfp8_alltoall_tp2_${dtype}.json"
done
```

再按实际需要扩大扫描范围：

```bash
for tp in 2 4 8; do
    "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node "$tp" \
        benchmarks/benchmark_mxfp8_all_reduce.py \
        --tokens 512 2048 4096 8192 16384 32768 \
        --hidden-sizes 2048 5120 \
        --dequant-dtype bf16 \
        --warmup 10 --iterations 50 --rounds 4 --breakdown \
        --output "mxfp8_alltoall_tp${tp}.json"
done
```

按实际空闲卡数选择 TP，不要并发运行多档扫描。支持 TP=2/4/8；tokens 必须能被
TP 整除，hidden size 必须是 64 的倍数。脚本不设最小 payload 门槛，目的是找出完整
路径的收益拐点。

结果 JSON 还记录两条路径的理论每 rank 发送量。对于 N 个 BF16 元素：

- BF16 ring AllReduce 模型约为 `4 * (TP-1)/TP * N` 字节；
- MXFP8 AllToAll 加 BF16 AllGather 约为
  `(1 + 1/32 + 2) * (TP-1)/TP * N` 字节。

后者理论上减少约 24.2%，但增加动态量化、打包、反量化 staging 和两次不同的集合通信，
因此最终结论必须依据完整路径实测。

已保存的 JSON 可以离线重新生成表格：

```bash
"$PYTHON" benchmarks/benchmark_mxfp8_all_reduce.py \
    --report-json mxfp8_alltoall_tp2_smoke.json --breakdown
```

微基准不运行模型推理。通过数值检查或获得通信提速后，仍需使用真实 Qwen3.6 prefill
shape 验证模型精度、TTFT、吞吐和进程总 HBM。

## 本地 CPU 单元测试

```bash
python3 -m pytest --noconftest -q \
    tests/ut/distributed/device_communicators/test_mxfp8_all_reduce_benchmark.py
```

覆盖 dtype 参数、算子类型编码、不同 rank 尺度的局部求和、独立 FP32 参考检查和
新旧 JSON 报告。CPU mock 不代表 A5 算子精度、峰值显存或性能已经验证。
