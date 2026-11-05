# A5 MXFP8 AllToAll/AllGather 性能单测

该脚本对普通 HCCL 集合通信组合方案做独立正确性和性能测试：

```text
BF16 [tokens, hidden]
  -> 每个 rank 独立动态量化为 E4M3FN 数据和 E8M0 块尺度
  -> 按行将数据和尺度打包并执行 AllToAll
  -> shard owner 反量化为 FP32，按来源 rank 求和
  -> 转为 BF16，并通过 AllGather 恢复完整结果
```

脚本只依赖 `torch` 和 `torch_npu`，不导入 vLLM，也不需要
`cann_ops_transformer`、原生 QuantAllReduce 或本地 C++ 扩展。测试结果只代表这条
AllToAll/AllGather 组合路径，不能用原生 MXFP8 QuantAllReduce 的结果代替。

## 先验证正确性

```bash
PYTHON=/usr/local/python3.12.16/bin/python3
export HCCL_NPU_SOCKET_PORT_RANGE="61000-61050"

"$PYTHON" -m torch.distributed.run --standalone --nproc-per-node 2 \
    benchmarks/benchmark_mxfp8_all_reduce.py \
    --tokens 128 --hidden-sizes 5120 \
    --check-only --output mxfp8_alltoall_tp2_check.json
```

脚本依次检查：

1. 全零、不同 rank 常数和正负抵消输入；
2. 不依赖动态量化结果的 rank/token/block 不同字节尺度用例；
3. `npu_anti_mx_quant` 相对独立 CPU E4M3FN/E8M0 字节解码的结果；
4. 固定量化 payload 经过 AllToAll、FP32 求和和 BF16 AllGather 后的结果；
5. 从 BF16 动态量化开始的完整路径结果。

任何检查失败都会终止，不会回退到 BF16 AllReduce。通信结果以独立字节解码后的
FP32 SUM 为参考；整体 relative L2 和最大绝对误差以原始 BF16 输入的 FP32 SUM 为参考。

## 性能冒烟测试

```bash
"$PYTHON" -m torch.distributed.run --standalone --nproc-per-node 2 \
    benchmarks/benchmark_mxfp8_all_reduce.py \
    --tokens 512 --hidden-sizes 2048 \
    --warmup 5 --iterations 20 --rounds 3 --breakdown \
    --output mxfp8_alltoall_tp2_smoke.json
```

完整路径使用 NPU Event 计时。每次迭代都恢复相同输入，恢复操作位于 event 区间外；
BF16 和候选路径按轮交替，单个样本取所有 rank 中的最大耗时，再计算中位数和 p95。
JSON 同时保存每轮中位数、全部样本、包含输入恢复的 wall time，以及 PyTorch allocator
可见的额外峰值内存。

`--breakdown` 独立测量五个阶段：

1. `dynamic_mx_quant`
2. `pack_payload`
3. `all_to_all`
4. `dequantize_and_sum`
5. `bf16_all_gather`

阶段测量包含额外 event 标记开销，各阶段中位数不能直接相加作为完整路径耗时。
判断收益应以完整路径的 `speedup = BF16 median / AllToAll median` 为准。

## 扫描 TP 和通信量

```bash
for tp in 2 4 8; do
    "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node "$tp" \
        benchmarks/benchmark_mxfp8_all_reduce.py \
        --tokens 512 2048 4096 8192 16384 32768 \
        --hidden-sizes 2048 5120 \
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

后者理论上减少约 24.2%，但增加动态量化、打包、FP32 staging 和两次不同的集合通信，
因此最终结论必须依据完整路径实测。

已保存的 JSON 可以离线重新生成表格：

```bash
"$PYTHON" benchmarks/benchmark_mxfp8_all_reduce.py \
    --report-json mxfp8_alltoall_tp2_smoke.json --breakdown
```

微基准不运行模型推理。通过数值检查或获得通信提速后，仍需使用真实 Qwen3.6 prefill
shape 验证模型精度、TTFT、吞吐和进程总 HBM。
