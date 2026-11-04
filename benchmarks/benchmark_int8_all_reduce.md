# A5 INT8 all-reduce 性能单测

脚本只依赖现有 `torch`、`torch_npu`，可单独复制到 A5 机器运行，不依赖 vLLM、融合量化通信算子或 C++ 编译。测试对象为 Eager prefill 常见的二维 BF16 激活 `[tokens, hidden_size]`，不修改服务配置或已有实现。

## 先跑通一档

在仓库根目录，用此前确认 NPU 可用的同一个 Python 环境运行：

```bash
python3 -m torch.distributed.run --standalone --nproc-per-node 4 \
    benchmarks/benchmark_int8_all_reduce.py \
    --tokens 512 --hidden-sizes 2048 \
    --warmup 5 --iterations 20 --rounds 3 --breakdown \
    --output int8_tp4_smoke.json
```

先检查零值、同号最大量化值两项是否 PASS。每个实际 payload 还会单独检查 INT8 求和结果、完整路径数值结果，再进行性能测量。任何算子或通信失败都会停止该次运行，不静默换算子或回退 BF16。

默认使用 `npu_quantize(..., div_mode=True)` 和 `npu_anti_quant(..., dst_dtype=bfloat16)`，与 PR 的分离式 INT8 路径一致。官方 API 文档：[量化](https://github.com/Ascend/op-plugin/blob/master/docs/en/custom_APIs/torch_npu/torch_npu-npu_quantize.md)、[反量化](https://github.com/Ascend/op-plugin/blob/master/docs/en/custom_APIs/torch_npu/torch_npu-npu_anti_quant.md)。这不代替目标版本的实际调用验证。

默认 `--amax-method norm` 使用 `vector_norm(ord=inf, dim=0)` 计算逐通道 absmax。`--amax-method abs` 改用 `abs().amax(dim=0)`，便于复现本地旧分支。结果 JSON 会记录该选择。

## 扫描 prefill 通信量与 TP

hidden size 2048、5120 分别覆盖此前讨论的两类 Qwen3.6 配置。最终应以实际部署中 BF16 all-reduce 的形状为准。

```bash
for tp in 2 4 8; do
    python3 -m torch.distributed.run --standalone --nproc-per-node "$tp" \
        benchmarks/benchmark_int8_all_reduce.py \
        --tokens 512 2048 4096 8192 16384 32768 \
        --hidden-sizes 2048 5120 \
        --warmup 10 --iterations 50 --rounds 4 --breakdown \
        --output "int8_tp${tp}.json"
done
```

该命令需要至少 8 张可见 NPU；按已有卡数选择 TP。不要并发执行几档测试。跨节点部署可使用对应的 torchrun rendezvous 参数，所有 rank 的脚本参数必须一致。

BF16 输入大小为 `tokens * hidden_size * 2` 字节。例如 hidden size 2048 下，8192/16384/32768 tokens 分别是 32/64/128 MiB。脚本对指定形状始终走 INT8 候选路径，不设置最小通信量门槛，以便找到收益拐点。

## 结果含义

测试完成后，终端按 hidden size 分组输出对比表；阶段耗时单独成表，不穿插 JSON：

```text
hidden_size = 2048

| Tokens | BF16 通信量(MiB) | BF16 耗时(ms) | INT8 完整耗时(ms) | 加速比 | 延迟变化     |
| ------ | ---------------- | ------------- | ----------------- | ------ | ------------ |
| 512    | 2.00             | 0.0773        | 0.2139            | 0.362x | 增加 176.62% |
| 16384  | 64.00            | 1.3987        | 0.8405            | 1.664x | 下降 39.91%  |
```

上述数值仅示范输出格式。误差与额外内存也会单独成表，其中相对 L2 误差显示为百分比。JSON 字段和数值单位保持原样。

已经跑完并保存的 JSON 可以直接转换成表格，无需重跑多卡测试：

```bash
python3 benchmarks/benchmark_int8_all_reduce.py --report-json int8_tp2.json --breakdown
```

测试过程中只输出简短进度；运行环境摘要与正确性检查结果保留。PyTorch/HCCL 自身的 warning 不被屏蔽。

- `BF16_ms`：原生 BF16 SUM all-reduce 的中位数耗时。
- `INT8_full_ms`：本地 absmax、FP32 MAX all-reduce、尺度计算与量化、INT8 SUM all-reduce、反量化与结果写回的完整中位数耗时。
- `speedup = BF16_ms / INT8_full_ms`：大于 1 表示该实现、该形状下微基准提速。
- `gain_pct = (1 - INT8_full_ms / BF16_ms) * 100`：正值表示完整路径延迟下降，负值表示变慢。
- `relative_L2`：INT8 方案相对原始输入 FP32 SUM 的相对 L2 误差。JSON 同时提供 BF16 基线的误差和 INT8 最大绝对误差。
- `extra_INT8_MiB`：排除原始输入与复用工作缓冲后的额外峰值分配内存；不含 allocator reserved 内存或进程全部 HBM。

判断收益时查看完整路径的 `speedup`，并检查 JSON 中的 p95 与各轮 `round_medians_ms`。若只是 INT8 通信阶段快，而完整路径慢，当前实现就没有收益。接近 1 的结果应增大采样次数并独立重跑，检查波动后再决定阈值。通过数值检查不表示模型精度已达标；这里使用合成 BF16 随机输入。

`--breakdown` 额外输出五个阶段的诊断耗时：

1. `local_amax`
2. `amax_all_reduce`
3. `scale_and_quantize`
4. `int8_all_reduce`
5. `dequantize_and_copy`

阶段诊断独立于主要计时执行，增加了 event 标记开销；各阶段中位数之和不等同于完整路径中位数。它用来判断收益被哪一段消耗。

## 计时与正确性约定

- 使用 NPU Event 计时，不在完整路径的阶段之间同步或读取 `.item()`。普通 HCCL 调用使用默认同步语义，让完成依赖回到当前流。
- 两条路径均原地修改工作张量。每次恢复相同输入，恢复操作位于 event 计时区间之外，避免重复 SUM 导致数值增长或溢出。
- 先预热，两条路径按轮交替测试顺序。每个样本跨 rank 取最大值，再计算中位数、p95。JSON 还记录包含恢复输入及 Python 调度的批量 wall time 与 `wall_speedup_including_reset`，供对照；它的测量口径与 event 计时不同。
- 不同 rank 使用不同随机种子；共享逐通道尺度经 MAX all-reduce 得到。`max_q=127//TP`，确保 INT8 SUM 不溢出，TP=2/4/8 对应 63/31/15。
- 先用同一份量化结果做 INT32 SUM，精确核对真实 INT8 SUM，捕获回绕；再用独立 FP32 乘尺度结果验证完整反量化路径，最后报告相对原始 FP32 SUM 的量化误差。
- 未执行模型推理，不能直接据此宣称 TTFT、吞吐或模型精度收益；有微基准收益后，再在真实 prefill 形状和模型负载上验证。

## 算子不可用时的显式对照

若当前软件版本不支持原生量化或反量化调用，可以在新的 torchrun 运行中显式选择普通 PyTorch 实现：

```bash
python3 -m torch.distributed.run --standalone --nproc-per-node 4 \
    benchmarks/benchmark_int8_all_reduce.py \
    --tokens 8192 16384 --hidden-sizes 2048 \
    --quant-backend torch --dequant-backend torch \
    --breakdown --output int8_tp4_torch_ops.json
```

也可只改 `--dequant-backend torch`。这些候选实现使用多次普通算子，必须依据 JSON 中记录的 backend 分别评价；不能把它们的结果当作原生量化/反量化版本的结果。这里始终使用普通 INT8 HCCL SUM，不依赖 `npu_quant_all_reduce`。

## 本地 CPU 检查

脚本的数学、路由、统计和计时控制有独立单元测试，不要求 CPU 机器安装 torch_npu：

```bash
python3 -m pytest --noconftest -q \
    tests/ut/distributed/device_communicators/test_int8_all_reduce_benchmark.py
```

CPU 检查不验证 NPU Event、HCCL 或原生量化算子。A5 性能结果只能由目标机器运行上述多卡命令获得。
