# A5 原生 MXFP8 all-reduce 精度与性能单测

采用 `benchmark_int8_all_reduce.py` 的独立多卡测试方式：先校验正确性，
再与 BF16 SUM all-reduce 比较完整流程耗时。脚本可单独复制到 A5 机器，
只依赖已有 `torch`、`torch_npu` 和 `cann_ops_transformer`，不依赖 vLLM、
本地 C++ 封装或 `tools/a5_quantized_comm` 目录。

测试 Eager prefill 常见的连续 BF16 激活 `[tokens, hidden_size]`。
候选路径为 `npu_dynamic_mx_quant -> 官方 quant_all_reduce -> BF16 写回`，
官方融合算子内部处理通信、反量化和归约。

## 先只验证精度

在已经确认有官方算子的 0.32 镜像中执行：

```bash
PYTHON=/usr/local/python3.12.16/bin/python3

"$PYTHON" -m torch.distributed.run --standalone --nproc-per-node 2 \
    benchmarks/benchmark_native_mxfp8_all_reduce.py \
    --tokens 128 --hidden-sizes 5120 \
    --check-only --output native_mxfp8_tp2_check.json
```

先后检查全零、不同 rank 常数、正负抵消和手工编码的不同 rank/token/block 尺度。
手工编码用例绕开动态量化算子，避免用同一算子互相验证。
每个实际 payload 还会独立检查原生 BF16 输出、原生 FP32 输出和完整 BF16 路径。
任何错误都会停止该次运行，不静默回退普通通信或旧 C++ 封装。

通过时先输出四项 `Correctness ...: PASS`，随后输出实际形状的精度结果，
最终打印 `PASS: all requested MXFP8 native communication and full-path checks passed.`。
`--check-only` 不进行预热和性能测量，结果中不生成 `speedup`。

## 再与 BF16 比较性能

```bash
"$PYTHON" -m torch.distributed.run --standalone --nproc-per-node 2 \
    benchmarks/benchmark_native_mxfp8_all_reduce.py \
    --tokens 512 --hidden-sizes 2048 \
    --warmup 5 --iterations 20 --rounds 3 --breakdown \
    --output native_mxfp8_tp2_smoke.json
```

终端输出列为：

```text
tokens hidden BF16_MiB BF16_ms MXFP8_full_ms speedup gain_pct MXFP8_rel_L2 BF16_rel_L2 extra_MXFP8_MiB
```

- `BF16_ms`：普通 BF16 SUM all-reduce 的中位数耗时。
- `MXFP8_full_ms`：动态 MX 量化、尺度布局、原生 QuantAllReduce 和 BF16 写回的完整中位数耗时。
- `speedup = BF16_ms / MXFP8_full_ms`：大于 1 表示这组微基准中完整路径提速。
- `gain_pct = (1 - MXFP8_full_ms / BF16_ms) * 100`：正值表示延迟下降。
- `MXFP8_rel_L2`、`BF16_rel_L2`：两条路径分别相对原始输入 FP32 SUM 的相对 L2 误差。
- `extra_MXFP8_MiB`：相对测量起点，PyTorch allocator 的额外峰值分配内存。

JSON 还包含原生 BF16/FP32 通信的误差、量化本身的误差、最大绝对误差、p95、
各轮中位数、原始样本以及包含恢复输入的批量 wall time。
原生通信 `pass=true` 代表结果满足诊断容差，量化误差默认报告，需结合模型判断是否可接受。

## 扫描 prefill 通信量与 TP

```bash
for tp in 2 4 8; do
    "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node "$tp" \
        benchmarks/benchmark_native_mxfp8_all_reduce.py \
        --tokens 512 2048 4096 8192 16384 32768 \
        --hidden-sizes 2048 5120 \
        --warmup 10 --iterations 50 --rounds 4 --breakdown \
        --output "native_mxfp8_tp${tp}.json"
done
```

按空闲可见卡数选择 TP，避免并发运行几档测试。所有 rank 参数应一致。
支持 TP=2/4/8、tokens 可被 TP 整除，hidden size 在 `[1024,8192]` 且 128 对齐。
BF16 输入大小为 `tokens * hidden_size * 2` 字节；脚本不设最小通信量阈值，
用于查找完整路径的收益拐点，而不是只计量 FP8 payload 的大小。

## 正确性与通信域约定

- 默认 HCCL 组用于 BF16 基线、普通 FP32 参考求和、统计和 barrier。
- 额外创建并显式初始化专用 HCCL 组，其非空 hcom 只交给官方 `quant_all_reduce`；不在该组上调用普通集合通信。
- 使用官方 Python facade，由它管理 context 和 buffer，不直接伪造底层 `torch.ops` 参数。
- E4M3FN 和 E8M0 按原始字节在 CPU 独立解析，逐 rank 反量化后在普通组中做 FP32 SUM。
  这条通信 golden 不依赖 `npu_anti_mx_quant`。
- 同一份 q/scale 分别检查 BF16、FP32 原生输出，然后检查重新量化的完整 BF16 路径。
  零值和常数等 sanity 用例还要求动态量化本身精确还原输入。
- 普通用例判据为 `abs(actual-rounded_reference) <= atol + rtol*sum(abs(rank_dequant))`，
  BF16 `rtol=2^-7`、FP32 `rtol=2e-5`、`atol=1e-6`；sanity 精确用例容差为零。
  这是测试容差，不是厂商精度保证。非有限值及错误 shape/dtype/device 会直接失败。
- 官方 torch extension 参数编码为 BF16 `15`、FP32 `6`、E4M3FN `292`、E8M0 `293`，
  这些不是原始 ACL dtype 编码。尺度只做字节 view 和布局处理，不做数值类型转换。

## 计时与内存口径

与 INT8 基准采用相同方法：NPU Event 计时，每次恢复相同输入，恢复位于 event 区间外；
两条路径都原地写回工作张量。先校验、预热，再按轮交替顺序测量。
每个样本跨 rank 取最大值，再计算中位数、p95。CPU golden、参考 collective、
`.item()` 和统计均不进入计时区间。

`--breakdown` 单独测量 `dynamic_mx_quant`、`scale_layout`、
`native_quant_all_reduce` 和 `copy_result` 四个阶段。
它包含额外 event 开销，各阶段中位数之和不等于完整路径中位数。
收益应依据 `MXFP8_full_ms`、p95、各轮中位数及 wall time 综合判断。

额外峰值只包含 PyTorch allocator 可见的瞬态分配；工作张量和已经创建的 context
位于测量起点之前，外部 HCCL context/buffer、allocator reserved 及进程全部 HBM
不在此数值范围内。正式部署仍需单独测量这些持久资源。

通过合成输入测试不代表模型精度、TTFT 或吞吐收益已经成立；后续仍需真实 prefill 形状和模型验证。

## 本地 CPU 单元测试

```bash
python3 -m pytest --noconftest -q \
    tests/ut/distributed/device_communicators/test_native_mxfp8_all_reduce_benchmark.py
```

覆盖已知 FP8/E8M0 编码、2/4/8 rank 的不同尺度仿真、失败结果检测、通信域选择、
输入恢复与计时边界、交替测量顺序及统计。只需 CPU PyTorch，不导入 torch_npu 或 vLLM。
CPU 仿真不证明原生算子、HCCL 或 NPU Event 已通过；A5 结果需由目标环境执行上述命令取得。

参数依据 [官方 Python 封装](https://gitcode.com/cann/ops-transformer/blob/master/mc2/quant_all_reduce/torch_extension/quant_all_reduce.py)
及 [C++ 绑定](https://gitcode.com/cann/ops-transformer/blob/master/mc2/quant_all_reduce/torch_extension/csrc/quant_all_reduce.cpp)。
