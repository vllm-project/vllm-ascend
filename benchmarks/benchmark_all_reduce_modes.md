# AI_CPU / AIV AllReduce benchmark

独立微基准，使用 `torch.distributed.all_reduce(..., SUM)` 比较请求 `AI_CPU` 和 `AIV` 时的性能。只依赖配套的 PyTorch、torch_npu、CANN 和至少两张 NPU，不导入 vLLM 或自定义算子。

## 单机对比

在 Ascend Linux 环境加载 CANN 环境并进入仓库后运行：

```bash
python3 benchmarks/benchmark_all_reduce_modes.py \
    --nproc-per-node 8 \
    --dtype bfloat16 \
    --sizes 1K 4K 16K 64K 256K 1M 4M 16M 64M \
    --warmup 20 --iterations 100 --rounds 3 \
    --hccl-buffsize-mib 200 \
    --output-dir all_reduce_tp8
```

- `--nproc-per-node` 按可见 NPU 数调整，默认 2。每个 rank 使用对应的 `LOCAL_RANK` 设备。
- `--sizes` 是**每个 rank 的输入张量字节数**，不是元素数或整组流量。`K/M/G` 和 `KiB/MiB/GiB` 均按 1024 计算，也可传纯字节数。Qwen 隐藏状态 `[tokens, hidden_size]` 的 BF16 大小为 `tokens * hidden_size * 2`。
- `--dtype` 支持 `bfloat16`（默认）、`float16`、`float32`；每种配置须被目标硬件、组网和软件栈支持。并非所有 A2/A3/950 组合都支持相同的展开模式和算子。
- 两个模式顺序运行在**独立 torchrun 作业**中，环境变量在 worker 导入 torch_npu、创建通信域前设置，避免 HCCL 缓存配置使切换失效。每个作业只创建一个 HCCL 通信域；控制同步、校验状态和时延收集使用 CPU/Gloo。
- `--hccl-buffsize-mib` 可省略；省略时两个作业继承同一个 `HCCL_BUFFSIZE`。算法、确定性等其他设置也保持继承并记录在 JSON 中。
- 不覆盖已有结果文件。重复实验请更换 `--output-dir`，或者省略它以使用带时间戳的目录。

无 NPU 时可以预览启动命令，无需安装 torch：

```bash
python3 benchmarks/benchmark_all_reduce_modes.py --nproc-per-node 4 --dry-run
```

## 计时与正确性

默认 `--timing event` 使用当前 NPU 流上的开始/结束 Event。调用 `all_reduce` 使用 `async_op=False`，让通信完成依赖返回当前流；批次结束后同步，再读取 Event 耗时。Python 下发空隙仍可能影响 Event 区间，这不是脱离框架的纯硬件耗时。

每次计时前恢复原始输入，复制位于 Event 区间之外，避免重复原地 SUM 溢出。预热及正确性检查不计入样本。每个 rank 输入 `rank + 1`，用独立计算的 `world_size * (world_size + 1) / 2` 检查全部输出元素；BF16/FP16/FP32 相对容差分别为 0.01/0.001/1e-5，NaN/Inf 不通过。每个消息大小在计时前及每轮计时后校验。

要测量包含 Python 调用及等待设备完成的单次端到端延迟，单独运行：

```bash
python3 benchmarks/benchmark_all_reduce_modes.py \
    --nproc-per-node 8 --sizes 4K 64K 1M 16M \
    --timing wall --output-dir all_reduce_tp8_wall
```

`wall` 模式在输入恢复后先同步，再测量一次 `all_reduce + NPU synchronize`，包含同步开销。不要混合比较 `event` 和 `wall` 的数值。

每次调用的样本先跨 rank 取最大值，再对所有轮次样本计算中位数、均值、p95。p95 使用 nearest-rank 定义；它表示该微基准的慢尾延迟。这里只做每轮起点的 CPU barrier，不在每个样本前插入 barrier，因此样本是各 rank 按顺序发起的对应通信调用，并非严格同步的全局起止时间。

默认先 AI_CPU 后 AIV；接近 1 的收益建议独立重跑，并通过两次单模式运行反转顺序，排除温度、系统负载与顺序偏差。脚本没有模拟计算通信重叠、其他通信域并发或真实模型推理。

## 输出

终端显示每个消息大小的校验结果以及最终对比表：

```text
     Payload   AI_CPU avg   AI_CPU med   AI_CPU p95      AIV avg      AIV med      AIV p95 Speedup(med)
      4096 B          ...          ...          ...          ...          ...          ...         ...x
```

- `AI_CPU.json` / `AIV.json`：各 rank 的设备、主机、torch/torch_npu 版本、相关环境变量、每轮中位数、跨 rank 取最大值后的全部时延样本，单位微秒。
- `comparison.json`：包含两个原始报告和对比结果。
- `comparison.csv`：每个消息大小一行，提供两个模式的 AVG/中位数/p95、加速比和延迟下降百分比。
- AVG 是所有轮次样本的算术平均值，每个样本先跨 rank 取最大值；终端进度和最终对比表均显示 AVG，CSV/对比 JSON 字段为 `ai_cpu_mean_us`、`aiv_mean_us`，单位微秒。
- `speedup_ai_cpu_over_aiv = AI_CPU median / AIV median`，大于 1 表示请求 AIV 的运行更快。
- `aiv_latency_reduction_pct = (1 - AIV median / AI_CPU median) * 100`，正值表示延迟下降。

**结果标记的是请求的模式，并非确认过的执行引擎。** CANN 在部分消息规模、资源不足或确定性约束下可能回退到 AI CPU；不支持的配置也可能报错或使用默认模式。脚本不将回退数据伪装成“纯 AIV kernel”数据，也不改写继承的确定性配置。需要确认执行引擎时，应结合目标版本的 HCCL 日志/Profiling；检查 JSON 中的 `HCCL_DETERMINISTIC`、`HCCL_ALGO`、`HCCL_BUFFSIZE` 等记录。参考 [HCCL 展开模式文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/hcclug/docs/en/user_guide/hccl_env/HCCL_OP_EXPANSION_MODE.md)。

## 单模式与多机

只运行一个模式，适合反转实验顺序或附加 profiler：

```bash
python3 benchmarks/benchmark_all_reduce_modes.py \
    --mode AIV --nproc-per-node 4 --sizes 4K 64K 1M \
    --output-dir aiv_only_run
```

也支持外部 torchrun。跨节点使用常规 rendezvous 参数，所有 rank 使用一致的脚本参数和 HCCL 配置；按实际网络设置 `GLOO_SOCKET_IFNAME`、`HCCL_SOCKET_IFNAME` 等。外部 torchrun 中必须指定单模式，两个模式分别启动，不能在同一个运行中切换：

```bash
HCCL_OP_EXPANSION_MODE=AI_CPU \
python3 -m torch.distributed.run \
    --nnodes 2 --nproc-per-node 8 --node-rank "$NODE_RANK" \
    --master-addr "$MASTER_ADDR" --master-port 29500 \
    benchmarks/benchmark_all_reduce_modes.py \
    --mode AI_CPU --sizes 4K 64K 1M 16M --output-dir multinode_ai_cpu
```

作业完成后改为 `--mode AIV`，换用新的输出目录，并在所有节点上重启作业。外部 torchrun 路径只由全局 rank 0 写该模式的 JSON，不自动生成跨作业对比文件。
