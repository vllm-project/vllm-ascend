# MiniMax-M3 QK norm、RoPE 与 KV cache 写入融合

## 实现

参考 Ascend PR [#15772](https://github.com/vllm-project/vllm-ascend/pull/15772)
和 vLLM 的
[`fusedMiniMaxQNormRopeKVInsertKernel`](https://github.com/vllm-project/vllm/blob/7566d83bd35cdfd7cd3f9e521a005d2fc1ed1c33/csrc/libtorch_stable/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu)。
实现独立于 #15772，可直接用于 Ascend main。

一个 Triton kernel 完成四组 Gemma RMSNorm、partial NeoX RoPE 和三组 cache 写入。
每个 program 处理一个 token，将主 Q/K、index Q/K 分别合并为连续 head 行向量计算，
直接按 positions 读取 cos/sin；不再调用外部 RoPE gather、slice contiguous 或 cache scatter。
只输出 Q 和 index Q，K/V/index K 直接写 cache。

输入支持 `[Q | K | V | index Q | index K]` 融合投影，也支持独立的 QKV/index 投影。
后者适用于 QKV 为 W8A8、index 为 BF16 的部署，不需要额外 concat/copy。
投影 GEMM 沿用模型原路径，不计入本次 prepare/cache 融合区域。

index 与主 QKV 无数据依赖，但分成两个 program 组不一定更快。
A3 上 192 tokens、8/1/1 heads 的对照：分组向量化后两路调度约 18.23μs，
同一个 program 处理两组约 14.40μs，因此采用后者，不引入多 stream/event 同步。
这属于实测选型，并不意味着所有 shape 都不适合并行。

模型通过 `minimax_m3_fused_sparse_forward` 不透明算子调用融合路径，
随后执行 indexer 和 sparse attention；不重复调用原 cache 插入。
无 attention metadata 的 dummy forward 只清零输出，不访问 cache。

## 适用范围与语义

- BF16 NPU 输入，main/index head dimension 均为 128，最多 512 tokens（含图 padding）。
  其他配置保留原路径；未扩大到大规模 prefill。
- 一维 positions、NeoX partial/full RoPE、四组 norm 使用相同 epsilon。
- 主 cache 为 `[num_blocks, block_size, heads, head_dim]`，index cache 为
  `[num_blocks, block_size, head_dim]`。按实际 stride 寻址，支持 block padding。
- **num_blocks 是动态分配容量，不硬编码为 5392，也不作为 kernel 编译参数。**
  main/index block size 可以不同。
- 独立 main/index slot mapping；负 slot 和实际 token 数之外的 padding 不写 cache。
  调度器须保证有效 slot 不越界、同一次调用无重复写入目标。
- FP32 norm，RoPE 前保留 BF16 舍入；主 K 在 cache 转换前保留模型 dtype 舍入。
  FP8 E4M3 cache 转换前 clamp 到 ±448，FP8 实机验证需要 A5。
- 不增加环境变量，不改变 MoE、FlashComm 或投机采样策略。

## A3 单算子性能

197 服务器，A3，CANN 9.2.0、torch 2.10.0、torch_npu 2.10.0.post4、
triton 3.5.0 / triton_ascend 3.2.2。BF16 输入，Q/KV/index heads 为 8/1/1，
head dimension 128，rotary dimension 64，block size 128，5392 blocks，独立 index 投影。

baseline 对应模型独立 index 投影的 prepare/cache 算子链：QKV RMSNorm/RoPE、
两次 index slice contiguous、两次 Gemma RMSNorm、index RoPE、主 KV 和 index K 插入。
包含运行时的主 Q/K `1 + weight`；两条路径都从投影输出开始计时。

| Tokens | Baseline（μs） | 融合（μs） | 加速比 |
| --- | --- | --- | --- |
| 1 | 32.73 | 3.07 | 10.67× |
| 4 | 37.40 | 3.45 | 10.85× |
| 8 | 41.71 | 3.66 | 11.41× |
| 16 | 50.96 | 4.55 | 11.20× |
| 32 | 59.96 | 6.66 | 9.01× |
| 64 | 64.47 | 8.48 | 7.60× |
| 128 | 67.47 | 11.34 | 5.95× |
| 192 | 76.05 | 14.40 | 5.28× |
| 256 | 86.53 | 17.50 | 4.95× |

16/1/1 heads、192 tokens 的独立测量为 **82.92 → 16.19μs**。
198 服务器复测 8/1/1 heads：192 tokens 为 **79.30 → 14.15μs**，
256 tokens 为 **85.54 → 17.20μs**，512 tokens 为 **104.93 → 28.63μs**；
16/1/1 heads、192 tokens 为 **82.58 → 15.95μs**。
上述数据为 ACLGraph 中每次捕获 20 次调用、重放 10 次、五组 NPU Event 测量的中位数。
没有把 gather/copy 挪到计时区外；投影输入和 cache 分配在两条路径计时前完成。

单次 eager profiler：197 的 16/1/1-head 融合 kernel 为 20.56μs，
198 的 8/1/1-head 为 29.16μs，融合区域均只有一个 kernel。
该采样与重复调用的 ACLGraph 中位数是不同指标，不能混用。
另一次同轮核对中，Event 为 14.40μs，profiler 内五次图重放的 kernel 为 21.22～21.36μs。
因此 **profiler 口径尚未达到 20μs**，交付目前测试中性能最好的正确版本。
20μs 仅在不带 profiler 的 Event 测量中达到；不能将其当作 PD 服务内延迟保证。
H20 的 3μs 是用户提供的数据，本次未复测。

## 验证与复现

```bash
pytest -q tests/ut/models/minimax_m3
pytest -q tests/e2e/pull_request/one_card/test_minimax_qknorm_rope_cache.py
python benchmarks/minimax_qknorm_rope_cache.py \
  --model-baseline --separate-index --cache-blocks 5392 \
  --tokens 1 4 8 16 32 64 128 192 256
python benchmarks/minimax_qknorm_rope_cache.py \
  --model-baseline --separate-index --q-heads 16 --tokens 192 \
  --profile-dir /tmp/minimax-cache-profile
```

UT 检查融合条件、fallback、dummy/fake、独立 index 投影无需 concat，以及 cache 插入顺序。
NPU 测试覆盖 1～513 tokens、多种 head 布局、partial/full RoPE、独立乱序/负 slot、
未写区域、非连续输入/cache、空输入、图重放更新 positions/slots 和并发内存流。
动态 cache 测试覆盖 3/64/5392/8192 blocks，访问分配末尾，positions 覆盖 0～262999。

197、198 两台机器均验证通过：模型 UT 118 项，NPU 测试 67 项，2 项 FP8/A5 用例跳过。

## PD 部署验证边界

按官方 5.3 节尝试 197/198 的 A3 1P1D：Prefill DP2/TP4/PP2，Decode DP4/TP4/PP1，
真实 W8A8 权重、EAGLE3（三个 draft tokens）、EP，Prefill FlashComm1，Decode ACLGraph。
Decode 四个 API 启动完成，197 跨机访问全部返回 health 200。
Prefill 的 Mooncake Ascend Direct 初始化失败：多个设备报
`EI0009: Device transport init error. Reason: The network port is down.`，
继而 `TransferEngine initialization failed with ret_value: -1`、Worker 退出。
主机 hccn_tool 确认所检查的设备 0、12 链路为 DOWN。
因此没有成功的 PD 请求、端到端精度或吞吐结果；本次按单算子方案验收性能。
未修改服务器物理网络配置，未将启动成功当作推理通过，也未用 dummy 权重代替真实验证。

原始 benchmark、测试、profiler 和部署日志保存在两台服务器的
`/home/codex-minimax-kv192-20261011/`；部署错误见
`runs/pd-baseline/logs/service.log`，单算子数据见 `runs/general-combined0.log`
和 `runs/final-q16.log`。完整 Linux pre-commit 受 Ruff 依赖安装失败阻断；
Windows 可执行的全部 hooks 通过，Linux 的 gitleaks、shellcheck、logger 检查单独通过。
