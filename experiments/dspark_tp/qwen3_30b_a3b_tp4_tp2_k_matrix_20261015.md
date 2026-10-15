# Qwen3-30B-A3B：TP4/TP2 不同 K 的 DSpark 对照（2026-10-15）

## 结论

在本次 7168 输入 + 1024 输出、256 请求/256 客户端并发的单轮在线压测中，**TP4 与 TP2 的所有 K=3/5/8 DSpark 组都慢于各自不开 DSpark 的目标模型基线**。TP4 最好的 DSpark 组为物理 K3，382.13 对 550.35 输出 token/s（−30.57%）；TP2 最好的为普通置信头 K3，407.43 对 482.21 token/s（−15.51%）。物理缩 K 相比普通置信头有正有负，单轮结果不能证明稳定收益。

`K=3/5` 的物理路径在日志记录的各 batch 桶均保持原 K；`K=8` 仅在 batch 桶 2 曾降至 7（TP4 又降至 5），桶 4–128 仍为 8。因此高并发测量期间，物理路径基本没有真正缩短草稿长度。`K=8` 的 8 条固定提示词 greedy 输出与无 DSpark 基线出现 1–2 条不一致，不能宣称这一配置精度无影响；K3/K5 在这个小样本检查中全部逐字一致。

## 环境与可复现命令

| 项目 | 设置 |
| --- | --- |
| 主机/容器 | `root@192.168.13.190` / `wxx-vllm-022` |
| vLLM-Ascend | `spec1`，`a035e172708df2a117f651ca02e22ff60c2ff70b` |
| vLLM | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| 模型 | `/home/vllm/weights/Qwen3-30B-A3B/` |
| DSpark 权重 | `/home/vllm/weights/CMB-Qwen3-30B-A3B-DSpark-preview` |
| 运行 | `VLLM_USE_V2_MODEL_RUNNER=1`、NPU 0–TP−1、`FULL_DECODE_ONLY`、异步调度 |
| 服务容量 | `max_model_len=8192`、`max_num_seqs=256`、`gpu_memory_utilization=0.93` |
| 负载 | 随机输入 7168 token，强制输出 1024 token，256 请求，客户端最大并发 256，temperature 0，seed 20261013；每组预热 4 请求，正式测量 1 次 |

在容器内加载 CANN 并运行串行矩阵；每组独立启动服务、做 8 条固定提示词检查、预热、正式压测，然后关闭服务。已通过的目录在脚本重启时跳过，不重复覆盖：

```bash
source /home/vllm/c00944022/cann/910/ascend-toolkit/set_env.sh
source /home/vllm/c00944022/cann/910/nnal/atb/set_env.sh
python3 /home/w00664509/dspark-vllm/hardware-vllm-ascend/vllm-ascend-spec1-20261013/experiments/dspark_tp/run_30b_k_matrix.py \
  --ascend-source /home/w00664509/dspark-vllm/hardware-vllm-ascend/vllm-ascend-spec1-20261013 \
  --vllm-source /home/w00664509/dspark-vllm/vllm-ced6857 \
  --output-root /home/w00664509/dspark-vllm/hardware-vllm-ascend/vllm-ascend-spec1-20261013/experiments/dspark_tp/results_qwen3_30b_tp4_tp2_k3_k5_k8_8k_b256_20261015 \
  --tp 4 2 --k 3 5 8 --num-prompts 256 --max-concurrency 256 \
  --max-num-seqs 256 --gpu-memory-utilization 0.93 --startup-timeout 1200
```

`run_case.py` 为服务和压测子进程设置 `VLLM_PLUGINS=ascend`、`VLLM_USE_V2_MODEL_RUNNER=1`、`ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`（TP4）或 `0,1`（TP2），以及以下源码/算子路径：

```bash
export PYTHONPATH=/home/w00664509/dspark-vllm/hardware-vllm-ascend/vllm-ascend-spec1-20261013:/home/w00664509/dspark-vllm/vllm-ced6857${PYTHONPATH:+:$PYTHONPATH}
export ASCEND_CUSTOM_OPP_PATH=/home/w00664509/dspark-vllm/hardware-vllm-ascend/vllm-ascend-spec1-sync-20260915/vllm_ascend/_cann_ops_custom/vendors/custom_transformer${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}
```

下面是可直接运行的 TP4、K3 物理组服务示例。TP2 时把可见设备改为 `0,1`、TP 改为 2；其他 K 把 `K` 改为 5 或 8。普通置信头组去掉 `--additional-config` 一行；不开 DSpark 组同时去掉 `--speculative-config` 和 `--additional-config` 两行。所有组的完整展开命令也记录在各自的 `metadata.json` 的 `serve_command` 字段中。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_PLUGINS=ascend
TP=4
K=3
vllm serve /home/vllm/weights/Qwen3-30B-A3B/ \
  --host 0.0.0.0 --port 8006 --max-model-len 8192 \
  --served-model-name auto --gpu-memory-utilization 0.93 \
  --tensor-parallel-size "$TP" --max-num-seqs 256 --async-scheduling \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --safetensors-load-strategy prefetch \
  --speculative-config "{\"method\":\"dspark\",\"model\":\"/home/vllm/weights/CMB-Qwen3-30B-A3B-DSpark-preview\",\"num_speculative_tokens\":$K,\"enable_adaptive_verification\":true}" \
  --additional-config '{"dynamic_spec_config":{"method":"dspark","physical_k":{}}}'
```

正式压测的命令（其中 `RESULTS_DIR` 为各组目录）：

```bash
vllm bench serve --backend openai --base-url http://127.0.0.1:8006 \
  --endpoint /v1/completions --model auto \
  --tokenizer /home/vllm/weights/Qwen3-30B-A3B/ \
  --dataset-name random --random-input-len 7168 --random-output-len 1024 \
  --num-prompts 256 --max-concurrency 256 --request-rate inf \
  --temperature 0 --ignore-eos --seed 20261013 \
  --save-result --result-dir "$RESULTS_DIR" --result-filename benchmark-1.json
```

## 性能结果

所有 14 组 `completed=256`、`failed=0`，输入/输出 token 总量均为 **1,835,008 / 262,144**。吞吐差异并非请求数或 token 数不同导致。以下“相对目标”以同 TP 的无 DSpark 组为分母；“相对置信头”以同 TP、同 K 的普通置信头组为分母。

| TP | 组别 | 输出 token/s | 相对目标 | 相对置信头 | 时长 s | 草稿接受/提议 | 最大运行请求 | 最高 KV 使用率 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 不开 DSpark | 550.35 | 基线 | — | 476.32 | — | 237 | 99.8% |
| 4 | 置信头 K3 | 356.73 | −35.18% | 基线 | 734.86 | 32.52% | 110 | 52.1% |
| 4 | 物理 K3 | 382.13 | −30.57% | +7.12% | 686.01 | 32.61% | 110 | 51.9% |
| 4 | 置信头 K5 | 363.88 | −33.88% | 基线 | 720.42 | 25.30% | 89 | 42.3% |
| 4 | 物理 K5 | 325.79 | −40.80% | −10.47% | 804.63 | 24.13% | 92 | 43.6% |
| 4 | 置信头 K8 | 286.69 | −47.91% | 基线 | 914.39 | 17.51% | 71 | 34.3% |
| 4 | 物理 K8 | 301.72 | −45.18% | +5.24% | 868.84 | 17.55% | 74 | 35.9% |
| 2 | 不开 DSpark | 482.21 | 基线 | — | 543.64 | — | 81 | 100.0% |
| 2 | 置信头 K3 | 407.43 | −15.51% | 基线 | 643.42 | 31.35% | 69 | 99.8% |
| 2 | 物理 K3 | 376.75 | −21.87% | −7.53% | 695.80 | 33.11% | 70 | 100.0% |
| 2 | 置信头 K5 | 355.07 | −26.37% | 基线 | 738.30 | 25.42% | 68 | 100.0% |
| 2 | 物理 K5 | 376.40 | −21.94% | +6.01% | 696.46 | 24.98% | 68 | 99.9% |
| 2 | 置信头 K8 | 368.18 | −23.65% | 基线 | 711.99 | 17.51% | 66 | 99.9% |
| 2 | 物理 K8 | 334.61 | −30.61% | −9.12% | 783.42 | 17.96% | 66 | 99.8% |

“草稿接受/提议”为 `server.log` 全程 `SpecDecoding metrics` 窗口中 `Accepted` 与 `Drafted` token 累加之比，包含少量精度检查和预热，不是用户请求成功率。`最大运行请求` 和 KV 使用率为服务定期日志的采样最大值，不保证捕捉瞬时峰值。客户端允许 256 并发，不表示 NPU 同时运行 256 条。每组只做一次正式测量；物理相对置信头约 5–10% 的正负差异尚缺少重复实验来确认稳定性。

## 物理 K、瓶颈与精度

物理组均有 `ASCEND_AV_PHYSICAL_K_COSTS` 和 `ASCEND_AV_PHYSICAL_K_DECISION` 日志，且没有 `ASCEND_AV_PHYSICAL_K_FALLBACK`，所以确实进入物理路径。决策日志显示：

| 配置 | 实际记录的 K 决策 |
| --- | --- |
| TP2/TP4，K3 | batch 桶 1、2、4、8、16、32、64、128 均选 3 |
| TP2/TP4，K5 | 上述桶均选 5 |
| TP2，K8 | 仅桶 2 选 7，其余桶均选 8 |
| TP4，K8 | 桶 2 先选 7、后选 5；其余桶均选 8 |

**直接证据：** K 增大后提议 token 的接受比例从约 31–33%（K3）降到约 25%（K5）、约 17.5%（K8），大量后续草稿计算没有转化为输出。服务明确警告 speculative 配置将 `max_num_scheduled_tokens` 设为 2048，可能影响性能；还警告 `max_cudagraph_capture_size=512` 小于潜在的 decode token 需求（K3 为 1024、K5 为 1536、K8 为 2048）。TP4 目标模型采样最高 237 条运行请求、KV 使用 99.8%，而 speculative 组仅 71–110 条、KV 34–52%；TP2 speculative 组为 66–70 条、KV 接近 100%，低于目标模型的 81 条。

**解释与边界：** 这组长输入、高并发任务包含大量 prefill；speculative 主要缩短 decode 的串行步数，却额外执行草稿和验证。低接受比例、固定 2048 调度 token 限额以及 TP4 未充分利用的 KV/并发容量，共同符合“额外开销超过省下的目标模型步数”的现象；但本轮没有算子级 profile，不能定量分摊各项开销。TP2 的 KV 接近满载，显存/缓存容量也可能限制其并发。物理 K 在占主导的 batch 桶并未缩短，故不能把其与普通置信头的差异归因于“成功缩 K”；若要验证物理缩短的收益，应先让高负载桶实际选出较小 K，并重复测量。

固定 8 条自然语言提示词以 `temperature=0`、`max_tokens=64`、seed 20261013 测得的逐字文本差异如下。K3/K5 的所有 TP、两种 speculative 模式均为 **0/8** 差异；K8 出现差异：

| TP | 置信头 K8 vs 目标 | 物理 K8 vs 目标 | 物理 K8 vs 置信头 K8 |
| ---: | ---: | ---: | ---: |
| 2 | 2/8（提示词 2、3） | 1/8（提示词 3） | 1/8（提示词 2） |
| 4 | 1/8（提示词 6） | 2/8（提示词 5、6） | 1/8（提示词 5） |

两种 K8 路径均有不一致，**不能据此单独归因于新增物理缩 K 功能**；也不能排除 K8 验证路径、模型数值差异或其他非确定性因素。需另做重复短提示词/token 级比对定位。随机长输入压测只核对成功数和 token 总量，没有保存每条输出供精度比较。上述 8 题检查不是广泛任务的精度评测。

下一步应先重复 K8 的固定提示词并做首次分歧 token/目标模型 logits 对照，确认精度风险；在此之前不宜把 K8 作为推荐配置。性能方面优先以表现最好的 K3 做小矩阵，显式调高 `max_num_batched_tokens` 并匹配 decode 图捕获容量，观察实际运行请求数、KV 使用率与吞吐是否恢复；再采集分阶段 prefill、draft、verify 耗时，避免盲目调大 K。任何收益结论都需至少重复多轮，并用真实提示词分布验证。

## 原始数据与复核

远端原始目录：

```text
/home/w00664509/dspark-vllm/hardware-vllm-ascend/vllm-ascend-spec1-20261013/experiments/dspark_tp/results_qwen3_30b_tp4_tp2_k3_k5_k8_8k_b256_20261015/
```

矩阵总日志在 `/tmp/qwen3_30b_tp4_tp2_k_matrix_20261015.log`，汇总状态在根目录 `matrix-status.json`。每组路径为 `tp{2,4}/{target,adaptive-k3,physical-k3,adaptive-k5,physical-k5,adaptive-k8,physical-k8}/`，其中 `metadata.json` 给出完整启动命令和提交，`benchmark-1.json` 给出正式性能，`server.log` 给出接受率、调度告警、物理 K 决策和 KV 状态，`accuracy.json` 保存固定提示词输出，`warmup.json` 为预热结果。原始目录也复制到本地仓库外的 `E:\ascendproject\canndev4\vllm-ascend-new\benchmark-data-20261015\`，未加入 Git。

本轮使用的 benchmark 驱动测试在本地通过 `3 passed`，在远端 CANN 环境下与相关物理 K 单测合计 `40 passed`；本轮没有修改推理路径代码。所有 14 个 `metadata.json` 均记录同一被测 vLLM-Ascend/vLLM 提交及 `passed` 状态。
