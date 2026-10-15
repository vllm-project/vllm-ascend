# Qwen3-30B-A3B：DSpark 正收益配置与物理 K 分析（2026-10-15）

## 结论

在 **TP2、随机输入 128 token、强制输出 512 token、客户端并发 1、K3** 的这组 decode-heavy 负载上，普通自适应置信头 DSpark 相对不开 DSpark 的目标模型，两轮输出吞吐分别提高 **51.81% / 53.66%**。这是本次搜索中测到的正收益配置，不代表 8K 长输入、高并发负载也有收益；后者的独立结果见 [TP4/TP2 K 矩阵](qwen3_30b_a3b_tp4_tp2_k_matrix_20261015.md)。

相同配置下，默认物理 K3 比普通置信头 K3 两轮分别慢 **17.56% / 23.92%**，但仍比不开 DSpark 快 **25.15% / 16.91%**。关键是：默认配置的候选集合只有 `(3,)`，运行决策也始终为 K3，因此**没有发生物理缩 K**。显式允许 `min_k=1` 后，候选集合变为 `(1, 2, 3)`，但 batch=1 的成本表仍将 K3 评为最便宜，运行仍选 K3；这组吞吐相对普通置信头一轮低 24.60%、另一轮高 33.27%，表明短样本测试波动明显。不能将默认物理组的降速归因于“缩短草稿”，也不能凭现有数据定量认定某一个物理路径操作就是降速根因。

四组的 8 条固定提示词 greedy 文本彼此逐字一致；每次正式压测均完成 16/16 请求、失败 0，输入/输出 token 总量同为 2048/8192。该检查范围有限，随机压测没有保存逐请求输出，不能替代完整精度评测。

## 环境和启动命令

| 项目 | 值 |
| --- | --- |
| 远端/容器 | `root@192.168.13.190` / `wxx-vllm-022` |
| 本地仓库/分支 | `E:\ascendproject\canndev4\vllm-ascend-new\vllm-ascend` / `spec1` |
| 远端仓库 | `/home/w00664509/dspark-vllm/hardware-vllm-ascend/vllm-ascend-spec1-20261013` |
| vLLM-Ascend / vLLM 提交 | `097aa51d5631f0497f53133069bf51c4bbd5fe09` / `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| 权重 | 目标 `/home/vllm/weights/Qwen3-30B-A3B/`；草稿 `/home/vllm/weights/CMB-Qwen3-30B-A3B-DSpark-preview` |
| 服务 | ModelRunnerV2、Ascend NPU 0/1、`FULL_DECODE_ONLY`、异步调度、`max_model_len=2048`、`max_num_seqs=4`、`gpu_memory_utilization=0.93` |
| 压测 | `temperature=0`、`ignore_eos`、seed `20261013`、每组独立服务、4 请求预热、16 请求正式测量 2 次、并发 1 |

以下为容器内直接启动普通置信头 K3 服务的完整命令。先进入 `wxx-vllm-022`，并加载环境：

```bash
source /home/vllm/c00944022/cann/910/ascend-toolkit/set_env.sh
source /home/vllm/c00944022/cann/910/nnal/atb/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_PLUGINS=ascend
export PYTHONPATH=/home/w00664509/dspark-vllm/hardware-vllm-ascend/vllm-ascend-spec1-20261013:/home/w00664509/dspark-vllm/vllm-ced6857${PYTHONPATH:+:$PYTHONPATH}
export ASCEND_CUSTOM_OPP_PATH=/home/w00664509/dspark-vllm/hardware-vllm-ascend/vllm-ascend-spec1-sync-20260915/vllm_ascend/_cann_ops_custom/vendors/custom_transformer${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}
vllm serve /home/vllm/weights/Qwen3-30B-A3B/ \
  --host 0.0.0.0 --port 8006 --max-model-len 2048 \
  --served-model-name auto --gpu-memory-utilization 0.93 \
  --tensor-parallel-size 2 --max-num-seqs 4 --async-scheduling \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --safetensors-load-strategy prefetch \
  --speculative-config '{"method":"dspark","model":"/home/vllm/weights/CMB-Qwen3-30B-A3B-DSpark-preview","num_speculative_tokens":3,"enable_adaptive_verification":true}'
```

不开 DSpark 时，去掉 `--speculative-config`；默认物理 K3 时，在置信头命令末尾追加 `--additional-config '{"dynamic_spec_config":{"method":"dspark","physical_k":{}}}'`；允许最低 K1 时把该参数改为 `'{"dynamic_spec_config":{"method":"dspark","physical_k":{"min_k":1}}}'`。每组 `metadata.json` 中的 `serve_command` 保存了实际展开的参数。

正式压测命令（`RESULTS_DIR` 指向下表对应目录，每轮分别使用 `benchmark-1.json` / `benchmark-2.json`）：

```bash
vllm bench serve --backend openai --base-url http://127.0.0.1:8006 \
  --endpoint /v1/completions --model auto \
  --tokenizer /home/vllm/weights/Qwen3-30B-A3B/ \
  --dataset-name random --random-input-len 128 --random-output-len 512 \
  --num-prompts 16 --max-concurrency 1 --request-rate inf \
  --temperature 0 --ignore-eos --seed 20261013 \
  --save-result --result-dir "$RESULTS_DIR" --result-filename benchmark-1.json
```

实验实际由 `experiments/dspark_tp/run_case.py` 串行完成服务启动、8 题检查、预热、正式压测及服务退出。容器内可按以下命令复现一组；切换 `--case target|adaptive|physical` 并为每组使用独立 `--result-tag`。最低 K1 组额外指定 `--physical-min-k 1`。复现时使用新的 `--output-root` 或 tag，不要覆盖下文原始数据：

```bash
python3 /home/w00664509/dspark-vllm/hardware-vllm-ascend/vllm-ascend-spec1-20261013/experiments/dspark_tp/run_case.py \
  --case physical --tp 2 \
  --ascend-source /home/w00664509/dspark-vllm/hardware-vllm-ascend/vllm-ascend-spec1-20261013 \
  --vllm-source /home/w00664509/dspark-vllm/vllm-ced6857 \
  --output-root /home/w00664509/dspark-vllm/hardware-vllm-ascend/vllm-ascend-spec1-20261013/experiments/dspark_tp/results_qwen3_30b_positive_search_20261015 \
  --model /home/vllm/weights/Qwen3-30B-A3B/ \
  --draft-model /home/vllm/weights/CMB-Qwen3-30B-A3B-DSpark-preview \
  --max-model-len 2048 --max-num-seqs 4 --gpu-memory-utilization 0.93 \
  --input-len 128 --output-len 512 --max-concurrency 1 \
  --num-prompts 16 --runs 2 --warmup-prompts 4 --startup-timeout 1200 \
  --num-speculative-tokens 3 --result-tag in128-out512-c1-n16
```

## 测量结果

| 组别 | 输出 token/s：第 1 / 2 轮 | 相对目标：第 1 / 2 轮 | 相对普通置信头：第 1 / 2 轮 | 平均 TPOT ms：第 1 / 2 轮 | 固定 8 题差异 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 不开 DSpark | 14.98 / 11.96 | 基线 | — | 66.39 / 83.27 | 基线 |
| 普通置信头 K3 | 22.74 / 18.38 | +51.81% / +53.66% | 基线 | 43.49 / 53.89 | 0/8 |
| 物理 K3，默认 `min_k=3` | 18.75 / 13.98 | +25.15% / +16.91% | −17.56% / −23.92% | 52.78 / 70.98 | 0/8 |
| 物理 K3，`min_k=1` | 17.15 / 24.49 | +14.46% / +104.79% | −24.60% / +33.27% | 57.82 / 40.32 | 0/8 |

TPOT 是每个输出间隔的均值；投机解码可一次接受多个 token，不能把 ITL 和单个 target decode 步耗时直接等同。以上同轮序号来自不同时间启动的独立服务，属于配对负载而非同时 A/B；每组的两轮共享同一服务。默认物理组相对普通置信头的方向两轮一致，但最低 K1 组方向相反，尚不足以断言路径本身有稳定的 20% 开销。小样本和随机提示词尤其容易受草稿接受模式及运行波动影响，下一步应采用交错启动顺序、更多请求及算子级计时来估计置信区间。

在同样 TP2、输入 256 / 输出 512、并发 4 的较长验证中，每轮 64 请求、16 预热：不开 DSpark 为 **56.51 / 55.69**，普通置信头 K3 为 **54.45 / 55.70 token/s**，没有正收益。先前只用每轮 16 请求的同配置探索结果为目标 **46.47 / 59.39**、置信头 **79.44 / 53.39**，两轮结论相反，因此未采用那个配置作为正收益依据。

## 物理 K 决策、接受率与瓶颈

- 默认物理组日志：`ASCEND_AV_PHYSICAL_K_COSTS candidates=(3,)`，batch 桶 1 的 `physical_k=3`。源码 `resolve_physical_k()` 默认 `min_k=3`，所以配置上就不可能从 K3 缩短。
- 显式 `min_k=1` 组：候选 `(1, 2, 3)`；batch 桶 1 的草稿成本分别约 **2984.21 / 30.68 / 19.68 ms**，决策仍为 `physical_k=3`。K1 成本异常高，可能是 profile 样本/图路径问题，不能把它直接解释为稳定运行时 K1 延迟；但成本表和决策足以确认本轮没有采用 K1/K2。
- 服务日志中 `Accepted/Drafted` 窗口累计：普通置信头 **11084/23406 = 47.35%**，默认物理 **11105/23442 = 47.37%**，最低 K1 组 **10789/24075 = 44.81%**。这些窗口覆盖固定题、预热和两轮压测，并非只对应正式压测；默认物理与普通置信头接受率基本相同，不能用接受率下降解释它的吞吐差。
- 所有组的服务日志采样均为最多 **1 条运行、0 条等待**，NPU KV 使用率最高 **0.1%**。因此本配置不是被 KV 容量或高并发队列卡住；它主要测单流 decode。
- 默认物理组虽然没有缩 K，仍进入了物理 K 的 worker/scheduler 包装与成本决策路径。额外的控制、状态传递或图分支是可疑开销来源，但这轮没有逐步 profile，且最低 K1 组的第二轮反向波动很大，所以**只能把它列为待验证机制，不能定量归因**。实际缩 K 的收益或劣化在此正收益配置上尚未被测到。

此前 TP2/TP4 的 K8、8K 长输入矩阵中出现固定提示词 greedy 不一致；本轮 K3 的 8 题一致不消除 K8 风险，故不把 K8 当作推荐配置。若继续优化，优先重测 batch=1 的 K1/K2/K3 草稿图计时，检查 K1 的 2984 ms 异常，再做固定 K 宽度、交错顺序的重复压测。只有运行日志明确出现 `physical_k<3` 后，才能讨论真正的“缩 K 性能”。

## 原始数据

共享远端根目录：

```text
/home/w00664509/dspark-vllm/hardware-vllm-ascend/vllm-ascend-spec1-20261013/experiments/dspark_tp/results_qwen3_30b_positive_search_20261015/
```

| 组别 | 根目录下路径 |
| --- | --- |
| 正收益目标 | `tp2/target-in128-out512-c1-n16/` |
| 正收益置信头 | `tp2/adaptive-in128-out512-c1-n16/` |
| 默认物理 K3 | `tp2/physical-in128-out512-c1-n16/` |
| 最低 K1 对照 | `tp2/physical-in128-out512-c1-n16-mink1/` |
| 并发 4 长验证目标/置信头 | `tp2/target-in256-out512-c4-n64-w16/`、`tp2/adaptive-in256-out512-c4-n64-w16/` |
| 并发 4 短探索目标/置信头 | `tp2/target-in256-out512-c4/`、`tp2/adaptive-in256-out512-c4/` |

各目录的 `metadata.json` 包含提交和服务命令，`benchmark-{1,2}.json` 包含正式统计，`benchmark-{1,2}.log` 是压测过程，`server.log` 包含接受率、物理 K 成本/决策及调度/KV 状态，`accuracy.json` 保存固定提示词的完整输出，`warmup.json` 为预热结果。目标并发 1 组最初误写入仅容器可见的 `/home/w00664509/dspark-vllm/hardware-vllm-ascend/vllm-ascend-spec1-20261015`，完成后用 `docker cp` 将整组结果复制到上面的共享目录；原件仍在容器内。此输出目录差异不改变服务配置，但复现时应使用共享目录。
