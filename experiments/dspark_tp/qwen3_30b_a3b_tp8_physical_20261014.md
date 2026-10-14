# Qwen3-30B-A3B TP8 物理缩 K 与普通置信头对照（2026-10-14）

## 目的和版本

本轮专门验证解除 TP8 物理缩 K 回退后，新路径相对原有 DSpark 置信头的性能与输出一致性。被测 vLLM-Ascend 为 `spec1` 的 `21da649d2ad31780d108c5e459367fae78f7c8ce`，相关单测 `37 passed`。普通置信头与物理缩 K 都在这个提交上运行，避免跨提交比较。旧的无 DSpark/置信头测试见 [上一轮记录](qwen3_30b_a3b_8k_b256_20261014.md)。

| 项目 | 设置 |
| --- | --- |
| 远端 | `root@192.168.13.190`，容器 `wxx-vllm-022` |
| vLLM-Ascend | `/home/w00664509/dspark-vllm/hardware-vllm-ascend/vllm-ascend-spec1-20261013` |
| vLLM | `/home/w00664509/dspark-vllm/vllm-ced6857` |
| 目标模型 | `/home/vllm/weights/Qwen3-30B-A3B/` |
| DSpark 权重 | `/home/vllm/weights/CMB-Qwen3-30B-A3B-DSpark-preview` |
| 推理 | ModelRunnerV2、TP8（NPU 0–7）、`FULL_DECODE_ONLY`、异步调度 |
| 容量 | `max_model_len=8192`、`max_num_seqs=256`、`gpu_memory_utilization=0.93` |
| 负载 | 随机输入 7168 token + 强制输出 1024 token，256 请求/256 最大并发，temperature 0，seed 20261013，预热 4 请求 |
| 结果目录 | `experiments/dspark_tp/results_qwen3_30b_a3b_8k_b256_tp8_physical_enabled_20261014/tp8/`（仅保留在远端工作树） |

## 实际命令

容器内先加载 CANN 环境：

```bash
source /home/vllm/c00944022/cann/910/ascend-toolkit/set_env.sh
source /home/vllm/c00944022/cann/910/nnal/atb/set_env.sh
cd /home/w00664509/dspark-vllm/hardware-vllm-ascend/vllm-ascend-spec1-20261013
```

脚本为服务和 benchmark 子进程设置以下环境（在直接运行后面的 `vllm serve` 命令时需要自行导出）：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_PLUGINS=ascend
export PYTHONPATH=/home/w00664509/dspark-vllm/hardware-vllm-ascend/vllm-ascend-spec1-20261013:/home/w00664509/dspark-vllm/vllm-ced6857${PYTHONPATH:+:$PYTHONPATH}
export ASCEND_CUSTOM_OPP_PATH=/home/w00664509/dspark-vllm/hardware-vllm-ascend/vllm-ascend-spec1-sync-20260915/vllm_ascend/_cann_ops_custom/vendors/custom_transformer${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}
```

每组使用 `experiments/dspark_tp/run_case.py` 独立启动和关闭服务，生成 `metadata.json`、`server.log`、`accuracy.json`、`warmup.json`、`benchmark-1.json`。以下两次运行除 `--case` 外参数完全相同，按 `physical`、`adaptive` 的顺序串行执行：

```bash
python3 experiments/dspark_tp/run_case.py \
  --case physical --tp 8 \
  --ascend-source /home/w00664509/dspark-vllm/hardware-vllm-ascend/vllm-ascend-spec1-20261013 \
  --vllm-source /home/w00664509/dspark-vllm/vllm-ced6857 \
  --output-root experiments/dspark_tp/results_qwen3_30b_a3b_8k_b256_tp8_physical_enabled_20261014 \
  --model /home/vllm/weights/Qwen3-30B-A3B/ \
  --draft-model /home/vllm/weights/CMB-Qwen3-30B-A3B-DSpark-preview \
  --max-model-len 8192 --max-num-seqs 256 --gpu-memory-utilization 0.93 \
  --input-len 7168 --output-len 1024 --max-concurrency 256 \
  --num-prompts 256 --runs 1 --warmup-prompts 4 --startup-timeout 1200

# 第二组仅改为 --case adaptive，其他参数不变。
```

两组共同服务参数为：

```bash
vllm serve /home/vllm/weights/Qwen3-30B-A3B/ \
  --host 0.0.0.0 --port 8006 --max-model-len 8192 \
  --served-model-name auto --gpu-memory-utilization 0.93 \
  --tensor-parallel-size 8 --max-num-seqs 256 --async-scheduling \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --safetensors-load-strategy prefetch \
  --speculative-config '{"method":"dspark","model":"/home/vllm/weights/CMB-Qwen3-30B-A3B-DSpark-preview","num_speculative_tokens":5,"enable_adaptive_verification":true}'
```

物理缩 K 组额外添加：

```bash
--additional-config '{"dynamic_spec_config":{"method":"dspark","physical_k":{}}}'
```

两组均由脚本发起以下正式请求（预热和固定提示词精度检查在此之前）：

```bash
vllm bench serve --backend openai --base-url http://127.0.0.1:8006 \
  --endpoint /v1/completions --model auto \
  --tokenizer /home/vllm/weights/Qwen3-30B-A3B/ \
  --dataset-name random --random-input-len 7168 --random-output-len 1024 \
  --num-prompts 256 --max-concurrency 256 --request-rate inf \
  --temperature 0 --ignore-eos --seed 20261013 \
  --save-result --result-dir RESULTS_DIR --result-filename benchmark-1.json
```

## 结果与判读

在本轮 8K、256 并发在线负载中，打开 TP8 物理缩 K 路径**没有性能收益**：输出吞吐相对普通置信头低 **4.85%**（174.58 对 183.48 token/s），测量时长多 5.10%。两组均完成 256/256 请求、没有失败，输入和输出 token 总量完全相同。由于物理组始终选 K=5，此结论只能说明“开启该路径、但未真正缩 K”的性能差异，不能证明 K<5 时的效果。

| 组别 | 请求成功/失败 | 时长 | 输出吞吐 | 总 token 吞吐 | 平均 TTFT | 平均 TPOT | 相对普通置信头 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 普通置信头（`adaptive`） | 256/0 | 1428.74 s | 183.48 token/s | 1467.84 token/s | 513679 ms | 335.12 ms | 基线 |
| 置信头 + 物理缩 K（`physical`） | 256/0 | 1501.58 s | 174.58 token/s | 1396.63 token/s | 596288 ms | 349.89 ms | **−4.85%** |

两组的输入/输出 token 均为 1,835,008 / 262,144。服务日志中物理组最高 `Running: 91 reqs`，普通置信头组最高 90；客户端并发 256 不代表 NPU 同时运行 256 路。物理组的 TP8 日志生成了 `K=3/4/5` 的 profiling 成本表，没有 `ASCEND_AV_PHYSICAL_K_FALLBACK`；因此此轮确实进入物理缩 K 实现路径。启动图捕获数量为 116 个，普通置信头组为 48 个；图捕获和 profiling 的启动时间不计入上表的正式测量时长。物理组截至结束在日志记录的各 batch 桶（1、2、4、8、16、32、64、128）决策均为 `physical_k=5`，未观察到真正缩短 K。不能把本轮结果称为“实际 K 下降的收益”。

两组 `accuracy.json` 的 8 条固定自然语言提示词 greedy 输出逐字节一致；这能发现此样本集中的输出变化，不能替代广泛任务的精度评测。随机长输入压测检查了请求成功数与吞吐，不保存逐条生成文本作精度对比。单轮结果不提供跨轮方差，且随机 token 负载不代表真实业务提示词分布。

本轮相关单测在容器的上述 CANN 环境下执行：

```bash
PYTHONPATH=/home/w00664509/dspark-vllm/hardware-vllm-ascend/vllm-ascend-spec1-20261013:/home/w00664509/dspark-vllm/vllm-ced6857 \
  python3 -m pytest -q \
  tests/ut/spec_decode/test_hardware_aware_policy.py \
  tests/ut/spec_decode/test_physical_k_profile.py
```

结果为 `37 passed`。两组固定提示词输出使用 `cmp -s tp8/physical/accuracy.json tp8/adaptive/accuracy.json` 核对一致；`benchmark-1.json` 中的 `completed`、`failed` 和 token 总量用于确认两组负载相等。

旧一轮置信头组曾为 177.07 token/s；本轮同配置重测为 183.48 token/s（+3.62%）。被测提交只解除物理路径的 TP8 运行门槛，普通置信头不进入该路径；这一个旧/新单轮差异不能归因为代码提升，也说明跨轮波动存在。因此本轮结论只使用同时期的 183.48 token/s 作对照。
