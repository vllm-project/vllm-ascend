# Qwen3-30B-A3B 分层硬件感知优化跟踪（2026-10-15）

## 当前结论

**尚未证明新增物理缩 K 相对普通置信头有稳定净收益。**已知 TP2、随机输入 128 / 输出 512、并发 1 时 DSpark 自身相对不开 DSpark 有收益，详见 [前一轮报告](qwen3_30b_a3b_positive_dspark_physical_k_20261015.md)。但原来的物理 K3 默认下限也是 3，配置上不会缩 K，仍运行额外控制路径。优化后该无效配置直接回到普通置信头路径，避免为不存在的候选宽度付费；该改动的吞吐效果尚待空闲主机复测。

K5 有真实的 K4 候选，但当前两次物理 K5 服务启动所得成本表互相矛盾：第一次 batch=1 的 K4/K5 草稿成本为 **11.70/14.91 ms**，策略在 K4/K5 之间切换；第二次为 **25.86/25.85 ms**，策略只选 K5。代码原先按 K3 全部样本、K4 全部样本、K5 全部样本的顺序画像，候选宽度与设备预热/时钟漂移混在一起；现已改成 K3/K4/K5 交错采样，并保留原始上游 K5 样本的相对顺序。每个较短 K/形状由 2 次增至 3 次，以免一次冷启动样本把中位数拉高。新画像策略尚未完成性能验证，不能把代码逻辑改善等同于已测吞吐收益。

本次测量期间，主机有 **64 个逻辑 CPU、load average 约 108–114、`vmstat` 显示 runnable 约 91–113、CPU idle 0%**；容器外有多组 Python/Ray 进程。单流解码对 CPU 调度敏感，环境足以干扰跨服务启动的 A/B 比较。以下性能数值保留为探索数据，**不是发布/推荐配置的收益证明**。未结束其他用户的进程。

## 代码修正与验证

### 2026-10-16：单并发请求切换的策略污染

检查运行时路径发现，物理 K 的 scheduler 状态按 batch bucket 保存；并发 1 的所有独立提示词共用 bucket 1。前一条提示词若选择 K4，下一条提示词会继承该稳定宽度，直到完整 K 探针或周期重评分。worker 原先也只按 bucket 每 128 步评分，没有在请求 ID 改变时立即使用新请求的完整宽度置信度。已有固定题的第 5 位接受率在不同时间段从约 0.02 到 0.95 波动，这种跨请求继承不适合作为新提示词的初始策略。它是代码层面可确认的机制问题；现有吞吐波动不能单独归因于此，因为测量期间主机 CPU 饱和。

提交 `a5cf87e22` 在单并发请求 ID 改变后清除旧的 bucket-1 K 状态，从安全的完整 K 重新开始；worker 在新请求首次拥有完整宽度草稿时立刻重新评分，不必等待 128 步。窄宽度阶段不反复计算置信度。新增 scheduler/worker 回归测试，远端容器中相关三份测试文件 **70 passed**。这个修改未经过空闲主机的配对吞吐复测，不能宣称已经产生性能收益。

另一个已确认的限制是启动成本表不稳定：同样的 TP2、batch 1，K4/K5 草稿成本在两次启动中分别为 11.70/14.91 ms 和 25.86/25.85 ms，导致第一次可选 K4、第二次仅选择 K5。较早的顺序画像已改为交错采样，但新画像仍待性能验证。物理 K4 还会少提出一个候选 token；只有草稿时延节省超过该 token 的预期接受收益和额外控制开销，才可能跑赢普通置信头。固定题的接受率差异很大，因此需要按提示词和实际选 K 分层分析，而不是只看总体吞吐。

| 修正 | 原因 | 提交 |
| --- | --- | --- |
| `min_k >= configured K` 时不启用物理路径 | K3 默认下限 3 没有任何缩短候选，旧实现仍有 worker/scheduler 控制成本 | `42f8dc1f0` |
| 较短 K 每形状采样 3 次取中位数 | 旧实现只取 2 次；一次冷启动可把中位数抬高，先前 K1 batch=1 曾报 2984 ms | `42f8dc1f0` |
| 缩窄后首次完整 K 探测立即重评 | 原先仅每 128 步打分，可能错过间隔 32 步的完整宽度探测 | `fa8aa7c6b` |
| 不同 K 的 profile 样本交错 | 原先按 K 整批测量，启动预热/设备时钟漂移可改变候选成本排序 | `617031b61` |

远端容器使用与服务一致的 vLLM 源码路径，执行以下单测：

```bash
source /home/vllm/c00944022/cann/910/ascend-toolkit/set_env.sh
source /home/vllm/c00944022/cann/910/nnal/atb/set_env.sh
export PYTHONPATH=/home/w00664509/dspark-vllm/hardware-vllm-ascend/vllm-ascend-spec1-20261013:/home/w00664509/dspark-vllm/vllm-ced6857
cd /home/w00664509/dspark-vllm/hardware-vllm-ascend/vllm-ascend-spec1-20261013
python3 -m pytest -q \
  tests/ut/spec_decode/test_physical_k_profile.py \
  tests/ut/spec_decode/test_hardware_aware_policy.py \
  tests/ut/spec_decode/test_hardware_aware_runtime.py
```

结果：`68 passed`。Windows 本地环境没有 `torch`，所以相关单测是在远端容器运行；本地执行了 `py_compile` 与 `git diff --check`。

## 已完成的 K5 探索测量

所有组使用目标 `/home/vllm/weights/Qwen3-30B-A3B/`、草稿 `/home/vllm/weights/CMB-Qwen3-30B-A3B-DSpark-preview`，TP2、ModelRunnerV2、`FULL_DECODE_ONLY`、异步调度、`max_model_len=2048`、`max_num_seqs=4`、显存比例 0.93。每组独立启动服务，8 条固定提示词检查、4 请求预热，随后 16 请求、并发 1、输入 128 / 强制输出 512、temperature 0、seed 20261013，正式测量两轮。每轮均 `completed=16`、`failed=0`；三组的 8 题逐字一致。

| 组别 | vLLM-Ascend 提交 | 两轮输出 token/s | 运行决策 | 判读 |
| --- | --- | ---: | --- | --- |
| 普通置信头 K5 | `b2625535b` | 27.02 / 27.07 | 无物理 K | 这次较稳定的对照 |
| 物理 K5，3 次低 K profile | `42f8dc1f0` | 23.04 / 31.69 | K4/K5 交替 | 一慢一快，未证实净收益 |
| 物理 K5，加完整宽度探测重评 | `fa8aa7c6b` | 24.22 / 16.10 | 仅 K5，成本表排除了 K3/K4 | 未实际缩 K；主机 CPU 饱和，不能单独归因于控制逻辑 |

上述三组的同序号“轮次”不是在同一进程/同时段并行的严格配对。普通置信头与物理组的服务启动间隔较长，且未持续记录各时段的 CPU 负载；不能直接把 16.10 或 31.69 token/s 全部归因于代码差异。要证明收益，须在主机负载可控时按交错顺序重复启动 A/B，并同时确认运行确实选到小于 5 的 K。较早的 K8/8K 负载有固定题 greedy 不一致，不能用 K8 代替安全的 K5 搜索。

## 复现命令与原始路径

容器内在上述 CANN 环境中运行；`--case adaptive` 不指定 `--physical-min-k`，物理组使用 `--case physical --physical-min-k 3`。各组使用唯一 tag，避免覆盖既有数据：

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
  --num-speculative-tokens 5 --physical-min-k 3 \
  --result-tag in128-out512-c1-n16-k5-p3
```

容器内服务、精度检查、预热及 benchmark 的完整实际命令保存在各组 `metadata.json` 和 `benchmark-{1,2}.log`。原始共享目录：

```text
/home/w00664509/dspark-vllm/hardware-vllm-ascend/vllm-ascend-spec1-20261013/experiments/dspark_tp/results_qwen3_30b_positive_search_20261015/tp2/
  adaptive-in128-out512-c1-n16-k5/
  physical-in128-out512-c1-n16-k5-p3/
  physical-in128-out512-c1-n16-k5-probe/
```

其中 `server.log` 包含 `ASCEND_AV_PHYSICAL_K_COSTS`、`ASCEND_AV_PHYSICAL_K_DECISION`、接受率及 KV/并发状态；`accuracy.json` 为 8 条固定题输出，`benchmark-{1,2}.json` 为吞吐及延迟。不要重用这些输出目录。最新交错采样提交 `617031b61` 尚无性能结果；下一步在 CPU 不饱和的窗口，先对同一负载做交错 A/B，再根据实际 K 决策决定是否调整候选宽度或负载。
