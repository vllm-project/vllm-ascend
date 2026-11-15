# CatCCOS A5 替换 MC2 的 AISBench 服务 A/B 计划

## 目标与比较边界

本计划比较同一台 A5、同一个 Qwen3-30B-A3B-Instruct-2507 服务在可复现公开负载和业务负载下的
两种配置：原生 MC2 与仅在原生 MC2 区间启用 CatCCOS。记录以下两种收益：

1. 相同输入、输出和到达速率下的 TTFT、TPOT、E2E 延迟变化。
2. 满足预先确定的延迟 SLO 时，可持续承载的最高请求速率，以及无速率限制时
   的稳定吞吐峰值。

当前分支在 A5 上仅当原生选路本来是 MC2、CatCCOS 容量也足够、且不是 draft
model 时选择 `FUSED_MC2`。超过原生 MC2 容量时仍由原有 A5 规则选择
`ALLGATHER` 或 `ALLTOALL`。CatCCOS 的 BF16 回退权重也采用与原生相同的
`maybe_trans_nz` 规则。这个约束使 A/B 的主要变量落在 MC2 算子链上。

在本文的 TP4、EP4、topK8、`enable_prefill_mc2=true`、调度 budget≥2048
配置中，原生 MC2 上限是 global M=2048，即 rank-local M=512：

| 调度步 global M | native | CatCCOS 配置 |
| ---: | --- | --- |
| 1～2048 | `MC2` | `FUSED_MC2`，调用 CatCCOS |
| 2049 及以上 | `ALLGATHER` | `ALLGATHER` |

表中 M 是**一次调度步中进入 MoE 的 token 总数**，不是请求的 prompt 长度，也
不是 AISBench 客户端并发。chunked prefill 与连续 batching 都可能改变实际 M。
若调度 budget 小于 2048，实际 MC2 上限也相应降低。

## 1. 固定服务和软件基线

两侧固定同一份模型权重、vLLM 0.23.0、vLLM-Ascend 提交、CatCCOS `.so`
构建、CANN/驱动、卡 4～7、TP4/EP4/DP1、BF16、eager、模型长度、调度
budget、`max_num_seqs`、前缀缓存策略和环境变量。只改变 `additional-config`
中的融合后端开关。记录以下命令的输出随结果归档：

```bash
git rev-parse HEAD
sha256sum /home/z00956592/catccos/build_torch_a5/lib/libcatccos_torch.so
vllm --version
# 在 AISBench checkout 中另外运行 git rev-parse HEAD。
```

本仓库提供**无 profiler** 的同参服务脚本。两侧依次启动，不能同时占用同组卡：

```bash
cd /home/z00956592/vllm-ascend-catccos

MAX_NUM_BATCHED_TOKENS=4096 MAX_NUM_SEQS=8 \
  bash tools/catccos_profiling/start_aisbench_service.sh native \
  > /home/z00956592/aisbench_native_serve.log 2>&1

# 停止 native、确认 worker 退出后，再启动 CatCCOS。
MAX_NUM_BATCHED_TOKENS=4096 MAX_NUM_SEQS=8 \
  bash tools/catccos_profiling/start_aisbench_service.sh catccos \
  > /home/z00956592/aisbench_catccos_serve.log 2>&1
```

脚本默认 `CATCCOS_MAX_TOKENS_PER_RANK=512`、`CATCCOS_MIN_TOKENS=1`，
两侧都打开 `enable_prefill_mc2`。它关闭 MoE profiling 和同步诊断，并保持
`catccos_sync_after_launch=false`。`MAX_NUM_SEQS=8` 是已用配置；若要测生产
的更高活跃序列数，先确认内存允许，然后**两侧一起**改为 16 或 32。
`--max-model-len` 必须容纳本轮最长的输入与输出，脚本可通过 `MAX_MODEL_LEN`
设置。脚本默认关闭前缀缓存以排除命中率差异；实际部署若开启缓存，另做一套
两侧都开启、前缀分布相同的 A/B。

服务健康检查：

```bash
curl -fsS http://127.0.0.1:28001/health
```

## 2. 准备 AISBench 客户端

在压测机安装并固定一个 AISBench 提交；安装方法见
[AISBench 官方仓库](https://github.com/AISBench/benchmark)。压测机到服务的
网络路径和客户端 CPU 资源在两侧保持相同。

复制 AISBench 的 `vllm_api_stream_chat.py` 为独立模型配置，例如
`catccos_ab_stream.py`，编辑以下字段：

```python
path="/home/weights/Qwen3-30B-A3B-Instruct-2507"  # 客户端 tokenizer
model="qwen3-catccos"
host_ip="127.0.0.1"
host_port=28001
stream=True
batch_size=8                  # 本轮客户端最大在途请求数；逐档修改
request_rate=-1              # 不限速并发发送；固定 QPS 轮改为正数
max_out_len=256              # 本轮输出上限；逐档修改
generation_kwargs=dict(temperature=0, ignore_eos=True)
```

`batch_size` 是 AISBench 的客户端并发，不等于 vLLM `max_num_seqs`。
本计划按 [AISBench 性能测试文档](https://github.com/AISBench/benchmark/blob/master/docs/source_en/base_tutorials/scenes_intro/performance_benchmark.md)
使用 `request_rate=-1` 做不限速轮；
固定 QPS 轮则设置正数。执行前记录 AISBench 提交并核对该版本的参数语义。
固定长度对照设置 `ignore_eos=True`，让两侧生成相同数量的 token；自然停止
的业务轮使用 `ignore_eos=False`，但要同时核对两侧输出长度分布。

### GSM8K 子集：本轮可直接执行的样本数

本轮用完整的 `gsm8k_gen_0_shot_cot_str_perf` 数据集配置；固定 0-shot 或
4-shot 提示模板，不在 A/B 两侧切换。不要选 `demo_gsm8k_*`：demo 配置只
读取 8 条，不能用于正式性能结论。先用前 200 条检查服务、实际输入输出
token 数、所选通信路径和错误率；并发与 QPS 的正式比较先用**同一批前 500
条**，每档至少重复 3 轮：

```bash
# 快速试跑；两侧使用相同的模型、数据集和参数。
ais_bench --models catccos_ab_stream \
  --datasets gsm8k_gen_0_shot_cot_str_perf --mode perf \
  --num-prompts 200 --num-warmups 20 \
  --work-dir /home/z00956592/aisbench_results

# 正式轮；修改模型配置中的 batch_size 或 request_rate 后，两侧各运行。
ais_bench --models catccos_ab_stream \
  --datasets gsm8k_gen_0_shot_cot_str_perf --mode perf \
  --num-prompts 500 --num-warmups 20 \
  --work-dir /home/z00956592/aisbench_results
```

AISBench 的 `--num-prompts 500` 默认按数据顺序取前 500 条，不随机抽样；
若自定义数据集配置带 `reader_cfg.test_range`，应核对实际加载的条数与
顺序。每轮归档数据集文件哈希、提示模板和逐请求结果。前 500 条足以作为
同输入 A/B 的起点，但不能代表生产请求分布。若高并发档在 2 分钟内跑完，
将**两侧同档**请求数增加到 800～1000。500 条中的最慢 1% 仅约 5 条，
不能单靠它判定 P99 改善；关键档位应使用至少 1000 个完成请求并重复测试。
`ignore_eos=True` 的固定输出长度轮只用于性能比较，GSM8K 精度另用自然
停止的配置评估。

本轮 GSM8K 是可复现的公开负载。若要评估实际生产收益，还需用业务
请求或 ShareGPT 类混合输入做补充；保留同一份文件、顺序和请求数。
ShareGPT 示例：

```bash
ais_bench --models catccos_ab_stream --datasets sharegpt_gen \
  --mode perf --num-prompts 1000 --num-warmups 20 \
  --work-dir /home/z00956592/aisbench_results
```

先按 AISBench 数据集配置下载并固定 ShareGPT 文件；每轮使用相同的文件、顺序
和请求数。若有业务请求日志，优先构造 AISBench 自定义数据集，保留输入长度、
目标输出长度和到达时间分布。没有 trace 时至少分短输入、平衡、长输入三档，
并用业务采样的比例组合，不把 ShareGPT 的分布直接当成生产分布。

辅助固定负载使用 AISBench `synthetic_gen_tokenid.py` 的副本，修改
`RequestCount`、`TokenIdConfig.RequestSize` 和 `PrefixLen=0`；输出长度由模型
配置的 `max_out_len` 控制。建议先用 128/256、512/128、2048/1、4096/1
四组输入/输出档位定位 decode、混合、prefill 与原生回退。AISBench 的
`tokenid` 数据会先解码为字符串再发送，服务端重新分词后的长度可能不同；
必须用 AISBench 的实际 `InputTokens` 检查档位。若要精确指定 MoE M，使用已有
的 `prompt_token_ids` 定点脚本，而不要用这个合成模式推断实际选路。

## 3. 两种负载扫描

### 并发扫描：找无速率限制的稳定上限

对 GSM8K、每个固定负载以及业务混合数据集，令 `request_rate=-1`，依次将 `batch_size`
设为 1、4、8、16、32、64。每档至少预热 20 次，正式轮按吞吐选择请求数，
使有效测量持续约 2 分钟或更久。若客户端实际并发达不到设定值，先检查
压测机的 CPU/网络和 AISBench worker；正式轮不要加 `--debug`，它会限制
客户端并行能力。记录每档实际并发、失败数、请求/s、输入/输出 token/s、
TTFT/TPOT/E2E 的 P50/P95/P99。继续增大并发直到吞吐进入平台期，或
延迟/失败率明显恶化。服务 `MAX_NUM_SEQS=8` 时，客户端并发超过 8 主要增加
排队压力；要测 16 或 32 条同时执行的序列，须将两侧服务的 `MAX_NUM_SEQS`
一起提高并另做一组 A/B。

### 固定 QPS：找满足业务 SLO 的容量

先从 native 的并发扫描取得近似饱和请求速率 `R`。固定客户端并发上限，
两侧使用**相同的绝对发送速率**：`0.25R、0.5R、0.75R、0.9R、1.0R、1.1R`，
再在拐点附近加密。修改模型配置的 `request_rate` 后重新运行相同命令。
在测试前写明业务 SLO，例如 TTFT P99、TPOT P99 和允许的失败率；不能看完
结果再设门槛。低于饱和点时，两侧完成吞吐都接近发送 QPS，因此此时主要
比较延迟。最终容量收益按下式计算：

```text
容量收益 = CatCCOS 满足 SLO 的最高持续 QPS / native 满足 SLO 的最高持续 QPS - 1
```

若使用真实时间戳 trace，按同一 trace 回放两次，再进行速率缩放；不要同时
改变输入分布和到达速率。若生产有突发流量，另加一组突发与空闲交替的 trace。

## 4. A/B 顺序与归因

同一档建议按 native → CatCCOS → CatCCOS → native 交错重复，至少各 3 个
有效轮次。每轮重启服务、完成预热，再运行 AISBench。不要混用冷启动数据；
无响应、超时或失败的请求必须计入结果。保留 AISBench 生成的配置、逐请求
CSV/JSON、服务日志及实际 token 数。

正式吞吐轮**不开 profiler**。另用
[服务内 profiling 流程](catccos_service_profiling.md)抓短轮，按
`global_M` 和 backend 统计 `catccos`、`native_mc2`、`native_allgather` 的
样本数，确认收益对应的真实替换覆盖率。重点核对：

- global M≤2048 的 CatCCOS/native 分别是 `FUSED_MC2`/`MC2`；
- global M>2048 的两侧都是 `ALLGATHER`；
- 非 TP 整倍数的 M 输出形状和 token 顺序正确；
- CatCCOS finalize 有 TP gather，没有重复外层 TP all-reduce。

4096/1 档是回退控制组：若两侧主要走 ALLGATHER 却仍有显著性能差异，检查
实际选路、输出 token 数、权重格式、内存占用和其他服务配置，不能把差值归因
于 MC2 被替换。精度方面先做逐层输出对照和 GSM8K 等整网验证；CatCCOS 使用
MXFP8 专家权重，性能 A/B 不能代替精度验收。

最终按负载档位交付一张表，至少包含：git 提交、模型与服务参数、AISBench
提交、实际输入/输出 token 分布、实际 backend 占比、设定 QPS/并发、实际并发、
成功率、请求/s、输入/输出 token/s、TTFT/TPOT/E2E P50/P95/P99、三轮波动
和容量收益。只有数据分布、输出长度与真实选路对齐后，才能解释总体收益。
