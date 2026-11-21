# Rex-Omni 单卡（Ascend 910B3）部署优化方法论

> 状态：已完成。基线、消融、全量验证与选型数据均落盘于 `results/`，日志在 `logs/`。
> 结论先行：默认部署配置取 **batch=200（一次性提交）+ max_pixels=1003520 + max_tokens=2048**
>
> + parser 去重 + 提前停止 + `async_scheduling=False` + `FULL_DECODE_ONLY` graph
> + `max_num_batched_tokens=32768` + 并行输入预处理（`REX_OMNI_VIS_WORKERS=8`、
> `--load-workers 8`）。全量 200 张 **3.65 img/s / 54.8 s**，相比初始可复现基线
> 0.472 img/s / 424 s **提升 7.74×**，精度在 run-to-run 噪声内（F1@0.5 0.755 vs
> 基线 0.690，mIoU-F1 0.549 vs 0.508，n=200）。
> 分阶段收益：参数 4.47× → 去重（精度）→ 提前停止 +17% → FULL graph +27% →
> 32k 分块 +3% → 并行预处理 +14.2%。

## 1. 目标与约束

+ 硬件：单卡 Ascend 910B3（64 GB HBM）
+ 模型：Rex-Omni（Qwen2.5-VL-3B，7.6 GB bf16）
+ 任务：开放词表目标检测（Dense200 box_eval）
+ 目标：精度不显著下降的前提下，降低单图延迟、提升吞吐
+ 精度容差：本环境下全量 200 张的 F1 run-to-run 波动约 **0.008–0.014**，
  小于该幅度的配置差异不应解读为真实精度变化（详见 §4.5）

## 2. 可复现基线协议

| 项 | 固定值 |
|---|---|
| 数据集 | Dense200 box_eval，标注 200 条 |
| 图像根目录 | `/home/workDir/rex_omni_eval/Rex-Omni-Eval` |
| prompt | `Detect [OBJ]. Output the bounding box coordinates in [x0, y0, x1, y1] format.` |
| 采样 | temperature=0, top_p=0.8, top_k=1, repetition_penalty=1.05 |
| 默认 max_tokens | 4096 |
| 默认 max_model_len | 4096 |
| min_pixels / max_pixels | 16×28×28 / 2560×28×28（2,007,040） |
| warmup | 1 张（不计入统计） |
| 精度指标 | 官方 `evaluation/metrics/other_metric.py`，IoU 0.5 P/R/F1 + IoU 0.5–0.95 平均 F1（下称 mIoU-F1） |
| 性能指标 | 冷加载时间、总推理耗时（不含加载）、img/s、输出 tok/s、平均输出 token |

> 说明 1：官方 `inference_text_prompt.py` 硬编码 `.to("cuda")`，NPU 上不可直接运行；
> 使用 `rex_omni.RexOmniWrapper`（vLLM backend）等价推理，输出经 `convert_for_metric.py`
> 转成官方指标格式。绝对精度可能与官方榜单有解析器差异，横向对比使用同一链路。
>
> 说明 2：**greedy 解码在本环境不是 batch-invariant**。同一批数据在 batch=64 与 batch=200
> 下仅 16/200 的原始输出完全一致（见 §4.5）。因此所有精度对比必须固定 batch 组成，
> 且单次差异小于噪声带时不下结论。

## 3. 基线结果

全量 200 张、固定分块 batch=64：

| 配置 | batch | 加载(s) | 总耗时(s) | img/s | 输出 tok/s | F1@0.5 | mIoU-F1 |
|---|---|---|---|---|---|---|---|
| base200_b64 | 64（64/64/64/8） | 60.7 | 424.1 | 0.472 | 290.9 | 0.6896 | 0.5075 |
| base20（参考，bs=1，仅前 20 张） | 1 | 76.5 | 434.5 | 0.046 | 32.4 | 0.6094 | 0.4205 |

全量基线细节：

+ 平均 prompt 1378.5 tok，平均输出 615.4 tok（P50 292，P90 1551，max 3800）
+ **26/200 的输出被 `max_model_len=4096` 截断**（`prompt+output` 顶到 4096）
+ 末尾 8 张的尾批仍耗时约 101 s，同步分块的长尾代价显著

## 4. 瓶颈分析

### 4.1 decode 主导

输出 token 呈双峰：简单图 10–400 tok，密集图 1600–3800 tok。prefill（图像 token）均值约 1346，
占比小于 decode。降低分辨率的收益有限，**减少无效输出 token 和消除批间屏障收益更大**。

### 4.2 低并发下 NPU 算力未饱和

3B 模型 bs=1 时 decode 仅 32–36 tok/s；bs=64 时 291 tok/s；bs=200 时 754 tok/s。
典型 memory-bound 场景，**并发是第一优先级**。

### 4.3 同步分块的批间屏障

`batch_size=64` 把 200 张切成 4 块，块间必须等最慢请求结束。改为一次性提交 200 条后，
vLLM 的 continuous batching 让尾批与其他请求重叠：同样 4096 token 预算下
**0.472 → 1.219 img/s（2.58×）**，总耗时 424 s → 164 s。

### 4.4 退化重复（关键发现）

超长输出并非有效检测框，而是模型陷入**退化重复**：同一个框（如 `<0> <0> <15> <2>`）
被循环生成。量化结果：

| 运行 | 退化样本（同一框重复 ≥20 次） | 单条最大重复次数 |
|---|---|---|
| base200_b64 | 26/200 | 753 |
| opt200_b200 (mt=4096) | 28/200 | 753 |
| opt200_b200_tok2048 | 24/200 | 403 |
| opt200_b200_acc (mml=8192) | 25/200 | 1572 |

推论：

+ 那 26 张“被截断”的样本，被截掉的主要是重复垃圾 token；
+ 把 `max_model_len` 提到 8192 后，模型继续重复到 8192（最大单条 7896 tok / 1572 个重复框），
  **精度没有提升**（§5.3）；
+ `max_tokens=2048` 截断这些重复尾巴，**精度不变而吞吐大增**。

### 4.5 batch 非确定性

同一配置不同 batch 组成下数值路径不同：b64 与 b200 两轮仅 16/200 输出完全一致。
同配置重复运行的 F1 波动 0.008–0.014（如 opt200_b200：0.6844 / 0.6984）。
**结论：精度对比需要重复实验或固定 batch 组成；±0.02 以内的子集差异不可信。**

## 5. 优化手段与消融

### 5.1 调度与并发

固定 40 张子集（`subset40.jsonl`，单变量），max_pixels=2M / max_tokens=4096：

| 配置 | batch | img/s | F1@0.5 |
|---|---|---|---|
| s40_b8 | 8 | 0.140 | 0.683 |
| s40_b16 | 16 | 0.227 | 0.677 |
| s40_b32 | 32 | 0.219 | 0.671 |
| s40_b64 | 64 | 0.400 | 0.641 |

+ 吞吐随 batch 单调上升，b32 略低于 b16 是尾批长尾导致的波动；
+ **全量一次性提交（batch=200）是最大单项收益**：同配置 0.472 → 1.219 img/s。

### 5.2 输入侧（max_pixels）

| 配置 | max_pixels | img/s | F1@0.5 |
|---|---|---|---|
| s40_b64 | 2,007,040 | 0.400 | 0.641 |
| s40_b64_mp1m | 1,003,520 | 0.418 | 0.683 |
| s40_b64_mp800k | 802,816 | 0.401 | 0.657 |

+ 1M 像素把平均 prompt 从 1378 降到 942（−32%），全量收益明显（§6）；
+ 800k 不再有额外收益，且精度噪声偏大，不推荐继续下调。

全量 200 张复核（推荐配置 + 提前停止，单次）：

| 配置 | max_pixels | avg prompt tok | 总耗时(s) | img/s | F1@0.5 | mIoU-F1 |
|---|---|---|---|---|---|---|
| earlystop_b200 | 1,003,520 | 942 | 82 | 2.449 | 0.7483 | 0.5431 |
| opt_mp800k_es | 802,816 | 832 | 87 | 2.301 | 0.7339 | 0.5269 |

再降到 800k 后 prompt 少 110 tok，但生成结果改变（平均输出 345 → 358 tok），
总耗时反而 +5 s，F1 −0.014。**1M 是实测甜点，不建议继续降分辨率。**

### 5.3 输出侧（max_tokens / max_model_len）

| 配置 | max_tokens | max_model_len | 截断类型 | img/s | F1@0.5 |
|---|---|---|---|---|---|
| s40_b64 | 4096 | 4096 | 26 条被上下文截断 | 0.400 | 0.641 |
| s40_b64_len8192 | 4096 | 8192 | 6 条顶满 max_tokens | 0.326 | 0.640 |
| s40_b64_len8k_tok8k | 8192 | 8192 | 仍 25 条被上下文截断 | 0.185 | 0.661 |
| s40_b64_tok2048 | 2048 | 4096 | 15 条顶满 max_tokens | 0.614 | 0.646 |

+ 只加 `max_model_len` 不加 `max_tokens` 无效：输出仍被 `max_tokens` 卡住，且 prefill/调度变慢；
+ 8k/8k 在 40 张子集上 +2.0pt F1，但**全量 200 张未复现**（§6），代价却是 3.5× 延迟；
+ `max_tokens=2048` 是全量最优：速度快、精度不降（尾巴本就是退化重复）。

### 5.4 重复惩罚

| 配置 | repetition_penalty | img/s | F1@0.5 | 退化样本 |
|---|---|---|---|---|
| opt200_b200_tok2048 | 1.05 | 1.774 | 0.6938 | 24/200 |
| opt200_b200_rp11 | 1.10 | 1.813 | 0.7014 | 21/200 |
| opt200_b200_rp12 | 1.20 | 1.855 | 0.6737 | 22/200 |

+ 1.1 没有实质改善（差异在噪声内）；
+ 1.2 开始损害正常框的召回（R 0.6874 → 0.6524），**不推荐**；
+ 退化重复是模型级失败模式，vLLM 层的 repetition penalty 无法根治，
  需要 logits processor 级约束（生成侧）或解析器去重（精度侧）。

### 5.5 引擎侧

+ `max_num_batched_tokens=32768`：40 张子集 0.399 vs 0.400 img/s，**无收益**；
+ ACL Graph 默认开启（日志确认 `PIECEWISE compilation enabled on NPU`），无需额外操作；
+ W8A8 量化：环境无 msmodelslim，未验证。

### 5.6 解析器去重（精度后处理）

**动机**：退化重复会把同一个框连续输出几十到上千次，官方 metric 把每个重复框都当作一条预测，
多余的全算 FP。生成侧暂时无法低成本根治，但解析侧可以无损去掉完全重复的预测。

**实现**（`rex_omni/parser.py`，按类别维护 `seen` 集合）：

+ box：以量化后的 `(x0,y0,x1,y1)` bin 元组为键；
+ point：以 `(x,y)` 为键；polygon：以完整坐标元组为键；
+ 保留首次出现，后续完全相同者丢弃。只做**完全相等**去重，不做 IoU/NMS，避免改变正常近邻框。

**全量结果**（推荐配置，仅 parser 去重差异）：

| 运行 | parser 去重 | 总耗时(s) | img/s | P@0.5 | R@0.5 | F1@0.5 | mIoU-F1 |
|---|---|---|---|---|---|---|---|
| opt200_b200_tok2048_mp1m | 否 | 95 | 2.113 | 0.7205 | 0.6917 | 0.7058 | 0.5144 |
| opt200_b200_tok2048_mp1m_r2 | 否 | 95 | 2.109 | 0.7228 | 0.6919 | 0.7070 | 0.5169 |
| rep_dedup_b200 | 是 | 96 | 2.081 | 0.7635 | 0.6879 | 0.7237 | 0.5262 |
| rep_dedup_b200_r2 | 是 | 98 | 2.035 | 0.7617 | 0.6811 | 0.7191 | 0.5233 |

+ 两次重复的 F1 为 0.7237 / 0.7191（Δ0.005），高于未去重的 0.7058 / 0.7070（Δ≈0.015），
  方向可复现；
+ 精度增益主要来自 P（+0.04），R 基本不变（差异在 batch 非确定性噪声内）；
+ 吞吐不变（2.04–2.11，与未去重运行的 2.11 同一波动带）；
+ **代价**：重复 token 仍然被生成，只是不再计入指标；真正的算力节省需要生成侧提前停止。

### 5.7 生成侧提前停止（logits processor）

**收益上限**：async scheduling 开启时，全量 200 张中有 20–23 条样本一直生成到
`max_tokens=2048`，占总输出 token 的 44–47%；关闭后为 7 条。若能在退化循环刚形成时停止，
收益可观。下面的实现与实验均在 `async_scheduling=False` 下进行（原因见后）。

**实现**（`rex_omni/early_stop.py`，vLLM 0.14.1 `LogitsProcessor` 子类）：

+ 通过 `LLM(logits_processors=[EarlyStopRepeatedBox])` 注册（V1 不支持 per-request processor）；
+ `update_state(BatchUpdate)` 保存 `added` 元组中 `output_tok_ids` 的**引用**，
  从而在 `apply(logits)` 时读到每个请求的实时生成历史；
+ 检测：尾部 5 个连续块逐 token 完全相等，块长 4–48，则判定为退化循环；
  用最后两个 token 对齐做预筛，避免每步全量比较；
+ 触发时把该行 logits 全部置 `-inf`，只保留模型 `eos_token_id`（151645），强制结束。

**两个环境陷阱**：

1. tokenizer 的 added-token 表把 151643+ 重映射为 `<0>..<999>`，其 `<|im_end|>` 是 152645，
   已超出模型词表 151936，**不能采样**；必须用 config 的 `eos_token_id=151645`。
2. vllm-ascend 覆写的 `_sample` 漏调 `update_async_output_token_ids()`，
   async scheduling 下 logits processor 只能看到 `-1` 占位符（会误判为重复并截断正常输出）。
   因此实验必须加 `async_scheduling=False`。

**全量结果**（均含 parser 去重；配对实验）：

| 运行 | 提前停止 | 总耗时(s) | img/s | 平均输出 tok | 顶满 2048 样本 | P@0.5 | R@0.5 | F1@0.5 | mIoU-F1 |
|---|---|---|---|---|---|---|---|---|---|
| opt200_async_off | 否 | 97.2 | 2.058 | 390.6 | 7 | 0.7794 | 0.7092 | 0.7426 | 0.5388 |
| opt200_async_off_r2 | 否 | 95.3 | 2.098 | 390.0 | 7 | 0.7784 | 0.7138 | 0.7447 | 0.5408 |
| **earlystop_b200** | 是 | **81.7** | **2.449** | 349.1 | **0** | 0.7838 | 0.7159 | **0.7483** | **0.5431** |
| **earlystop_b200_r2** | 是 | **82.2** | **2.432** | 345.4 | **0** | 0.7839 | 0.7149 | **0.7478** | **0.5425** |

+ 吞吐 **+17%**（2.078 → 2.441 img/s 均值），耗时 −15%，输出 token −11%；
+ 7 条顶满样本全部被截短（323–1564 tok），非退化样本没有大额 token 变化；
+ F1 +0.005（0.7437 → 0.7481 均值），在 run-to-run 噪声内，**无精度损失**；
+ 触发阈值 `repeats=5` 时，实际只触发了退化样本，正常重复框未受影响。
+ **墙钟与长尾强相关**：同配置补跑两次为 82.0 s / F1 0.7416 / mIoU 0.5382 和
  67.3 s / F1 0.7337 / mIoU 0.5333。后者是 301 框长尾样本更早触发早停、输出更短，
  速度 +18% 但 F1 −0.01；82 s 是保守且精度更高的工作点。

**附带发现：async scheduling 改变生成结果。** 开/关两轮 200 条中仅 15 条输出完全一致；
关闭后 F1 从 0.7237 / 0.7191（async 开，§5.6）升到 0.7426 / 0.7447，
吞吐基本不变（2.08 vs 2.11 img/s）。机制未定位，但两轮重复稳定，暂作为经验结论记录。

### 5.8 NPU profiling：瓶颈到底在哪

三个互补实验：

**(a) 系统采样。** 跑推荐配置全量，同时每 ~0.8 s 采 `npu-smi info -t usages -i 0`：

| 指标 | 全程均值 | decode 为主的后 2/3 |
|---|---|---|
| AICore Usage | 41.4% | 45.3% |
| HBM Bandwidth Usage | 18.2% | 21.8% |
| HBM 容量占用 | 80%（KV cache 已分配） | 80% |

AICore 与 HBM 带宽都没有打满，先排除了「算力饱和」和「KV/权重带宽饱和」。

> **2026-11-05 校正**：41.4% 是**含空转**的全程均值。逐 0.3 s 时间线显示运行中段有
> 约 9–10 s AICore 恒为 0（见 §5.11），生成段 AICore 中位数 38%、**p90 = 100%**。
> 准确表述应为「低并发长尾不受算力/带宽限制，满载段会打满算力」——结论方向不变，
> 但不要用这个均值推断"整段都算力富余"。

**(b) 单步延迟随 batch 缩放。** 固定 `max_tokens=32`，先测 `max_tokens=1` 得 prefill，
再算 `(t32-t1)/31`（in-process engine，绝对值比正式多进程高约 10%）：

| batch | prefill (s) | 32 token 解码 (s) | 单步延迟 (ms) |
|---|---|---|---|
| 1 | 0.10 | 0.96 | 27.8 |
| 8 | 0.58 | 1.51 | 30.2 |
| 32 | 4.40 | 5.51 | 35.8 |
| 200 | 24.83 | 27.46 | 85.0 |

> 注：表中 `prefill (s)` 列是 `max_tokens=1` 的墙钟差值，含有图像加载与请求预处理，
> 比真实 prefill 偏大（见 §5.11(f) 的校正口径），但用于算单步延迟不受影响。

bs=1 → bs=8 单步几乎不变，说明有一笔约 28 ms 的**固定 per-step 开销**；
这正是全量墙钟里长尾的来源：301 框的样本要跑约 1560 步，
后 ~1300 步只剩个位数请求，`1300 × 28 ms ≈ 36 s`。

**(c) kernel 级 profiling（`torch_npu.profiler`，Level1）。**
vLLM V1 默认在子进程跑 EngineCore，driver 侧 profiler 采不到 NPU kernel；
需 `VLLM_ENABLE_V1_MULTIPROCESSING=0`。bs=1、32 token 的结果：

| 指标 | 数值 |
|---|---|
| Stage（设备侧） | 1.033 s |
| Computing（kernel 时间） | 0.305 s → **9.5 ms/step** |
| Free（kernel 间空档） | 0.729 s → **22.8 ms/step** |
| 每步 host ACL API 总耗时 | **约 14 ms** |

+ kernel 构成：MatMulV2 67.7%（6.4 ms/step）、FusedInferAttentionScore 10.7%、
  RopeWithSinCosCache 6.3%、AddRmsNormBias 2.5%、SwiGlu 2.3%；
+ host API 明细（每步）：launch 节点 2.0 ms、attention 1.34 ms、ReshapeAndCache
  Execute 1.22 ms + Setup 1.03 ms、`aclrtSynchronizeDevice` 1.05 ms、
  `aclrtLaunchKernelWithHostArgs` 1.01 ms、InplaceFill 0.99 ms、InplaceCopy 1.68 ms；
  每步约 196 次 kernel launch + 36 次 graph execute；
+ bs=200、64 token 的 profile 里 Computing 18.5 s / Free 3.8 s（设备侧算得较满），
  说明**满载阶段是 kernel-bound，长尾阶段是 dispatch-bound**。

**(d) async scheduling 能否藏掉这笔开销？** 多进程、`max_tokens=32` 对比：

| batch | async off (ms/step) | async on (ms/step) |
|---|---|---|
| 1 | 25.9 | 25.0 |
| 8 | 28.4 | 28.5 |
| 32 | 30.9 | 30.0 |

基本无差异。原因是每步必须等采样结果才能决定下一步，host 下发仍在关键路径上；
async scheduling 重叠不了这部分。

**结论**：

1. 不是 KV 带宽（HBM ~20%）、不是 AICore 饱和（均值 41–45%，且含约 10 s 空转；
   逐秒看生成段 p90 打满）、不是 HBM 容量；
2. 低并发下单步 28 ms 里 kernel 只有 ~9.5 ms，其余是算子下发/同步/调度；
   36 层每层若干小算子，每步 ~200 次 launch，是 vllm-ascend 当前的固定成本；
3. 满载 200 并发时 kernel 时间升到主导，dispatch 被部分掩盖，但墙钟仍被长尾拖住；
4. 因此 **W8A8 量化收益存疑**（低并发不是带宽/算力受限，量化主要改善这两者），
   配置参数也已到顶；剩余空间在框架层的 graph 覆盖/算子融合，或减少 decode 步数。
   后续用算子级统计量化了前者的上限：bs=1 decode 中 MatMulV2 已占设备时间 67.7%，
   可融合算子合计约 11%，现实收益约 5–10%，**优先级低于减少 decode 步数**。

### 5.9 ACL graph 全覆盖（FULL_DECODE_ONLY）：兑现 5.8 的结论

vllm-ascend 0.14.0rc1 默认 `CUDAGraphMode.PIECEWISE`：只把部分子图归并，
attention、ReshapeAndCache 等仍在 graph 外逐算子下发。实验性的 `FULL_DECODE_ONLY`
把 `splitting_ops` 置空，整个 decode forward（含 attention 与 cache update）
捕获成一张 ACL graph，每步只剩一次 replay。

```python
compilation_config={
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [1, 8, 16, 32, 64, 128, 200, 256],
}
```

**(a) 单步延迟**（`max_tokens=32`，同 5.8(b) 的测法，in-process engine）：

| batch | PIECEWISE (ms) | FULL_DECODE_ONLY (ms) | 加速 |
|---|---|---|---|
| 1 | 27.8 | **10.4** | 2.7× |
| 8 | 30.2 | **12.3** | 2.5× |
| 32 | 35.8 | **17.1** | 2.1× |

bs=1 的 10.4 ms 已经接近 5.8(c) 测到的纯 kernel 时间 9.5 ms——下发空档基本消失。

**(b) 全量 200 张**（其余配置同推荐部署配置：去重 + 提前停止 + async off）：

| 配置 | 总耗时(s) | img/s | 输出 tok/s | F1@0.5 | mIoU-F1 |
|---|---|---|---|---|---|
| PIECEWISE（推荐基线） | 82 | 2.449 | 879 | 0.7483 | 0.5431 |
| **FULL_DECODE_ONLY #1** | **65.0** | **3.078** | 1137 | **0.7457** | **0.5418** |
| **FULL_DECODE_ONLY #2** | **63.9** | **3.128** | 1188 | **0.7490** | **0.5437** |
| **两次均值** | **64.5** | **3.103** | 1163 | **0.7474** | **0.5428** |

+ 端到端 **+27%**（2.441 → 3.103 img/s），与 profiling 的预测吻合：
  长尾约 1300 步 × 节省 13–17 ms ≈ 17 s；
+ 图捕获只多花 3 s、0.26 GiB，load 总时长 50 s（与 PIECEWISE 同量级）；
+ 提前停止在 FULL graph 下正常工作：平均输出 348.9 / 358.3 token（基线 349.1），
  最大 1977 / 1796 < 2048，没有退化样本漏网；
+ greedy 输出与 PIECEWISE 分别有 183/200、200/200 不同，但两次 FULL graph 之间
  也只有 18/200 相同——这是本环境 greedy 的 run-to-run 非确定性（批内归约顺序等），
  不是 FULL graph 特有；输出长度分布与精度均一致（F1 0.7457 / 0.7490，
  mIoU-F1 0.5418 / 0.5437，差异在噪声内）。

**注意事项**：

1. `cudagraph_capture_sizes` 必须覆盖线上 batch 分桶；本配置覆盖 1–256。官方警告
   捕获过多尺寸可能 OOM 或挂起，必要时降低 `gpu_memory_utilization` 或减少尺寸。
2. 该模式在 vllm-ascend 仍标记为 experimental，升级版本后需回归。
3. 依赖 token 历史的 logits processor（提前停止）必须在 `async_scheduling=False`
   下运行，与 5.7 的结论一致；FULL graph 不改变这一约束。

### 5.10 prefill 分块（`max_num_batched_tokens`）

200 条共约 18.8 万 prompt token，默认 8192 要切约 23 个 chunk；提到 32768 只需约 6 个：

| max_num_batched_tokens | 总耗时(s) | img/s | F1@0.5 | mIoU-F1 |
|---|---|---|---|---|
| 8192（默认，两次均值） | 64.5 | 3.103 | 0.7474 | 0.5428 |
| **32768 #1** | **62.4** | **3.205** | 0.7474 | 0.5425 |
| **32768 #2** | **62.7** | **3.189** | 0.7474 | 0.5427 |
| **32768 均值** | **62.6** | **3.197** | **0.7474** | **0.5426** |

+3%，两次 F1 完全一致；load 时间 +2 s，未见 OOM。这是继 FULL graph 之后
唯一还有正收益的配置项，再提到 65536 预计 <1%，未测。

### 5.11 CPU 输入预处理多线程（`OMP_NUM_THREADS`）：否决（2026-11-05）

**(a) 动机。** §5.8(a) 的均值掩盖了一段空转。把 `npu-smi` 采样间隔从 0.8 s 降到
0.3 s 并逐秒看时间线，每次运行都有约 9–10 s AICore 恒为 0：

```text
t=  6.3s  AICore 27%   ← warmup
t=  8.9s  AICore  0%   ← ⬇ 约 9 s 无任何 NPU 活动
t= 18.0s  AICore 100%  ← 生成开始
```

同期 vLLM 把输入预处理（图像 resize/normalize/patchify）强制单线程：
`vllm/v1/executor/multiproc_executor.py:887` 与 `v1/engine/input_processor.py:502`
在 `OMP_NUM_THREADS` 未设置时把 torch 线程压到 1。容器 cgroup 配额为 8 CPU。
离线测 200 张预处理：**1 线程 25.3 s（126.7 ms/图）→ 8 线程 8.5 s（42.6 ms/图），3.0×**。

**(b) 预期。** 把这 9–10 s 空转换回 3–4 s，全量 62.6 → 54–56 s（+13~16%）。

**(c) 实测**（推荐配置不变，仅设置 `OMP_NUM_THREADS`，每次全量 200）：

| 运行 | 总耗时 | img/s | 平均输出 tok | 输出 tok/s | F1@0.5 | mIoU-F1 |
|---|---|---|---|---|---|---|
| 历史基线 #1 | 62.40 | 3.205 | 357.2 | 1216 | 0.7474 | 0.5425 |
| 历史基线 #2 | 62.72 | 3.189 | 351.1 | 1189 | 0.7474 | 0.5427 |
| **同场对照（t=1）** | 62.57 | 3.196 | 364.0 | 1235 | 0.7554 | 0.5493 |
| OMP=8 #1 | 61.03 | 3.277 | 355.7 | 1240 | 0.7460 | 0.5424 |
| OMP=8 #2 | 65.16 | 3.069 | 357.9 | 1164 | 0.7493 | 0.5437 |

OMP=8 两轮均值 63.1 s，比同场对照**慢 0.8%**；五轮输出 tok/s 在 1164–1240（±3%），
覆盖全部差异；精度不变（同一条代码路径）。

**(d) 决定性证据：隔离探针。** 固定 `max_tokens=1`（去掉 decode 变量），只改线程数：

| 指标 | t=1 | t=8 |
|---|---|---|
| 端到端 | 26.62 s | **26.15 s（−0.5 s）** |
| `Adding requests`（200 条预处理） | 14 s | **7 s（−7 s）** |

预处理墙钟砍掉一半，端到端只快 0.5 s → **CPU 预处理不在关键路径上**。机制：
vLLM V1 的 EngineCore 是独立进程的常驻循环，请求流式送入；前 ~35 张（凑满第一个
32768-token prefill chunk）预处理完引擎就开始跑，剩余请求的预处理被 NPU 的
prefill/decode 完全掩盖。

**(e) 结论与教训。** 不设置 `OMP_NUM_THREADS`，推荐配置不变。
**"CPU 侧看起来很忙"不等于它在关键路径上**——用 `max_tokens=1` 这类隔离探针
把待优化段单独量一次，比任何理论估算都便宜（本轮原本估 +13~16%，实际 0%）。

**(f) 校正后的时间结构（并行化前，最终配置 62.6 s）。** 用同一探针叠加 0.3 s 采样
（`results/mt1_sampled/npu_util.csv`，时间线 `idle 7.2→16.4 s / active 16.4→33.8 s`）
拆分。早期版本把 `max_tokens=1` 的 26.6 s 整体当作 prefill，是错的：

| 阶段 | 耗时 | 占比 | 是否在关键路径 |
|---|---|---|---|
| 图像加载（PIL open + JPEG 解码） | 3.6 s | 6% | **是**（`batch_dt` 之外，单线程） |
| 引擎启动 / 首批请求预处理 | 5.6 s | 9% | 是，但提高线程数未能前移 |
| prefill（18.8 万 prompt token，6 × 32k chunk） | 17.4 s | 28% | 是，AICore 接近打满 |
| decode（约 7 万输出 token） | 36.0 s | 58% | 是，主要在长尾低并发段 |

> **2026-11-05 续**：表中的“引擎启动 5.6 s”已证伪——它是 Rex-Omni wrapper 里
> 逐条串行的 `process_vision_info()`，不是引擎启动（§5.12）。两段 CPU 工作并行化后，
> 实际的引擎启动只有约 0.3 s，新时间结构见 §5.12(f)。

据此，剩余优先级：decode 步数（投机解码）> 图像加载 3.6 s 并行解码 >
定位 5.6 s 启动延迟 > 图覆盖/算子融合（算子级统计给出的上限约 5–10%，
bs=1 decode 中 MatMulV2 已占设备时间 67.7%，不可融合）> W8A8。

### 5.12 并行输入预处理：+14.2%（2026-11-05）

**(a) 定位。** §5.11(f) 留下 5.6 s 来源不明的启动延迟。用
`profiling/startup_trace.py`（`VLLM_ENABLE_V1_MULTIPROCESSING=0` 单进程引擎，
monkey-patch `InputProcessor.process_inputs` 与 `LLMEngine.step` 打时间戳）看到：
200 条请求的输入预处理是**逐条串行**完成的，最后一条开始时引擎已空等数秒。
两段 CPU 工作都在请求入队之前，因此都在关键路径上：

| 阶段 | 位置 | 串行耗时（200 张） |
|---|---|---|
| PIL `open` + `convert("RGB")` | `run_eval.py` 循环 | 3.6 s（17.9 ms/图） |
| `process_vision_info()`（to_rgb + smart_resize + resize） | `wrapper.py` 逐条循环 | **7.3 s（36.4 ms/图）** |

**(b) 设计。** 两段都改用 `ThreadPoolExecutor`。PIL 在 JPEG 解码和 resize 时释放
GIL，线程池可以真正并行。wrapper 侧线程数由 `REX_OMNI_VIS_WORKERS` 控制
（默认 8，1 回退串行）；评测脚本用 `--load-workers`（默认 8，1 = 串行）。
并行化只改“谁先准备好输入”，不改变任何图像张量。

**(c) 微基准**（`profiling/vispar_microbench.py`，纯 CPU，`OMP_NUM_THREADS=1`）：

| 阶段 | 串行 | 8 线程 | 加速 |
|---|---|---|---|
| PIL 读图 | 3.59 s | 0.54 s | 6.6× |
| `process_vision_info` | 7.28 s | 1.04 s | 7.0× |
| 合计 | 10.87 s | **1.58 s** | −9.3 s |

**(d) 正确性。** 串行与 8 线程输出的图像张量逐字节一致、顺序一致；
全量 200 条的 `num_prompt_tokens` 与对照组完全相同（平均 942.0），无漏图/错位。

**(e) 全量实测**（推荐配置不变，每次 200 张）：

| 运行 | 总耗时 | img/s | 平均输出 tok | 输出 tok/s | F1@0.5 | mIoU-F1 |
|---|---|---|---|---|---|---|
| 同场对照（串行，`omp1_b200`） | 62.57 | 3.196 | 364.0 | 1235 | 0.7554 | 0.5493 |
| 仅 wrapper 并行（`vispar_b200`） | 58.00 | 3.448 | 351.9 | 1295 | 0.7428 | 0.5376 |
| **两段并行 #1（`vispar_load_b200`）** | **54.46** | **3.672** | 361.1 | 1341 | 0.7539 | 0.5488 |
| **两段并行 #2（`vispar_load_b200_r2`）** | **55.09** | **3.631** | 362.4 | 1330 | 0.7569 | 0.5483 |
| **两段并行均值** | **54.8** | **3.65** | 361.8 | 1335 | 0.7554 | 0.5486 |

端到端 **+14.2%**，精度差 ≤0.013（噪声带内）。NPU 首段空转由 9.8 s 降到
0.9 / 1.9 s（`npu_util.csv` 逐 0.3 s 采样）。

**(f) 新时间结构（最终配置 54.8 s）**：

| 阶段 | 耗时 | 占比 | 说明 |
|---|---|---|---|
| 图像加载（8 线程） | 约 0.5 s | 1% | 串行时 3.6 s |
| `process_vision_info`（8 线程） | 约 1.0 s | 2% | 串行时 7.3 s；即 §5.11(f) 的“5.6 s 引擎启动” |
| 引擎启动 / 首个 prefill chunk | 约 0.3 s | <1% | 真正的固定启动开销很小 |
| prefill | 约 17.4 s | 32% | 与并行化无关，按对照推算 |
| decode | 约 36.0 s | 66% | 与并行化无关，按对照推算 |

**(g) 适用边界。** 收益来自“消除等待”，不是“少算”；图片数少（bs=1–8）时收益按
比例缩小；`workers` 不应超过容器 CPU 配额（本例 8 核）。这是客户端侧优化，线上用
`vllm serve` 时预处理在服务进程中按请求并发，不存在“200 张一次性串行预处理”的
排队，收益口径不同。与 §5.11 的对照表明，判据不是“CPU 还是 NPU”，而是这段工作
有没有被别的执行流掩盖。

## 6. 精度-性能帕累托与选型（全量 200）

| 配置 | batch | max_pixels | max_tokens | max_model_len | 总耗时(s) | img/s | 输出 tok/s | F1@0.5 | mIoU-F1 |
|---|---|---|---|---|---|---|---|---|---|
| base200_b64 | 64 | 2,007,040 | 4096 | 4096 | 424 | 0.472 | 291 | 0.6896 | 0.5075 |
| opt200_b200 | 200 | 2,007,040 | 4096 | 4096 | 164 / 170 | 1.219 / 1.179 | 754 | 0.6844 / 0.6984 | 0.5046 / 0.5136 |
| opt200_b200_tok2048 | 200 | 2,007,040 | 2048 | 4096 | 113 / 107 | 1.774 / 1.867 | 928 | 0.6938 / 0.7013 | 0.5111 / 0.5154 |
| **opt200_b200_tok2048_mp1m** | 200 | **1,003,520** | **2048** | 4096 | **95 / 95** | **2.113 / 2.109** | 1106 | **0.7058 / 0.7070** | **0.5144 / 0.5169** |
| opt200_b200_acc | 200 | 2,007,040 | 8192 | 8192 | 329 | 0.607 | 666 | 0.6860 | 0.5051 |
| opt200_b200_rp11 | 200 | 2,007,040 | 2048 | 4096 | 110 | 1.813 | 896 | 0.7014 | — |
| opt200_b200_rp12 | 200 | 2,007,040 | 2048 | 4096 | 108 | 1.855 | 905 | 0.6737 | — |
| **rep_dedup_b200** | 200 | 1,003,520 | 2048 | 4096 | 96 | 2.081 | 1049 | **0.7237** | **0.5262** |
| **rep_dedup_b200_r2** | 200 | 1,003,520 | 2048 | 4096 | 98 | 2.035 | 1060 | **0.7191** | **0.5233** |
| opt200_async_off | 200 | 1,003,520 | 2048 | 4096 | 97 | 2.058 | 835 | 0.7426 | 0.5388 |
| opt200_async_off_r2 | 200 | 1,003,520 | 2048 | 4096 | 95 | 2.098 | 851 | 0.7447 | 0.5408 |
| **earlystop_b200** | 200 | 1,003,520 | 2048 | 4096 | **82** | **2.449** | 879 | **0.7483** | **0.5431** |
| **earlystop_b200_r2** | 200 | 1,003,520 | 2048 | 4096 | **82** | **2.432** | 895 | **0.7478** | **0.5425** |
| profile_run2（同配置补跑） | 200 | 1,003,520 | 2048 | 4096 | 82 | 2.438 | 878 | 0.7416 | 0.5382 |
| profile_run（同配置补跑） | 200 | 1,003,520 | 2048 | 4096 | 67 | 2.971 | 988 | 0.7337 | 0.5333 |
| opt_mp800k_es | 200 | 802,816 | 2048 | 4096 | 87 | 2.301 | 859 | 0.7339 | 0.5269 |
| fullgraph_b200 | 200 | 1,003,520 | 2048 | 4096 | 65.0 | 3.078 | 1137 | 0.7457 | 0.5418 |
| fullgraph_b200_r2 | 200 | 1,003,520 | 2048 | 4096 | 63.9 | 3.128 | 1188 | 0.7490 | 0.5437 |
| **fullgraph_mbt32k** | 200 | 1,003,520 | 2048 | 4096 | **62.4** | **3.205** | 1216 | **0.7474** | **0.5425** |
| **fullgraph_mbt32k_r2** | 200 | 1,003,520 | 2048 | 4096 | **62.7** | **3.189** | 1189 | **0.7474** | **0.5427** |
| omp1_b200（同场对照，2026-11-05） | 200 | 1,003,520 | 2048 | 4096 | 62.6 | 3.196 | 1235 | 0.7554 | 0.5493 |
| omp8_b200（OMP=8，已否决） | 200 | 1,003,520 | 2048 | 4096 | 61.0 | 3.277 | 1240 | 0.7460 | 0.5424 |
| omp8_b200_r2（OMP=8，已否决） | 200 | 1,003,520 | 2048 | 4096 | 65.2 | 3.069 | 1164 | 0.7493 | 0.5437 |
| vispar_b200（仅 wrapper 并行） | 200 | 1,003,520 | 2048 | 4096 | 58.0 | 3.448 | 1295 | 0.7428 | 0.5376 |
| **vispar_load_b200（推荐配置）** | 200 | 1,003,520 | 2048 | 4096 | **54.5** | **3.672** | 1341 | **0.7539** | **0.5488** |
| **vispar_load_b200_r2（推荐配置）** | 200 | 1,003,520 | 2048 | 4096 | **55.1** | **3.631** | 1330 | **0.7569** | **0.5483** |

> 从 `opt200_async_off*` 起的所有行关闭 async scheduling；`earlystop_*`、`fullgraph_*`
> 与 `opt_mp800k_es` 另加 `--early-stop-repeats 5`；`fullgraph_*` 额外启用
> FULL_DECODE_ONLY ACL graph，`fullgraph_mbt32k*` 另加 `max_num_batched_tokens=32768`；
> 所有行均含 parser 去重。`omp*` 三行（2026-11-05）与 `fullgraph_mbt32k*`
> 配置完全相同，仅设置 `OMP_NUM_THREADS`（8/8/1），用于 §5.11 的否决实验；
> `vispar_*` 三行在 `omp1_b200` 基础上加并行预处理（`vispar_b200` 只开 wrapper 侧），
> 见 §5.12。

选型：

+ **推荐部署配置**：`batch=200`（一次性提交/continuous batching）+ `max_pixels=1003520`
    + `max_tokens=2048` + `max_model_len=4096` + `repetition_penalty=1.05`
    + parser 去重 + 生成侧提前停止（`repeats=5`）+ `async_scheduling=False`
    + `cudagraph_mode=FULL_DECODE_ONLY`（捕获尺寸 `[1,8,16,32,64,128,200,256]`）
    + `max_num_batched_tokens=32768` + `REX_OMNI_VIS_WORKERS=8` + `--load-workers 8`
  → **3.65 img/s，200 张 54.8 s，摊销 0.274 s/图，F1@0.5 0.7554（0.7539 / 0.7569），
  mIoU-F1 0.5486（0.5488 / 0.5483）**
+ 相对初始可复现基线（0.472 img/s / 424 s）：**吞吐 7.74×，F1 +0.066**
+ 相对单卡 bs=1（0.046 img/s）：**约 79×**
+ 只做参数优化（不含去重/提前停止/graph）时：2.11 img/s，F1 0.706；去重是零成本
  精度补丁，提前停止是 +17% 的算力优化，FULL graph 是 +27% 的框架层优化，
  32k prefill 分块再 +3%，并行输入预处理再 +14.2%，五者可独立启用
+ 精度模式（8k/8k）在子集上的增益未在全量复现，且慢 3.5×，**不建议作为默认**

## 7. 可复用方法论

1. **先固定协议再优化**：数据集子集、prompt、采样参数、warmup 全部固定，否则数字不可比。
2. **先量化噪声再看差异**：greedy 解码可能不是 batch-invariant；本环境全量 F1 波动 0.008–0.014，
   小于噪声带的“提升”不可信。关键配置至少重复 2 次。
3. **先定位瓶颈再动手**：拆分 prefill/decode、看输出 token 分布，避免盲目调分辨率。
   本例 decode 主导，所以减小输出 token 和消除批间屏障收益最大。
4. **先审输出质量再调预算**：看到截断/超长输出，先检查是否是退化重复。
   “上下文不够”可能只是表象，加预算反而更慢且无效。
5. **单变量消融 + 全量复核**：子集消融定位方向，候选配置必须在全量上验证；
   子集上的 ±0.02 常常是噪声。
6. **连续批处理优于固定分块**：固定 chunk 的块间屏障由最慢请求决定；
   一次性提交让 continuous batching 摊平长尾。
7. **精度性能同链路**：所有配置用同一推理包装器、同一 metric 脚本、同一截断处理。
8. **先验证框架接口的真实语义**：logits processor 拿到的 `output_token_ids` 在
   async scheduling 下可能是 `-1` 占位符；先用小样本打印真实值，再写检测逻辑。
9. **配对实验**：关闭 async scheduling 会同时影响生成结果与吞吐，必须跑“同开关、仅差一个变量”
   的配对基线，否则无法归因。
10. **固化产物**：脚本 + 配置 + 日志 + 结果 JSON 全部落盘，保证可复现。
11. **先证明在工作在关键路径上，再优化它**：CPU 预处理从 14 s 优化到 7 s，端到端
    只快 0.5 s——请求流式送入、引擎常驻，后续请求的预处理被 NPU 计算掩盖。
    用 `max_tokens=1` 之类的隔离探针把待优化段单独量一次，成本极低。
12. **均值会骗人，先看时间线**：AICore 全程均值 41% 让人以为算力富余，
    逐秒看却是「约 10 s 全 0 + 生成段 p90 打满」两段拼起来的；采样间隔
    （0.8 s vs 0.3 s）决定了你能不能看见这段空转。
13. **给关键路径上的每个参与者打时间戳**：`startup_trace.py` 用 monkey-patch 记录
    每条请求的预处理起止与引擎每步的起止，才看清“引擎启动慢”其实是客户端串行预处理。
    端到端 profile 只能告诉你“这里空了 5.6 s”，打点才能告诉你“是谁让它空的”。
14. **并行化前先确认目标能并行**：PIL 解码/resize 释放 GIL，线程池就有 6.6–7.0×；
    若目标是纯 Python 计算，同样写法可能零收益。微基准（`vispar_microbench.py`）
    比直接改端到端便宜得多。
15. **客户端与推理栈要分别评估**：本轮 14.2% 来自 wrapper 与评测脚本，未改推理栈；
    先按“谁能贡献关键路径时间”分层，而不是默认只优化 NPU 侧。

## 8. 复现命令

```bash
# 推荐部署配置（全量 200：去重 + 提前停止 + FULL decode graph + 32k 分块 + 并行预处理）
cd /home/workDir/00_Software/Rex-Omni
REX_OMNI_VIS_WORKERS=8 python3 /home/workDir/rex_omni_deploy/run_eval.py \
  --jsonl /home/workDir/rex_omni_eval/Rex-Omni-Eval/_annotations/box_eval/Dense200.jsonl \
  --image-root /home/workDir/rex_omni_eval/Rex-Omni-Eval \
  --out /home/workDir/rex_omni_deploy/results/vispar_load_b200/answer.jsonl \
  --tag vispar_load_b200 --limit 200 --batch-size 200 \
  --max-pixels 1003520 --max-tokens 2048 --max-model-len 4096 \
  --repetition-penalty 1.05 --gpu-mem 0.8 --load-workers 8 \
  --early-stop-repeats 5 \
  --extra '{"async_scheduling": false, "max_num_batched_tokens": 32768, "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,8,16,32,64,128,200,256]}}'

# 精度评分（必须用绝对路径，score.sh 内部会 cd 到仓库）
bash /home/workDir/rex_omni_deploy/score.sh \
  /home/workDir/rex_omni_deploy/results/vispar_load_b200

# 40 张子集消融（自动跳过已有结果）
bash /home/workDir/rex_omni_deploy/sweep.sh
```

对照实验：

+ 纯吞吐优化：去掉 `--early-stop-repeats`、`--extra`，并 `git checkout rex_omni/parser.py`；
+ 仅去重：保留 parser 补丁，去掉 `--early-stop-repeats`、`--extra`（`results/rep_dedup_b200[_r2]`）；
+ 提前停止对照：保留 parser 补丁与 `--extra`，去掉 `--early-stop-repeats`（`results/opt200_async_off[_r2]`）；
+ graph 对照：保留提前停止，`--extra` 只留 `{"async_scheduling": false}`
  （`results/earlystop_b200[_r2]`，82 s / 2.44 img/s）；
+ 并行预处理对照：上一条配置加完整 `--extra` 即 `omp1_b200`（62.6 s / 3.20 img/s，
  串行预处理）；再加 `REX_OMNI_VIS_WORKERS=8 --load-workers 8` 即推荐配置
  （`results/vispar_load_b200[_r2]`，54.8 s / 3.65 img/s）。只开 wrapper 侧为
  `vispar_b200`（58.0 s / 3.45 img/s）。

## 9. 局限与后续工作

+ 瓶颈定位见 §5.8（`npu-smi` 利用率/带宽 + `torch_npu.profiler` + 单步缩放）；
  **时间结构已两次校正（§5.11(f)、§5.12(f)）**：并行化后为 图像加载 0.5 s（1%）+
  `process_vision_info` 1.0 s（2%）+ 引擎启动约 0.3 s（<1%）+ prefill 17.4 s（32%）+
  decode 36.0 s（66%）。早期「prefill 约 25 s」把读图与预处理算进了 prefill；
  此后一度把 5.6 s 归因为引擎启动，实际是客户端串行 `process_vision_info()`（§5.12）；
  CPU 侧输入预处理已从 10.9 s 压到 1.6 s，**基本无剩余空间**；
  输出格式已接近下限（约 5 tok/框，300 框样本 ~1560 tok），继续压缩输出长度空间很小，
  下一步是减少 decode 步数（投机解码），这是唯一还能拿到两位数收益的方向；
  图覆盖/算子融合经算子级统计评估上限约 5–10%，优先级低于上述几项；
+ 分辨率已扫到边界：1M → 800k 全量反而 +5 s、F1 −0.014（§5.2），说明剩余优化不在
  输入侧；
+ per-request 延迟默认不可见：vLLM 的 `LLM` 默认 `disable_log_stats=True`，
  `RequestOutput.metrics` 为 None。设 `--extra '{"disable_log_stats": false}'` 后可取
  TTFT/TPOT 及 queue/prefill/decode 拆分。时间戳字段与 `async_scheduling` 无关：
  本栈 async on 的 b200 运行也回填了全部四个字段；早期一次 b64 取不到，是那次
  运行的 wrapper 版本早于时间戳补丁。**归因运行时用引擎 TTFT**（queue+prefill），
  离线 `LLM.generate` 的 TTFT 在大 batch 下含前端 add/output 积压。详见
  CASE_STUDY_V2.md §1.2/§6；
+ 退化重复已在解析侧去重（§5.6）并在生成侧提前停止（§5.7）：全量上顶满 2048 的样本
  从 7/7 降到 0，输出 token −11%、吞吐 +17%；残留问题：vllm-ascend 的 async scheduling
  与 logits processor 不兼容，需关闭 async scheduling（可能损失部分重叠收益）；
+ async scheduling 开关导致 F1 约 0.02 的差异，机制未定位，需在更多数据上复核；
+ W8A8 量化未验证（缺 msmodelslim）；
+ 精度为单一 prompt/单一包装器结果，绝对值与官方榜单可能有解析器差异。
