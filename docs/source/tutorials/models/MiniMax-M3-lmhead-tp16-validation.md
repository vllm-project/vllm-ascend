# MiniMax-M3 / EAGLE3 LM-head TP16 验证

## 环境与结论

2026-10-14，在 197 的 `cxy-minimax-m3-pr18344-20261012` 容器上验证。
仅停止并替换 cxy 的旧服务；`lh1012` 在芯片 0 上的进程保持运行。

- vLLM：官方 `v0.30.0`，`ced6857afa0ea7b2e3f0846a62e1394e90f15607`。
- Ascend：`d94a832bb9f88ec19e60ddfe7b554eaf22811053` 加本文适配；保留容器原有 PP 修复。
- 硬件：A3，16 个 Ascend 910 芯片。
- 主模型：`/models/minimax/Eco-Tech_MiniMax-M3-w8a8-0626`。
- Draft：`/models/minimax/MiniMax-M3-EAGLE3`，每轮 3 个 draft token。
- 配置：DP4 / TP4 / EP16 / PP1，V2 runner，文本模式，真实完整权重。
- 容量：`max_model_len=131072`、`max_num_seqs=16`，不代表已验证满长度满并发性能。

原实现直接配置 LM-head TP16 会在启动阶段报错：
`finegrained tp sizes must divide by data_parallel_size.`
原因是原 fine-grained TP 只沿 DP 维度创建进程组，DP4 下最大为 4；它替换
LM-head 的标准 TP 切分，不会自动与 TP4 相乘。

适配后主模型和 EAGLE3 均可启动并推理，32 条运行时记录覆盖两种模型的全部
16 个 rank。每个 LM-head 权重从 `[50016, 6144]` 变成 `[12504, 6144]`，
LM-head TP 为 16，attention 仍为 TP4。

## 实现

1. 对大于 DP 的 LM-head TP，要求它是标准 TP 的整数倍且整除 DP × TP；PCP 必须为 1。
2. 按 PP stage 创建跨 DP × TP 的 LM-head 组，保持完整的标准 TP 组。
3. TP 内先分割重复的 hidden-state token 行，再进行 LM-head all-gather、GEMM、
   all-to-all；最后在标准 TP 组内恢复 token 行顺序。非整除行数先补零、最后裁剪。
4. 不超过 DP 的原有配置保持 DP-axis 行为；embedding、o_proj、MLP 的限制不变。

没有修改 upstream vLLM。运行时 shape 采集代码在采集结束后已恢复。
权重更小不等同于端到端更快：此验证没有测量新增通信后的 TPOT 或吞吐收益。

## 启动与复现

容器内完整启动脚本：`/workspace/lmhead-tp16-20261014/serve.sh`。
在原 DP4 / TP4 启动参数的 `additional_config` 中添加：

```json
{
  "finegrained_tp_config": {
    "lmhead_tensor_parallel_size": 16
  }
}
```

```bash
bash /workspace/lmhead-tp16-20261014/serve.sh 1
```

服务地址为 `http://192.168.13.197:8000`，模型名 `minimax-m3`。
GET `/v1/models` 返回 200，4 个并发加法请求全部返回 200 且答案正确。
ACLGraph 捕获及 draft prefill/decode replay 均已执行，flashcomm1 启用。
此次未测试多模态，也未使用 dummy 权重。

## GPQA 20 题（TP16，NZ 修改前）

从容器已有 `/workspace/accuracy/gpqa.json` 固定选取索引 0–19；沿用其原始
messages 和答案标签，不按历史对错筛选。评测文件 SHA256：
`651ec4858ea125aeef29de491217de0def59c536d4dae7570a16b904741624f3`。

使用 temperature=0、seed=42，从最终回答中的 `ANSWER: A/B/C/D` 提取答案。
首轮并发 8，输出上限 8192：20 题全部完成，15 题正确（75%）；无 HTTP 错误，
4 题因输出预算截断而无法提取答案，1 题正常结束但回答错误。

保留首轮原始结果，将索引 7、8、12、15 的输出上限提高到 32768、并发 4 单独复测。
复测中 7、12 题答对，15 题正常结束但答错，8 题在 32768 token 处仍截断。
合并结果为 **17/20（85%）**，无 HTTP 错误，2 题答错，1 题截断未提取到答案。
原始首轮 15/20 与提高预算后的结果分别保留，不能视为统一输出预算的单轮测试。
逐题汇总见容器证据目录的 `combined-summary.json`。20 题仅用于功能与小样本精度检查，
不代表完整 GPQA 得分，也不构成与关闭 LM-head TP 的严格精度对照。

```bash
python /workspace/lmhead-tp16-20261014/gpqa20_eval.py --runner tp16 --concurrency 8
python /workspace/lmhead-tp16-20261014/gpqa_retry_eval.py --runner tp16-32k-retry --concurrency 4
```

## NZ 修改

PR 基于 Ascend `releases/v0.30.0` 的 `acd91d31e` 整理。
未量化 LM-head 使用独立的 `AscendUnquantizedLMHeadMethod`，在权重加载后调用
现有 `maybe_trans_nz`。`weight_nz_mode=2` 时按现有硬件策略将 BF16/FP16 转为 NZ；
不新增环境变量，不在每次 forward 中转换。量化 head 保留原量化方法；
与 embedding 绑定的权重保持原布局，避免改变 lookup 的存储格式。

新增 10 项单测覆盖 mode 0/1/2、BF16/FP16/FP32 和 tied weights，连同 release
版本的词表模块测试共 37 项通过。上面的 GPQA 结果来自 NZ 修改前，不能当作
TP16+NZ 的精度结果；没有进行 TP4/TP16 的端到端性能对照。

真实权重 TP16+NZ 服务启动成功，4 个并发加法请求均返回正确答案。
32 条记录确认主模型和 EAGLE3 在全部 16 个 rank 上的 LM-head 权重均为
`[12504, 6144]`、`FRACTAL_NZ`；采集后已移除临时打印代码。

单卡 BF16 微基准使用 `[10,6144] × [12504,6144]ᵀ`，7 轮交替测试 ND/NZ：

| 模式 | ND 中位数 | NZ 中位数 | 耗时降低 |
| --- | --- | --- | --- |
| Eager | 125.16 μs | 110.22 μs | 11.9% |
| NPUGraph | 125.73 μs | 110.44 μs | 12.2% |

NZ 权重通过本次新增的加载后处理方法生成。测试输入的输出最大绝对差为 0，
argmax 一致率为 100%。计时不包含加载时转换，也不包含 TP 通信；
这里 M=10 是单算子测试维度，不等同于 fine-grained TP 聚合后的实际行数。
因此不能据此推算整个服务的 TPOT 收益。

## 测试与证据

相关单测共 202 项通过，覆盖配置校验、进程组不跨 PP、不同 DP 输入的完整 logits
恢复、非整除 token 行、vocab 切分和 V2 draft 采样。

```bash
pytest -q tests/ut/test_ascend_config.py \
  tests/ut/distributed/test_parallel_state.py \
  tests/ut/ops/test_vocab_parallel_embedding.py \
  tests/ut/ops/test_lmhead_dp_tp.py \
  tests/ut/worker/test_model_runner_v2_finegrained_tp.py
```

本地执行了 `bash format.sh ci`。Ruff、Markdown 等检查通过；部分 shell hooks
因 Windows 上缺少 `/bin/bash` 或 `shellcheck` 未能运行，全量检查未全部通过。
没有运行整个仓库的全部 UT/ST。

容器内证据目录：`/workspace/lmhead-tp16-20261014/`：

- `unmodified.log`：原实现配置校验失败。
- `adapted.log`、`head-shapes.txt`：启动、图执行和全部 rank 的 LM-head 形状。
- `smoke-1.json` 至 `smoke-4.json`：真实并发请求输出。
- `gpqa20.json`、`gpqa20_eval.py`：固定题目与评分程序。
- `tp16/gpqa/`：首轮逐题原始输出及汇总。
- `tp16-32k-retry/gpqa/`：截断题复测输出及汇总。
- `combined-summary.json`：保留逐题来源的合并评分。
- `tp16-nz.log`、`nz-head-formats.txt`：TP16+NZ 启动及全部 rank 的格式记录。
- `nz-smoke-*.json`：TP16+NZ 的 4 个真实请求输出。
- `nz-bench.py`、`nz-bench/result.json`：单算子 ND/NZ 精度与耗时对照。
