# 淬火行动-yxr2-Qwen3-32B-W4A4 性能测试报告

- 交付人：yxr2
- 跟踪 Issue：https://github.com/vllm-project/vllm-ascend/issues/11759
- 测试日期：2026-07-22

已完成 Qwen3-32B-W4A4 在线推理服务的基础性能验证。200 个请求全部成功，结论：**PASS**。

## 1. 测试范围

本次测试按照 Issue #11759 的要求，使用 vLLM benchmark 工具验证 Qwen3-32B-W4A4 在线推理服务的性能，记录可复现的测试命令、完整指标、问题与规避方案。

## 2. 测试环境

以下硬件和软件配置来自 Issue #11759 的任务书：

| 项目 | 配置 |
| --- | --- |
| 镜像 | `quay.io/ascend/vllm-ascend:v0.18.0` |
| 模型 | `vllm-ascend/Qwen3-32B-W4A4` |
| 硬件 | Atlas 800I A2，单卡 |
| Ascend HDK | 25.5.0.B078 |
| 内存 | 1 TB |
| 硬盘 | 300 GB |

本次 benchmark 日志确认 Ascend 平台插件已激活，并注册了 vLLM-Ascend 模型加载器。模型 tokenizer 使用实验环境中的本地路径：

```text
/workspace/shared_assets/models/Qwen/Qwen3-32B-W4A4
```

## 3. 性能测试方法

### 3.1 测试参数

| 参数 | 配置 |
| --- | ---: |
| API endpoint | `/v1/completions` |
| Dataset | Random |
| 输入长度 | 1024 tokens |
| 输出长度 | 128 tokens |
| 请求数量 | 200 |
| 请求速率 | 1 RPS |
| Burstiness | 1.0 |
| 最大并发数 | 不限制，由请求到达和服务处理速度决定 |

命令中未显式指定输出长度；benchmark 参数回显显示 `random_output_len=128`，因此本次测试输出长度为 128 tokens。

### 3.2 测试命令

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 vllm bench serve \
    --model qwen3-32b-w4a4 \
    --tokenizer /workspace/shared_assets/models/Qwen/Qwen3-32B-W4A4 \
    --endpoint /v1/completions \
    --dataset-name random \
    --random-input 1024 \
    --num-prompts 200 \
    --request-rate 1 \
    --save-result \
    --result-dir ./
```

该命令连接默认地址 `http://127.0.0.1:8000`，并将测试结果保存到当前目录。

## 4. 性能测试结果

### 4.1 请求与吞吐指标

| 指标 | 结果 |
| --- | ---: |
| Successful requests | 200 |
| Failed requests | 0 |
| 请求成功率 | 100% |
| Benchmark duration | 207.09 s |
| Total input tokens | 204800 |
| Total generated tokens | 25600 |
| Request throughput | 0.97 req/s |
| Output token throughput | 123.62 tok/s |
| Peak output token throughput | 274.00 tok/s |
| Total token throughput | 1112.58 tok/s |
| Peak concurrent requests | 19 |

### 4.2 延迟指标

| 指标 | Mean | Median | P99 |
| --- | ---: | ---: | ---: |
| TTFT | 346.31 ms | 313.00 ms | 713.66 ms |
| TPOT | 64.66 ms | 64.79 ms | 77.52 ms |
| ITL | 64.66 ms | 55.54 ms | 165.83 ms |

其中：

- TTFT（Time to First Token）表示首 Token 延迟；
- TPOT（Time per Output Token）表示除首 Token 外的平均单 Token 生成耗时；
- ITL（Inter-token Latency）表示相邻输出 Token 之间的延迟。

### 4.3 原始结果摘要

```text
============ Serving Benchmark Result ============
Successful requests:                     200
Failed requests:                         0
Request rate configured (RPS):           1.00
Benchmark duration (s):                  207.09
Total input tokens:                      204800
Total generated tokens:                  25600
Request throughput (req/s):              0.97
Output token throughput (tok/s):         123.62
Peak output token throughput (tok/s):    274.00
Peak concurrent requests:                19.00
Total token throughput (tok/s):          1112.58
Mean TTFT (ms):                          346.31
Median TTFT (ms):                        313.00
P99 TTFT (ms):                           713.66
Mean TPOT (ms):                          64.66
Median TPOT (ms):                        64.79
P99 TPOT (ms):                           77.52
Mean ITL (ms):                           64.66
Median ITL (ms):                         55.54
P99 ITL (ms):                            165.83
==================================================
```

保存的原始结果文件为：

```text
openai-1.0qps-qwen3-32b-w4a4-20260722-060936.json
```

## 5. 功能验证

性能测试共向 `/v1/completions` 端点发送 200 个在线推理请求，全部完成且无失败请求。这证明在本次 1024 输入 Token、128 输出 Token、1 RPS 的测试场景下：

- 在线推理服务可以正常接收请求；
- 模型能够完成 Token 生成；
- 服务在持续 207.09 秒的测试期间未出现请求失败。

本次使用随机 Token 数据集，验证目标是服务功能和性能，不作为语义精度评测结果。

## 6. 问题、规避方案与 FAQ

### 6.1 离线环境下 tokenizer 可能尝试访问 Hugging Face Hub

**现象：** 使用服务别名作为 tokenizer 时，benchmark 可能尝试连接 Hugging Face Hub。

**规避方案：** 开启离线模式，并显式指定本地 tokenizer 路径：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
vllm bench serve \
    --tokenizer /workspace/shared_assets/models/Qwen/Qwen3-32B-W4A4 \
    ...
```

本次测试在离线模式下正常完成。

### 6.2 新版 benchmark 不再默认使用 greedy sampling

日志包含以下提示：

```text
vllm bench serve no longer sets temperature==0 (greedy) in requests by default.
```

本次测试采用服务端默认采样配置。如果需要与旧版 benchmark 的 greedy 行为保持一致，应显式增加：

```bash
--temperature 0
```

### 6.3 测试结束时出现 swigvarlink 弃用警告

测试结束后出现：

```text
DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

该警告发生在全部请求完成之后，未影响 benchmark 结果，200 个请求均执行成功。

## 7. 社区文档验证反馈

参考文档：

https://docs.vllm.ai/projects/ascend/en/v0.18.0/tutorials/models/Qwen3-32B-W4A4.html

验证结论与建议：

1. 文档给出的 Qwen3-32B-W4A4 单卡部署和 benchmark 流程能够指导用户完成测试。
2. 建议在 benchmark 示例中明确输入长度、输出长度、请求数和请求速率，便于不同环境之间对比结果。
3. 建议补充离线环境使用本地 tokenizer 的示例，避免 benchmark 尝试访问远程模型仓库。
4. 建议明确新版 benchmark 的采样行为，并说明可使用 `--temperature 0` 获得确定性 greedy 输出。
5. Issue #11759 测试方法中的部分链接将中文说明拼入 URL，建议将“的”和“Run basic benchmark”等说明移到链接之外。

## 8. 最终结论

Qwen3-32B-W4A4 在线推理性能测试通过：

- 200 个请求全部成功，失败请求为 0；
- 请求吞吐为 0.97 req/s；
- 输出 Token 吞吐为 123.62 tok/s；
- 总 Token 吞吐为 1112.58 tok/s；
- 平均 TTFT 为 346.31 ms，平均 TPOT 为 64.66 ms；
- 离线 tokenizer 配置有效，测试结果已成功保存为 JSON 文件。

本次随机数据集测试能够证明服务功能和基础性能正常；独立语义精度评测不在本次 benchmark 的覆盖范围内。
