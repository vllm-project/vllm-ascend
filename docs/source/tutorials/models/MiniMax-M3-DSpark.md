# MiniMax-M3 QuaRot + DSpark

## Introduction

本文记录 vLLM Ascend `releases/v0.30.0` 上旋转量化 MiniMax-M3 与
[NVIDIA MiniMax-M3-DSpark](https://huggingface.co/nvidia/MiniMax-M3-DSpark) 的适配与验证。
主模型为 `MiniMax-M3_quarot_iter_0626_5`，DSpark 为未旋转的 BF16/FP32 检查点。

DSpark 使用主模型第 `1,12,23,35,46,57` 层输出（从 0 开始），对应残差流边界
`2,13,24,36,47,58`。MiniMax-M3 原先只为 EAGLE3 启用这些输出；本次将 DSpark
加入同一接口，保持残差相加、序列并行 gather 和原有 EAGLE3 行为。

### 两份词表权重

令主模型隐藏状态为 `h_rot = h @ Q`。现有 DSpark 对齐逻辑执行：

- 主模型保留旋转后的 embedding 和 lm_head。
- DSpark 检查点没有 embedding，因此单独加载主模型的词表分片，计算
  `E_draft = E_rot @ Q.T`，保存为独立 embedding，不修改主模型参数。
- DSpark 检查点自带未旋转的 lm_head，直接保留，不与主模型共享。
- 六路隐藏状态的投影分别转换为 `W_fc_i @ Q`，使投影接收旋转后的主模型输出。

矩阵来自主模型 `quant_model_description.json` 中
`optional.quarot.rotation_map.global_rotation`。加载过程会先建立临时 embedding
共享引用，再由 `post_process` 替换为独立的反旋转副本；推理使用的是替换后的权重。

## Supported Features

本轮在 Atlas A3（16 × 64 GiB）使用真实权重验证 MRV2、TP4 × DP4、EP、
FlashComm1 和主模型 FULL_DECODE_ONLY。DSpark draft 使用 eager。
服务及 GPQA 请求成功，但精度和冒烟题仍有问题，不能视为正确性验收通过。
多模态、原生 MTP 和 A2 未验证。

### 生态 Flash Attention

参考 PR #16501 接入 [flash-attention-npu](https://github.com/MinghuasLab/flash-attention-npu)。
仅为 MiniMax M3 及其配套 EAGLE3 / DSpark 的兼容普通注意力选择 FA3；
模型专用 sparse/indexer 路径保持原实现。未安装算子或配置不兼容时沿用原 backend。

滑窗大小来自 attention 层配置，因果性来自运行时 metadata，不写死窗口或 mask。
窗口 token 数 W 转换为左距离 W-1；因果右距离为 0，非因果右距离为 W-1。
连续 K/V 策略同时覆盖主模型和相关 draft。

完整 Q 和请求边界始终传入算子。FULL_DECODE_ONLY 下，单请求 query 长度上界
按 1 + num_speculative_tokens 规划，本配置为 8；该值不是整个 batch 的 token 数。
实际单请求 query 长度超过该上界时保留真实长度，prefill 不截断。

## Environment Preparation

验证镜像 ID：`5bdd40411c1a`。

```text
swr.cn-north-12.myhuaweicloud.com/atlas_inference/vllm-ascend:daily-releases-v0.30.0-a3-cann9.2.1_20261021210600-20261021-r21
```

按 [MiniMax-M3 部署文档](MiniMax-M3.md) 挂载设备、驱动、权重及工作目录。
源码位于 `/vllm-workspace/vllm` 和 `/vllm-workspace/vllm-ascend`。
vLLM commit 为 `ced6857afa0ea7b2e3f0846a62e1394e90f15607`，
Ascend 基于 `5156ddd6c` 加本补丁。PyTorch 为 `2.10.0+cpu`，
torch-npu 为 `2.10.0.post7`，CANN 为 `9.2.1`，
Transformers 为 `5.14.1`，未升级依赖。

先在服务容器中安装生态算子。本次验证的源码 commit 如下，构建版本为 0.4.2：

```bash
cd /workspace
git clone https://github.com/MinghuasLab/flash-attention-npu.git
cd flash-attention-npu
git checkout 920da315f077ab712675644212915549669f6d09
git submodule update --init csrc/catlass
export FLASH_ATTN_BUILD_VERSION=v3
export FLASH_ATTN_BUILD_NPU=910
export MAX_JOBS=16
python -m pip install . --no-build-isolation --no-deps
```

## Deployment

下面为实际验证命令，IP、网卡和权重路径按环境修改。

```bash
#!/usr/bin/env bash
set -eo pipefail
ulimit -c 0
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export VLLM_USE_V2_MODEL_RUNNER=1
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=1024
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export VLLM_HOST_IP=192.168.13.197 HCCL_IF_IP=192.168.13.197
export HCCL_SOCKET_IFNAME=enp209s0f0 GLOO_SOCKET_IFNAME=enp209s0f0 TP_SOCKET_IFNAME=enp209s0f0
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd /workspace
exec vllm serve /mnt/weight/MiniMax/MiniMax-M3_quarot_iter_0626_5 \
 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
 --host 0.0.0.0 --port 11223 --served-model-name minimax-m3 \
 --trust-remote-code --quantization ascend --max-model-len 133000 \
 --tensor-parallel-size 4 --data-parallel-size 4 --api-server-count 1 \
 --max-num-batched-tokens 32768 --long-prefill-token-threshold 4096 \
 --enable-expert-parallel --max-num-seqs 32 --distributed_executor_backend mp \
 --gpu-memory-utilization 0.92 --reasoning-parser minimax_m3 \
 --limit-mm-per-prompt '{"image":1,"video":0}' \
 --enable-chunked-prefill --enable-prefix-caching --disable-hybrid-kv-cache-manager \
 --speculative-config '{"model":"/home/weights/MiniMax-M3-DSpark","method":"dspark","num_speculative_tokens":7,"enforce_eager":true}' \
 --additional-config '{"enable_cpu_binding":true,"enable_flashcomm1":true,"ascend_compilation_config":{"fuse_norm_quant":false},"multistream_overlap_shared_expert":true,"enable_shared_expert_dp":true,"weight_nz_mode":2}'
```

`HCCL_BUFFSIZE=1024` 用于扩大通信缓冲区。原 200 MiB 配置在本轮
MoE dispatch 报所需 216,006,656 bytes、实际 209,715,200 bytes，扩大后完成启动和评测。

`speculative-config.enforce_eager` 仅用于 draft，规避此前 Q/K norm 融合头数冲突。
主模型仍运行图模式。`--disable-hybrid-kv-cache-manager` 规避混合缓存分组容量问题，
不关闭滑窗 mask。133,000 是请求上下文上限，max-num-seqs 32 不代表支持
32 条满长请求同时运行。

## Functional Verification

```bash
curl -fsS http://127.0.0.1:11223/v1/models
curl -fsS http://127.0.0.1:11223/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"minimax-m3","messages":[{"role":"user","content":"What is 2 + 2?"}],"temperature":0,"max_tokens":256,"chat_template_kwargs":{"thinking_mode":"enabled"}}'
```

本轮健康检查和推理返回 HTTP 200，但上述冒烟题最终 content 为 `2`，正确性不通过。
不能仅凭启动成功或请求状态码认定适配通过。

## Accuracy Evaluation

使用固定 GPQA short-10，原始 CSV 行号为
`[6,23,26,35,57,59,70,173,188,189]`，沿用此前 baseline 的提示词与选项顺序。
这 10 题按此前 baseline 输出长度选择，是回归子集，不代表无偏的 GPQA 精度。

AISBench 参数为 0-shot、并发 8、temperature 0.6、top_p 0.95、seed 42、
thinking enabled、最大输出 65,536 tokens。测试服务器已有配置和数据：

```bash
ais_bench /workspace/accuracy/short10_aisbench.py \
  --num-prompts 10 --num-warmups 0 --dump-eval-details --dump-extract-rate \
  -w /workspace/accuracy/dspark-fa3-short10-new
```

该配置及数据为服务器上的验证资产，不随本 PR 提交；新环境需按
[MiniMax-M3 文本评测说明](MiniMax-M3.md#82-text-evaluation) 准备 AISBench，
并保持同一题目、选项和参数才能比较。

| 配置 | 原始评分 | 统一去除最终回答加粗标记后 |
| --- | --- | --- |
| 此前非推测 baseline，同 10 题 | 8/10（80%） | 9/10（90%） |
| 此前 EAGLE3，推测长度 3 | 8/10（80%） | 9/10（90%） |
| 本轮生态 FA3 + DSpark，推测长度 7 | 7/10（70%） | 8/10（80%） |

第 59 行明确回答加粗的 B，原始提取器误提取 A。归一化对所有最终回答统一去除
`**` 后使用原提取器，不从思考过程补分。

第 23、173 行在思考过程已分别得到 D（18）和 C（5 MeV），但最终回答遗漏选项及
最后结论，按最终回答口径仍计错。此前有严重重复的第 188 行本次正确回答 B，
共 3,332 字符，无重复长行。全套同一长行（至少 40 字符）最多出现 6 次，
未发现替换字符 U+FFFD 或异常 ASCII 控制字符；没有复现上百次长句循环。

本轮未达到约 90% 的目标。不同运行同时改变 draft 和算子等配置，不能单独归因
精度变化，也不能据此认定重复问题已彻底解决。

验证资产位于服务器宿主机目录
`/workspace/cxy-minimax-m3-dspark-v030-20261022`，容器内对应 `/workspace`：

- `serve-dspark-fa3.sh`、`serve-dspark-fa3.log`：启动配置和日志。
- `accuracy/dspark-fa3-short10/20261022_172239/`：AISBench 原始预测和评分。
- `accuracy/dspark-fa3-short10/audit-summary.json`：统一评分及重复检查。
- `accuracy/dspark-fa3-short10/row-23-full.txt`、`row-173-full.txt`：完整原始回答。
- `accuracy/dspark-fa3-short10/smoke.json`：冒烟请求原始返回。

## Performance

GPQA 区间的 metrics 前后差值排除冒烟请求：

- 接受 draft token：14,567；提出 draft token：30,842。
- token 采信率：**47.23%**。
- draft 轮数：4,406；含 bonus 的平均接受长度：**4.306**。

不要平均日志中的百分比；本轮没有相同条件的吞吐基线，不能据此报告加速比。

## Validation

开发环境验证：相关单测 819 passed、1 skipped；实际 NPU 算子数值及图重放验证
112 passed。覆盖 GQA、prefill/decode、推测验证、配置化因果/非因果滑窗和缓存页面变化。
验证用测试文件保留在开发环境，按本 PR 范围要求不提交 tests 目录改动。
格式检查使用 `bash format.sh ci`，真实权重 GPQA 结果如上；
算子和单测通过不能替代模型正确性验收。
