# Kimi-K2.6 (W4A8) — Model Runner V1 / V2 可复现配置（A3 单机 16 die）

> 本文件是 `test_kimi_k26_mrv1_mrv2_parity.py` 的配套记录。所有数值均为 2026-11-19 在
> univpn `198.186.1.2` 上实测，不是估算。

## 1. 镜像

| 项 | 值 |
|---|---|
| tag | `quay.io/ascend/vllm-ascend:v0.30.0-a3` |
| content digest | `sha256:e6c22b94f4ab3634ad20ae4ad28b73b79ef59a1e9eb0372454affd8c186e36a6` |
| local image id | `sha256:77aea90962507771cfaf5304c926ee3f3539907eb59d37edb560177e7e7b4033` |
| 说明 | 本机该 tag 由 `quay.nju.edu.cn/ascend/vllm-ascend:v0.30.0-a3` 镜像而来，两者 image id 相同，因此 content digest 与任务书一致；`docker images --digests` 对该 tag 自身未记录 RepoDigest，故以 image id 建立同一性 |

## 2. 组件版本（容器内实测）

| 组件 | 版本 |
|---|---|
| vllm | `0.30.0` |
| vllm-ascend | `0.30.0`（源码树 git `c09d3e20b9c45c80b34ae3781440d90904a44023` = tag `v0.30.0`） |
| CANN | `9.2.1`（镜像内置；宿主驱动 `25.5.0`、宿主 CANN `9.1.0` 只提供驱动/固件） |
| torch | 镜像内置（`torch_npu` 同源） |
| Python | `3.12.16` |
| OS / kernel（宿主） | openEuler 22.03 LTS-SP4 / `5.10.0-216.0.0.115.oe2203sp4.aarch64` |

`tox`/CI 侧的精确版本由 `_software_versions()` 在运行时写入结果 JSON 的 `software` 字段，
以运行时值为准。

## 3. 硬件

| 项 | 值 |
|---|---|
| 机器 | Atlas 800T A3，单机 |
| NPU | 16 × Ascend910（`19e5:d803`），64 GiB HBM / die |
| 使用方式 | 16 die 全部占用（TP8 × DP2，EP 跨 16 rank） |

## 4. 模型与权重指纹

| 项 | 目标模型 | 草稿模型 |
|---|---|---|
| 路径 | `/models/Kimi-K2.6-w4a8` | `/models/Kimi-K2.5-DFlash` |
| 结构 | `KimiK25ForConditionalGeneration`（文本塔 61 层，384 experts，MLA） | `DFlashDraftModel`（6 层，block_size 8） |
| 量化 | W4A8（experts int4/per-channel + int8/per-token act） | bf16 |
| 分片 | 126 × `quant_model_weights-*-of-00126.safetensors`，535,908,987,345 B | 2 |
| index sha256 | `eaea6655258a65e5fdb90f191d7bba07180247654da4c085040533f958f9da55` | `f8f42e43954db93c471bc7b63c2dc82a87c5fe0d0e0d1ce2b6b3c50b4c7f69b9` |
| config sha256 | `83d3abc77267a7690e736d57be3b238b0231d143b5b76a7c77d19c107d1c4509` | `ef214622b1d52ccc2e60409ae2d6c303bf6b375477eb94370384c15df0c49bb7` |

权重来自 ModelScope `Eco-Tech/Kimi-K2.6-w4a8`（非 git 仓库），因此用 **index 文件 sha256 + 分片数 + 总字节数**
作为可复现的 revision 标识；`KIMI_MODEL_REVISION` 仅在换成 HF 镜像时使用。

## 5. 并行与运行配置

| 参数 | 值 |
|---|---|
| TP / DP / EP | `8 / 2 / 16`（`--enable-expert-parallel`） |
| 图模式 | `{"cudagraph_mode": "FULL_DECODE_ONLY"}`（max capture size 64） |
| 上下文 | `--max-model-len 65536` |
| 批 | `--max-num-seqs 8`，`--max-num-batched-tokens 8192` |
| 显存 | `--gpu-memory-utilization 0.85`（实测峰值 58.9 GB/die） |
| 前缀缓存 | 关闭（`--no-enable-prefix-caching`） |
| 推测解码 | `{"method":"dflash","model":"<draft>","num_speculative_tokens":7}` |
| 多模态 | `--mm-processor-cache-gb 0 --mm-encoder-tp-mode data` |
| 加载 | `--safetensors-load-strategy prefetch` |
| 前端解析 | `--tool-call-parser kimi_k2 --reasoning-parser kimi_k2 --enable-auto-tool-choice`（缺后者时带 `tools` 的请求会被 400 拒绝，已实测） |
| seed | `1234` |

## 6. 完整可复制命令

```bash
IMG=quay.io/ascend/vllm-ascend:v0.30.0-a3
MODEL=/models/Kimi-K2.6-w4a8
DRAFT=/models/Kimi-K2.5-DFlash

docker run -d --name kimi_mrv_parity --privileged --shm-size 64g \
  -v /dev:/dev \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v $MODEL:$MODEL:ro -v $DRAFT:$DRAFT:ro -v /work:/work \
  -e LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/dcmi \
  --entrypoint bash $IMG -c "exec sleep infinity"

# 注意：容器内任何 python/vllm 调用都必须走 `bash -lc`（上面的 LD_LIBRARY_PATH
# 覆盖了镜像自带值，只有 login shell 会 source CANN profile 从而找到 libhccl.so）。

# 服务（两 runner 只差 VLLM_USE_V2_MODEL_RUNNER）
VLLM_USE_V2_MODEL_RUNNER=0 \
HCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 TP_SOCKET_IFNAME=eth0 \
HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=800 \
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True OMP_PROC_BIND=false OMP_NUM_THREADS=10 \
TASK_QUEUE_ENABLE=1 TOKENIZERS_PARALLELISM=false \
vllm serve $MODEL \
  --host 0.0.0.0 --port 8000 --quantization ascend \
  --tensor-parallel-size 8 --data-parallel-size 2 --enable-expert-parallel \
  --trust-remote-code --safetensors-load-strategy prefetch --no-enable-prefix-caching \
  --max-num-seqs 8 --max-model-len 65536 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.85 --mm-processor-cache-gb 0 --mm-encoder-tp-mode data \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative-config '{"method":"dflash","model":"'$DRAFT'","num_speculative_tokens":7}' \
  --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 --enable-auto-tool-choice \
  --seed 1234
```

测试入口：

```bash
cd /vllm-workspace/vllm-ascend
KIMI_MODEL=$MODEL KIMI_DRAFT_MODEL=$DRAFT HCCL_SOCKET_IFNAME=eth0 \
pytest -sv tests/e2e/nightly/single_node/model_runner_v2/test_kimi_k26_mrv1_mrv2_parity.py
```

## 7. 判定口径

| 维度 | 规则 |
|---|---|
| greedy token | 逐 token 相同；首个 token 必须一致（否则 `prefill_token_mismatch`） |
| greedy logprob | `|Δ| ≤ 0.0689`（index 0）/ `≤ 0.1378`（index ≥ 1）；token 不一致时用右侧 top-20 反查左侧 token |
| sampled | 只比较跨 runner token 重合率；**不断言相等** —— 实测同一 runner 内相同 seed 也不可复现（详见报告 §精度） |
| API 契约 | `content` 中不得出现 `<|tool_calls_section_begin|>` 等 Kimi 控制标记；tool call 的 arguments 必须是合法 JSON |
| 性能 | 默认只记录；`KIMI_STRICT_PERF=1` 时按 10% 容差断言（单轮无配对，噪声需要 ABBA 四轮才能分离） |
