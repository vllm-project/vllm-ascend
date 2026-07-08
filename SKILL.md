# DeepSeek-R1-Distill ACLGRAPH Adaptation Notes

## 适用范围

本记录适用于 DeepSeek-R1-Distill Qwen/Llama 系列在 vLLM Ascend 上的 ACLGRAPH 适配。当前已用真实权重验证：

- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`

不要把 Qwen 7B/14B、Llama 70B 或其他变体声明为已验证，除非补齐对应 real-weight 日志。

## 模型路由决策

- Qwen-backed Distill 走 `Qwen2ForCausalLM` / `qwen2` 路径。
- Llama-backed Distill 走 `LlamaForCausalLM` / `llama` 路径。
- 两个代表检查点都是 dense text-only BF16 权重，在 310P 上用 `--dtype float16` 运行。
- 不新增 DeepSeek MoE/MLA patch；只有确认现有 Qwen/Llama 路径无法工作时才考虑最小 patch。

## 验证顺序

1. 检查 `config.json`：`architectures`、`model_type`、`torch_dtype`、`max_position_embeddings`、是否有 MoE/MTP/multimodal 字段。
2. 下载真实权重到 `${CACHE_ROOT}`，默认 `CACHE_ROOT=/data/.cache`，优先 ModelScope 或 hf-mirror。
3. 先跑 eager real-weight smoke，确认加载和一次非空推理。
4. 再跑 ACLGRAPH real-weight smoke，必须包含 `FULL_DECODE_ONLY`、有界 `cudagraph_capture_sizes`、一次非空推理和 `Replaying aclgraph` 日志。
5. accuracy 评测与 graph 证据分开：`lm_eval` 可用 eager 保持稳定，ACLGRAPH 用专门 E2E 覆盖。
6. 大模型多卡先跑 TP eager smoke，再跑 ACLGRAPH，避免把权重加载/HCCL 问题误判为 graph capture 问题。
7. Qwen-32B accuracy 基线使用 `gsm8k`、`limit=32`、`num_fewshot=5`、TP4 eager 采集；实测 `exact_match,strict-match=0.0`，`exact_match,flexible-extract=0.125`。

## 310P ACLGRAPH 参数

Qwen 1.5B 通过参数：

```bash
--dtype float16 \
--max-model-len 4096 \
--max-num-seqs 4 \
--max-num-batched-tokens 4096 \
--gpu-memory-utilization 0.70 \
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}'
```

Llama 8B 通过参数：

```bash
--dtype float16 \
--max-model-len 4096 \
--max-num-seqs 2 \
--max-num-batched-tokens 4096 \
--gpu-memory-utilization 0.80 \
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2]}'
```

Qwen 32B TP4 通过参数：

```bash
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-2,3,4,5}
if [ -z "${QWEN32B_MODEL_PATH:-}" ]; then
  QWEN32B_MODEL_PATH=$(CACHE_DIR="${MODELSCOPE_CACHE}" uv run python - <<'PY'
import os
from modelscope import snapshot_download

print(snapshot_download(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    cache_dir=os.environ["CACHE_DIR"],
))
PY
)
fi
export QWEN32B_MODEL_PATH

--dtype float16 \
--tensor-parallel-size 4 \
--distributed-executor-backend mp \
--max-model-len 2048 \
--max-num-seqs 1 \
--max-num-batched-tokens 2048 \
--gpu-memory-utilization 0.60 \
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}'
```

Qwen 32B TP2 压力验证通过参数：

```bash
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1}

--dtype float16 \
--tensor-parallel-size 2 \
--distributed-executor-backend mp \
--max-model-len 1024 \
--max-num-seqs 1 \
--max-num-batched-tokens 1024 \
--gpu-memory-utilization 0.90 \
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}'
```

310P 不要省略 `--max-model-len`。配置理论值是 131072，但当前实测通过值是小模型 4096、Qwen-32B TP4 2048、Qwen-32B TP2 1024。

## 环境与缓存

```bash
export CACHE_ROOT=${CACHE_ROOT:-/data/.cache}
export ASCEND_TOOLKIT_HOME=${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/cann-9.1.0-beta.1}
export ASCEND_NNAL_HOME=${ASCEND_NNAL_HOME:-/usr/local/Ascend/nnal}
source "${ASCEND_TOOLKIT_HOME}/set_env.sh"
source "${ASCEND_NNAL_HOME}/atb/set_env.sh"
export UV_CACHE_DIR=${UV_CACHE_DIR:-${CACHE_ROOT}/uv}
export HF_HOME=${HF_HOME:-${CACHE_ROOT}/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${CACHE_ROOT}}
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-${CACHE_ROOT}/modelscope}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-${CACHE_ROOT}/vllm}
export HCCL_OP_EXPANSION_MODE=AIV
```

如果缺少 `libatb.so`，安装 `Ascend-cann-nnal` 后 source `${ASCEND_NNAL_HOME}/atb/set_env.sh`。不要同时 source ATB 和 ASDSIP。

## 算子与运行时注意事项

- `FULL_DECODE_ONLY` 可降低 310P capture 压力。
- 大模型先从 `cudagraph_capture_sizes=[1]` 开始；TP4 通过后再考虑扩大 capture sizes。
- Qwen-32B TP4 每 rank 权重约 16.08 GiB；TP2 每 rank 权重约 32.06 GiB，显存边界明显更紧。
- Qwen-32B TP4 graph capture 约 3 秒，NPU graph memory 约 0.15 GiB；TP2 同样约 0.15 GiB。
- stream resource exhaustion 时先减小 `cudagraph_capture_sizes`，再降低 `max-num-seqs` 或 `max-model-len`。
- `Graph capturing finished` 只证明 capture 完成；必须在真实请求后看到 `Replaying aclgraph`。
- `--enforce-eager` 是隔离手段，不是 ACLGRAPH 通过证据。
- 避免在热路径新增 NPU tensor `.item()`，防止 CPU/NPU 同步。

## 不适用功能

- Multimodal：N/A，代表检查点是 text-only。
- MoE / Expert Parallel / FlashComm：N/A，代表检查点不是 MoE。
- MTP：N/A，配置和权重未包含 nextn/MTP 层。
- Llama-70B 和未列出的 Distill 变体：checkpoint-dependent，需要单独 real-weight 验证。

## PR/Issue 摘要模板

Task #21 / #9079 summary:

- Added DeepSeek-R1-Distill Qwen/Llama accuracy configs.
- Added DeepSeek-R1-Distill Qwen-32B accuracy config with measured `gsm8k` metrics: strict-match `0.0`, flexible-extract `0.125`.
- Added 310P ACLGRAPH E2E coverage with `FULL_DECODE_ONLY`.
- Verified real weights for Qwen-1.5B and Llama-8B from `${MODELSCOPE_CACHE}`.
- Verified Qwen-32B real weights on 310P TP4 with `ASCEND_RT_VISIBLE_DEVICES=2,3,4,5`, `max_model_len=2048`, and capture sizes `[1]`.
- Verified Qwen-32B TP2 pressure path with `ASCEND_RT_VISIBLE_DEVICES=0,1`, `max_model_len=1024`, and capture sizes `[1]`; prefer TP4 for headroom.
- Qwen graph smoke: capture sizes `[1,2,4]`, non-empty chat output, `Replaying aclgraph`.
- Llama graph smoke: capture sizes `[1,2]`, non-empty chat output, `Replaying aclgraph`.
- Qwen-32B graph smoke: non-empty output, `Graph capturing finished`, and `Replaying aclgraph`.
- Multimodal, MoE, EP, FlashComm, and MTP are not applicable for these dense text-only checkpoints.
