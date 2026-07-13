# DeepSeek-R1-Distill

## Introduction

`DeepSeek-R1-Distill` 包含 Qwen 和 Llama 两类稠密文本模型。本教程覆盖已验证的代表检查点：

- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`

这些检查点分别复用 vLLM 中的 `Qwen2ForCausalLM` 和 `LlamaForCausalLM` 路径，不需要新增 DeepSeek MoE/MLA 专属模型代码。配置中的理论最大上下文为 131072，本次在 Atlas 300I DUO (310P) 上验证的实用 `max-model-len` 为 4096；Qwen-32B 多卡验证使用 2048。

## Supported Features

| Feature | Status | Notes |
| --- | --- | --- |
| BF16 checkpoint loading | Supported | 310P 验证时使用 `--dtype float16`，日志显示从 BF16 cast 到 FP16。 |
| ACLGRAPH | Supported | 310P real-weight 验证通过，使用 `FULL_DECODE_ONLY` 和有界 capture sizes。 |
| Tensor parallel | Supported | Qwen-32B 在 310P 上通过 TP4 ACLGRAPH；TP2 通过压力验证但推荐优先使用 TP4。 |
| Multimodal | N/A | 代表检查点是 text-only dense 模型。 |
| MoE / EP / FlashComm | N/A | 代表检查点不是 MoE。 |
| MTP | N/A | 配置和权重未提供 MTP/nextn 层。 |

不要把未验证的大尺寸 Distill 变体标记为已验证 ACLGRAPH 支持。当前 Llama-70B 未验证；需要先完成 real-weight startup、一次非空推理输出和 graph replay 证据。

## Environment Preparation

### Storage And Cache

本教程默认使用 `/data/.cache` 作为下载和运行缓存目录；如环境不同，可只覆盖 `CACHE_ROOT`。

```bash
export CACHE_ROOT=${CACHE_ROOT:-/data/.cache}
export UV_CACHE_DIR=${UV_CACHE_DIR:-${CACHE_ROOT}/uv}
export HF_HOME=${HF_HOME:-${CACHE_ROOT}/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${CACHE_ROOT}}
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-${CACHE_ROOT}/modelscope}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-${CACHE_ROOT}/vllm}
```

### Runtime Environment

310P 验证需要加载 CANN 和 NNAL/ATB 环境。不要同时加载 ATB 和 ASDSIP。

```bash
export ASCEND_TOOLKIT_HOME=${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/cann-9.1.0-beta.1}
export ASCEND_NNAL_HOME=${ASCEND_NNAL_HOME:-/usr/local/Ascend/nnal}
source "${ASCEND_TOOLKIT_HOME}/set_env.sh"
source "${ASCEND_NNAL_HOME}/atb/set_env.sh"
export HCCL_OP_EXPANSION_MODE=AIV
export ASCEND_RT_VISIBLE_DEVICES=0
```

如果运行时报 `libatb.so` 缺失，请先安装 `Ascend-cann-nnal`，然后重新 source `${ASCEND_NNAL_HOME}/atb/set_env.sh`。

### Model Weights

优先使用 ModelScope 或 hf-mirror，并把权重放在 `${CACHE_ROOT}` 下。

使用 ModelScope Python API：

```bash
CACHE_DIR="${MODELSCOPE_CACHE}" uv run python -c "import os; from modelscope import snapshot_download; snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', cache_dir=os.environ['CACHE_DIR'])"
CACHE_DIR="${MODELSCOPE_CACHE}" uv run python -c "import os; from modelscope import snapshot_download; snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Llama-8B', cache_dir=os.environ['CACHE_DIR'])"
CACHE_DIR="${MODELSCOPE_CACHE}" uv run python -c "import os; from modelscope import snapshot_download; snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-32B', cache_dir=os.environ['CACHE_DIR'])"
```

如果环境提供 `vllm_modelscope`，也可以使用它下载同名模型。使用 Hugging Face mirror 时设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=${HF_HOME:-${CACHE_ROOT}/huggingface}
```

## Deployment

以下命令是 310P 上通过 real-weight 验证的 ACLGRAPH 路径。310P 不建议省略 `--max-model-len`，避免按 131072 自动建大 mask 导致 OOM。

### Qwen 1.5B

```bash
if [ -z "${QWEN_MODEL_PATH:-}" ]; then
  QWEN_MODEL_PATH=$(CACHE_DIR="${MODELSCOPE_CACHE}" uv run python - <<'PY'
import os
from modelscope import snapshot_download

print(snapshot_download(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    cache_dir=os.environ["CACHE_DIR"],
))
PY
)
fi
export QWEN_MODEL_PATH

vllm serve "${QWEN_MODEL_PATH}" \
  --served-model-name deepseek-r1-distill-qwen-1.5b \
  --dtype float16 \
  --max-model-len 4096 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.70 \
  --additional-config '{"ascend_compilation_config": {"fuse_norm_quant": false}}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}' \
  --port 8000
```

### Llama 8B

```bash
if [ -z "${LLAMA_MODEL_PATH:-}" ]; then
  LLAMA_MODEL_PATH=$(CACHE_DIR="${MODELSCOPE_CACHE}" uv run python - <<'PY'
import os
from modelscope import snapshot_download

print(snapshot_download(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    cache_dir=os.environ["CACHE_DIR"],
))
PY
)
fi
export LLAMA_MODEL_PATH

vllm serve "${LLAMA_MODEL_PATH}" \
  --served-model-name deepseek-r1-distill-llama-8b \
  --dtype float16 \
  --max-model-len 4096 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.80 \
  --additional-config '{"ascend_compilation_config": {"fuse_norm_quant": false}}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2]}' \
  --port 8000
```

### Qwen 32B TP4

Qwen-32B 在 310P 上用真实权重验证了 TP4 eager 和 TP4 ACLGRAPH。当前主验证拓扑是逻辑设备 `2,3,4,5`，这组设备在测试机器上由物理 NPU5/NPU6 提供并显示为 PHB 关系。实际机器上请先用 `npu-smi info -m` 和 `npu-smi info -t topo` 确认可用拓扑。

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

vllm serve "${QWEN32B_MODEL_PATH}" \
  --served-model-name deepseek-r1-distill-qwen-32b \
  --dtype float16 \
  --tensor-parallel-size 4 \
  --distributed-executor-backend mp \
  --max-model-len 2048 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.60 \
  --additional-config '{"ascend_compilation_config": {"fuse_norm_quant": false}}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}' \
  --port 8000
```

实测 TP4 ACLGRAPH 边界：

- `ASCEND_RT_VISIBLE_DEVICES=2,3,4,5`
- `tensor_parallel_size=4`
- `max_model_len=2048`
- `max_num_seqs=1`
- `max_num_batched_tokens=2048`
- `gpu_memory_utilization=0.60`
- `cudagraph_capture_sizes=[1]`
- 每个 TP rank 权重约 16.08 GiB
- graph capture 约 3 秒，NPU graph memory 约 0.15 GiB

TP2 也完成了压力验证，使用 `ASCEND_RT_VISIBLE_DEVICES=0,1`、`max_model_len=1024`、`gpu_memory_utilization=0.90`、capture sizes `[1]`，每个 rank 权重约 32.06 GiB。TP2 显存边界更紧，文档推荐生产或回归验证优先使用 TP4。

Eager 模式只用于隔离问题，不作为 ACLGRAPH 支持证据：

```bash
vllm serve <model-path> --dtype float16 --max-model-len 4096 --enforce-eager --port 8000
```

## Functional Verification

服务启动后必须发起真实请求。`Application startup complete` 不是充分证据。

```bash
curl -sf http://127.0.0.1:8000/v1/models

curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-r1-distill-qwen-1.5b",
    "messages": [{"role": "user", "content": "Reply with the word graph."}],
    "temperature": 0,
    "max_tokens": 16
  }'
```

ACLGRAPH 证据需要同时包含：

- 日志出现 `Graph capturing finished`
- 一次 `/v1/chat/completions` 返回 HTTP 200 且输出非空
- 请求后日志出现 `Replaying aclgraph`

## Accuracy Evaluation

本次使用 `tests/e2e/models/test_lm_eval_correctness.py` 在 310P 上完成 real-weight eager accuracy 验证。accuracy 配置使用 eager，是为了把质量评测与 graph-mode 证据分开；ACLGRAPH 覆盖由 `_310p/test_deepseek_r1_distill_aclgraph_310p.py` 提供。

| Model | Task | num_fewshot | limit | strict-match | flexible-extract |
| --- | --- | ---: | ---: | ---: | ---: |
| DeepSeek-R1-Distill-Qwen-1.5B | gsm8k | 5 | 32 | 0.0 | 0.0 |
| DeepSeek-R1-Distill-Llama-8B | gsm8k | 5 | 32 | 0.0 | 0.1875 |
| DeepSeek-R1-Distill-Qwen-32B | gsm8k | 5 | 32 | 0.0 | 0.125 |

运行示例：

```bash
export VLLM_USE_MODELSCOPE=True
export HF_ENDPOINT=https://hf-mirror.com
export REPORT_DIR=${REPORT_DIR:-/tmp/deepseek_r1_distill_accuracy}
uv run pytest -q tests/e2e/models/test_lm_eval_correctness.py \
  --config tests/e2e/models/configs/DeepSeek-R1-Distill-Qwen-1.5B.yaml \
  --tp-size 1 \
  --report-dir "${REPORT_DIR}"
```

Qwen-32B accuracy 基线使用 TP4 和 eager 模式采集，ACLGRAPH 支持仍由四卡 `_310p` smoke 测试证明。

## Performance

310P 上建议从小 capture sizes 开始，例如 Qwen-1.5B `[1,2,4]`、Llama-8B `[1,2]`、Qwen-32B TP4 `[1]`。如果遇到 stream resource exhaustion、capture begin 失败或 OOM：

- 保持 `FULL_DECODE_ONLY`
- 减小 `cudagraph_capture_sizes`
- 降低 `max-model-len` 或 `max-num-seqs`
- 使用 `--enforce-eager` 只做问题隔离

吞吐测试可使用：

```bash
if [ -z "${QWEN_MODEL_PATH:-}" ]; then
  QWEN_MODEL_PATH=$(CACHE_DIR="${MODELSCOPE_CACHE}" uv run python - <<'PY'
import os
from modelscope import snapshot_download

print(snapshot_download(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    cache_dir=os.environ["CACHE_DIR"],
))
PY
)
fi
export QWEN_MODEL_PATH
export BENCH_RESULT_DIR=${BENCH_RESULT_DIR:-/tmp/deepseek_r1_distill_bench}

vllm bench serve \
  --model "${QWEN_MODEL_PATH}" \
  --served-model-name deepseek-r1-distill-qwen-1.5b \
  --dataset-name random \
  --random-input 200 \
  --num-prompts 200 \
  --request-rate 1 \
  --save-result \
  --result-dir "${BENCH_RESULT_DIR}"
```
