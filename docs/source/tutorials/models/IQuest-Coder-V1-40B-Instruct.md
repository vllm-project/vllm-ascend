# IQuest-Coder-V1-40B-Instruct

## Introduction

`IQuestLab/IQuest-Coder-V1-40B-Instruct` is a 40B-parameter code LLM released by IQuestLab. Despite a custom architecture name (`IQuestCoderForCausalLM`) in its `config.json`, the model is a **standard LLaMA backbone** — verified field-by-field against `config.json` and the safetensors key layout (`q_proj/k_proj/v_proj.../lm_head` are identical to `LlamaForCausalLM`). The custom name does not imply a custom structure.

Key architecture facts (relevant for tensor parallelism):

| Field | Value |
|---|---|
| hidden_size | 5120 |
| num_hidden_layers | 80 |
| num_attention_heads | 40 |
| num_key_value_heads | 8 |
| head_dim | 128 |
| vocab_size | 76800 |
| dtype | bf16 |
| max_position_embeddings | 131072 |

Because the model ships `auto_map` remote code (a custom config/tokenizer class), `--trust-remote-code` is **required** and cannot be bypassed. The remote code was written for the transformers 4.x era; on newer transformers (5.x) it is the first suspect if loading fails (see [Known Limitations](#9-known-limitations)).

## Supported Features

- **Tensor parallelism**: TP must divide both 40 (attention heads) and 8 (KV heads) → **TP ∈ {1, 2, 4, 8}** (TP3 fails during weight sharding).
- **Precision**: bf16.
- **Architecture mapping**: `--hf-overrides '{"architectures": ["LlamaForCausalLM"]}'` (see [Deployment](#5-online-service-deployment)).
- **Execution mode**: enforce-eager recommended for serving; chunked prefill disabled in the validated configuration.

## Environment Preparation

### Model Weight

`IQuest-Coder-V1-40B-Instruct` (BF16) requires ~80GB of disk and **2 x Ascend 910B (64GB) NPUs** for TP2 serving. Reserve at least **120GB** of free disk before downloading.

The weights are non-gated. Download with resumable transfer (HF mirror shown; plain `hf download` works from anywhere):

```bash
export HF_ENDPOINT=https://hf-mirror.com   # omit if huggingface.co is reachable directly
hf download IQuestLab/IQuest-Coder-V1-40B-Instruct
```

`hf download` stores files under `~/.cache/huggingface/hub` and is **resumable** — re-running the same command after an interruption continues from the last completed shard. Verify completeness by checking the 16-file manifest (15 safetensors shards + config/tokenizer/index files):

```bash
hf download IQuestLab/IQuest-Coder-V1-40B-Instruct   # exits 0 when nothing is left to fetch
du -sh ~/.cache/huggingface/hub/models--IQuestLab--IQuest-Coder-V1-40B-Instruct  # ~80GB
```

ModelScope mirror (fallback if HF is unreachable):

```bash
modelscope download --model IQuestLab/IQuest-Coder-V1-40B-Instruct
```

It is recommended to place the model weight in a shared cache directory, such as `/root/.cache/` or a local model path like `/data/vllm-workspace/models/IQuest-Coder-V1-40B-Instruct`.

### Installation

`IQuest-Coder-V1-40B-Instruct` can be deployed with `vllm-ascend` in a compatible runtime environment.

You can use the official docker image for deployment:

```bash
export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
docker run --rm \
  --name vllm-ascend \
  --shm-size=1g \
  --device /dev/davinci0 \
  --device /dev/davinci1 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /root/.cache:/root/.cache \
  -v /data/vllm-workspace/models:/data/vllm-workspace/models \
  -p 8000:8000 \
  -it $IMAGE bash
```

If you do not want to use the docker image, you can also build from source:

- Install `vllm-ascend` from source, refer to [installation](../../installation.md).

## Online Service Deployment

The custom architecture name `IQuestCoderForCausalLM` is unknown to vLLM's model registry. Instead of patching the registry (an alias entry that rots across vLLM versions), map the architecture at load time with **`--hf-overrides`** — a configuration-only, version-stable approach:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 vllm serve IQuestLab/IQuest-Coder-V1-40B-Instruct \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --trust-remote-code \
  --hf-overrides '{"architectures": ["LlamaForCausalLM"]}' \
  --no-enable-chunked-prefill \
  --port 8000
```

Why `hf-overrides` rather than a registry alias patch:

| | alias patch | `--hf-overrides` (used here) |
|---|---|---|
| Code change | source registry entry | none (CLI/config only) |
| Version drift | alias rots across vLLM versions | survives |
| Evidence | CI threshold runs | dual-run equivalence (identical results with either route) |

Safety of the mapping: the safetensors key layout of `IQuestCoderForCausalLM` is identical to `LlamaForCausalLM` (verified key-by-key), so no weight renaming or structural adaptation happens — only the class used for instantiation changes.

## Functional Verification

Once your server is started, you can query the model with a simple prompt:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "IQuestLab/IQuest-Coder-V1-40B-Instruct",
    "prompt": "def quicksort(arr):",
    "max_tokens": 128,
    "temperature": 0.0
  }'
```

A valid Python implementation in the response indicates that the model is deployed correctly.

## Accuracy Evaluation

Accuracy is tracked through the e2e harness:

```bash
pytest -sv tests/e2e/models/test_lm_eval_correctness.py \
  --config tests/e2e/models/configs/IQuest-Coder-V1-40B-Instruct.yaml \
  --tp-size 2
```

The evaluation setting encoded in the config:

- Tasks: `humaneval` (pass@1), `mbpp` (pass_at_1).
- **Completion mode** (`apply_chat_template: false`) is required for this model: with the chat template applied, the model's reasoning preamble is truncated by humaneval's `until` sequences (`\ndef`, `\nclass`, ...) before any code is written, which scores 0.0.
- Few-shot counts follow each task's own default (humaneval 0-shot, mbpp 3-shot); a global `num_fewshot: 0` forces mbpp into 0-shot and also collapses its score.
- Environment: `HF_ALLOW_CODE_EVAL=1` (code_eval metric) + `confirm_run_unsafe_code=True` (lm_eval evaluator gate) + `HF_HUB_OFFLINE=1` (weights/datasets pre-cached; avoids hub stalls).
- Thresholds (RTOL = 0.05, `np.isclose` per metric in the harness), locally calibrated on Atlas A2 TP2 (3 runs, median):

| Task | Metric | Threshold | Status |
|---|---|---|---|
| humaneval | pass@1 | 0.689 | calibrated (runs: 0.6768 / 0.689 / 0.6951) |
| mbpp | pass_at_1 | 0.724 | calibrated (runs: 0.718 / 0.734 / 0.724) |

Thresholds are finalized by the nightly accuracy run registered in `.github/workflows/configs/nightly_config.yaml` (accuracy-group-3, two-card runner — required because `serve.tensor_parallel_size: 2` overrides the CLI `--tp-size`).

## Memory Scheduling

TP2 memory account (paper):

- Weights: ~40GB bf16 per card.
- Runtime reserve (eager mode + activations): ~3GB + ~2.6GB.
- KV cache budget: ~12GB per card at `gpu_memory_utilization=0.85` → **160KB/token/card** (80 layers x 8 KV heads x 128 head_dim x 2 (K+V) x 2 bytes = 160KB per token per card).
    - 4096 tokens → ~0.62GB; 8192 tokens → ~1.25GB.
- Concurrency safety line: 4–6 concurrent sequences (logits peaks ~2.3GB/request).

Runtime HBM sampling recipe (run while an eval/serving round is in flight):

```bash
while true; do date; npu-smi info; sleep 30; done | tee hbm_sampling.log
```

Compare the sampled peak HBM against the paper account above: weights (~40GB/card) + KV (160KB/token x seq length x concurrency) should account for the observed usage within ~15%.

Measured during the local calibration rounds (TP2/8192, `gpu_memory_utilization=0.85`):

| Card | Paper budget (util 0.85) | Measured peak | Delta |
|---|---|---|---|
| NPU 0 | 55718MB (54.4GiB) | 56821MB (86.7% of 64GB) | +2.0% (activations / non-torch allocations) |
| NPU 1 | 55718MB (54.4GiB) | 56583MB (86.3% of 64GB) | +1.6% |

The measured peaks match the paper account within the expected overhead; no OOM occurred across the calibration rounds.

## Performance Tuning

- If `--max-model-len 8192` OOMs, fall back to `4096` (validated precedent) or reduce concurrency (`--max-num-seqs 4`).
- For long-context work (up to 131072 positional embeddings) lower `gpu_memory_utilization` headroom for the KV pool and reduce `--max-num-seqs` accordingly.
- Chunked prefill is disabled in the validated configuration; re-enable only with a dedicated memory sweep.

## Known Limitations

- **transformers 5.x remote-code compatibility**: the model requires `transformers>=4.52.4` but the repo pins a 5.x transformers. The `auto_map` remote code (config/tokenizer classes) is written for the 4.x API. If model loading fails with a modeling/tokenizer error, this is the first suspect — test against a known-good transformers version, or use an isolated virtualenv for this model (keep the main environment untouched).
- **TP3 is invalid**: 40 attention heads and 8 KV heads cannot be sharded 3-way; the TP constraint is {1, 2, 4, 8}.
- **Slow tokenizer**: the remote tokenizer code path is slow on first load; pre-check with `backend_tokenizer` where supported.
- **Custom architecture name**: without `--hf-overrides` (or an equivalent registry alias), `vllm serve` fails at architecture resolution.
