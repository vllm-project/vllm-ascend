# LoRA Adapters

## Feature Introduction

LoRA serves a frozen base model together with low-rank adapters. The client selects an adapter by setting `model` to the `name` registered in `--lora-modules`. The base weights stay loaded.

Argument semantics and adapter file layout follow the [vLLM LoRA guide](https://docs.vllm.ai/en/latest/features/lora/). In the launch command, `...` stands for the other flags of that deployment.

## Supported Hardware

LoRA on vLLM-Ascend is supported on Atlas 300I Duo, Atlas A2, and Atlas A3.

LoRA applies to both dense models and mixture-of-experts (MoE) models. The base model stays frozen. Each request selects an adapter by the `name` registered at startup. The current MoE support status is as follows:

| MoE mode | Tensor parallel (AllGather) | Expert parallelism (All-to-All) |
| --- | --- | --- |
| Non-quantized | Supported | Supported |
| W8A8 dynamic quantization | Supported | Supported |
| W4A8 dynamic quantization (per-channel) | Supported | Supported (AllGather in graph mode, All-to-All in eager mode) |

Other MoE quantization methods (including per-group W4A8 and weights with antiquant offsets), Fused MC2, and dynamic EPLB are not supported with LoRA.

## LoRA Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--enable-lora` | flag | off | Enables LoRA and loads the adapters given by `--lora-modules`. |
| `--lora-modules` | JSON | none | Registers an adapter. Fields are `name`, `path`, and `base_model_name`. A MoE adapter may also set `is_3d_lora_weight`. |
| `--max-loras` | int | 1 | Maximum number of adapters active in one batch. |
| `--max-lora-rank` | int | 16 | Upper bound on the adapter rank `r`. Must be one of 8, 16, 32, 64, and must be greater than or equal to `r`. |
| `--fully-sharded-loras` | flag | off | Shards the full LoRA computation across tensor-parallel ranks. By default only part of that computation is sharded. |

### `--enable-lora`

`--enable-lora` turns LoRA serving on. Adapters listed in `--lora-modules` are loaded only when this flag is set. Runtime load and unload also require it.

### `--lora-modules`

`name` is the value the client sends as `model`. `path` is the PEFT adapter directory, which contains `adapter_config.json` and the weight file. `base_model_name` records the base checkpoint the adapter was trained on. The client does not send `base_model_name`.

For a fused PEFT MoE adapter, set `is_3d_lora_weight` to `true`. The expert LoRA weights are then one stacked tensor, such as `experts.gate_up_proj` and `experts.down_proj`. Per-expert keys use `false`. Non-MoE models ignore this field. Layout rules are in the upstream [LoRA guide](https://docs.vllm.ai/en/latest/features/lora/).

### `--max-loras` and `--max-lora-rank`

`--max-lora-rank` must be one of 8, 16, 32, or 64, and it must be at least the rank `r` in `adapter_config.json`. `--max-loras 1` keeps a single device slot. That is the setting used by the fast path in the next section.

### `--fully-sharded-loras`

Without this flag, tensor parallelism shards only part of the LoRA computation. `--fully-sharded-loras` shards the rest as well. Use it when the sequence is long, the rank is high, or the tensor-parallel size is large. It does not change the request `model` name. The checked launches in this page leave it off.

## Performance

**Single-slot fast path.** On Atlas A2 and Atlas A3, `--max-loras 1`, `--fully-sharded-loras` left off, BF16 weights, a rank in {8, 16, 32, 64}, and at most 128 tokens select two kernels. Shrink splits the long input dimension (split-K) so more compute units work on a small token count. Expand is fused: it reduces those partials, applies the adapter mask and scale, multiplies by B, and adds the delta into the base output in one launch. More than one active adapter, `--fully-sharded-loras`, or a rank outside that set stays on the general SGMV/BGMV path.

**Fully sharded LoRA.** Add `--fully-sharded-loras` when partial sharding is the limiter. This path does not use the fused single-slot kernels above.

## Usage Example

### Qwen3.5-27B

```bash
vllm serve /path/to/Qwen3.5-27B \
    ... \
    --enable-lora \
    --max-loras 1 \
    --max-lora-rank 8 \
    --lora-modules '{"name": "mix-lora", "path": "/path/to/Qwen3.5-27B-lora", "base_model_name": "Qwen/Qwen3.5-27B"}'
```

To use the fully sharded path, add `--fully-sharded-loras` on the same command.

### DeepSeek-V4.1

This path enables language-model LoRA projections (`wq_a`, `wq_b`, `wkv`, `wo_a`, `wo_b`, `gate_up_proj`, `down_proj`, and shared experts) on DeepSeek-V4.1:

- HF text-backbone `.attn.` names are mapped to `self_attn` and re-rooted under `language_model.`. Vision and aligner modules stay frozen and are excluded.
- LoRA with expert parallelism uses AllGather when ACL graph capture is enabled and All-to-All under `--enforce-eager`. MC2 and Fused MC2 are not supported.
- Active `wo_a` adapters require a single local output-projection group per rank (`n_local_groups == 1`), no fine-grained output-projection TP, and `--fully-sharded-loras` left off.
- W4A8 MoE LoRA requires per-channel weight scales without antiquant offsets or dynamic EPLB.
- Under `DP=1`, layers with no active expert slot keep the fused base kernel; under `EP` with `DP>1`, layers with loaded expert slots conservatively use the LoRA path because remote ranks may route adapter tokens.

Set `WEIGHTS_DIR` and `ADAPTER_ROOT` to compatible local artifacts before launching:

```bash
vllm serve "${WEIGHTS_DIR}" \
    --served-model-name deepseek-lora \
    --tensor-parallel-size 8 --enable-expert-parallel \
    --quantization ascend --enable-lora --max-loras 1 --max-lora-rank 8 \
    --lora-modules "adapter=${ADAPTER_ROOT}" \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_fused_mc2":0}'
```

For eager validation, replace `--compilation-config ...` with `--enforce-eager`. Verify base requests, a zero adapter, and a non-zero adapter with identical prompts and sampling parameters.

### Request

The two calls differ only in `model`. A base request uses the served model name and runs the frozen base weights only. A LoRA request uses the `name` from `--lora-modules` and adds that adapter on top of the same base. `base_model_name` inside `--lora-modules` is not a request name.

Base:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen3.5-27B",
        "messages": [
            {"role": "user", "content": "Introduce LoRA in one sentence."}
        ],
        "max_tokens": 64,
        "temperature": 0
    }'
```

LoRA:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "mix-lora",
        "messages": [
            {"role": "user", "content": "Introduce LoRA in one sentence."}
        ],
        "max_tokens": 64,
        "temperature": 0
    }'
```

`Qwen3.5-27B` is the served base name from this example. If the launch sets `--served-model-name`, use that string instead. A successful call returns HTTP 200 and a non-empty `choices[0].message.content`.

## Dynamic LoRA

The server can load and unload adapters after startup. Start it with `--enable-lora`, and set:

```bash
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
```

Load an adapter. `lora_name` is the `model` value used by later requests. `lora_path` is the adapter directory.

```bash
curl -X POST http://127.0.0.1:8000/v1/load_lora_adapter \
    -H "Content-Type: application/json" \
    -d '{"lora_name": "mix-lora", "lora_path": "/path/to/lora"}'
```

Unload that adapter:

```bash
curl -X POST http://127.0.0.1:8000/v1/unload_lora_adapter \
    -H "Content-Type: application/json" \
    -d '{"lora_name": "mix-lora"}'
```

A successful load returns `Success: LoRA adapter 'mix-lora' added successfully.` A successful unload returns `Success: LoRA adapter 'mix-lora' removed successfully.` After unload, requests with `"model": "mix-lora"` no longer select that adapter. The base model remains available.
