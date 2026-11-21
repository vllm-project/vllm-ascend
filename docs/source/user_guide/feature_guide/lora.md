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

### DeepSeek-V4 / DeepSeek-V4.1

This path enables language-model LoRA projections (`wq_a`, `wq_b`, `wkv`, `wo_a`, `wo_b`, `gate_up_proj`, `down_proj`, and shared experts) on DeepSeek-V4 and DeepSeek-V4.1:

- HF text-backbone `.attn.`, `.ffn.`, `.w1.`, `.w2.`, and `.w3.` names map onto `self_attn`, `mlp`, `gate_proj`, `down_proj`, and `up_proj` (and re-root under `language_model.` for DeepSeek-V4.1). Vision and aligner modules stay frozen and are excluded.
- Unmatched adapter keys emit a `WARNING` when the adapter is loaded so typos or mismatched 2D/3D MoE key layouts are surfaced immediately instead of failing silently.
- **MoE adapter formats**:
    - **2D per-expert keys (default)**: e.g. `model.layers.<L>.mlp.experts.<E>.{gate_proj,up_proj,down_proj}.lora_{A,B}.weight` or HF-spelled `model.layers.<L>.ffn.experts.<E>.{w1,w2,w3}.lora_{A,B}.weight`.
    - **3D fused expert keys**: e.g. `model.layers.<L>.mlp.experts.base_layer.lora_{A,B}.weight` and `model.layers.<L>.mlp.experts.lora_{A,B}.weight`. Because DeepSeek-V4 / DeepSeek-V4.1 use 2D routed expert weights, loading a 3D fused MoE adapter requires launching the server with `--enable-mixed-moe-lora-format` and setting `"is_3d_lora_weight": true` in `--lora-modules` (or the dynamic load request).
- **Graph capture & parallel mode**:
    - LoRA with expert parallelism uses AllGather when ACL graph capture is enabled and All-to-All under `--enforce-eager`. MC2 and Fused MC2 are not supported.
    - Under `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`, ACL graph decode replay is **partially enabled**: decoder-step ACL graph replay is active, while Ascend decoder bounded replay is disabled because LoRA token mappings must follow the active replay batch.
- **Quantization & constraints**:
    - W4A8 MoE LoRA requires per-channel weight scales without antiquant offsets or dynamic EPLB. On 8-way TP, W4A8 MoE checkpoints require `--enable-expert-parallel` so the grouped-matmul K dimension satisfies the kernel alignment constraint.
    - Active `wo_a` adapters require a single local output-projection group per rank (`n_local_groups == 1`), no fine-grained output-projection TP, and `--fully-sharded-loras` left off.
    - Under `DP=1`, layers with no active expert slot skip MoE LoRA delta kernels; under `EP` with `DP>1`, layers with loaded expert slots conservatively run the MoE LoRA path because remote ranks may route adapter tokens. To preserve zero-delta numerical equivalence and dispatch/combine dtype consistency between base and adapter requests in a LoRA-enabled engine, shared experts and LoRA-capable quantized MoE stay on the linear-wrapper / floating-point unfused quantization boundary path whenever an adapter is loaded on those layers.
- **Latency & throughput characteristics**:
    - Enabling `--enable-lora` routes wrapped base linear layers (including shared experts) through unfused LoRA wrappers instead of fused base kernels, adding a fixed structural decode overhead (~19–23% measured on Atlas A3 W4A8 for both attention-only and MoE adapters, independent of adapter weights).
    - Always compare base vs adapter latency inside the **same LoRA-enabled engine** (rather than against a separate non-LoRA engine, which runs different base/shared-expert fused kernels) and report exact decode conditions (output length, batch size, and graph/bounded-replay mode).

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
