# LoRA Adapters

## Feature Introduction

LoRA serves a frozen base model together with low-rank adapters. The client selects an adapter by setting `model` to the `name` registered in `--lora-modules`. The base weights stay loaded.

Argument semantics and adapter file layout follow the [vLLM LoRA guide](https://docs.vllm.ai/en/latest/features/lora/). In the launch command, `...` stands for the other flags of that deployment.

## Supported Hardware

LoRA on vLLM-Ascend is supported on Atlas 300I Duo, Atlas A2, and Atlas A3.
On Ascend 950, DeepSeek V4's FP8 `wo_a` projection accepts BF16 LoRA adapters
with the default `--max-lora-rank 16`.

LoRA applies to both dense models and mixture-of-experts (MoE) models. The base model stays frozen. Each request selects an adapter by the `name` registered at startup. The current MoE support status is as follows:

| MoE mode | Tensor parallel (AllGather) | Expert parallelism (All-to-All) |
| --- | --- | --- |
| Non-quantized | Supported | Supported |
| W8A8 dynamic quantization | Supported | Supported |

Other MoE quantization methods, Fused MC2, and EPLB are not supported with LoRA.

### DeepSeek V4 projections

DeepSeek V4 applies adapters to `compressor.wkv`, `compressor.wgate`,
`indexer.compressor.wkv`, `indexer.compressor.wgate`, and `wo_a`. Requests using
different adapters can share a batch with base-model requests. For `wo_a`,
LoRA-A is shared across attention groups and LoRA-B contains each group's
output rows. With `--fully-sharded-loras`, this projection retains a full
LoRA-A on each tensor-parallel rank because the groups consume different inputs.

Adapter files must use Ascend's `self_attn` module names. For example, the
supported projection names in layer 2 are:

| Projection | Adapter module name |
| --- | --- |
| Grouped output | `model.layers.2.self_attn.wo_a` |
| Main compressor KV | `model.layers.2.self_attn.compressor.wkv` |
| Main compressor gate | `model.layers.2.self_attn.compressor.wgate` |
| Indexer compressor KV | `model.layers.2.self_attn.indexer.compressor.wkv` |
| Indexer compressor gate | `model.layers.2.self_attn.indexer.compressor.wgate` |

Append `.lora_A.weight` or `.lora_B.weight` to these names for the stored
adapter tensor keys. The PEFT prefix `base_model.model.` is optional; for
example, `base_model.model.model.layers.2.self_attn.wo_a.lora_A.weight`.
Replace the layer index with the layer containing the trained projection.

The adapter loader does not automatically convert upstream DeepSeek V4
`.attn.` names to `.self_attn.`. Convert both A and B tensor keys before
loading such an adapter, preserving their values and shapes, and update any
corresponding full names or name patterns in `adapter_config.json`. Changing
only `target_modules` does not rename stored tensors: an `.attn.wo_a` key can
pass suffix validation but its weights will not reach the Ascend module.

The deployment filter `--lora-target-modules` accepts leaf names such as
`wo_a`, `wkv`, and `wgate`, or exact full module names from the table. Use a
full name to select one compressor projection; `compressor.wkv` alone does
not match. Full targets must use `self_attn` and omit the PEFT prefix and
`.lora_A.weight` / `.lora_B.weight` suffixes.

DeepSeek V4 LoRA requires `--prefill-context-parallel-size 1`,
`--decode-context-parallel-size 1`, and
`finegrained_tp_config.oproj_tensor_parallel_size=1` in `--additional-config`
(their defaults). Other values fail at initialization because these token
exchanges do not carry adapter mappings. Standard tensor parallelism is
independent of this restriction.

With `--enable-lora`, DeepSeek V4 automatically disables DSA multi-stream
overlap, even when `multistream_dsv4_dsa_overlap` is `true` in
`--additional-config`. This keeps concurrent adapter operations from interfering
with one another and can reduce throughput.

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

For a fused PEFT MoE adapter, set `is_3d_lora_weight` to `true`. The expert LoRA weights are then one stacked tensor, such as `experts.gate_up_proj` and `experts.down_proj`. Per-expert keys use `false`. Non-MoE models ignore this field. Layout rules are in the upstream [vLLM LoRA guide](https://docs.vllm.ai/en/latest/features/lora/).

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
