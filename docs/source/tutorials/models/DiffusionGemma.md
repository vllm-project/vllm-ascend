# DiffusionGemma

DiffusionGemma denoises a canvas of tokens in parallel. vLLM Ascend uses the
upstream model, scheduler and sampler and adapts attention metadata and
ACLGraph updates for Ascend.

## Supported features

The BF16 `google/diffusiongemma-26B-A4B-it` checkpoint was validated with
tensor parallel size 1 on Ascend 950DT.

| Feature | Validated configuration |
| --- | --- |
| Text and image generation | Eager OpenAI-compatible serving |
| Chunked prefill and prefix caching | Eager text, cache enabled and disabled |
| ACLGraph | FULL and PIECEWISE replay with async scheduling and up to four active sequences |
| Structured decisions | Upstream server, text and image input |
| Context length tested | 2,048 for serving; 32,768 configured for the long-context JevBench cases |

## Deployment

Use a matching vLLM Ascend image with the Ascend custom operators available.
The following recipe serves text requests and uses the graph configuration
validated on a single 950DT.

```bash
vllm serve google/diffusiongemma-26B-A4B-it \
    --served-model-name dgemma --dtype bfloat16 --language-model-only \
    --max-model-len 2048 --max-num-seqs 4 --max-num-batched-tokens 256 \
    --async-scheduling --enable-chunked-prefill \
    --diffusion-config '{"canvas_length":64}' \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[64,128,256]}'
```

Omit `--language-model-only` for image input. For eager validation, replace
`--compilation-config` with `--enforce-eager`. Use the default generation
parameters from the model because DiffusionGemma pins its temperature schedule.

## Functional verification

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"dgemma","messages":[{"role":"user","content":"Name the capital of France."}],"max_tokens":128}'
```

For structured decisions, launch the
[upstream structured diffusion server](https://github.com/vllm-project/vllm/tree/main/examples/features/structured_diffusion)
from the vLLM source directory. This server accepts `/v1/systemone`.

```bash
python examples/features/structured_diffusion/structured_server.py \
    --upstream http://127.0.0.1:8000 --model dgemma \
    --tokenizer google/diffusiongemma-26B-A4B-it --canvas 64 \
    --host 127.0.0.1 --port 8011
```

```bash
curl -s http://127.0.0.1:8011/v1/systemone \
    -H 'Content-Type: application/json' \
    -d '{"state":"I cannot sign in.","questions":{"access":{"type":"noul","instructions":"Is the user blocked?"}}}'
```

## Evaluation

The single-card validation covered System One Bench `ag_news` and
`typed-decisions` with 2,200 decisions and JevBench with 231 public cases.
To measure serving throughput, drive the generation and decision endpoints
separately and keep the concurrency, scheduling, cache and canvas settings
fixed when comparing eager and graph modes.
