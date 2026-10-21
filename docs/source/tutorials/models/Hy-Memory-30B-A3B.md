# Hy-Memory-30B-A3B

## 1 Introduction

[Hy-Memory-30B-A3B](https://github.com/Tencent-Hunyuan/Hy-Memory) is a compact
long-term-memory **writer** model (~3B active parameters) released by Tencent
Hunyuan. It executes the write path of external memory frameworks (Mem0, MemOS,
Hy-Memory, EverMemOS): given a conversation, it decides which facts to retain
and emits them as structured memory entries. The model is a fine-tune of the
Hy3 (Hunyuan V3) MoE backbone and uses the native `HYV3ForCausalLM`
architecture, which vLLM serves with its built-in implementation — no custom
code or `trust_remote_code` is required.

This guide describes how to run the model on **Atlas 300I Duo (310P3)** with
vLLM Ascend.

Key architecture facts (from `config.json`): 48 layers, hidden size 2048,
32 query heads / 4 KV heads (GQA, head_dim 128, per-head QK-Norm), 128 routed
experts with top-8 routing plus 1 shared expert (expert intermediate size 768),
sigmoid routing with correction bias and weight re-normalization, first layer
dense, vocabulary 120832. Total weights are ~60 GB, which **exceeds the 44 GB
HBM of a single 310P3 chip — tensor parallelism with TP≥2 is required**. One
Atlas 300I Duo card carries two chips, so TP=2 fits on a single card.

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md)
for the model's support status. The results below were verified on Atlas 300I
Duo (310P3) in FP16 with `vllm-ascend` **v0.27.1rc1**:

- Chat completion and the memory-extraction workflow from the model card
  (system prompt + `reasoning_effort=no_think`), including exact JSON memory
  output
- Chinese / English generation and basic reasoning
- TP=2 both within a single card (two chips, HCCS interconnect) and across two
  cards (PCIe); decode throughput is memory-bandwidth bound and identical in
  both topologies
- Chunked prefill and automatic prefix caching (engine defaults)

## 3 Prerequisites

### 3.1 Model Weight

| Weight Version | Download Links |
|----------------|----------------|
| `Hy-Memory-30B-A3B` | [ModelScope](https://modelscope.cn/models/Tencent-Hunyuan/Hy-Memory-30B-A3B) / [HuggingFace](https://huggingface.co/tencent/Hy-Memory-30B-A3B) |

The checkpoint ships BF16 weights (~60 GB, 51 safetensors shards). Atlas 300I
Duo does not support BF16, so pass `--dtype float16` when serving (weights are
cast on load; no config edit is needed).

>**Path description**: Download the model weights to a directory of your choice and record it. Ensure the model path in the subsequent deployment command matches this directory.

### 3.2 Docker Image

Use the standard 310P image, e.g. `quay.io/ascend/vllm-ascend:{vllm_ascend_version}-310p`.

>**Known issue — version requirement**: use `v0.27.1rc1-310p` or later. On the
>`v0.26.0rc2-310p` image the server starts and produces fluent text, but
>`hy_v3` output degrades to echoing the user input or looping on
>`<think></think>` (the model code switched from `FusedMoE` to
>`FusedMoEFactory` between 0.26 and 0.27, which resolves the correct 310P MoE
>implementation; on 0.26 the first token distribution is still healthy but the
>selected experts do not contribute correctly).

## 4 Installation

### 4.1 Docker Image Installation

=== "Atlas 300I DUO"

    ```shell
    export IMAGE=quay.io/ascend/vllm-ascend:v0.27.1rc1-310p
    docker run -d \
        --name vllm-ascend-hymem \
        --shm-size=32g \
        -e ASCEND_VISIBLE_DEVICES=14,15 \
        -v /data/models/Hy-Memory-30B-A3B:/model:ro \
        -p 8321:8321 \
        $IMAGE bash -c 'vllm serve /model \
            --served-model-name hymem \
            --dtype float16 \
            --tensor-parallel-size 2 \
            --max-model-len 4096 \
            --enforce-eager \
            --host 0.0.0.0 --port 8321'
    ```

`ASCEND_VISIBLE_DEVICES` selects the chips. On the 300I Duo each card exposes
two chips (two `/dev/davinciN` nodes), so `14,15` addresses the two chips of
**one physical card** (HCCS interconnect). Placing the two chips of different
cards also works (e.g. `9,15`); decode throughput is unchanged because decode
is HBM-bandwidth bound, while cross-card prefill loses some interconnect
bandwidth.

## 5 Deployment Verification

Once the server is up (`Application startup complete` in the container log):

=== "Chat"

    ```shell
    curl http://127.0.0.1:8321/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
          "model": "hymem",
          "messages": [{"role": "user", "content": "你好，请用一句话介绍你自己。"}],
          "max_tokens": 64, "temperature": 0
        }'
    ```

=== "Memory extraction (model-card workflow)"

    ```shell
    curl http://127.0.0.1:8321/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
          "model": "hymem",
          "messages": [
            {"role": "system", "content": "Extract explicit user facts and preferences from the conversation. Write each memory as a concise, self-contained statement. Do not infer or invent information. Return only a JSON object with a \"memories\" list of strings."},
            {"role": "user", "content": "Hi, Im Alex. I prefer window seats on flights and Im allergic to peanuts."},
            {"role": "assistant", "content": "Got it, Alex! Ill remember you prefer window seats and have a peanut allergy."},
            {"role": "user", "content": "Also, I usually work out in the mornings before work."}
          ],
          "max_tokens": 256, "temperature": 0,
          "chat_template_kwargs": {"reasoning_effort": "no_think"}
        }'
    ```

Expected output for the memory extraction call:

```json
{
  "memories": [
    "Alex prefers window seats on flights.",
    "Alex is allergic to peanuts.",
    "Alex usually works out in the mornings before work."
  ]
}
```

## 6 Benchmark

Validated on Atlas 300I Duo (310P3), FP16, vLLM Ascend v0.27.1rc1, TP=2,
`--enforce-eager` (greedy decoding, temperature 0):

| Metric | TP=2, single card (2 chips, HCCS) | TP=2, two cards (PCIe) |
|--------|-----------------------------------|------------------------|
| Decode, batch=1 (256 tok) | 4.6 tok/s | 4.6 tok/s |
| Decode, batch=8 (256 tok each) | 31.9 tok/s aggregate | - |
| Prefill (1227-token prompt) | ~678 tok/s | - |
| Correctness (math reasoning / memory extraction / zh-en chat) | PASS | PASS |

Decode is HBM-bandwidth bound (3B active parameters per token across 128
experts), so the chip interconnect (HCCS vs PCIe) does not affect decode
throughput; single-card placement is still preferred to keep cross-card PCIe
free during prefill.

`enforce-eager` disables ACL graph capture; W8A8 quantization (310P supports
W8A8 for other MoE models) is untested for this model and is the most
promising follow-up for decode speed-up.
