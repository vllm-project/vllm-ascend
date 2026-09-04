# DeepSeek-V4-Flash-Vision-Exp (Experimental)

## 1 Introduction

DeepSeek-V4-Flash-Vision-Exp is a multimodal mixture-of-experts model in the
DeepSeek-V4 family. It combines the DeepSeek-V4 language model with a vision
encoder and aligner, and accepts text, single-image, and multi-image requests
through the OpenAI-compatible chat API.

Support on vLLM Ascend is experimental and is available on the `main` branch
with the matching vLLM 0.27.x revision. This guide documents the validated
Ascend W8A8 configuration on one Atlas 800 A3 server. Currently, only colocated
deployment is supported: Prefill and Decode run in the same service. Do not use
Prefill-Decode disaggregation for this model.

## 2 Supported Features

Refer to the [Supported Models](../../user_guide/support_matrix/supported_models.md)
for the complete support matrix and the
[Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration.

The configuration in this guide has been validated with W8A8 weights,
TP4/DP4/EP16, automatic prefix caching, and `FULL_DECODE_ONLY` ACL Graph on one
Atlas 800 A3 server (128GB × 8 NPUs). BF16 serving, DSA context parallelism,
FlashComm1, and Prefill-Decode disaggregation are not covered by this guide.

## 3 Prerequisites

### 3.1 Model Weights and Hardware

| Model | Download | Hardware requirements |
| --- | --- | --- |
| DeepSeek-V4-Flash-Vision-Exp-w8a8-QuaRot | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4-Flash-Vision-Exp-w8a8-QuaRot) | One Atlas 800 A3 server (128GB × 8 NPUs) |
| DeepSeek-V4-Flash-Vision-Exp | [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp) | Original model weights; use the ModelScope W8A8 QuaRot checkpoint for the deployment in this guide |

The launch command below uses the public Ascend W8A8 QuaRot checkpoint from
ModelScope. The checkpoint includes the ModelSlim quantization description and
the multimodal `bias_vl` tensors required by vLLM Ascend.

Download or prepare the checkpoint in a directory of your choice and record
the absolute path. The examples use `<YOUR_MODEL_PATH>`; replace it with that
path, for example
`/data/weights/DeepSeek-V4-Flash-Vision-Exp-w8a8-QuaRot`.

### 3.2 Software

Use a vLLM Ascend `main` image built after DeepSeek-V4-Flash-Vision-Exp support
was merged, together with the vLLM revision recorded in
`.github/vllm-main-verified.commit`. Mixing other vLLM and vLLM Ascend
revisions is not supported.

## 4 Installation

This guide supports only the Atlas A3 series. Follow the
[prebuilt image installation guide](../../getting_started/installation.md#installation-prebuilt-image)
or build the `main` branch with `Dockerfile.a3`.

Expose all 16 logical devices of the 8 NPUs to the container and mount the model
directory at the same absolute path used by the serving command. Verify the
installed packages before starting the service:

```shell
python -m pip show vllm vllm-ascend
```

Both packages must be present, and their revisions must match the selected
vLLM Ascend `main` checkout.

## 5 Online Service Deployment

### 5.1 Single-Node Colocated Deployment

The validated topology uses four data-parallel engines. Each engine uses four
tensor-parallel ranks, while expert parallelism spans all 16 ranks.

```shell
# Replace <YOUR_MODEL_PATH> with the actual path recorded in Section 3.1.
export MODEL_PATH="<YOUR_MODEL_PATH>"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export HCCL_BUFFSIZE=1024

vllm serve "$MODEL_PATH" \
  --served-model-name dsv4-vision \
  --max-model-len 130000 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.9 \
  --data-parallel-size 4 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --tokenizer-mode deepseek_v4 \
  --quantization ascend \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":32}' \
  --block-size 32 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --port 8900
```

Key parameters:

- `--data-parallel-size 4` and `--tensor-parallel-size 4` consume all 16
  visible logical devices backed by the server's 8 NPUs.
  `--enable-expert-parallel` distributes MoE experts across the ranks.
- `--max-model-len 130000` limits the combined input and output length of one
  request. Reduce it if the service cannot allocate enough KV cache.
- `--max-num-seqs 32` is the maximum number of sequences scheduled by each DP
  engine. Reduce it first if runtime memory pressure is observed.
- `--block-size 32` is required by the validated DeepSeek-V4 prefix-cache and
  sparse-attention configuration.
- `FULL_DECODE_ONLY` captures decode execution while keeping Prefill outside
  the full decode graph.

Wait until all four DP engines finish loading weights and graph capture. A
successful startup includes output similar to:

```text
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

For general startup issues, refer to the [Public FAQ](../../faqs.md).

### 5.2 Prefill-Decode Disaggregation

Prefill-Decode disaggregation is not currently supported for
DeepSeek-V4-Flash-Vision-Exp. Use the colocated deployment in Section 5.1.

## 6 Functional Verification

Set `IMAGE_URL` to an HTTP(S) URL that is reachable from the serving container,
then send a multimodal chat-completions request:

```shell
export IMAGE_URL="<YOUR_IMAGE_URL>"

curl -sS -o response.json -w '%{http_code}\n' \
  http://127.0.0.1:8900/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"dsv4-vision\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"image_url\", \"image_url\": {\"url\": \"${IMAGE_URL}\"}},
        {\"type\": \"text\", \"text\": \"Describe this image.\"}
      ]
    }],
    \"temperature\": 0,
    \"max_tokens\": 256
  }"

jq -e '.choices[0].message.content | length > 0' response.json
```

Expected output:

```text
200
true
```

The generated description is stored in
`response.json` under `.choices[0].message.content`. To use local image files,
mount their parent directory into the container and add
`--allowed-local-media-path <DIRECTORY>` to the serving command.

## 7 Accuracy Evaluation

The merged implementation was validated on OCRBench V1 with 1,000 samples. It
completed without request errors and scored **826/1000 (82.6)** using the W8A8
checkpoint, concurrency 64, temperature 1.0, top-p 0.95, maximum output length
8192, seed 7, and thinking enabled with high reasoning effort.

For reproducing an accuracy evaluation, refer to
[Using AISBench](../../developer_guide/evaluation/using_ais_bench.md). Results
depend on the checkpoint, prompt template, decoding parameters, and dataset
version; record all of them when comparing runs.

## 8 Performance Evaluation

No production performance baseline is published for this experimental model.
Use the [AISBench performance evaluation guide](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation)
with the deployment in Section 5.1, and report image resolution, input/output
lengths, request concurrency, TTFT, TPOT, ITL, and throughput.

## 9 Performance Tuning

The values in Section 5.1 are the validated starting point, not globally
optimal settings. Tune `--max-num-seqs`, `--max-num-batched-tokens`, and
`--gpu-memory-utilization` together for the target image sizes and sequence
lengths. Keep the single-node A3 TP4/DP4/EP topology and `--block-size 32`
until an alternative configuration has been validated.

Refer to the
[performance tuning guide](../../developer_guide/performance_and_debug/optimization_and_tuning.md)
for general tuning methods.

## 10 Limitations

- Only single-node colocated serving on one Atlas 800 A3 server
  (128GB × 8 NPUs) is documented.
- Prefill-Decode disaggregation, DSA context parallelism, and FlashComm1 are
  not supported in the documented configuration.
- BF16 full-model validation and production performance qualification are not
  complete.
- With TP4 ACL Graph, graph capture batch sizes must be multiples of four.
- DSpark uses target-side multimodal Prefill followed by text-only speculative
  Decode; the draft model does not consume image embeddings or propose tokens
  inside image spans.
