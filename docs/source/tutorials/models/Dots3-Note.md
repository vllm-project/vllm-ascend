# Dots3 Note

## 1 Introduction

Dots3 Note is a large language model based on the Dots3 Note MoE + MLA architecture, with audio and vision multimodal encoders. The text, image, and audio forms are not mutually exclusive and can be enabled simultaneously; when device memory is limited, it is recommended to enable the required forms on demand. The forms cannot be switched dynamically within the same process, and the service must be restarted when switching.

| Form | Input | Typical scenario | Modal configuration |
|---|---|---|---|
| text-only | Text | Long-context dialogue, text reasoning | `--language-model-only` |
| image | Text + images (≤ 7 per request) | MMMU and other vision tasks | `--limit-mm-per-prompt '{"image":7,"video":0,"audio":0}'` |
| audio | Text + audio | MMAR and other audio tasks | `--limit-mm-per-prompt '{"image":0,"video":0,"audio":1}'` |

If you only need to verify text capability, select text-only; select image for vision tasks such as MMMU; select audio for audio tasks such as MMAR. After a form is enabled, all subsequent startup and functional validation commands must use the same form.

This document is validated and written based on **vLLM-Ascend v0.22.1rc1** (with vLLM 0.22.1). Dots3 Note is fully supported in this version, and all **v0.22.1rc1 and later versions** can run stably.

## 2 Supported Features

Dots3 Note supports the following features on vLLM-Ascend:

| Feature | Support | Description |
|---|---|---|
| Model architecture | Dots3 Note MoE + MLA | MoE mixture of experts + multi-head latent attention |
| Multimodal | audio + vision | Audio / vision encoders, enabled per form (see §5.1) |
| MTP speculative decoding | ✅ (text-only / audio) | MTP3 + draft eager; disabled for image (Model Runner V1 limitation) |
| FusedMC2 | ✅ | Fused `dispatch_ffn_combine` / `mega_moe` operators for MoE (`--additional-config`) |
| Prefix caching | ✅ | `--enable-prefix-caching`, reuses KV for similar prompts |

> This chapter describes the validation scope and does not mean that all hardware and software version combinations are covered. When reproducing the results in this document, prefer the validation environment and component versions listed in Chapters 3–4.

## 3 Prerequisites

Before installation, complete the checks in this chapter on the host. Only after the hardware, drivers, firmware, and model weights are all ready should you proceed to Chapter 4.

### 3.1 Hardware Introduction

The validation environment of this document is as follows:

| Item | Specification |
|---|---|
| Server | Single-node Atlas A3 inference series (Atlas 800I A3, board model `IT22HMDA_4_S`) |
| NPU | Atlas A3 products (8 cards, each dual-die (2 chips)), 16 chips in total, corresponding to `/dev/davinci[0-15]` |
| Memory | 64 GB device memory per chip |
| Chip software version | SOC_VERSION = `ascend910_9391` (A3 series) |
| Host form | Single-node deployment (not multi-node) |

Run the following checks on the host:

```bash
npu-smi info
ls -l /dev/davinci{0..15} /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc
```

Pass criteria: `npu-smi info` recognizes 8 cards with 16 chips in total, and all chips report normal device memory status; the device files required for container startup all exist. If the number of devices or device files differs from this document, adjust the device mapping in Section 4.2 and the subsequent parallel parameters accordingly; the TP16 configuration cannot be applied directly.

### 3.2 Driver Version Requirements

The software environment validated in this document is as follows:

| Item | Version |
|---|---|
| NPU driver (npu-smi) | 26.0.rc1 |
| Firmware | 9.0.0.0.205 |
| CANN | 9.0.0 |
| torch-npu | 2.10.0 |

The driver and firmware are installed on the host and mounted into the container through paths such as `/usr/local/Ascend/driver/version.info` and `/usr/local/Ascend/driver/lib64/`. First run the following on the host:

```bash
npu-smi info
cat /usr/local/Ascend/driver/version.info
```

If `npu-smi info` output shows 8 cards with 16 chips in total and normal device memory, the host-side NPU is ready. After entering the container, run `npu-smi info` again to confirm that the devices and driver are also visible inside the container.

### 3.3 Model Weight

Ensure that the model weights are fully downloaded and confirm their visible path **inside the container**. All subsequent startup commands reuse the following variables:

```bash
MODEL_PATH=/path/to/dots3_note   # Model weight directory inside the container
SERVED_NAME=dots3_note       # Model name exposed by the service

test -d "$MODEL_PATH"
```

If the model weights reside on the host, mount the host directory into the container at container startup, and set `MODEL_PATH` to the in-container path after mounting. `SERVED_NAME` must be consistent with the model name in the client requests in Chapter 6; otherwise the requests return 404.

## 4 Installation

This chapter provides a single installation method: use the official prebuilt image directly.

### 4.1 Components

| Component | Version | Description |
|---|---|---|
| vLLM | 0.22.1 | Inference serving framework, provides OpenAI-compatible service (`vllm serve`) |
| vLLM-Ascend (vllm-ascend) | 0.22.1rc1 | Community-maintained Ascend NPU hardware plugin that connects the NPU to vLLM through the vLLM hardware pluggable interface, aligned with the vLLM version |
| torch-npu | 2.10.0 | PyTorch NPU operator library, paired with the torch version |
| CANN | 9.0.0 | Ascend software stack (development kit + operator packages), in-container path `/usr/local/Ascend/cann-9.0.0` |

> This document recommends the all-in-one image `quay.io/ascend/vllm-ascend:dots3-note-prev-a3-openeuler`. The image bundles vLLM, vLLM-Ascend, torch-npu, and CANN, so no separate installation is required.

### 4.2 Image Acquisition and Build

The runtime uses the official vLLM-Ascend all-in-one image, which already contains vLLM, vLLM-Ascend, torch-npu, CANN, and NNAL. Choose the image matching your hardware series and OS:

| Image | Hardware | OS |
|---|---|---|
| `quay.io/ascend/vllm-ascend:dots3-note-prev` | Atlas A2 | Ubuntu |
| `quay.io/ascend/vllm-ascend:dots3-note-prev-openeuler` | Atlas A2 | openEuler |
| `quay.io/ascend/vllm-ascend:dots3-note-prev-a3` | Atlas A3 | Ubuntu |
| `quay.io/ascend/vllm-ascend:dots3-note-prev-a3-openeuler` | Atlas A3 | openEuler |

**Use the official prebuilt image directly (recommended)**

The following commands are all executed on the host:

```bash
export IMAGE=quay.io/ascend/vllm-ascend:dots3-note-prev-a3-openeuler
docker pull "$IMAGE"
```

Before startup, set the actual path of the model weights on the host:

```bash
export HOST_MODEL_PATH=/path/to/dots3_note
test -d "$HOST_MODEL_PATH"
```

Atlas A3 requires mapping all 16 `davinci` devices and mounting the drivers. The command below has expanded all devices and can be copied directly; if the host directories differ, adjust the mount source paths first:

```bash
docker run -it --rm \
    --name vllm-ascend \
    --shm-size=1g \
    --net=host \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci8 \
    --device /dev/davinci9 \
    --device /dev/davinci10 \
    --device /dev/davinci11 \
    --device /dev/davinci12 \
    --device /dev/davinci13 \
    --device /dev/davinci14 \
    --device /dev/davinci15 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -v "$HOST_MODEL_PATH":/models/dots3_note:ro \
    "$IMAGE" bash
```

The default working directory of the container is `/workspace`, and vLLM and vLLM-Ascend are installed in development mode (`pip install -e`). After entering the container, set the paths and service name shared by subsequent commands:

```bash
export MODEL_PATH=/models/dots3_note
export SERVED_NAME=dots3_note
```

If you need to read local image, audio, or other media files later, add the corresponding read-only directory mounts at `docker run` time, and make sure Chapters 5 and 6 use in-container paths.

After entering the container, complete the following minimal checks before continuing with the deployment:

```bash
npu-smi info
python -c "import vllm, vllm_ascend, torch_npu; print(vllm.__version__)"
pip show vllm vllm-ascend torch-npu
test -d "$MODEL_PATH"
```

### 4.3 Environment Variables

Set the following environment variables inside the container before starting the service. They are common to all three deployment forms; the variable descriptions come from the official vLLM-Ascend documentation and common practice:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export LD_PRELOAD=/usr/lib64/libjemalloc.so.2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_OP_EXPANSION_MODE=AIV
```

| Variable | Description |
|---|---|
| `ASCEND_RT_VISIBLE_DEVICES` | Specifies the NPU device IDs visible to the process (all 16 chips visible) |
| `LD_PRELOAD` (jemalloc) | Preloads the jemalloc memory allocator to reduce memory fragmentation in multi-threaded scenarios |
| `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` | Enables the virtual memory feature to mitigate fragmentation caused by dynamic memory size adjustments during inference (official OOM troubleshooting recommendation) |
| `TASK_QUEUE_ENABLE=1` | Enables the NPU task queue to dispatch operators asynchronously |
| `CPU_AFFINITY_CONF=1` | Enables CPU core binding to reduce cross-NUMA access |
| `HCCL_OP_EXPANSION_MODE=AIV` | Expands collective communication operators to AIV cores to improve multi-card communication bandwidth (example configuration from official tutorials) |

After setting, first confirm the key variables and the preloaded library:

```bash
test -f /usr/lib64/libjemalloc.so.2
echo "$ASCEND_RT_VISIBLE_DEVICES"
```

If the jemalloc path does not exist in the selected image, install it first or change it to the actual path in the image; do not start the service with an invalid `LD_PRELOAD`.

## 5 Online Service Deployment {: #5-online-service-deployment }

This chapter starts one service form at a time. For the first deployment, it is recommended to start text-only first to complete the basic pipeline validation, then switch to image or audio according to the test objective.

### 5.1 Deployment Modes

The three deployment forms are compared as follows:

| Form | Input | Typical scenario | Modal configuration | Context | KV cache | MTP | Weight per worker |
|---|---|---|---|---|---|---|---|
| text-only | Text | Long-context dialogue, text reasoning | `--language-model-only` | 32768 | `--gpu-memory-utilization 0.92` | ✅ MTP3 | 36.26 GB |
| image | Text + images (≤ 7 per request) | MMMU and other vision tasks | `--limit-mm-per-prompt '{"image":7,"video":0,"audio":0}'` | 32768 | `--kv-cache-memory-bytes 2816M` | ❌ | ~48.8 GB (+12.5 GB) |
| audio | Text + audio | MMAR and other audio tasks | `--limit-mm-per-prompt '{"image":0,"video":0,"audio":1}'` | 4096 | `--kv-cache-memory-bytes 4G` | ✅ MTP3 | 37.88 GB (+1.62 GB) |

All three forms share 16 chips with TP16 + MoE expert parallelism (EP16). Before startup, confirm:

```bash
test -d "$MODEL_PATH"
npu-smi info
```

When switching forms, first press `Ctrl+C` in the service terminal to stop the process normally, then check whether there are residual workers. Only when the normal stop does not clean up completely, use the following commands in order:

```bash
pgrep -af 'VLLM::Worker|vllm serve'
pkill -TERM -f 'VLLM::Worker'

# Last resort when the process still cannot exit
pkill -9 -f VLLM::Worker
```

After confirming that the old process has exited and port 8000 is released, start another form to avoid residual processes occupying NPU memory.

> **Concurrency suggestion**: text-only / audio use `--max-num-seqs 4`; image can use `--max-num-seqs 16`. The 2816M KV for the image form is the upper limit allowed by the FusedMC2 computation peak under 16 concurrency (about 3.64 GiB per card); increasing KV to ≥3072M will cause OOM. If a larger KV capacity is needed, reduce the concurrency to 8.

### 5.2 text-only Online Deployment

Suitable for pure text dialogue, mathematical reasoning, and long-context scenarios. Inside the container, confirm that the environment variables in Section 4.3 have been set, then start:

```bash
vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --tensor-parallel-size 16 \
    --enable-expert-parallel \
    --speculative-config \
      '{"method":"mtp","num_speculative_tokens":3,"enforce_eager":true}' \
    --additional-config \
      '{"enable_fused_mc2":1}' \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --safetensors-load-strategy lazy \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --async-scheduling \
    --language-model-only \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 4 \
    --default-chat-template-kwargs '{"enable_thinking":false}' \
    --generation-config vllm \
    --compilation-config \
      '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[16],"max_cudagraph_capture_size":16}' \
    --port 8000
```

Keep this terminal running. After seeing the success log listed in Section 5.6, run the health check in another container terminal or host terminal, then proceed to Section 6.1.

### 5.3 image Online Deployment

The vision form disables MTP (Model Runner V1 limitation) and forces chunked vision prefill to avoid peak OOM. `MEDIA_DIR` must be an in-container path and must contain the test images in Section 6.2:

```bash
export MEDIA_DIR=/path/to/local-image-root   # Local image media root directory (whitelist)
test -d "$MEDIA_DIR"
```

Start inside the container:

```bash
vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --tensor-parallel-size 16 \
    --enable-expert-parallel \
    --additional-config \
      '{"enable_fused_mc2":1}' \
    --max-model-len 32768 \
    --kv-cache-memory-bytes 2816M \
    --safetensors-load-strategy lazy \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --async-scheduling \
    --allowed-local-media-path "$MEDIA_DIR" \
    --limit-mm-per-prompt '{"image":7,"video":0,"audio":0}' \
    --mm-processor-cache-gb 0 \
    --mm-encoder-tp-mode data \
    --skip-mm-profiling \
    --max-num-batched-tokens 1024 \
    --max-num-seqs 16 \
    --generation-config vllm \
    --compilation-config \
      '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[16]}' \
    --port 8000
```

Keep this terminal running. After passing the health check in Section 5.6, proceed to Section 6.2.

### 5.4 audio Online Deployment

`MEDIA_ROOT` must be an in-container path and cover the local audio directory that the service needs to read:

```bash
export MEDIA_ROOT=/path/to/local-audio-root   # Local audio media root directory (whitelist)
test -d "$MEDIA_ROOT"
```

Start inside the container:

```bash
vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --tensor-parallel-size 16 \
    --enable-expert-parallel \
    --speculative-config \
      '{"method":"mtp","num_speculative_tokens":3,"enforce_eager":true}' \
    --additional-config \
      '{"enable_fused_mc2":1}' \
    --max-model-len 4096 \
    --kv-cache-memory-bytes 4G \
    --safetensors-load-strategy lazy \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --async-scheduling \
    --limit-mm-per-prompt '{"image":0,"video":0,"audio":1}' \
    --allowed-local-media-path "$MEDIA_ROOT" \
    --mm-encoder-tp-mode data \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 4 \
    --default-chat-template-kwargs '{"enable_thinking":false}' \
    --generation-config vllm \
    --compilation-config \
      '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,16],"max_cudagraph_capture_size":16}' \
    --port 8000
```

Keep this terminal running. After passing the health check in Section 5.6, proceed to Section 6.3.

### 5.5 Parameter Description

The parameter classification and descriptions refer to the vLLM official [Engine Arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/) documentation (engine arguments of `vllm serve`) and the vLLM-Ascend documentation. The tables below cover all parameters used by the three forms; when troubleshooting, first check this section, then modify the startup command.

#### 5.5.1 Common Parameters (used by all three forms)

| Parameter | Value | Description |
|---|---|---|
| `--served-model-name` | `dots3_note` | The model name exposed by the API; client requests must use this name (the default `default` returns 404) |
| `--tensor-parallel-size` | 16 | Tensor parallel group size; model weights are sharded across 16 chips |
| `--enable-expert-parallel` | On | MoE layers use expert parallelism instead of tensor parallelism; each rank loads only the expert weights of its own EP shard |
| `--safetensors-load-strategy` | `lazy` | safetensors weight loading strategy: `lazy` means memory-mapped, on-demand loading from files, reducing peak startup memory |
| `--enable-prefix-caching` | On | Automatic prefix caching; similar prompts reuse the KV cache, improving throughput |
| `--enable-chunked-prefill` | On | Chunked prefill to prevent a single long request from occupying the whole batch, reducing TTFT fluctuation |
| `--async-scheduling` | On | Asynchronous scheduling, decoupling scheduling from inference, improving throughput |
| `--max-num-seqs` | 4 for text-only / audio; 16 for image | Maximum number of concurrent sequences per batch; should match the client concurrency and the memory budget of the corresponding form |
| `--generation-config` | `vllm` | Generation config source: `vllm` means not loading the model's own generation config and using vLLM defaults |
| `--compilation-config` | see §5.5.4 | Compilation / graph capture configuration (`VLLM_COMPILE`, `FULL_DECODE_ONLY`, capture sizes) |
| `--additional-config` | see §5.5.4 | vLLM-Ascend additional configuration (FusedMC2) |
| `--port` | 8000 | Service listening port |

#### 5.5.2 Modality-Related Parameters

| Parameter | text-only | image | audio | Description |
|---|---|---|---|---|
| `--language-model-only` | ✅ | — | — | Load only the language model without loading multimodal encoders |
| `--limit-mm-per-prompt` | — | `{"image":7,...}` | `{"audio":1,...}` | Upper limit of each modality input per request; when the corresponding modality is `0`, that encoder is not loaded (see §5.1). image uses 7 because up to 7 images per request are supported; for single-image scenarios, it can be tightened to 1 |
| `--max-model-len` | 32768 | 32768 | 4096 | Model context length (prompt + output) |
| `--max-num-batched-tokens` | 8192 | 1024 | 4096 | Maximum tokens per batch; image uses 1024 to chunk large-image prefill, avoiding FusedMC2 peak OOM |
| `--allowed-local-media-path` | — | `MEDIA_DIR` | `MEDIA_ROOT` | Directory whitelist that allows the API to read local media files (security-sensitive; enable only in trusted environments) |
| `--mm-processor-cache-gb` | — | 0 | — | Multimodal processor cache size (GB) |
| `--mm-encoder-tp-mode` | — | `data` | `data` | Tensor parallel mode of the multimodal encoder (data = replicate the encoder on each rank) |
| `--skip-mm-profiling` | — | ✅ | — | Skip multimodal memory profiling |
| `--speculative-config` | MTP3 | — | MTP3 | Speculative decoding configuration (MTP3 + draft eager); not set for the image form (Model Runner V1 vision limitation) |
| `--default-chat-template-kwargs` | `enable_thinking:false` | — | `enable_thinking:false` | Default chat template arguments, disabling thinking output |

#### 5.5.3 KV Cache Configuration

| Form | KV cache | Reason |
|---|---|---|
| text-only | `--gpu-memory-utilization 0.92` | Automatic profiling, validated |
| image | `--kv-cache-memory-bytes 2816M` | FusedMC2 peak needs about 3.64 GiB per card; too large a KV (3328M) causes OOM; combined with `--max-num-batched-tokens 1024` to chunk large-image prefill |
| audio | `--kv-cache-memory-bytes 4G` | Single-request budget is about 1,131 tokens (text 360 + audio 259 + output 512); 1G provides only 2,220 tokens, insufficient, while 4G provides 8,956 tokens (2.19x) |

> **Why image/audio use explicit `--kv-cache-memory-bytes` (measured)**: automatic profiling under v0.22.1 can allocate a large KV (audio gets 21,565 tokens at 0.92 and 25,686 tokens at 0.95), but the actual peak memory during graph capture exceeds the limit and causes the worker to exit, with logs showing `Worker proc VllmWorker-* died unexpectedly`. Explicit KV is more conservative than automatic profiling and reserves headroom for the about 3.76 GiB peak of graph capture, so it is the currently validated stable approach.
>
> If you insist on automatic profiling, you can set `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` to disable the ACL graph memory estimation and try again; this approach has not been validated on single modality and is not recommended as the default configuration for external deployment testing.

#### 5.5.4 Compilation and vLLM-Ascend Additional Configuration

| Parameter | Value | Description |
|---|---|---|
| `--compilation-config.mode` | `VLLM_COMPILE` | Compilation mode |
| `--compilation-config.cudagraph_mode` | `FULL_DECODE_ONLY` | ACL graph captures only the decode stage, reducing capture peak memory |
| `--compilation-config.cudagraph_capture_sizes` | text-only `[16]`; image `[16]`; audio `[4,8,16]` | Graph capture batch sizes; TP16 + sequence parallelism requires multiples of 16, and audio `[4,8]` are automatically removed at startup |
| `--compilation-config.max_cudagraph_capture_size` | 16 (text-only / audio) | Maximum graph capture batch size, `= max_num_seqs × (1 + num_speculative_tokens) = 4 × 4` |
| `--additional-config.enable_fused_mc2` | `1` | Enables FusedMC2 (fused `mega_moe` operators for MoE) |

### 5.6 Service Verification

Do not judge startup success only by process existence. First wait for model loading and graph capture to complete; the success log should contain:

```text
GPU KV cache size: <tokens>
Maximum concurrency for ... tokens per request: ...
Graph capturing finished ...
Application startup complete.
```

Then run the health check in another terminal:

```bash
curl -sf http://127.0.0.1:8000/v1/models
```

The returned result should contain the model name `dots3_note` and the `max_model_len` of the corresponding form. At the same time, confirm that the service terminal has no `worker died`, OOM, or continuous restart logs. Only when both conditions are met is the service considered ready, and then proceed to Chapter 6.

If the health check fails, troubleshoot in the following order: whether the service is still loading, whether port 8000 is listening, whether `--served-model-name` is correct, and whether the service terminal shows NPU OOM or abnormal worker exit.

## 6 Functional Verification

The goal of functional verification is to use a real request to confirm that the "client → OpenAI-compatible interface → model inference → response" pipeline is usable. Only execute the subsection corresponding to the current service form.

### 6.1 text-only

Execute in another terminal:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "dots3_note",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "temperature": 0,
        "max_tokens": 640
    }'
```

Pass criteria: the HTTP request succeeds, the `model` in the response matches the service name, and `choices[0].message.content` contains a valid text answer.

### 6.2 image

First confirm that the test image is within the `MEDIA_DIR` whitelist configured in Section 5.3, and replace the path below with the actual file:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "dots3_note",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "file:///path/to/image.png"}},
                {"type": "text", "text": "What is shown in this image?"}
            ]
        }],
        "temperature": 0,
        "max_tokens": 256
    }'
```

Pass criteria: the HTTP request succeeds, the response contains valid text related to the image content, and the service terminal shows no media path rejection, HTTP 400, or OOM.

> The local path must be within the `--allowed-local-media-path` whitelist; base64 data URLs are submitted over HTTP and do not require the local path whitelist.

### 6.3 audio

First confirm that `/path/to/audio.wav` exists. The following command relies on GNU coreutils that support `base64 -w0`:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d @<(printf '{
        "model":"dots3_note",
        "messages":[{
            "role":"user",
            "content":[
                {"type":"audio_url","audio_url":{"url":"data:audio/wav;base64,'; base64 -w0 /path/to/audio.wav; printf '"}},
                {"type":"text","text":"What does the speaker say?"}
            ]
        }],
        "temperature":0,
        "max_tokens":1024
    }')
```

Pass criteria: the HTTP request succeeds, the response contains valid text related to the audio content, and the service terminal shows no audio decoding, HTTP 400, or OOM errors.

## 7 Accuracy Evaluation (Reference)

> This chapter provides an accuracy evaluation guide for reference only. It covers the GSM8K dataset, which is used to evaluate text-only math reasoning. The evaluation commands in this chapter are for reference; the deployment and service startup procedures in Chapters 3-5 are the prerequisite for running them.

GSM8K is used to evaluate pure text mathematical reasoning. The `gsm8k` dataset is built into evalscope, downloaded automatically from ModelScope, and includes a 4-shot template and `\boxed{}` answer extraction. Before evaluating, confirm the text-only service in Section 5.2 is running, and the model name in the client requests must be `dots3_note` (the default model name `default` returns 404).

First install evalscope inside the container:

```bash
# Install inside the container (for domestic networks, use the Tsinghua mirror)
pip install evalscope -i https://pypi.tuna.tsinghua.edu.cn/simple

# Verify the installation
evalscope eval --help
python -c "import evalscope; print(evalscope.__version__)"
```

Then run a smoke test first, followed by the full evaluation:

```bash
# Smoke test (5 samples)
evalscope eval \
  --model dots3_note \
  --api-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets gsm8k \
  --generation-config '{"temperature": 0, "max_tokens": 2048}' \
  --eval-batch-size 4 \
  --limit 5

# Full evaluation (1319 samples, remove --limit)
evalscope eval \
  --model dots3_note \
  --api-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets gsm8k \
  --generation-config '{"temperature": 0, "max_tokens": 2048}' \
  --eval-batch-size 4
```

> The built-in dataset requires access to ModelScope. For domestic networks, set `export MODELSCOPE_API_URL=https://modelscope.cn` before running.

## 8 Performance Evaluation (Reference)

> This chapter provides a performance evaluation guide for reference only. The evaluation methods and commands in this chapter are for reference; the deployment and service startup procedures in Chapters 3-5 are the prerequisite for running them.

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### 8.2 Using vLLM Benchmark

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.
