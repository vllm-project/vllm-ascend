# MiniMax-H3

## 1 Introduction

MiniMax-H3 is MiniMax's audio-visual generation model. It generates video together with synchronized audio from a text prompt (text-to-video-audio, `t2va`), and is served through vLLM's omni mode via the `/v1/videos/sync` endpoint instead of the usual chat/completions APIs.

This document shows the main verification steps for MiniMax-H3 on Ascend NPU, including environment preparation, single-node deployment on A3 and A2 series, and functional verification.

MiniMax-H3 is served from the dedicated `vllm-omni` image rather than the standard `vllm-ascend` image, because it requires the omni runtime, the VAE parallel decoder, and the video generation API server.

## 2 Supported Features

Please refer to the [Supported Features List](../../user_guide/support_matrix/supported_models.md) for the model support matrix.

Please refer to the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration information.

!!! note

    MiniMax-H3 runs in omni (generation) mode. Text-generation features such as chunked prefill, automatic prefix caching, speculative decoding, and prefill-decode disaggregation do not apply to this model.

## 3 Prerequisites

### 3.1 Model Weight

| Model | Hardware Requirement | Download |
| ----- | -------------------- | -------- |
| MiniMax-H3 | Atlas 800I A3 (64GB, 8 cards)<br>Atlas 800I A2 (64GB, 8 cards) | [Download](https://modelscope.cn/models/MiniMax/MiniMax-H3) |

Download the weights to a local directory and export it as `MODEL_ROOT`. The audio-visual generation weights used by the commands below live in the `FL2VA` subdirectory of the downloaded repository:

```bash
export MODEL_ROOT=/path/to/MiniMax-H3
ls "${MODEL_ROOT}/FL2VA"
```

## 4 Installation

### 4.1 Docker Image Installation

MiniMax-H3 uses the dedicated `vllm-omni` image. Pull the tag that matches your hardware.

=== "A3 series"

    **Docker Pull:**

    ```bash
    # Please replace the tag with the actual published version for your environment.
    export IMAGE=quay.io/ascend/vllm-omni:minimax-h3-a3
    docker pull $IMAGE
    ```

    **Docker Run:**

    ```bash
    docker run \
        --name vllm-omni-env \
        --shm-size=128g \
        --ipc host \
        --net host \
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
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /usr/local/sbin:/usr/local/sbin \
        -v /path/to/MiniMax-H3:/workspace/MiniMax-H3 \
        -it -d $IMAGE bash
    ```

    !!! note

        A3 has 8 NPUs with a dual-die design (16 chips total: `/dev/davinci[0-15]`). If you are on a shared machine, map only the chips you need.

=== "A2 series"

    **Docker Pull:**

    ```bash
    # Please replace the tag with the actual published version for your environment.
    export IMAGE=quay.io/ascend/vllm-omni:minimax-h3
    docker pull $IMAGE
    ```

    **Docker Run:**

    ```bash
    docker run \
        --name vllm-omni-env \
        --shm-size=128g \
        --ipc host \
        --net host \
        --device /dev/davinci0 \
        --device /dev/davinci1 \
        --device /dev/davinci2 \
        --device /dev/davinci3 \
        --device /dev/davinci4 \
        --device /dev/davinci5 \
        --device /dev/davinci6 \
        --device /dev/davinci7 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /usr/local/sbin:/usr/local/sbin \
        -v /path/to/MiniMax-H3:/workspace/MiniMax-H3 \
        -it -d $IMAGE bash
    ```

!!! tip

    The mounts above are the minimum required for NPU driver access plus the model weight directory. Add additional `-v` mounts as needed for your environment.

**Installation Verification:**

```bash
docker ps | grep vllm-omni-env
```

Expected result: the container is listed with status `Up`.

## 5 Online Service Deployment {: #5-online-service-deployment }

!!! note

    Replace `${MODEL_ROOT}` with the actual path of your downloaded weights inside the container.

### 5.1 Single-Node Online Deployment

Video generation is a long-running request. `VLLM_OMNI_VIDEO_SYNC_TIMEOUT` controls how long the server waits for a synchronous `/v1/videos/sync` request to finish, in seconds. It must be set to a value large enough for your resolution, duration, and step count. The value below (1800 seconds) is validated for 1280x720, 8-second, 60-step generation.

=== "A3 series"

    ```bash
    export MODEL_ROOT=/workspace/MiniMax-H3
    export PORT=8091
    export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800

    vllm serve "${MODEL_ROOT}/FL2VA" \
        --omni \
        --host 0.0.0.0 \
        --port "${PORT}" \
        --trust-remote-code \
        --num-gpus 8 \
        --tensor-parallel-size 4 \
        --usp 2 \
        --ring 1 \
        --enable-layerwise-offload \
        --text-encoder-tp-size 8 \
        --vae-patch-parallel-size 8 \
        --vae-parallel-mode tile \
        --vae-use-tiling
    ```

=== "A2 series"

    ```bash
    export MODEL_ROOT=/workspace/MiniMax-H3
    export PORT=8091
    export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800

    vllm serve "${MODEL_ROOT}/FL2VA" \
        --omni \
        --host 0.0.0.0 \
        --port "${PORT}" \
        --trust-remote-code \
        --num-gpus 8 \
        --tensor-parallel-size 8 \
        --usp 1 \
        --ring 1 \
        --enable-layerwise-offload \
        --vae-patch-parallel-size 8 \
        --vae-parallel-mode tile \
        --vae-use-tiling
    ```

Key parameter descriptions:

| Parameter | Description |
| --------- | ----------- |
| `--omni` | Enables omni (audio-visual generation) mode and the `/v1/videos/*` endpoints |
| `--num-gpus` | Total number of NPUs used by the instance |
| `--tensor-parallel-size` | Tensor parallel size for the diffusion transformer |
| `--usp` | Unified sequence parallel (Ulysses) degree; must be consistent with `--ring`, `--tensor-parallel-size`, and `--num-gpus` |
| `--ring` | Ring attention parallel degree |
| `--enable-layerwise-offload` | Offloads weights layer by layer to reduce peak NPU memory |
| `--text-encoder-tp-size` | Tensor parallel size for the text encoder |
| `--vae-patch-parallel-size` | Number of NPUs used for parallel VAE decoding |
| `--vae-parallel-mode` | VAE parallel strategy; `tile` splits the latent into tiles |
| `--vae-use-tiling` | Enables tiled VAE decoding to lower peak memory at high resolution |

**Common Issues Tip:** If you encounter OOM or startup issues, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html) for troubleshooting. For MiniMax-H3 specific issues, refer to [Chapter 10 FAQ](#10-faq).

**Service Verification:**

```bash
curl http://127.0.0.1:8091/health
```

Expected result: HTTP 200 with an empty body.

## 6 Functional Verification

Once the server is started, submit a text-to-video-audio (`t2va`) request. The request is synchronous and returns the generated MP4 file directly, so `--max-time` must be at least as large as `VLLM_OMNI_VIDEO_SYNC_TIMEOUT`.

```bash
curl -sS --max-time 1800 -X POST "http://127.0.0.1:8091/v1/videos/sync" \
  -F 'prompt=傍晚小厨房的真人实拍的手 与手绘发光2d动画 融合在一起的影像。夕阳余晖残留在窗边，生活感十足的小厨房里有旧木桌、洗到一半的马克杯、起雾的玻璃瓶、悬挂的抹布。画面带有智能手机单手拍摄的手抖、近距离对焦的犹豫、逆光曝光波动。要像在家中慌忙拍下某个不可思议事件的自然质感，不要广告影像的精心整理。声音只用厨房环境声与手绘生物柔和的电子音、小小的叫声。' \
  -F 'width=1280' -F 'height=720' -F 'fps=24' \
  -F 'num_inference_steps=60' -F 'flow_shift=12' -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":8,"audio_flow_shift":3.0}' \
  -o "out_t2va.mp4"
```

Request field descriptions:

| Field | Description |
| ----- | ----------- |
| `prompt` | Text description of the video and audio to generate |
| `width` / `height` | Output resolution in pixels |
| `fps` | Output frame rate |
| `num_inference_steps` | Number of denoising steps; higher values improve quality and increase latency |
| `flow_shift` | Flow matching timestep shift for the video branch |
| `seed` | Random seed; fix it to reproduce a result |
| `extra_params.task` | `t2va` generates video with synchronized audio |
| `extra_params.duration` | Video length in seconds |
| `extra_params.audio_flow_shift` | Flow matching timestep shift for the audio branch |

Expected result: HTTP 200, and `out_t2va.mp4` is written to the current directory as a playable 8-second 1280x720 MP4 containing both the video track and the generated audio track. Verify it with:

```bash
ls -lh out_t2va.mp4
ffprobe out_t2va.mp4
```

Expected result: `ffprobe` reports one video stream (1280x720, 24 fps) and one audio stream.

## 7 Accuracy Evaluation

MiniMax-H3 is an audio-visual generation model, so token-level accuracy harnesses such as AISBench and lm_eval do not apply. Assess output quality by fixing `seed`, `num_inference_steps`, and `flow_shift`, then comparing the generated MP4 against a reference sample for prompt adherence, temporal consistency, and audio-video synchronization.

## 8 Performance Evaluation

Measure the wall-clock latency of a single synchronous generation request, keeping resolution, duration, `fps`, and `num_inference_steps` fixed across runs:

```bash
time curl -sS --max-time 1800 -X POST "http://127.0.0.1:8091/v1/videos/sync" \
  -F 'prompt=A cat walking on a wooden table at sunset.' \
  -F 'width=1280' -F 'height=720' -F 'fps=24' \
  -F 'num_inference_steps=60' -F 'flow_shift=12' -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":8,"audio_flow_shift":3.0}' \
  -o "bench_t2va.mp4"
```

## 9 Performance Tuning

> **Note**: The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on resolution, duration, step count, and the number of available NPUs.

### 9.1 Recommended Configurations

| Hardware | Total NPUs | Tensor Parallel | USP | Ring | Text Encoder TP | VAE Patch Parallel | Layerwise Offload |
| -------- | ---------- | --------------- | --- | ---- | --------------- | ------------------ | ----------------- |
| Atlas 800I A3 | 8 | 4 | 2 | 1 | 8 | 8 | On |
| Atlas 800I A2 | 8 | 8 | 1 | 1 | - | 8 | On |

### 9.2 Tuning Guidelines

- Reduce `num_inference_steps` to trade quality for latency. It is the dominant latency factor.
- Lower `width`, `height`, or `extra_params.duration` if you hit NPU memory limits.
- Keep `--vae-use-tiling` and `--vae-parallel-mode tile` enabled at 720p and above; tiled VAE decoding is what keeps peak memory within budget.
- `--enable-layerwise-offload` trades some latency for a large reduction in peak memory. Disable it only if you have memory headroom and need lower latency.
- On A3, raising `--usp` moves parallelism from tensor parallel to sequence parallel, which usually helps for longer videos. Ensure the parallel degrees stay consistent with `--num-gpus`.

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for general tuning methods.

## 10 FAQ

For common environment, installation, and general parameter issues, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html). This chapter only covers MiniMax-H3 specific issues.

- **Q: The request fails with a timeout even though the server is still generating.**

    A: Increase `VLLM_OMNI_VIDEO_SYNC_TIMEOUT` before starting the server and pass a `--max-time` on the client that is at least as large. Long durations, high resolutions, and large `num_inference_steps` all increase generation time.

- **Q: Why does `/v1/chat/completions` return 404?**

    A: MiniMax-H3 runs in omni generation mode and only exposes the video generation endpoints such as `/v1/videos/sync`. There is no chat/completions API for this model.

- **Q: The server runs out of memory during VAE decoding at 1280x720.**

    A: Confirm that `--vae-use-tiling` and `--vae-parallel-mode tile` are set and that `--vae-patch-parallel-size` matches the number of available NPUs. If it still runs out of memory, lower the output resolution or duration.

- **Q: Startup fails with a mismatch between `--num-gpus` and the parallel degrees.**

    A: `--tensor-parallel-size`, `--usp`, and `--ring` must be consistent with `--num-gpus`. Use the validated combinations in Section 9.1 as a starting point.
