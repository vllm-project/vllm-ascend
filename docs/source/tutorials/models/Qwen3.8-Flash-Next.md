# Qwen3.8-Flash-Next

## 1 Introduction

!!! note

    The "PD Separation Deployment" mentioned in this document is provided only as an example/reference design to demonstrate the basic process of deploying Prefill and Decode in separated mode. It is not recommended for use in the production environment.

    **For the production environment on the customer side, [MindIE Motor](https://www.hiascend.com/document/detail/en/mindie/latest/quickstart/docs/en/user_guide/quick_start.md) + vLLM-Ascend is recommended.**

Qwen3.8-Flash-Next is a multimodal Mixture-of-Experts (MoE) model and an experimental preview of the architecture that will underpin Qwen4. Its language model combines Gated DeltaNet (GDN) and Qwen Sparse Attention (QSA), gated residual connections, Position Learning Enhancement (PLE), and a native Multi-Token Prediction (MTP) head.

This tutorial covers model preparation, installation, colocated and prefill/decode (PD) deployment, functional verification, and evaluation on A2 series, A3 series, and Ascend 950 DT. Section 5.4 describes basic PD deployment with a MemCache three-tier KV cache pool on A3 series and Ascend 950 DT.

Qwen3.8-Flash-Next (experimental) was first supported in `vllm-ascend:v0.26.0rc`. The original A3 series W8A8 configuration was validated with text and multimodal input. The deployment examples below use W4A8 checkpoints (W4A8-MXFP on Ascend 950 DT) and additional performance settings; they require separate runtime and accuracy validation. For the best performance, it is recommended to use **vLLM-Ascend v0.32.0rc** (`releases/v0.32.0rc`).

!!! warning

    Automatic prefix caching is experimental. For A2 series and A3 series colocated deployment, enable it with `--enable-prefix-caching --mamba-cache-mode align` and revalidate NPU memory capacity for the target context length and concurrency.

## 2 Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_features.md) to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get feature configuration details.

## 3 Prerequisites

### 3.1 Model Weight

| Weight Version | Hardware Requirements | Download Links |
| --- | --- | --- |
| `Qwen3.8-Flash-Next-w8a8-mtp` (quantized version) | A3 series | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/Qwen3.8-Flash-Next-w8a8-mtp) |
| `Qwen3.8-Flash-Next-w4a8` (quantized version) | A2 series / A3 series | |
| `Qwen3.8-Flash-Next-w4a8-mxfp` (quantized version) | Ascend 950 DT | |

The examples use ModelScope model IDs with `VLLM_USE_MODELSCOPE=True`. Both PD roles must load the same checkpoint.

## 4 Installation

### 4.1 Docker Image Installation

Refer to [using docker](../../getting_started/installation.md#installation-prebuilt-image) for the complete installation guide.

Use the v0.32.0rc image for the target hardware.

=== "A2 series"

    ```bash
    # Reuse the host model cache at /root/.cache.
    export IMAGE=quay.io/ascend/vllm-ascend:v0.32.0rc
    export NAME=vllm-ascend

    docker run --rm \
      --name $NAME \
      --net=host \
      --shm-size=128g \
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
      -v /usr/local/dcmi:/usr/local/dcmi \
      -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
      -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
      -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
      -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
      -v /etc/ascend_install.info:/etc/ascend_install.info \
      -v /etc/hccn.conf:/etc/hccn.conf:ro \
      -v /root/.cache:/root/.cache \
      -it $IMAGE bash
    ```

=== "A3 series"

    ```bash
    # Reuse the host model cache at /root/.cache.
    export IMAGE=quay.io/ascend/vllm-ascend:v0.32.0rc-a3
    export NAME=vllm-ascend

    docker run --rm \
      --name $NAME \
      --net=host \
      --shm-size=128g \
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
      -v /etc/hccn.conf:/etc/hccn.conf:ro \
      -v /root/.cache:/root/.cache \
      -it $IMAGE bash
    ```

=== "Ascend 950 DT"

    ```bash
    # Reuse the host model cache at /root/.cache.
    export IMAGE=quay.io/ascend/vllm-ascend:v0.32.0rc-a5
    export NAME=vllm-ascend

    docker run --rm \
      --name $NAME \
      --net=host \
      --shm-size=500g \
      --device /dev/davinci_manager \
      --device /dev/hisi_hdc \
      --device /dev/ummu \
      --device /dev/uburma \
      --device /dev/davinci0 \
      --device /dev/davinci1 \
      --device /dev/davinci2 \
      --device /dev/davinci3 \
      --device /dev/davinci4 \
      --device /dev/davinci5 \
      --device /dev/davinci6 \
      --device /dev/davinci7 \
      -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
      -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
      -v /root/host:/root/host \
      -v /usr/local/sbin:/usr/local/sbin \
      -v /usr/local/dcmi:/usr/local/dcmi \
      -v /var/log/npu:/usr/slog \
      -v /etc/ascend_install.info:/etc/ascend_install.info \
      -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
      -v /etc/hixlep:/etc/hixlep \
      -v /root/.cache:/root/.cache \
      -it $IMAGE bash
    ```

Ensure the CANN and ATB environments are initialized in the serving shell. If the image does not initialize them automatically, source the corresponding `set_env.sh` scripts from their installation directories. Then verify that vLLM and vLLM-Ascend can be imported:

```bash
python -c "import vllm, vllm_ascend; print('vllm and vllm-ascend are ready')"
```

### 4.2 Source Code Installation

For source installation on A2 series, A3 series, or Ascend 950 DT, follow the [installation guide](../../getting_started/installation.md#installation-existing-cann-install).

Initialize the CANN and ATB environments in the serving shell as described in the installation guide. For multi-node deployment, set up the environment on each node.

### 4.3 Docker for Three-Tier PD Pooling

Use the following containers for the A3 series and Ascend 950 DT basic PD pooling configuration in [Section 5.4](#54-pd-separation-with-memcache-three-tier-kv-cache-pool). For colocated, ordinary PD, and image-and-text 10K deployment, use Section 4.1. The commands below retain `--privileged=true` from the [MemCache Docker example](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#step-1-prerequisite-checks); this tutorial has not validated a configuration with reduced container privileges.

Before creating the container, prepare the dedicated SSD or loop device using the storage preparation steps in Section 5.4. Replace `<SSD_DEVICE_PATH>` with its actual host path in both sides of the device mapping. For multiple devices, add one `--device` entry per device and join their container paths with colons in the storage configuration file, for example `ubsio.disk.path = /dev/loop0:/dev/loop1` when those are the allocated devices. Create loop devices on the host, not inside the container.

Set `MEMCACHE_CONFIG_DIR` to the existing directory containing `mmc-local.conf` for the selected hardware. Docker mounts this directory at the same absolute path and passes the variable into the container. All configuration paths below use this variable. If starting new storage services with Section 5.4, prepare `mmc-meta.conf` in the same directory and, for A3 standalone storage, prepare `mmc-local-standalone.conf` there as well.

Verify that all host device and file paths exist before running Docker. The 950 DT example retains the device, driver, and communication-tool mounts from the [MemCache backend guide](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#scenario-2-memcache-backend).

Additional communication-file mounts depend on the installed stack. The [950 DT container requirements](../features/pd_disaggregation_mooncake_multi_node.md#atlas-850850e950-superpod-950pr950dt-container-requirements) describe when `/lib/route.conf` is needed; CANN 9.2.0 and later do not require it. If the target environment requires `/etc/hccl_rootinfo.json` or other communication configuration files, add the mounts specified by that environment. These files are not MemCache configuration files and are not assumed to exist in the commands below.

=== "A3 series"

    ```bash
    # Reuse the host model cache at /root/.cache.
    export IMAGE=quay.io/ascend/vllm-ascend:v0.32.0rc-a3
    export NAME=vllm-ascend-qwen38-pool
    export MEMCACHE_CONFIG_DIR="<EXISTING_MEMCACHE_CONFIG_DIR>"

    docker run --rm \
      --name $NAME \
      -e MEMCACHE_CONFIG_DIR \
      --net=host \
      --privileged=true \
      --shm-size=128g \
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
      --device "<SSD_DEVICE_PATH>:<SSD_DEVICE_PATH>" \
      --device /dev/devmm_svm \
      --device /dev/hisi_hdc \
      -v /usr/local/dcmi:/usr/local/dcmi \
      -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
      -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
      -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
      -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
      -v /etc/ascend_install.info:/etc/ascend_install.info \
      -v /etc/hccn.conf:/etc/hccn.conf:ro \
      -v /root/.cache:/root/.cache \
      -v "${MEMCACHE_CONFIG_DIR}:${MEMCACHE_CONFIG_DIR}" \
      -it $IMAGE bash
    ```

=== "Ascend 950 DT"

    ```bash
    # Reuse the host model cache at /root/.cache.
    export IMAGE=quay.io/ascend/vllm-ascend:v0.32.0rc-a5
    export NAME=vllm-ascend-qwen38-pool
    export MEMCACHE_CONFIG_DIR="<EXISTING_MEMCACHE_CONFIG_DIR>"

    docker run --rm \
      --name $NAME \
      -e MEMCACHE_CONFIG_DIR \
      --net=host \
      --privileged=true \
      --shm-size=500g \
      --device /dev/davinci_manager \
      --device "<SSD_DEVICE_PATH>:<SSD_DEVICE_PATH>" \
      --device /dev/hisi_hdc \
      --device /dev/ummu \
      --device /dev/uburma \
      --device /dev/davinci0 \
      --device /dev/davinci1 \
      --device /dev/davinci2 \
      --device /dev/davinci3 \
      --device /dev/davinci4 \
      --device /dev/davinci5 \
      --device /dev/davinci6 \
      --device /dev/davinci7 \
      -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
      -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
      -v /root/host:/root/host \
      -v /usr/local/sbin:/usr/local/sbin \
      -v /usr/local/dcmi:/usr/local/dcmi \
      -v /var/log/npu:/usr/slog \
      -v /etc/ascend_install.info:/etc/ascend_install.info \
      -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
      -v /etc/hixlep:/etc/hixlep \
      -v /etc/hccn.conf:/etc/hccn.conf:ro \
      -v /usr/bin/urma_admin:/usr/bin/urma_admin:ro \
      -v /root/.cache:/root/.cache \
      -v "${MEMCACHE_CONFIG_DIR}:${MEMCACHE_CONFIG_DIR}" \
      -it $IMAGE bash
    ```

Start MetaService first. For A3, start the standalone MemCache storage process as described in Section 5.4 and wait for successful initialization before starting Prefill on devices 0–7 and Decode on devices 8–15 in separate terminals. The standalone process stores KV cache and does not run model inference. For 950 DT, start the P/D engines after MetaService; their workers initialize the embedded LocalServices. The disk is reinitialized when the storage service starts with `force_new_disk=true`; use only storage dedicated to this pool.

After entering the container, check that the model and storage dependencies are available:

```bash
python -c "import vllm, vllm_ascend, mooncake, memcache_hybrid, memfabric_hybrid; print('Model and pool dependencies are available')"
```

An import check does not validate version compatibility; complete the MemCache dependency checks in Section 5.4. `--shm-size` controls the container's `/dev/shm` limit and does not allocate the MemCache DRAM pool or the UBS IO buffers.

## 5 Online Service Deployment {: #5-online-service-deployment }

### 5.1 Single-Node Online Deployment

Prefill and decode run in the same service. Select the hardware tab below.

Before starting:

- Ensure the serving environment can access ModelScope to download the checkpoint.
- Verify available NPU memory before changing context length, concurrency, or memory utilization.

=== "A2 series"

    This W4A8 deployment uses TP8 × DP1, a maximum context length of 131072 tokens, and up to 8 active sequences.

    ```bash
    export VLLM_USE_MODELSCOPE=True
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_ENGINE_READY_TIMEOUT_S=4800

    vllm serve Eco-Tech/Qwen3.8-Flash-Next-w4a8 \
        --host 0.0.0.0 \
        --port 8088 \
        --served-model-name qwen3.8-flash-next \
        --trust-remote-code \
        --quantization ascend \
        --tensor-parallel-size 8 \
        --data-parallel-size 1 \
        --enable-expert-parallel \
        --max-model-len 131072 \
        --max-num-seqs 8 \
        --max-num-batched-tokens 4096 \
        --gpu-memory-utilization 0.93 \
        --enable-prefix-caching \
        --mamba-cache-mode align \
        --async-scheduling \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_xml \
        --reasoning-parser qwen3 \
        --engram-config '{"cpu_offload":true,"dp_shared_memory":false}' \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' \
        --additional-config '{"enable_cpu_binding":true,"enable_shared_expert_dp":true,"enable_fused_mc2":0,"enable_flashcomm1":true,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"fuse_norm_quant":false,"enable_npugraph_ex":true,"enable_static_kernel":false,"enable_super_kernel":false}}'
    ```

    **Key parameters:**

    - `--quantization ascend` loads the W4A8 checkpoint; TP8 × DP1 runs one model instance across eight devices.
    - Model Runner V2, async scheduling, and expert parallelism are enabled. `FULL_DECODE_ONLY` captures decode graphs, and MTP drafts three tokens per step.
    - `--mamba-cache-mode align` aligns GDN and PLE state checkpoints for prefix caching.
    - `cpu_offload=true` offloads Engram embedding tables to host memory; `dp_shared_memory=false` disables sharing for this DP1 configuration. Use a checkpoint with supported n-gram layers and provide sufficient host memory.
    - `--additional-config` enables CPU binding, FlashComm1, shared expert DP, and shared expert overlap. Shared expert DP requires expert parallelism and TP > 1; `enable_fused_mc2=0` disables fused MC2 and is compatible with overlap. Norm/quantization fusion is disabled.
    - `--enable-auto-tool-choice --tool-call-parser qwen3_xml` enables function calling; `--reasoning-parser qwen3` separates reasoning from the final answer.
    - Multimodal requests require a checkpoint with a vision encoder. For text-only serving, add `--language-model-only` to skip loading it.
    - `enable_npugraph_ex=true` enables NPU graph extension; static kernels and super kernels remain disabled.

=== "A3 series"

    This W4A8 deployment uses TP8 × DP1, a maximum context length of 131072 tokens, and up to 8 active sequences.

    ```bash
    export VLLM_USE_MODELSCOPE=True
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_ENGINE_READY_TIMEOUT_S=4800

    vllm serve Eco-Tech/Qwen3.8-Flash-Next-w4a8 \
        --host 0.0.0.0 \
        --port 8088 \
        --served-model-name qwen3.8-flash-next \
        --trust-remote-code \
        --quantization ascend \
        --tensor-parallel-size 8 \
        --data-parallel-size 1 \
        --enable-expert-parallel \
        --max-model-len 131072 \
        --max-num-seqs 8 \
        --max-num-batched-tokens 4096 \
        --gpu-memory-utilization 0.93 \
        --enable-prefix-caching \
        --mamba-cache-mode align \
        --async-scheduling \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_xml \
        --reasoning-parser qwen3 \
        --engram-config '{"cpu_offload":true,"dp_shared_memory":false}' \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' \
        --additional-config '{"enable_cpu_binding":true,"enable_shared_expert_dp":true,"enable_fused_mc2":0,"enable_flashcomm1":true,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"fuse_norm_quant":false,"enable_npugraph_ex":true,"enable_static_kernel":true,"enable_super_kernel":false}}'
    ```

    **Key parameters:**

    - `--quantization ascend` loads the W4A8 checkpoint; TP8 × DP1 runs one model instance across eight devices.
    - Model Runner V2, async scheduling, and expert parallelism are enabled. `FULL_DECODE_ONLY` captures decode graphs, and MTP drafts three tokens per step.
    - `--mamba-cache-mode align` aligns GDN and PLE state checkpoints for prefix caching.
    - `cpu_offload=true` offloads Engram embedding tables to host memory; `dp_shared_memory=false` disables sharing for this DP1 configuration. Use a checkpoint with supported n-gram layers and provide sufficient host memory.
    - `--additional-config` enables CPU binding, FlashComm1, shared expert DP, and shared expert overlap. Shared expert DP requires expert parallelism and TP > 1; `enable_fused_mc2=0` disables fused MC2 and is compatible with overlap. Norm/quantization fusion is disabled.
    - `--enable-auto-tool-choice --tool-call-parser qwen3_xml` enables function calling; `--reasoning-parser qwen3` separates reasoning from the final answer.
    - Multimodal requests require a checkpoint with a vision encoder. For text-only serving, add `--language-model-only` to skip loading it.
    - `enable_npugraph_ex=true` and `enable_static_kernel=true` enable NPU graph extension and static kernels; super kernels remain disabled.

=== "Ascend 950 DT"

    This W4A8-MXFP deployment uses TP4 × DP1, a maximum context length of 141312 tokens, and up to 32 active sequences.

    ```bash
    export VLLM_USE_MODELSCOPE=True
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export VLLM_ENGINE_READY_TIMEOUT_S=4800
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export TASK_QUEUE_ENABLE=1
    export HCCL_BUFFSIZE=1024
    export HCCL_NPU_SOCKET_PORT_RANGE=auto

    vllm serve Eco-Tech/Qwen3.8-Flash-Next-w4a8-mxfp \
        --host 0.0.0.0 \
        --port 8087 \
        --served-model-name qwen3.8-flash-next-950-dt \
        --trust-remote-code \
        --quantization ascend \
        --tensor-parallel-size 4 \
        --data-parallel-size 1 \
        --max-model-len 141312 \
        --max-num-seqs 32 \
        --max-num-batched-tokens 4096 \
        --gpu-memory-utilization 0.90 \
        --enable-prefix-caching \
        --async-scheduling \
        --enable-expert-parallel \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_xml \
        --reasoning-parser qwen3 \
        --engram-config '{"cpu_offload":true,"dp_shared_memory":false}' \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --additional-config '{"enable_cpu_binding":true,"enable_shared_expert_dp":true,"enable_fused_mc2":0,"enable_flashcomm1":true,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"fuse_norm_quant":false,"enable_npugraph_ex":true,"enable_static_kernel":true,"enable_super_kernel":false}}' \
        --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'
    ```

    **Key parameters:**

    - `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3` selects the four devices for TP4.
    - `VLLM_USE_V2_MODEL_RUNNER=1` selects Model Runner V2.
    - `--quantization ascend` loads the Ascend-compatible W4A8-MXFP checkpoint.
    - `--max-model-len 141312` and `--max-num-seqs 32` set context and concurrency limits; verify memory capacity before increasing them.
    - `--async-scheduling` enables asynchronous scheduling; `--enable-expert-parallel` enables expert parallelism for MoE layers.
    - `--additional-config` enables CPU binding, FlashComm1, shared expert DP, and shared expert multistream overlap. Shared expert DP requires expert parallelism and TP > 1. `enable_fused_mc2=0` disables fused MC2 and is compatible with shared expert overlap.
    - NPU graph extension and static kernels are enabled; super kernels and norm/quantization fusion are disabled.
    - `FULL_DECODE_ONLY` enables decode graph capture. The MTP configuration uses three draft tokens.
    - Prefix caching is enabled. The current runtime automatically selects `mamba_cache_mode=align` when prefix caching is enabled and the mode is left at its default `none`; the A2 series and A3 series examples set the same mode explicitly. This is not a hardware-specific difference.
    - `--enable-auto-tool-choice --tool-call-parser qwen3_xml` enables Function Calling; `--reasoning-parser qwen3` separates reasoning from the final answer.

**Service Verification:**

Wait for model loading and graph capture to complete, then check the service from the serving host. Use port `8088` for A2 series and A3 series or `8087` for 950 DT:

```bash
curl --fail "http://127.0.0.1:8088/v1/models"
```

The response should list the configured model name. After startup, use [Section 6](#6-functional-verification) to verify generation.

### 5.2 Basic PD Separation Deployment

Install compatible vLLM and vLLM-Ascend, then follow the [Mooncake installation guide](../features/pd_disaggregation_mooncake_single_node.md#install-mooncake). Select the hardware and parallel configuration below. Replace the IP-address and network-interface placeholders with the actual values for each host before running the commands. Keep the host addresses consistent across terminals; `nic_name` must identify the interface that owns `local_ip`. Run the model commands in separate terminals and start the proxy after all model APIs are ready.

=== "A2 series"

    This W4A8 deployment uses two eight-device A2 series hosts. Prefill runs on `PREFILL_HOST` and Decode runs on `DECODE_HOST`, each using devices 0–7. Each role uses TP2 × DP4 and loads the same checkpoint.

    Before starting:

    - Expose the devices used by each host and mount `/etc/hccn.conf`. Use `--shm-size=128g` and provide sufficient host memory for Engram CPU offload with DP shared memory. Replace the IP and network-interface placeholders at the start of each command block. Set `local_ip` to the current host address and `nic_name` to its network interface.
    - Run P and D in separate terminals. After both APIs are ready, start the proxy on the prefill host from the vLLM-Ascend repository root.

    **Prefill:**

    ```bash
    PREFILL_HOST="<PREFILL_LOCAL_IP>"
    DECODE_HOST="<DECODE_LOCAL_IP>"
    local_ip="${PREFILL_HOST}"
    nic_name="<SERVICE_NETWORK_INTERFACE>"

    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY

    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export VLLM_ENGINE_READY_TIMEOUT_S=4800
    export VLLM_HOST_IP="${local_ip}"
    export HCCL_IF_IP="${local_ip}"
    export GLOO_SOCKET_IFNAME="${nic_name}"
    export TP_SOCKET_IFNAME="${nic_name}"
    export HCCL_SOCKET_IFNAME="${nic_name}"
    export no_proxy="localhost,127.0.0.1,${PREFILL_HOST},${DECODE_HOST}"
    export NO_PROXY="${no_proxy}"
    export LD_LIBRARY_PATH="$(python -c 'import mooncake; print(next(iter(mooncake.__path__)))'):${LD_LIBRARY_PATH:-}"
    export VLLM_USE_MODELSCOPE=True
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export HCCL_NPU_SOCKET_PORT_RANGE=61400-61499
    export HCCL_BUFFSIZE=8192

    vllm serve Eco-Tech/Qwen3.8-Flash-Next-w4a8 \
        --host 0.0.0.0 \
        --port 25568 \
        --served-model-name qwen38-flash-next-a2-pd \
        --trust-remote-code \
        --quantization ascend \
        --tensor-parallel-size 2 \
        --data-parallel-size 4 \
        --max-model-len 141312 \
        --max-num-seqs 32 \
        --max-num-batched-tokens 8192 \
        --gpu-memory-utilization 0.90 \
        --enable-prefix-caching \
        --async-scheduling \
        --enable-expert-parallel \
        --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
        --compilation-config '{"cudagraph_mode":"NONE"}' \
        --additional-config '{"enable_shared_expert_dp":true,"enable_flashcomm1":true,"enable_prefill_mc2":true,"enable_fused_mc2":0,"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"fuse_norm_quant":false}}' \
        --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' \
        --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_producer","kv_port":32600,"kv_connector_extra_config":{"prefill":{"dp_size":4,"tp_size":2},"decode":{"dp_size":4,"tp_size":2}}}'
    ```

    **Decode:**

    ```bash
    PREFILL_HOST="<PREFILL_LOCAL_IP>"
    DECODE_HOST="<DECODE_LOCAL_IP>"
    local_ip="${DECODE_HOST}"
    nic_name="<SERVICE_NETWORK_INTERFACE>"

    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY

    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export VLLM_ENGINE_READY_TIMEOUT_S=4800
    export VLLM_HOST_IP="${local_ip}"
    export HCCL_IF_IP="${local_ip}"
    export GLOO_SOCKET_IFNAME="${nic_name}"
    export TP_SOCKET_IFNAME="${nic_name}"
    export HCCL_SOCKET_IFNAME="${nic_name}"
    export no_proxy="localhost,127.0.0.1,${PREFILL_HOST},${DECODE_HOST}"
    export NO_PROXY="${no_proxy}"
    export LD_LIBRARY_PATH="$(python -c 'import mooncake; print(next(iter(mooncake.__path__)))'):${LD_LIBRARY_PATH:-}"
    export VLLM_USE_MODELSCOPE=True
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export HCCL_NPU_SOCKET_PORT_RANGE=61500-61599
    export HCCL_BUFFSIZE=512

    vllm serve Eco-Tech/Qwen3.8-Flash-Next-w4a8 \
        --host 0.0.0.0 \
        --port 25569 \
        --served-model-name qwen38-flash-next-a2-pd \
        --trust-remote-code \
        --quantization ascend \
        --tensor-parallel-size 2 \
        --data-parallel-size 4 \
        --max-model-len 141312 \
        --max-num-seqs 32 \
        --max-num-batched-tokens 8192 \
        --gpu-memory-utilization 0.90 \
        --enable-prefix-caching \
        --async-scheduling \
        --enable-expert-parallel \
        --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --additional-config '{"enable_flashcomm1":true,"enable_cpu_binding":true,"enable_shared_expert_dp":true,"multistream_overlap_shared_expert":true,"scheduler_config":{"recompute_scheduler_enable":true},"ascend_compilation_config":{"fuse_norm_quant":false,"enable_npugraph_ex":true,"enable_static_kernel":true,"enable_super_kernel":false}}' \
        --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' \
        --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_consumer","kv_port":32700,"kv_connector_extra_config":{"prefill":{"dp_size":4,"tp_size":2},"decode":{"dp_size":4,"tp_size":2}}}'
    ```

    **Proxy:**

    ```bash
    PREFILL_HOST="<PREFILL_LOCAL_IP>"
    DECODE_HOST="<DECODE_LOCAL_IP>"

    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY
    export no_proxy="localhost,127.0.0.1,${PREFILL_HOST},${DECODE_HOST}"
    export NO_PROXY="${no_proxy}"

    python examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
        --host 0.0.0.0 --port 25570 \
        --prefiller-hosts "${PREFILL_HOST}" --prefiller-ports 25568 \
        --decoder-hosts "${DECODE_HOST}" --decoder-ports 25569
    ```

    **Service Verification:**

    From the prefill host, check that both APIs list `qwen38-flash-next-a2-pd`:

    ```bash
    PREFILL_HOST="<PREFILL_LOCAL_IP>"
    DECODE_HOST="<DECODE_LOCAL_IP>"

    curl --fail "http://${PREFILL_HOST}:25568/v1/models"
    curl --fail "http://${DECODE_HOST}:25569/v1/models"
    ```

    After both checks succeed, run the Proxy command in a separate terminal.

    Once the proxy is listening, verify the complete PD path:

    ```bash
    curl --fail "http://127.0.0.1:25570/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "qwen38-flash-next-a2-pd",
            "messages": [{"role": "user", "content": "Who are you?"}],
            "max_tokens": 1024,
            "temperature": 0
        }'
    ```

    Inspect the prefill, decode, and proxy terminal output if the request fails.

    **Key parameters:**

    - Both roles enable prefix caching; the current runtime automatically changes the default Mamba cache mode from `none` to `align`.
    - Engram requires a checkpoint with supported n-gram layers. `cpu_offload=true` keeps Engram embedding tables in host memory. `dp_shared_memory=true` shares the corresponding table shards among node-local DP replicas; replicas must share the same host and shared-memory namespace. Both options are booleans. This DP configuration satisfies the DP > 1 requirement; elastic EP is not supported with shared memory.
    - Prefill sets `cudagraph_mode=NONE`, enables `enable_prefill_mc2=true`, and disables fused MC2. Verify MC2 token limits for TP2 × DP4 before increasing the batch token budget.
    - Decode enables `FULL_DECODE_ONLY`, NPU graph extension, static kernels, and recompute scheduling. Super kernels remain disabled.
    - Both roles use three MTP draft tokens.
    - `enable_fused_mc2=0` disables fused MC2, so it does not conflict with `multistream_overlap_shared_expert=true`. The decode command relies on the runtime's default value of zero.
    - Keep the Mooncake DP4 / TP2 declarations synchronized with both services' parallel settings.
    - For Function Calling and reasoning parsing, add `--enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3` to both roles and verify proxy forwarding. Multimodal input and thinking control also require validation through the proxy with the W4A8 checkpoint.

=== "A3 series"

    This W4A8 PD deployment uses one 16-device A3 series host specified by `local_ip`. Prefill uses TP4 × PP2 × DP1 on devices 0–7; Decode uses TP2 × DP4 on devices 8–15. Both roles load the same checkpoint. The proxy routes requests through one Prefill API endpoint and one Decode API endpoint (1P1D); Decode runs four DP replicas behind its endpoint.

    Before starting:

    - Expose the devices used by each host and mount `/etc/hccn.conf`. Use `--shm-size=128g` and provide sufficient host memory for Engram CPU offload with DP shared memory. Replace the IP and network-interface placeholders at the start of each command block. Set `local_ip` to the current host address and `nic_name` to its network interface.
    - Run P and D in separate terminals. After both APIs are ready, start the proxy on the prefill host from the vLLM-Ascend repository root.

    **Prefill:**

    ```bash
    local_ip="<LOCAL_SERVICE_IP>"
    nic_name="<SERVICE_NETWORK_INTERFACE>"

    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY

    unset VLLM_PP_LAYER_PARTITION
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export VLLM_ENGINE_READY_TIMEOUT_S=4800
    export VLLM_HOST_IP="${local_ip}"
    export HCCL_IF_IP="${local_ip}"
    export GLOO_SOCKET_IFNAME="${nic_name}"
    export TP_SOCKET_IFNAME="${nic_name}"
    export HCCL_SOCKET_IFNAME="${nic_name}"
    export no_proxy="localhost,127.0.0.1,${local_ip}"
    export NO_PROXY="${no_proxy}"
    export LD_LIBRARY_PATH="$(python -c 'import mooncake; print(next(iter(mooncake.__path__)))'):${LD_LIBRARY_PATH:-}"
    export VLLM_USE_MODELSCOPE=True
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export HCCL_NPU_SOCKET_PORT_RANGE=61400-61499
    export HCCL_BUFFSIZE=8192

    vllm serve Eco-Tech/Qwen3.8-Flash-Next-w4a8 \
        --host 0.0.0.0 \
        --port 25568 \
        --served-model-name qwen38-flash-next-a3-pd \
        --trust-remote-code \
        --quantization ascend \
        --tensor-parallel-size 4 \
        --pipeline-parallel-size 2 \
        --data-parallel-size 1 \
        --max-model-len 141312 \
        --max-num-seqs 32 \
        --max-num-batched-tokens 8192 \
        --gpu-memory-utilization 0.90 \
        --enable-prefix-caching \
        --async-scheduling \
        --enable-expert-parallel \
        --mamba-cache-mode align \
        --mamba-ssm-cache-dtype bfloat16 \
        --engram-config '{"cpu_offload":true,"dp_shared_memory":false}' \
        --enforce-eager \
        --additional-config '{"enable_shared_expert_dp":true,"enable_flashcomm1":true,"enable_prefill_mc2":true,"enable_fused_mc2":0,"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"fuse_norm_quant":false}}' \
        --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}' \
        --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_producer","kv_port":32600,"kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":4,"pp_size":2},"decode":{"dp_size":4,"tp_size":2}}}'
    ```

    **Decode:**

    ```bash
    local_ip="<LOCAL_SERVICE_IP>"
    nic_name="<SERVICE_NETWORK_INTERFACE>"

    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY

    unset VLLM_PP_LAYER_PARTITION
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export VLLM_ENGINE_READY_TIMEOUT_S=4800
    export VLLM_HOST_IP="${local_ip}"
    export HCCL_IF_IP="${local_ip}"
    export GLOO_SOCKET_IFNAME="${nic_name}"
    export TP_SOCKET_IFNAME="${nic_name}"
    export HCCL_SOCKET_IFNAME="${nic_name}"
    export no_proxy="localhost,127.0.0.1,${local_ip}"
    export NO_PROXY="${no_proxy}"
    export LD_LIBRARY_PATH="$(python -c 'import mooncake; print(next(iter(mooncake.__path__)))'):${LD_LIBRARY_PATH:-}"
    export VLLM_USE_MODELSCOPE=True
    export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
    export HCCL_NPU_SOCKET_PORT_RANGE=61500-61599
    export HCCL_BUFFSIZE=512

    vllm serve Eco-Tech/Qwen3.8-Flash-Next-w4a8 \
        --host 0.0.0.0 \
        --port 25569 \
        --served-model-name qwen38-flash-next-a3-pd \
        --trust-remote-code \
        --quantization ascend \
        --tensor-parallel-size 2 \
        --data-parallel-size 4 \
        --max-model-len 141312 \
        --max-num-seqs 32 \
        --max-num-batched-tokens 8192 \
        --gpu-memory-utilization 0.90 \
        --enable-prefix-caching \
        --async-scheduling \
        --enable-expert-parallel \
        --mamba-cache-mode align \
        --mamba-ssm-cache-dtype bfloat16 \
        --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --additional-config '{"enable_flashcomm1":true,"enable_cpu_binding":true,"enable_shared_expert_dp":true,"multistream_overlap_shared_expert":true,"scheduler_config":{"recompute_scheduler_enable":true},"ascend_compilation_config":{"fuse_norm_quant":false,"enable_npugraph_ex":true,"enable_static_kernel":true,"enable_super_kernel":false}}' \
        --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":false}' \
        --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_consumer","kv_port":32700,"kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":4,"pp_size":2},"decode":{"dp_size":4,"tp_size":2}}}'
    ```

    **Proxy:**

    ```bash
    local_ip="<LOCAL_SERVICE_IP>"

    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY
    export no_proxy="localhost,127.0.0.1,${local_ip}"
    export NO_PROXY="${no_proxy}"

    python examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
        --host 0.0.0.0 --port 25570 \
        --prefiller-hosts "${local_ip}" --prefiller-ports 25568 \
        --decoder-hosts "${local_ip}" --decoder-ports 25569
    ```

    **Service Verification:**

    Check the startup logs for TP4 × PP2 × DP1 on Prefill and TP2 × DP4 with `enable_static_kernel=true` on Decode. From the serving host, check that both APIs list `qwen38-flash-next-a3-pd`:

    ```bash
    curl --fail "http://127.0.0.1:25568/v1/models"
    curl --fail "http://127.0.0.1:25569/v1/models"
    ```

    After both checks succeed, run the Proxy command in a separate terminal.

    Once the proxy is listening, verify the complete PD path:

    ```bash
    curl --fail "http://127.0.0.1:25570/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "qwen38-flash-next-a3-pd",
            "messages": [{"role": "user", "content": "Who are you?"}],
            "max_tokens": 1024,
            "temperature": 0
        }'
    ```

    Inspect the prefill, decode, and proxy terminal output if the request fails.

    **Key parameters:**

    - Both roles enable prefix caching and explicitly set `--mamba-cache-mode align` and `--mamba-ssm-cache-dtype bfloat16`.
    - Engram requires a checkpoint with supported n-gram layers. `cpu_offload=true` keeps Engram embedding tables in host memory. `dp_shared_memory=true` shares the corresponding table shards among node-local DP replicas; replicas must share the same host and shared-memory namespace. Both options are booleans. Prefill uses DP1 and sets `dp_shared_memory=false`; Decode uses DP4 and sets it to `true`. Elastic EP is not supported with shared memory.
    - Prefill distributes the model across two pipeline stages, with TP4 in each stage. It runs in eager mode (`--enforce-eager`), enables `enable_prefill_mc2=true`, and disables fused MC2. Verify MC2 token limits for TP4 × PP2 × DP1 before increasing the batch token budget.
    - Decode uses four TP2 replicas without pipeline parallelism. It enables `FULL_DECODE_ONLY`, NPU graph extension, `enable_static_kernel=true`, and recompute scheduling. Static kernels are enabled for Decode; Prefill retains eager execution. Super kernels remain disabled.
    - Both roles use three MTP draft tokens; Prefill sets MTP `enforce_eager=true` and Decode sets it to `false`. Both roles use `--max-num-batched-tokens 8192`; Prefill retains `HCCL_BUFFSIZE=8192` and Decode retains `HCCL_BUFFSIZE=512`.
    - `enable_fused_mc2=0` disables fused MC2, so it does not conflict with `multistream_overlap_shared_expert=true`. The decode command relies on the runtime's default value of zero.
    - Keep the Mooncake declarations synchronized on both roles: Prefill uses DP1 / TP4 / PP2, and Decode uses DP4 / TP2. No custom layer partition is specified; clear `VLLM_PP_LAYER_PARTITION` in both terminals to use the runtime's default partition.
    - For Function Calling and reasoning parsing, add `--enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3` to both roles and verify proxy forwarding. Multimodal input and thinking control also require validation through the proxy with the W4A8 checkpoint.

=== "Ascend 950 DT"

    This W4A8-MXFP deployment uses eight devices on one Ascend 950 DT host. Prefill and decode each use TP2 × DP2, with a separate API process for each DP rank. MooncakeConnectorV2 transfers KV cache between the two roles.

    | Setting | Prefill | Decode |
    | --- | --- | --- |
    | Rank 0 / rank 1 devices | 0,1 / 2,3 | 4,5 / 6,7 |
    | TP × DP | 2 × 2 | 2 × 2 |
    | Rank 0 / rank 1 API ports | 8088 / 8089 | 8092 / 8093 |
    | DP RPC port | 13388 | 13389 |
    | Mooncake role / KV base port | `kv_producer` / 30000 | `kv_consumer` / 30100 |
    | Maximum context / sequences / batched tokens | 141312 / 64 / 4096 | 141312 / 64 / 4096 |

    All four processes load `Eco-Tech/Qwen3.8-Flash-Next-w4a8-mxfp`. Set `local_ip` to the serving host address and `nic_name` to its network interface in each command block. Provide sufficient host memory for Engram CPU offload and shared memory.

    Run each rank command in a separate terminal. Start the proxy after all four APIs are ready. Port 8087 is also used by the colocated example, so stop that service before starting this proxy.

    **Prefill:**

    Rank 0:

    ```bash
    local_ip="<LOCAL_SERVICE_IP>"
    nic_name="<SERVICE_NETWORK_INTERFACE>"

    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY

    export VLLM_HOST_IP="${local_ip}"
    export HCCL_IF_IP="${local_ip}"
    export GLOO_SOCKET_IFNAME="${nic_name}"
    export TP_SOCKET_IFNAME="${nic_name}"
    export HCCL_SOCKET_IFNAME="${nic_name}"
    export VLLM_USE_MODELSCOPE=True
    export ASCEND_RT_VISIBLE_DEVICES=0,1
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export VLLM_ENGINE_READY_TIMEOUT_S=4800
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export TASK_QUEUE_ENABLE=1
    export HCCL_BUFFSIZE=1024
    export HCCL_NPU_SOCKET_PORT_RANGE=auto
    export LD_LIBRARY_PATH="$(python -c 'import mooncake; print(next(iter(mooncake.__path__)))'):${LD_LIBRARY_PATH:-}"

    vllm serve Eco-Tech/Qwen3.8-Flash-Next-w4a8-mxfp \
        --host 0.0.0.0 \
        --port 8088 \
        --served-model-name qwen38-flash-next-950-dt-pd \
        --trust-remote-code \
        --quantization ascend \
        --tensor-parallel-size 2 \
        --data-parallel-size 2 \
        --data-parallel-size-local 1 \
        --data-parallel-rank 0 \
        --data-parallel-address "${local_ip}" \
        --data-parallel-rpc-port 13388 \
        --max-model-len 141312 \
        --max-num-seqs 64 \
        --max-num-batched-tokens 4096 \
        --gpu-memory-utilization 0.90 \
        --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
        --enable-prefix-caching \
        --async-scheduling \
        --enable-expert-parallel \
        --compilation-config '{"cudagraph_mode":"NONE"}' \
        --additional-config '{"enable_cpu_binding":true,"enable_shared_expert_dp":true,"enable_fused_mc2":0,"enable_flashcomm1":true,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"fuse_norm_quant":false,"enable_npugraph_ex":true,"enable_static_kernel":false,"enable_super_kernel":false}}' \
        --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' \
        --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_producer","kv_port":30000,"kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":2},"decode":{"dp_size":2,"tp_size":2}}}'
    ```

    Rank 1:

    ```bash
    local_ip="<LOCAL_SERVICE_IP>"
    nic_name="<SERVICE_NETWORK_INTERFACE>"

    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY

    export VLLM_HOST_IP="${local_ip}"
    export HCCL_IF_IP="${local_ip}"
    export GLOO_SOCKET_IFNAME="${nic_name}"
    export TP_SOCKET_IFNAME="${nic_name}"
    export HCCL_SOCKET_IFNAME="${nic_name}"
    export VLLM_USE_MODELSCOPE=True
    export ASCEND_RT_VISIBLE_DEVICES=2,3
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export VLLM_ENGINE_READY_TIMEOUT_S=4800
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export TASK_QUEUE_ENABLE=1
    export HCCL_BUFFSIZE=1024
    export HCCL_NPU_SOCKET_PORT_RANGE=auto
    export LD_LIBRARY_PATH="$(python -c 'import mooncake; print(next(iter(mooncake.__path__)))'):${LD_LIBRARY_PATH:-}"

    vllm serve Eco-Tech/Qwen3.8-Flash-Next-w4a8-mxfp \
        --host 0.0.0.0 \
        --port 8089 \
        --served-model-name qwen38-flash-next-950-dt-pd \
        --trust-remote-code \
        --quantization ascend \
        --tensor-parallel-size 2 \
        --data-parallel-size 2 \
        --data-parallel-size-local 1 \
        --data-parallel-rank 1 \
        --data-parallel-address "${local_ip}" \
        --data-parallel-rpc-port 13388 \
        --max-model-len 141312 \
        --max-num-seqs 64 \
        --max-num-batched-tokens 4096 \
        --gpu-memory-utilization 0.90 \
        --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
        --enable-prefix-caching \
        --async-scheduling \
        --enable-expert-parallel \
        --compilation-config '{"cudagraph_mode":"NONE"}' \
        --additional-config '{"enable_cpu_binding":true,"enable_shared_expert_dp":true,"enable_fused_mc2":0,"enable_flashcomm1":true,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"fuse_norm_quant":false,"enable_npugraph_ex":true,"enable_static_kernel":false,"enable_super_kernel":false}}' \
        --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' \
        --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_producer","kv_port":30000,"kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":2},"decode":{"dp_size":2,"tp_size":2}}}'
    ```

    **Decode:**

    Rank 0:

    ```bash
    local_ip="<LOCAL_SERVICE_IP>"
    nic_name="<SERVICE_NETWORK_INTERFACE>"

    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY

    export VLLM_HOST_IP="${local_ip}"
    export HCCL_IF_IP="${local_ip}"
    export GLOO_SOCKET_IFNAME="${nic_name}"
    export TP_SOCKET_IFNAME="${nic_name}"
    export HCCL_SOCKET_IFNAME="${nic_name}"
    export VLLM_USE_MODELSCOPE=True
    export ASCEND_RT_VISIBLE_DEVICES=4,5
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export VLLM_ENGINE_READY_TIMEOUT_S=4800
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export TASK_QUEUE_ENABLE=1
    export HCCL_BUFFSIZE=1024
    export HCCL_NPU_SOCKET_PORT_RANGE=auto
    export LD_LIBRARY_PATH="$(python -c 'import mooncake; print(next(iter(mooncake.__path__)))'):${LD_LIBRARY_PATH:-}"

    vllm serve Eco-Tech/Qwen3.8-Flash-Next-w4a8-mxfp \
        --host 0.0.0.0 \
        --port 8092 \
        --served-model-name qwen38-flash-next-950-dt-pd \
        --trust-remote-code \
        --quantization ascend \
        --tensor-parallel-size 2 \
        --data-parallel-size 2 \
        --data-parallel-size-local 1 \
        --data-parallel-rank 0 \
        --data-parallel-address "${local_ip}" \
        --data-parallel-rpc-port 13389 \
        --max-model-len 141312 \
        --max-num-seqs 64 \
        --max-num-batched-tokens 4096 \
        --gpu-memory-utilization 0.90 \
        --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
        --enable-prefix-caching \
        --async-scheduling \
        --enable-expert-parallel \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --additional-config '{"enable_cpu_binding":true,"enable_shared_expert_dp":true,"enable_fused_mc2":0,"enable_flashcomm1":true,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"fuse_norm_quant":false,"enable_npugraph_ex":true,"enable_static_kernel":true,"enable_super_kernel":false},"scheduler_config":{"recompute_scheduler_enable":true}}' \
        --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' \
        --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_consumer","kv_port":30100,"kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":2},"decode":{"dp_size":2,"tp_size":2}}}'
    ```

    Rank 1:

    ```bash
    local_ip="<LOCAL_SERVICE_IP>"
    nic_name="<SERVICE_NETWORK_INTERFACE>"

    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY

    export VLLM_HOST_IP="${local_ip}"
    export HCCL_IF_IP="${local_ip}"
    export GLOO_SOCKET_IFNAME="${nic_name}"
    export TP_SOCKET_IFNAME="${nic_name}"
    export HCCL_SOCKET_IFNAME="${nic_name}"
    export VLLM_USE_MODELSCOPE=True
    export ASCEND_RT_VISIBLE_DEVICES=6,7
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export VLLM_ENGINE_READY_TIMEOUT_S=4800
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export TASK_QUEUE_ENABLE=1
    export HCCL_BUFFSIZE=1024
    export HCCL_NPU_SOCKET_PORT_RANGE=auto
    export LD_LIBRARY_PATH="$(python -c 'import mooncake; print(next(iter(mooncake.__path__)))'):${LD_LIBRARY_PATH:-}"

    vllm serve Eco-Tech/Qwen3.8-Flash-Next-w4a8-mxfp \
        --host 0.0.0.0 \
        --port 8093 \
        --served-model-name qwen38-flash-next-950-dt-pd \
        --trust-remote-code \
        --quantization ascend \
        --tensor-parallel-size 2 \
        --data-parallel-size 2 \
        --data-parallel-size-local 1 \
        --data-parallel-rank 1 \
        --data-parallel-address "${local_ip}" \
        --data-parallel-rpc-port 13389 \
        --max-model-len 141312 \
        --max-num-seqs 64 \
        --max-num-batched-tokens 4096 \
        --gpu-memory-utilization 0.90 \
        --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
        --enable-prefix-caching \
        --async-scheduling \
        --enable-expert-parallel \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --additional-config '{"enable_cpu_binding":true,"enable_shared_expert_dp":true,"enable_fused_mc2":0,"enable_flashcomm1":true,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"fuse_norm_quant":false,"enable_npugraph_ex":true,"enable_static_kernel":true,"enable_super_kernel":false},"scheduler_config":{"recompute_scheduler_enable":true}}' \
        --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' \
        --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_consumer","kv_port":30100,"kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":2},"decode":{"dp_size":2,"tp_size":2}}}'
    ```

    **Proxy:**

    Run the following command from the vLLM-Ascend repository root after all four model APIs are ready:

    ```bash
    local_ip="<LOCAL_SERVICE_IP>"

    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY

    python examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
        --host "${local_ip}" \
        --port 8087 \
        --prefiller-hosts "${local_ip}" "${local_ip}" \
        --prefiller-ports 8088 8089 \
        --decoder-hosts "${local_ip}" "${local_ip}" \
        --decoder-ports 8092 8093
    ```

    **Service Verification:**

    Check that all four APIs list `qwen38-flash-next-950-dt-pd` before starting the proxy:

    ```bash
    local_ip="<LOCAL_SERVICE_IP>"

    curl --fail "http://${local_ip}:8088/v1/models"
    curl --fail "http://${local_ip}:8089/v1/models"
    curl --fail "http://${local_ip}:8092/v1/models"
    curl --fail "http://${local_ip}:8093/v1/models"
    ```

    Once the proxy is listening, verify the complete PD path:

    ```bash
    local_ip="<LOCAL_SERVICE_IP>"

    curl --fail "http://${local_ip}:8087/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"qwen38-flash-next-950-dt-pd","messages":[{"role":"user","content":"Who are you?"}],"max_tokens":1024,"temperature":0}'
    ```

    **Key parameters:**

    - Each command starts one DP rank with TP2. Keep each role's DP address, RPC port, and model settings consistent across its two ranks.
    - Both roles enable Model Runner V2, async scheduling, and expert parallelism (EP). EP is distinct from expert parallel load balancing (EPLB); these commands do not enable EPLB.
    - `cpu_offload=true` keeps Engram embedding tables in host memory; `dp_shared_memory=true` shares the corresponding table shards among node-local DP replicas. Both options are booleans. DP replicas must share the same host and shared-memory namespace; this DP2 configuration satisfies the DP > 1 requirement and does not enable elastic EP.
    - Prefill enables prefix caching, which automatically selects Mamba cache mode `align` from the default `none`, and uses `cudagraph_mode=NONE`; static kernels are disabled.
    - Decode enables prefix caching, `FULL_DECODE_ONLY`, static kernels, and recompute scheduling.
    - Both roles enable CPU binding, shared expert DP, FlashComm1, shared expert multistream overlap, and NPU graph extension. Fused MC2, norm/quantization fusion, and super kernels are disabled.
    - Both roles use three MTP draft tokens.
    - Keep the Mooncake DP2 / TP2 declarations synchronized with the service topology and route both ranks of each role through the proxy. `kv_port` is a base port shared by ranks of the same role: the connector adds the DP serving index × TP × PP × PCP, then the local device index. With TP2 and PP = PCP = 1, prefill rank 0 uses ports 30000–30001 and rank 1 uses 30002–30003; decode uses 30100–30103. Do not increment the configured base port for rank 1.
    - For Function Calling and reasoning parsing, add `--enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3` to all four model commands and verify proxy forwarding. Validate multimodal input and thinking control through the proxy with the selected checkpoint.

### 5.3 Image-and-Text 10K PD Separation Deployment

This case uses the same eight-device Ascend 950 DT host as the basic PD example in [Section 5.2](#52-basic-pd-separation-deployment). For image-and-text 10K validation, prefill uses TP2 × PP2 × DP1 with a `25,23` layer partition, and decode uses TP2 × DP2 behind one API endpoint. Both roles use 8192 batch tokens and `HCCL_BUFFSIZE=512`. Both roles enable prefix caching; decode also enables static kernels.

The 10K target is the processed input length, including text and image tokens. Both roles retain `--max-model-len 143360`; keep the input plus requested output within this limit. Use the W4A8-MXFP checkpoint from [Section 3.1](#31-model-weight) with its vision encoder and image processor.

The complete Prefill, Decode, and proxy commands are provided below.

=== "Ascend 950 DT"

    Initialize the environment as described in [Section 4](#4-installation) and run Prefill and Decode in separate terminals. Both roles load `Eco-Tech/Qwen3.8-Flash-Next-w4a8-mxfp` from ModelScope.

    **Prefill:**

    ```bash
    export VLLM_USE_MODELSCOPE=True
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export VLLM_ENGINE_READY_TIMEOUT_S=4800
    export VLLM_ASCEND_PLE_BACKEND=torch
    export VLLM_PP_LAYER_PARTITION=25,23
    export HCCL_NPU_SOCKET_PORT_RANGE=61400-61499
    export HCCL_BUFFSIZE=512
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

    vllm serve Eco-Tech/Qwen3.8-Flash-Next-w4a8-mxfp \
        --host 0.0.0.0 \
        --port 8247 \
        --served-model-name qwen38-megamoe-sp-pd \
        --trust-remote-code \
        --quantization ascend \
        --tensor-parallel-size 2 \
        --pipeline-parallel-size 2 \
        --data-parallel-size 1 \
        --max-model-len 143360 \
        --max-num-seqs 64 \
        --max-num-batched-tokens 8192 \
        --gpu-memory-utilization 0.9 \
        --enable-prefix-caching \
        --async-scheduling \
        --enable-expert-parallel \
        --engram-config '{"cpu_offload":true,"dp_shared_memory":false}' \
        --compilation-config '{"cudagraph_mode":"NONE"}' \
        --additional-config '{"enable_shared_expert_dp":true,"enable_flashcomm1":true,"multistream_overlap_shared_expert":true,"enable_cpu_binding":true,"ascend_compilation_config":{"fuse_norm_quant":false}}' \
        --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}' \
        --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_producer","kv_port":32470,"kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":2,"pp_size":2,"pp_layer_partition":"25,23"},"decode":{"dp_size":2,"tp_size":2}}}'
    ```

    **Decode:**

    ```bash
    export VLLM_USE_MODELSCOPE=True
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export VLLM_ENGINE_READY_TIMEOUT_S=4800
    export VLLM_ASCEND_PLE_BACKEND=torch
    unset VLLM_PP_LAYER_PARTITION
    export HCCL_NPU_SOCKET_PORT_RANGE=61500-61599
    export HCCL_BUFFSIZE=512
    export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7

    vllm serve Eco-Tech/Qwen3.8-Flash-Next-w4a8-mxfp \
        --host 0.0.0.0 \
        --port 8248 \
        --served-model-name qwen38-megamoe-sp-pd \
        --trust-remote-code \
        --quantization ascend \
        --tensor-parallel-size 2 \
        --data-parallel-size 2 \
        --max-model-len 143360 \
        --max-num-seqs 64 \
        --max-num-batched-tokens 8192 \
        --gpu-memory-utilization 0.9 \
        --enable-prefix-caching \
        --async-scheduling \
        --enable-expert-parallel \
        --engram-config '{"cpu_offload":true,"dp_shared_memory":false}' \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --additional-config '{"enable_shared_expert_dp":true,"enable_flashcomm1":true,"multistream_overlap_shared_expert":true,"enable_cpu_binding":true,"ascend_compilation_config":{"fuse_norm_quant":false,"enable_npugraph_ex":true,"enable_static_kernel":true,"enable_super_kernel":false},"enable_fused_mc2":0}' \
        --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":false}' \
        --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_consumer","kv_port":32480,"kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":2,"pp_size":2,"pp_layer_partition":"25,23"},"decode":{"dp_size":2,"tp_size":2}}}'
    ```

    **Proxy:**

    After both model APIs are ready, run the proxy from the vLLM-Ascend repository root:

    ```bash
    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY

    python examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
        --host "127.0.0.1" \
        --port 8246 \
        --prefiller-hosts "127.0.0.1" \
        --prefiller-ports 8247 \
        --decoder-hosts "127.0.0.1" \
        --decoder-ports 8248
    ```

    **Service Verification:**

    Wait until both model APIs are ready, then start the proxy as described above. Check that prefill uses TP2/PP2/DP1, decode uses TP2/DP2 with static kernels enabled, and both roles report `max_model_len=143360`. Send a 10K image-and-text sample through the proxy using the request below. Replace `<IMAGE_URL>` with the sample image URL and `<TEXT_PROMPT>` with its text; the image must be accessible from the serving environment.

    ```bash
    curl --fail "http://127.0.0.1:8247/v1/models"
    curl --fail "http://127.0.0.1:8248/v1/models"
    curl --fail "http://127.0.0.1:8246/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"qwen38-megamoe-sp-pd","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"<IMAGE_URL>"}},{"type":"text","text":"<TEXT_PROMPT>"}]}],"max_tokens":1024,"temperature":0}'
    ```

    Expected result: HTTP 200 with a response that addresses both the image and text. Check `usage.prompt_tokens` against the 10K validation target and confirm that the request completes without image-processing, KV-transfer, or out-of-memory errors. Supply the actual validation sample before using this request as a 10K test.

    **Key parameters:**

    - Prefill partitions the layers across two pipeline stages with `VLLM_PP_LAYER_PARTITION=25,23`. Keep `pp_size=2` and `pp_layer_partition="25,23"` in both Mooncake configurations consistent with this partition.
    - Decode uses TP2 × DP2 behind one API endpoint on port `8248`; the proxy routes to that endpoint rather than separate DP-rank APIs.
    - Both roles enable prefix caching. Prefill disables graph capture and sets MTP `enforce_eager=true`. Decode uses `FULL_DECODE_ONLY`, sets MTP `enforce_eager=false`, enables static kernels, and disables super kernels.
    - Both roles enable Engram CPU offload with `dp_shared_memory=false`, use `HCCL_BUFFSIZE=512`, and set `VLLM_ENGINE_READY_TIMEOUT_S=4800` to allow sufficient engine initialization time. Provide sufficient host memory for the checkpoint and the decode DP replicas.
    - This case retains `VLLM_ASCEND_PLE_BACKEND=torch` from the supplied scripts. It only selects a backend in runtimes that implement this variable; an unknown-variable warning means it has no effect.

### 5.4 PD Separation with MemCache Three-Tier KV Cache Pool

This example extends [Section 5.2](#52-basic-pd-separation-deployment) with three-tier KV caching: NPU HBM, pooled DRAM, and SSD storage through MemCache/UBS IO. `MultiConnector` combines the P-to-D transport with `AscendStoreConnector`. This section covers A3 series and Ascend 950 DT basic PD deployment only. The image-and-text TP2 × PP2 case in Section 5.3 is a separate configuration.

The A3 example extends the complete 1P1D serving path in Section 5.2: TP4 × PP2 × DP1 Prefill, TP2 × DP4 Decode, and the proxy. Prefill retains eager execution, and Decode retains static kernels. MemCache adds external KV storage to this deployment. The 950 DT connector settings are adapted from its pooling reference scripts. Validate these configurations with the installed runtime and checkpoint; the examples do not establish measured cache-hit rates or performance results.

**Prepare the environment:**

For Docker deployment, use [Section 4.3](#43-docker-for-three-tier-pd-pooling) and complete the host disk preparation below before creating the container. `MEMCACHE_CONFIG_DIR` identifies the same existing directory on the host and inside the container. When running without Docker, set this variable to the existing configuration directory before starting any storage service or P/D engine.

Follow the [MemCache backend guide](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#scenario-2-memcache-backend) for the matching HDK/CANN, communication libraries, container mounts, and host-memory preparation. Use compatible `memcache-hybrid` and `memfabric-hybrid` packages, both at least version 1.3.0 as required by that guide, together with Mooncake. A3 HCCS `device_sdma` requires a compatible LingQu Computing Network version (at least 1.5). Confirm that the installed vLLM-Ascend registers the connector named in the hardware tab.

```bash
pip show memcache-hybrid memfabric-hybrid
```

Use the reported package location to find the MemCache and MemFabric library directories. Use `MEMCACHE_CONFIG_DIR` for the existing configuration directory used by the reference scripts. Replace `<MEMCACHE_LIB_DIR>`, `<MEMFABRIC_LIB_DIR>`, and `<PYTHON_LIB_DIR>` with the installed library paths. All P/D processes must use the same checkpoint and connect to the same MetaService and ConfigStore.

**Prepare SSD storage:**

On every host contributing SSD storage, inspect the disks and available filesystem space:

```bash
lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS
```

Use a dedicated disk that has no partitions, mount points, filesystem signatures, or data to retain. Set `ubsio.disk.path` to the verified device path; separate multiple devices with `:`. Do not copy another host's disk names without checking them locally.

If no dedicated disk is available, use a loop-backed file for testing. The following example allocates 100 GiB in an existing `/data` filesystem; verify its free space and choose a capacity suitable for the test before running it. Use a new backing-file name for each deployment:

```bash
df -h /data
test ! -e /data/qwen38-memcache-ssd.img || exit 1
sudo fallocate -l 100G /data/qwen38-memcache-ssd.img
sudo losetup --find --show --direct-io=on /data/qwen38-memcache-ssd.img
```

Record the returned device path and use it as `<SSD_DEVICE_PATH>` below. Pass the actual disk or loop device into each container that accesses it, for example by adding `--device /dev/loop0:/dev/loop0` only when `/dev/loop0` is the allocated device. A loop device is an alternative to a dedicated disk, not an additional requirement. Its underlying storage must be suitable for the intended SSD test.

!!! warning

    `ubsio.standalone.force_new_disk=true` reinitializes the configured cache disks when the service starts. Use only storage dedicated to this experiment. Do not share those disks with another active pool or restart storage services during a cache-reuse test.

**Configure MetaService:**

If starting a new MetaService, prepare `mmc-meta.conf` in the configuration directory for the selected hardware, as mounted in Section 4.3. Reuse the existing MetaService when it already serves this pool. Use an address reachable from every participating host; `127.0.0.1` is suitable only when all clients share that host/network namespace. Keep ports 5000, 6000, and 8000 available for this deployment:

```ini
ock.mmc.meta_service_url = tcp://<META_SERVICE_IP>:5000
ock.mmc.meta_service.config_store_url = tcp://<META_SERVICE_IP>:6000
ock.mmc.meta_service.metrics_url = http://<META_SERVICE_IP>:8000
ock.mmc.log_level = info
ock.mmc.evict_threshold_high = 70
ock.mmc.evict_threshold_low = 60
```

**Configure storage and P/D engines:**

The hardware tabs below provide LocalService settings and replacements for the `--kv-transfer-config` argument in Section 5.2. Keep the existing device assignments, API ports, model IDs, TP/PP/DP topology, and proxy commands unless a change is explicitly listed. Replace the existing connector argument rather than appending a second one. Keep prefix caching and Model Runner V2 enabled.

Add these exports in every Prefill and Decode terminal before its `vllm serve` command. Preserve the existing Torch, torch_npu, and Mooncake library paths:

```bash
export VLLM_USE_MODELSCOPE=True
export PYTHONHASHSEED=0
export MMC_LOCAL_CONFIG_PATH="${MEMCACHE_CONFIG_DIR}/mmc-local.conf"
export LD_LIBRARY_PATH="<MEMCACHE_LIB_DIR>:<MEMFABRIC_LIB_DIR>:<PYTHON_LIB_DIR>:${LD_LIBRARY_PATH:-}"
```

The `dram.size` and `ubsio.mem.size_in_gb` allocations are per DRAM-enabled LocalService. Budget their aggregate across the host, plus Engram CPU offload, vLLM, and operating-system memory. `world_size=256` is the maximum supported LocalService count, not the model DP size. For memory sizing and limits, see [MemCache SSD cache](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#step-7-enable-memcache-ssd-cache).

=== "A3 series"

    Use the single-host layout from Section 5.2: TP4 × PP2 × DP1 for Prefill and TP2 × DP4 for Decode. This tab follows the HCCS SSD deployment with a separate MemCache storage process. The vLLM clients do not allocate DRAM or own the SSD. Configure `${MEMCACHE_CONFIG_DIR}/mmc-local.conf` as follows:

    ```ini
    ock.mmc.meta_service_url = tcp://<META_SERVICE_IP>:5000
    ock.mmc.local_service.config_store_url = tcp://<META_SERVICE_IP>:6000
    ock.mmc.log_level = info
    ock.mmc.local_service.world_size = 256
    ock.mmc.local_service.protocol = device_sdma
    ock.mmc.local_service.dram.size = 0GB
    ock.mmc.local_service.max.dram.size = 1024GB
    ock.mmc.local_service.storage.enabled = false
    ```

    Prepare `${MEMCACHE_CONFIG_DIR}/mmc-local-standalone.conf` for the storage process. Replace `<POOL_DRAM_SIZE>` with the validated pool capacity including its unit, such as `100GB`; reserve a further 50 GB for UBS IO and sufficient memory for Engram and vLLM. Both files must use the same MetaService, ConfigStore, protocol, and `world_size`.

    ```ini
    ock.mmc.meta_service_url = tcp://<META_SERVICE_IP>:5000
    ock.mmc.local_service.config_store_url = tcp://<META_SERVICE_IP>:6000
    ock.mmc.log_level = info
    ock.mmc.local_service.world_size = 256
    ock.mmc.local_service.protocol = device_sdma
    ock.mmc.local_service.dram.size = <POOL_DRAM_SIZE>
    ock.mmc.local_service.max.dram.size = 1024GB
    ock.mmc.local_service.storage.enabled = true
    ubsio.disk.path = <SSD_DEVICE_PATH>
    ubsio.mem.size_in_gb = 50
    ubsio.standalone.device_count = 1
    ubsio.standalone.force_new_disk = true
    ```

    Export the following in the storage, Prefill, and Decode terminals. Clear incompatible RoCE/UB settings inherited from another experiment:

    ```bash
    unset HCCL_INTRA_ROCE_ENABLE ASCEND_GLOBAL_RESOURCE_CONFIG ASCEND_LOCAL_COMM_RES
    export ACL_OP_INIT_MODE=1
    export ASCEND_ENABLE_USE_FABRIC_MEM=1
    ```

    Retain the A3 1P1D settings from Section 5.2: Prefill uses TP4 × PP2 × DP1 and Decode uses TP2 × DP4 with `enable_static_kernel=true`; P/D batch token budgets are both 8192, P uses `HCCL_BUFFSIZE=8192`, and D uses `HCCL_BUFFSIZE=512`. Keep `MooncakeConnectorV2` as the P-to-D transport and add `AscendStoreConnector` through `MultiConnector`. The storage settings above are additional requirements for the HCCS SSD pool; they do not replace the model execution settings. Validate the combined PP and three-tier pooling configuration on the target runtime.

    **Prefill connector argument:**

    ```text
    --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_producer","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_producer","kv_port":32600,"kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":4,"pp_size":2},"decode":{"dp_size":4,"tp_size":2}}},{"kv_connector":"AscendStoreConnector","kv_role":"kv_producer","kv_connector_extra_config":{"backend":"memcache","lookup_rpc_port":"qwen38_pool_p","use_layerwise":false}}]}}'
    ```

    **Decode connector argument:**

    ```text
    --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_consumer","kv_port":32700,"kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":4,"pp_size":2},"decode":{"dp_size":4,"tp_size":2}}},{"kv_connector":"AscendStoreConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"backend":"memcache","lookup_rpc_port":"qwen38_pool_d","use_layerwise":false}}]}}'
    ```

=== "Ascend 950 DT"

    Use the eight-device TP2 × DP2 layout from Section 5.2. Apply the common exports and connector replacement to all four P/D rank commands. This retains the documented per-rank API endpoints. For UB, configure `${MEMCACHE_CONFIG_DIR}/mmc-local.conf` as follows:

    ```ini
    ock.mmc.meta_service_url = tcp://<META_SERVICE_IP>:5000
    ock.mmc.local_service.config_store_url = tcp://<META_SERVICE_IP>:6000
    ock.mmc.log_level = info
    ock.mmc.local_service.world_size = 256
    ock.mmc.local_service.protocol = device_urma
    ock.mmc.local_service.dram.size = 10GB
    ock.mmc.local_service.max.dram.size = 1024GB
    ock.mmc.local_service.storage.enabled = true
    ubsio.disk.path = <SSD_DEVICE_PATH>
    ubsio.mem.size_in_gb = 10
    ubsio.standalone.device_count = 8
    ubsio.standalone.force_new_disk = true
    ```

    Add the UB environment setting in every rank terminal and clear the HCCS/RoCE settings from other experiments:

    ```bash
    unset HCCL_INTRA_ROCE_ENABLE ASCEND_ENABLE_USE_FABRIC_MEM ASCEND_GLOBAL_RESOURCE_CONFIG
    export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
    ```

    The shown settings budget 80 GB for pool DRAM and 80 GB for UBS IO on this eight-worker host, before the other memory requirements. This tab covers UB. UBOE requires `device_uboe` and the matching resource configuration and mounts from the MemCache backend guide. Do not mix UB and UBOE settings.

    **Prefill connector argument:**

    ```text
    --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_producer","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_producer","kv_port":30000,"kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":2},"decode":{"dp_size":2,"tp_size":2}}},{"kv_connector":"AscendStoreConnector","kv_role":"kv_producer","kv_connector_extra_config":{"backend":"memcache","lookup_rpc_port":"qwen38_pool_p","use_layerwise":false}}]}}'
    ```

    **Decode connector argument:**

    ```text
    --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_consumer","kv_port":30100,"kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":2},"decode":{"dp_size":2,"tp_size":2}}},{"kv_connector":"AscendStoreConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"backend":"memcache","lookup_rpc_port":"qwen38_pool_d","use_layerwise":false}}]}}'
    ```

**Start the storage services:**

Start one MetaService for this pool, unless the matching service is already running:

```bash
export MMC_META_CONFIG_PATH="${MEMCACHE_CONFIG_DIR}/mmc-meta.conf"
python -c "from memcache_hybrid import MetaService; MetaService.main()"
```

For A3 HCCS only, open another terminal in the prepared container, set the HCCS exports from the A3 tab, and start the standalone storage process before either engine. Keep it running:

```bash
export MMC_LOCAL_CONFIG_PATH="${MEMCACHE_CONFIG_DIR}/mmc-local-standalone.conf"
export LD_LIBRARY_PATH="<MEMCACHE_LIB_DIR>:<MEMFABRIC_LIB_DIR>:<PYTHON_LIB_DIR>:${LD_LIBRARY_PATH:-}"
python -u - <<'PY'
import time
from memcache_hybrid import DistributedObjectStore

store = DistributedObjectStore()
result = store.init(0)
if result != 0:
    raise RuntimeError(f"MemCache initialization failed: {result}")
print("Successfully initialized MemCache", flush=True)
while True:
    time.sleep(1)
PY
```

Wait for successful MemCache initialization. In both A3 engine terminals, `MMC_LOCAL_CONFIG_PATH` must point to `mmc-local.conf`, not the standalone file. 950 DT uses embedded LocalServices initialized by the engine workers and does not run this standalone process.

**Start Prefill, Decode, and Proxy:**

Run the matching commands from Section 5.2 in separate terminals with the exports and connector replacements above. Keep `VLLM_ENGINE_READY_TIMEOUT_S=4800`. All P/D connector topology declarations must match the actual TP/PP/DP layout. After every model API is ready, start the same proxy command and run the service verification for that hardware. Use the existing API address and served model name from Section 6.

**Verify three-tier cache reuse:**

1. Verify the effective engine configuration. For A3, confirm TP4 × PP2 × DP1 and eager execution on Prefill, and TP2 × DP4, `FULL_DECODE_ONLY`, and `enable_static_kernel=true` on Decode. Send a request through the proxy and verify output correctness and normal P-to-D transfer.
2. Send requests with reusable prefixes and different suffixes. Observe MemCache lookup, put, get, and hit activity. A successful response or a local prefix-cache hit alone does not establish external pool reuse.
3. Use enough distinct prefixes to exceed the configured DRAM eviction watermark and observe SSD eviction activity, including `evict_to_ssd` when exposed by the installed metrics version. Keep the storage processes running throughout the test.
4. Request an evicted prefix again. Verify SSD read/restore activity and correct generated output. Distinguish this from a hit still served from NPU or DRAM memory; SSD writes alone do not prove successful restoration.
5. Record the runtime versions, storage configuration, input/output lengths, pool hit/eviction/restore metrics, TTFT, and output-quality results. Use the same workload for the basic-PD comparison.

Inspect `/var/log/memcache_hybrid`, `/var/log/memfabric_hybrid`, and `/var/log/ubsio/` for initialization and storage failures. Metric names and available counters depend on the installed MemCache version.

**Key parameters:**

- Keep `PYTHONHASHSEED=0` consistent across P/D workers. `lookup_rpc_port` identifies a local lookup IPC endpoint; the connector appends the DP index. Use distinct P/D identifiers and different identifiers for other pools sharing the same IPC directory.
- Keep `use_layerwise=false` for this reference configuration. Prefill performs external prefix lookup and cache writes; Decode receives the request's KV/state through the P-to-D connector. Optional Decode-side pool loading/writing is not enabled by these examples.
- Do not copy DeepSeek-specific cache block sizes into Qwen3.8. Its hybrid attention state requires compatible model and connector implementations. The KV Cache Pool guide excludes hybrid-attention models from `kv_load_failure_policy=recompute`; do not add that fallback to these commands.
- The A3 HCCS storage ownership differs from the embedded 950 DT example: only its standalone process enables SSD and contributes DRAM. Do not give both the vLLM clients and the standalone process ownership of the same disks.

## 6 Functional Verification

Use the OpenAI-compatible `/v1/chat/completions` API to verify generation. Use the API address and served model name for your deployment. Replace the host, port, and model name in the requests below as needed:

| Deployment | API address | Model name |
| --- | --- | --- |
| A2 series colocated | `http://127.0.0.1:8088` | `qwen3.8-flash-next` |
| A3 series colocated | `http://127.0.0.1:8088` | `qwen3.8-flash-next` |
| Ascend 950 DT colocated | `http://127.0.0.1:8087` | `qwen3.8-flash-next-950-dt` |
| A2 series PD proxy | `http://127.0.0.1:25570` | `qwen38-flash-next-a2-pd` |
| A3 series PD proxy | `http://127.0.0.1:25570` | `qwen38-flash-next-a3-pd` |
| Ascend 950 DT PD proxy | `http://<LOCAL_SERVICE_IP>:8087` | `qwen38-flash-next-950-dt-pd` |
| Ascend 950 DT image-and-text 10K PD proxy | `http://127.0.0.1:8246` | `qwen38-megamoe-sp-pd` |

For PD, complete the proxy check in [Section 5.2](#52-basic-pd-separation-deployment), or [Section 5.3](#53-image-and-text-10k-pd-separation-deployment) for the image-and-text 10K case. For three-tier pooling, also complete the SSD eviction and restore checks in [Section 5.4](#54-pd-separation-with-memcache-three-tier-kv-cache-pool).

### 6.1 Basic Chat Completion

```bash
curl "http://127.0.0.1:8088/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.8-flash-next",
        "messages": [
            {"role": "user", "content": "Who are you?"}
        ],
        "max_tokens": 1024,
        "temperature": 0
    }'
```

Expected result: the service returns HTTP 200 OK with a generated answer in `choices[0].message.content`.

### 6.2 Multimodal Chat Completion

Use a W4A8 checkpoint that includes the vision encoder to verify image-and-text input. Replace `<IMAGE_URL>` with an image URL accessible from the serving environment:

```bash
curl "http://127.0.0.1:8088/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.8-flash-next",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "<IMAGE_URL>"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Describe this image."
                    }
                ]
            }
        ],
        "max_tokens": 1024,
        "temperature": 0
    }'
```

Expected result: the service returns HTTP 200 OK and describes the supplied image in `choices[0].message.content`.

### 6.3 Function Calling

Function Calling requires the following startup options, which are included in the colocated examples in Section 5; for PD separation, see [Section 5.2](#52-basic-pd-separation-deployment):

```text
--enable-auto-tool-choice --tool-call-parser qwen3_xml
```

Send a request containing the available tools and set `tool_choice` to `auto`:

```bash
curl "http://127.0.0.1:8088/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.8-flash-next",
        "messages": [
            {
                "role": "user",
                "content": "What is the weather in Beijing?"
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name"
                            }
                        },
                        "required": ["city"]
                    }
                }
            }
        ],
        "tool_choice": "auto",
        "max_tokens": 1024,
        "temperature": 0
    }'
```

When the model chooses the function, the parsed result appears in `choices[0].message.tool_calls`. The caller must execute the function and send its result back to the model in a follow-up message.

### 6.4 Reasoning Parser and Thinking Control

Reasoning parsing requires the following startup option, which is included in the colocated examples in Section 5; for PD separation, see [Section 5.2](#52-basic-pd-separation-deployment):

```text
--reasoning-parser qwen3
```

Qwen3.8-Flash-Next uses thinking mode by default. To request reasoning explicitly, set `enable_thinking` to `true` (or omit it). The parser returns the reasoning separately from the final answer:

```bash
curl "http://127.0.0.1:8088/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.8-flash-next",
        "messages": [
            {
                "role": "user",
                "content": "Who are you?"
            }
        ],
        "max_tokens": 1024,
        "temperature": 0,
        "top_p": 1,
        "chat_template_kwargs": {
            "enable_thinking": true
        }
    }'
```

The final answer is returned in `choices[0].message.content`, while the parsed reasoning is returned separately in the `reasoning_content` field of the response.

To disable thinking and request a direct answer, use the following request:

```bash
curl "http://127.0.0.1:8088/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.8-flash-next",
        "messages": [
            {
                "role": "user",
                "content": "Who are you?"
            }
        ],
        "max_tokens": 1024,
        "temperature": 0,
        "top_p": 1,
        "chat_template_kwargs": {
            "enable_thinking": false
        }
    }'
```

## 7 Accuracy Evaluation

### 7.1 Using AISBench

Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for evaluation setup and usage. The result below is for reference only.

The result below is from the original A3 series W8A8 TP8 × DP1 deployment. Evaluate the W4A8 and W4A8-MXFP configurations separately.

| Hardware | Dataset | Metric | Score |
| --- | --- | --- | --- |
| A3 series | GPQA Diamond | Accuracy | 90.91 |

## 8 Performance Evaluation

Use the endpoint and served model name for the deployment under test. Record request throughput, output token throughput, time to first token (TTFT), and time per output token (TPOT). Compare configurations using the same input/output lengths and request concurrency.

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### 8.2 Using vLLM Benchmark

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

## 9 Performance Tuning

### 9.1 Recommended Configurations

Use the configuration for your hardware and deployment mode. The table summarizes the commands in [Section 5](#5-online-service-deployment); tune context length, concurrency, and memory utilization for the target workload.

| Hardware | Deployment / role | Devices | TP × PP × DP | Max model length | Max sequences | Batch tokens |
| --- | --- | --- | --- | --- | --- | --- |
| A2 series | Colocated | 0–7 | 8 × 1 × 1 | 131072 | 8 | 4096 |
| A3 series | Colocated | 0–7 | 8 × 1 × 1 | 131072 | 8 | 4096 |
| Ascend 950 DT | Colocated | 0–3 | 4 × 1 × 1 | 141312 | 32 | 4096 |
| A2 series | Prefill | 0–7 on prefill host | 2 × 1 × 4 | 141312 | 32 | 8192 |
| A3 series | Prefill | 0–7 | 4 × 2 × 1 | 141312 | 32 | 8192 |
| Ascend 950 DT | Prefill | 0–3 | 2 × 1 × 2 | 141312 | 64 | 4096 |
| A2 series | Decode | 0–7 on decode host | 2 × 1 × 4 | 141312 | 32 | 8192 |
| A3 series | Decode | 8–15 | 2 × 1 × 4 | 141312 | 32 | 8192 |
| Ascend 950 DT | Decode | 4–7 | 2 × 1 × 2 | 141312 | 64 | 4096 |
| Ascend 950 DT | Prefill (image-and-text 10K) | 0–3 | 2 × 2 × 1 | 143360 | 64 | 8192 |
| Ascend 950 DT | Decode (image-and-text 10K) | 4–7 | 2 × 1 × 2 | 143360 | 64 | 8192 |

For A3 series PD deployment, use the Prefill and Decode rows together: each role occupies eight devices, for 16 devices in total. Prefill uses eager execution with PP2; Decode enables static kernels with TP2 × DP4.

For A3 series and Ascend 950 DT MemCache three-tier pooling, retain the basic PD token budgets above and apply the hardware-specific storage and communication settings in [Section 5.4](#54-pd-separation-with-memcache-three-tier-kv-cache-pool). Pool and UBS IO memory must be budgeted separately from Engram CPU offload.

### 9.2 Tuning Guidelines

Refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for tuning methods and the [Feature Matrix](../../user_guide/support_matrix/feature_matrix.md) for feature descriptions.

#### 9.2.1 Tool-Assisted Tuning with msmodeling

MindStudio-Modeling (msmodeling) tunes serving and benchmark parameters using measured results and its knowledge base.

For instructions, see [Optix-for-PD-disaggregation](https://gitcode.com/Ascend/msmodeling/blob/master/docs/en/quick_start/optix_agent_pd_disaggregation.md).

### 9.3 Model Runner V2 Decode Submission

For supported Qwen3.8-Flash-Next attention implementations, Model Runner V2 reduces CPU synchronization during decode by keeping exact token counts on the device and using estimated upper bounds on the CPU. This optimization is enabled automatically and requires no configuration switch. The resolved target architecture must be `Qwen4ExpForCausalLM` or `Qwen4ExpForConditionalGeneration`. Other models retain their existing behavior.

After KV cache initialization, the model checks whether the target and MTP draft attention implementations support this optimization. If supported, it skips copying corrected token counts from the device to the host and the CPU wait for that copy. Exact positions, accepted token counts, KV cache slots, and recurrent state continue to use device-side values. KV-transfer connector behavior is unchanged.

This optimization also applies to text and ordinary non-realtime multimodal requests.

Unknown target/draft consumers, context parallelism, sparse KV offload and realtime media retain exact CPU synchronization. The runner's existing PP exact-count protection still applies. There is no extra TP8, DP1, PP1, SP-off, prefix-cache-off, PLE weight-offload, FULL-graph or fixed-K requirement. Underlying model/runtime restrictions still apply.

The decision is fixed for the model state's initialized lifetime so production and consumption cannot switch independently between batches. Changes to runtime configuration or attention consumers require model-state reinitialization. Initialization drains the old copy stream once before enabling the exemption.

## 10 FAQ

For common environment, installation, and parameter issues, refer to the [Public FAQs](../../faqs.md).
