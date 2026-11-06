# Qwen3.8-Flash-Next

## 1 Introduction

!!! note

    The "PD Separation Deployment" mentioned in this document is provided only as an example/reference design to demonstrate the basic process of deploying Prefill and Decode in separated mode. It is not recommended for use in the production environment.

    **For the production environment on the customer side, [MindIE Motor](https://www.hiascend.com/document/detail/en/mindie/latest/quickstart/docs/en/user_guide/quick_start.md) + vLLM-Ascend is recommended.**

Qwen3.8-Flash-Next is a multimodal Mixture-of-Experts (MoE) model and an experimental preview of the architecture that will underpin Qwen4. Its language model combines Gated DeltaNet (GDN) and Qwen Sparse Attention (QSA), gated residual connections, Position Learning Enhancement (PLE), and a native Multi-Token Prediction (MTP) head.

This tutorial covers model preparation, installation, colocated and prefill/decode (PD) deployment, functional verification, and evaluation on A2 series, A3 series, and Ascend 950 DT. Section 5.4 describes basic PD deployment with a MemCache three-tier KV cache pool on A2 series, A3 series, and Ascend 950 DT.

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

Select the image for your target hardware.

!!! note

    Run Docker on the host: `IMAGE` selects the image, and `NAME` sets the container name used by `docker exec`. The documentation build supplies the image version; when copying the Markdown source, replace the version template with the release version of the documentation you are using.

=== "A2 series"

    ```bash
    # Reuse the host model cache at /root/.cache.
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a2
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
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3
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
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a5
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

The A2/A3 `/etc/hccn.conf` mount supplies the communication configuration for the PD examples; check that the host file exists, and omit this mount if running only colocated deployment.

Ensure the CANN and ATB environments are initialized in the serving shell. If the image does not initialize them automatically, source the corresponding `set_env.sh` scripts from their installation directories. Then verify that vLLM and vLLM-Ascend can be imported:

```bash
python -c "import vllm, vllm_ascend; print('vllm and vllm-ascend are ready')"
```

### 4.2 Source Code Installation

For source installation on A2 series, A3 series, or Ascend 950 DT, follow the [installation guide](../../getting_started/installation.md#installation-existing-cann-install).

Initialize the CANN and ATB environments in the serving shell as described in the installation guide. For multi-node deployment, set up the environment on each node.

### 4.3 Docker for Three-Tier PD Pooling

Use the following containers for the A2 series, A3 series, and Ascend 950 DT basic PD pooling configuration in [Section 5.4](#54-pd-separation-with-memcache-three-tier-kv-cache-pool). For colocated, ordinary PD, and image-and-text 10K deployment, use Section 4.1. The commands below retain `--privileged=true` from the [MemCache Docker example](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#step-1-prerequisite-checks); this tutorial has not validated a configuration with reduced container privileges.

The examples below use one container per host for MetaService, the P/D engines with their embedded LocalServices, and the proxy. Run the following preparation on the **host**, then choose the matching hardware tab; Section 5.4 starts the services inside that container.

For A2, repeat the host preparation and create one container on each of the two hosts from Section 5.2. Each host uses its own SSD and personal configuration directory. Run Prefill, MetaService, and Proxy on the prefill host, and Decode on the decode host; both hosts connect to the same MetaService and ConfigStore.

!!! note "A2 host HDK requirement"

    A2 three-tier pooling requires **HDK 25.5.1 or later on both hosts**. HDK is installed on the host, not in the Docker image. Run `npu-smi info` on each host to check the version; if it is older, follow the [HDK upgrade guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/softwareinst/instg/instg_0029.html) and check again before starting the containers.

**Prepare the SSD device on the host:**

```bash
lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS
```

Choose a dedicated device with no partitions, mount points, filesystem signatures, or data to retain. Do not use a system disk. If a suitable device is already available, skip the loop-file commands below.

For a test without a dedicated disk, select a new backing-file path on an existing SSD filesystem. Set `LOOP_BACKING_SIZE` to the capacity planned for this deployment, including its unit. Its parent directory must exist and have sufficient free space:

```bash
export LOOP_BACKING_FILE="<ABSOLUTE_PATH_TO_NEW_BACKING_FILE>"
export LOOP_BACKING_SIZE="<BACKING_FILE_SIZE_WITH_UNIT>"
df -h "$(dirname "${LOOP_BACKING_FILE}")"
test ! -e "${LOOP_BACKING_FILE}" || exit 1
sudo fallocate -l "${LOOP_BACKING_SIZE}" "${LOOP_BACKING_FILE}"
sudo losetup --find --show --direct-io=on "${LOOP_BACKING_FILE}"
```

Record the device returned by `losetup` and use that device path for `SSD_DEVICE_PATH` below; `LOOP_BACKING_FILE` is the host backing file, not the device path. A loop device is a testing alternative, not an additional requirement for a deployment with a dedicated disk.

**Prepare the personal configuration directory and device mapping:**

Set `MEMCACHE_CONFIG_DIR` to your chosen absolute host directory for personal configuration copies, separate from the image's package-template directory, and `SSD_DEVICE_PATH` to the actual dedicated or loop device. Run these commands in the same host terminal used for `docker run`:

```bash
export MEMCACHE_CONFIG_DIR="<ABSOLUTE_PERSONAL_CONFIG_DIR>"
export SSD_DEVICE_PATH="<ACTUAL_SSD_DEVICE_PATH>"
mkdir -p "${MEMCACHE_CONFIG_DIR}"
test -b "${SSD_DEVICE_PATH}" || exit 1
lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS "${SSD_DEVICE_PATH}"
sudo blkid "${SSD_DEVICE_PATH}"
```

Proceed only after confirming that the selected device has no filesystem signature or mount point; `blkid` normally prints nothing and returns a nonzero status for a blank device. The commands below map one device. For multiple devices, add one `--device` mapping per device and separate the matching `ubsio.disk.path` entries with `:`.

!!! warning

    In Section 5.4, `MEMCACHE_FORCE_NEW_DISK` controls `ubsio.standalone.force_new_disk`. Setting it to `true` reinitializes cache storage at startup. Select its value for the target deployment and use only dedicated cache storage.

!!! note

    Set `MEMCACHE_CONFIG_DIR` with the host-side `export` above before running Docker: `-e MEMCACHE_CONFIG_DIR` passes its value into the container, while `-v "${MEMCACHE_CONFIG_DIR}:${MEMCACHE_CONFIG_DIR}"` mounts the actual directory at the same path. Inside the container, Section 5.4 uses this variable to prepare the personal configuration files and set `MMC_META_CONFIG_PATH` and `MMC_LOCAL_CONFIG_PATH`; the image's original templates remain unchanged.

`SSD_DEVICE_PATH` is expanded by the host shell in `--device`; it is not passed into the container as an environment variable. In Section 5.4, set `MEMCACHE_SSD_PATHS` to the same mapped paths for the selected hardware platform.

Verify that all host device and file paths exist before running Docker. The 950 DT example retains the device, driver, and communication-tool mounts from the [MemCache backend guide](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#scenario-2-memcache-backend).

=== "A2 series"

    ```bash
    # Reuse the host model cache at /root/.cache.
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a2
    export NAME=vllm-ascend-qwen38-pool

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
      --device /dev/davinci_manager \
      --device "${SSD_DEVICE_PATH}:${SSD_DEVICE_PATH}" \
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

=== "A3 series"

    ```bash
    # Reuse the host model cache at /root/.cache.
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3
    export NAME=vllm-ascend-qwen38-pool

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
      --device "${SSD_DEVICE_PATH}:${SSD_DEVICE_PATH}" \
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
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a5
    export NAME=vllm-ascend-qwen38-pool

    docker run --rm \
      --name $NAME \
      -e MEMCACHE_CONFIG_DIR \
      --net=host \
      --privileged=true \
      --shm-size=500g \
      --device /dev/davinci_manager \
      --device "${SSD_DEVICE_PATH}:${SSD_DEVICE_PATH}" \
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

Keep this container running. Use Section 5.4 to prepare the configurations and start MetaService, P/D, and Proxy in order. To open another shell in the same container, run the following from a new **host** terminal:

```bash
docker exec -it vllm-ascend-qwen38-pool bash
```

The container inherits `MEMCACHE_CONFIG_DIR`, including in new `docker exec` shells, but exports made inside one shell are not shared with other shells. Repeat the library and service-specific exports from Section 5.4 in each service terminal; if you change `NAME` when creating the container, use that name in the `docker exec` command above.

Keep `MEMCACHE_CONFIG_DIR` set to the path passed at container creation; changing the host variable later does not update the running container's environment or mounts. Files written to this mounted directory from inside the container are the same personal copies visible on the host.

After entering the container, check that the model and storage dependencies are available:

```bash
python -c "import vllm, vllm_ascend, mooncake, memcache_hybrid, memfabric_hybrid; print('Model and pool dependencies are available')"
```

An import check does not validate version compatibility; complete the MemCache dependency checks in Section 5.4. `--shm-size` controls the container's `/dev/shm` limit and does not allocate the MemCache DRAM pool or the UBS IO buffers.

!!! note "950 DT pooling runtime"

    Use a runtime that supports Qwen3.8-Flash-Next and MemCache three-tier pooling. Initialize CANN/ATB as described in Section 4.1. If the image does not initialize its repository, custom operators, and model libraries automatically, apply the following in each service shell; use actual container paths, with library directories separated by `:`.

    ```bash
    export ASCEND_REPO="<VLLM_ASCEND_REPOSITORY_IN_CONTAINER>"
    export MODEL_RUNTIME_LIB_DIRS="<MOONCAKE_TORCH_AND_TORCH_NPU_LIBRARY_DIRECTORIES>"
    export PYTHONPATH="${ASCEND_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
    source "${ASCEND_REPO}/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash"
    export LD_LIBRARY_PATH="${MODEL_RUNTIME_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
    ```

    Resolve the model-library directories from the active environment's installed Mooncake, Torch, and torch_npu packages. Section 5.4 supplies the separate MemCache/MemFabric library paths and retains the reference process limits (`ulimit -l unlimited`, `ulimit -s 65536`).

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

!!! note

    Set `PREFILL_HOST`, `DECODE_HOST`, `local_ip`, and `nic_name` where shown in each command block; these are shell variables, and a new terminal does not inherit values assigned in another terminal. The `VLLM_HOST_IP`/`HCCL_IF_IP` and socket-interface exports use those values for communication, while `no_proxy`/`NO_PROXY` bypass HTTP proxies for the listed service addresses.

=== "A2 series"

    This W4A8 deployment uses two eight-device A2 series hosts. Prefill runs on `PREFILL_HOST` and Decode runs on `DECODE_HOST`, each using devices 0–7. Prefill uses TP4 × PP2 × DP1, and Decode uses TP2 × DP4, matching the A3 model topology. Both roles load the same checkpoint; the proxy routes requests through one API endpoint per role (1P1D).

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

    unset VLLM_PP_LAYER_PARTITION
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
    PREFILL_HOST="<PREFILL_LOCAL_IP>"
    DECODE_HOST="<DECODE_LOCAL_IP>"
    local_ip="${DECODE_HOST}"
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

    Check the startup logs for TP4 × PP2 × DP1 on Prefill and TP2 × DP4 with `enable_static_kernel=true` on Decode. From the prefill host, check that both APIs list `qwen38-flash-next-a2-pd`:

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

    - Both roles enable prefix caching and explicitly set `--mamba-cache-mode align` and `--mamba-ssm-cache-dtype bfloat16`.
    - Engram requires a checkpoint with supported n-gram layers. `cpu_offload=true` keeps Engram embedding tables in host memory. `dp_shared_memory=true` shares the corresponding table shards among node-local DP replicas; replicas must share the same host and shared-memory namespace. Both options are booleans. Prefill uses DP1 and sets `dp_shared_memory=false`; Decode uses DP4 and sets it to `true`. Elastic EP is not supported with shared memory.
    - Prefill distributes the model across two pipeline stages, with TP4 in each stage. It runs in eager mode (`--enforce-eager`), enables `enable_prefill_mc2=true`, and disables fused MC2. Verify MC2 token limits for TP4 × PP2 × DP1 before increasing the batch token budget.
    - Decode uses four TP2 replicas without pipeline parallelism. It enables `FULL_DECODE_ONLY`, NPU graph extension, `enable_static_kernel=true`, and recompute scheduling. Static kernels are enabled for Decode; Prefill retains eager execution. Super kernels remain disabled.
    - Both roles use three MTP draft tokens; Prefill sets MTP `enforce_eager=true` and Decode sets it to `false`. Both roles use `--max-num-batched-tokens 8192`; Prefill retains `HCCL_BUFFSIZE=8192` and Decode retains `HCCL_BUFFSIZE=512`.
    - `enable_fused_mc2=0` disables fused MC2, so it does not conflict with `multistream_overlap_shared_expert=true`. The decode command relies on the runtime's default value of zero.
    - Keep the Mooncake declarations synchronized on both roles: Prefill uses DP1 / TP4 / PP2, and Decode uses DP4 / TP2. No custom layer partition is specified; clear `VLLM_PP_LAYER_PARTITION` in both terminals to use the runtime's default partition.
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

`MultiConnector` combines Mooncake P-to-D transport with MemCache-backed KV caching in HBM, DRAM, and SSD. A2/A3 reuse the model commands from [Section 5.2](#52-basic-pd-separation-deployment); the 950 DT tab contains its complete pooling commands. This section does not cover colocated deployment or the image-and-text 10K case.

**Before starting:**

Start the matching container from [Section 4.3](#43-docker-for-three-tier-pd-pooling), with its dedicated SSD mapping and `MEMCACHE_CONFIG_DIR`. Run the following steps inside the container. A2 requires HDK ≥ 25.5.1 on both hosts; A3 HCCS requires LingQu Computing Network 1.5 or later. Follow the [MemCache backend guide](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#scenario-2-memcache-backend) for compatible runtime dependencies.

```bash
free -h
python -m pip show memcache-hybrid memfabric-hybrid
```

Both packages must be at least version 1.3.0. If missing, install compatible versions and repeat the import check in Section 4.3:

```bash
python -m pip install "memcache-hybrid>=1.3.0" "memfabric-hybrid>=1.3.0"
```

Use the package `Location` reported above to set the library directories. Run these exports in every service terminal, preserving the initialized model runtime's library paths:

```bash
export MEMCACHE_LIB_DIR="<MEMCACHE_LIBRARY_DIRECTORY>"
export MEMFABRIC_LIB_DIR="<MEMFABRIC_LIBRARY_DIRECTORY>"
export PYTHON_LIB_DIR="$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
export LD_LIBRARY_PATH="${MEMCACHE_LIB_DIR}:${MEMFABRIC_LIB_DIR}:${PYTHON_LIB_DIR}:${LD_LIBRARY_PATH:-}"
```

`MEMCACHE_LIB_DIR` and `MEMFABRIC_LIB_DIR` are the respective package `Location` plus `/memcache_hybrid/lib` and `/memfabric_hybrid/lib`. Set the template directory in the configuration-editing terminal:

```bash
export MEMCACHE_TEMPLATE_DIR="<IMAGE_MEMCACHE_CONFIG_DIR>"
ls "${MEMCACHE_TEMPLATE_DIR}"
```

The template path varies with the image and Python version; `/usr/local/python3.11.10/lib/python3.11/site-packages/memcache_hybrid/config/` is only an example inside the container. Copy templates into your personal mounted `MEMCACHE_CONFIG_DIR`. `cp -n` preserves existing files, while `cat >` replaces them; retain required image-specific settings when editing an existing configuration.

**A2/A3 configuration preparation:**

For A2, prepare LocalService files on both hosts and MetaService only on the prefill host. For A3, prepare them in the single serving container. Both roles must use the same reachable MetaService and ConfigStore endpoints. When reusing a running pool, retain its MetaService configuration and skip its creation and startup.

```bash
cp -n "${MEMCACHE_TEMPLATE_DIR}/mmc-local.conf" "${MEMCACHE_CONFIG_DIR}/mmc-local.conf"
export MEMCACHE_META_URL="tcp://<META_SERVICE_IP>:<META_SERVICE_PORT>"
export MEMCACHE_CONFIG_STORE_URL="tcp://<CONFIG_STORE_IP>:<CONFIG_STORE_PORT>"
```

For a new pool, prepare MetaService:

```bash
cp -n "${MEMCACHE_TEMPLATE_DIR}/mmc-meta.conf" "${MEMCACHE_CONFIG_DIR}/mmc-meta.conf"
export MEMCACHE_METRICS_URL="http://<METRICS_IP>:<METRICS_PORT>"
export MEMCACHE_EVICT_HIGH="<HIGH_EVICTION_THRESHOLD_PERCENT>"
export MEMCACHE_EVICT_LOW="<LOW_EVICTION_THRESHOLD_PERCENT>"

cat > "${MEMCACHE_CONFIG_DIR}/mmc-meta.conf" <<EOF
ock.mmc.meta_service_url = ${MEMCACHE_META_URL}
ock.mmc.meta_service.config_store_url = ${MEMCACHE_CONFIG_STORE_URL}
ock.mmc.meta_service.metrics_url = ${MEMCACHE_METRICS_URL}
ock.mmc.log_level = info
ock.mmc.evict_threshold_high = ${MEMCACHE_EVICT_HIGH}
ock.mmc.evict_threshold_low = ${MEMCACHE_EVICT_LOW}
EOF
```

Set each host's storage values from its deployment configuration and memory budget. Use the device paths mapped in Section 4.3, separated by `:` for multiple devices:

```bash
export MEMCACHE_SSD_PATHS="<MAPPED_SSD_DEVICE_PATHS_SEPARATED_BY_COLONS>"
export MEMCACHE_WORLD_SIZE="<POOL_LOCAL_SERVICE_CAPACITY>"
export MEMCACHE_DRAM_SIZE="<DRAM_SIZE_PER_LOCAL_SERVICE_WITH_UNIT>"
export MEMCACHE_MAX_DRAM_SIZE="<MAX_DRAM_SIZE_WITH_UNIT>"
export MEMCACHE_UBS_IO_MEM_GB="<UBS_IO_MEMORY_GB_PER_LOCAL_SERVICE>"
export MEMCACHE_DEVICE_COUNT="<DRAM_ENABLED_LOCAL_SERVICE_COUNT_ON_THIS_HOST>"
export MEMCACHE_FORCE_NEW_DISK="<true_OR_false_FOR_THIS_DEPLOYMENT>"
```

`dram.size` and `ubsio.mem.size_in_gb` apply per DRAM-enabled LocalService. Budget their host-wide total alongside Engram, vLLM, and operating-system memory. `world_size` is pool capacity; `device_count` is the host's DRAM-enabled LocalService count, not the SSD count. The file-writing commands expand these variables into literal configuration values.

In every A2/A3 P/D terminal, apply the library exports above and these common exports before the hardware-specific settings:

```bash
export VLLM_USE_MODELSCOPE=True
export PYTHONHASHSEED=0
export MMC_LOCAL_CONFIG_PATH="${MEMCACHE_CONFIG_DIR}/mmc-local.conf"
```

=== "A2 series"

    Keep the A2 model configuration from Section 5.2: TP4 × PP2 × DP1 for Prefill and TP2 × DP4 for Decode, on separate eight-device hosts. On each host, use the common MetaService and ConfigStore endpoints and pool `world_size`, but set SSD paths and memory values for that host. Write the RoCE configuration:

    ```bash
    : "${MEMCACHE_CONFIG_DIR:?Set the personal configuration directory in Section 4.3}"

    cat > "${MEMCACHE_CONFIG_DIR}/mmc-local.conf" <<EOF
    ock.mmc.meta_service_url = ${MEMCACHE_META_URL}
    ock.mmc.local_service.config_store_url = ${MEMCACHE_CONFIG_STORE_URL}
    ock.mmc.log_level = info
    ock.mmc.local_service.world_size = ${MEMCACHE_WORLD_SIZE}
    ock.mmc.local_service.protocol = device_rdma
    ock.mmc.local_service.dram.size = ${MEMCACHE_DRAM_SIZE}
    ock.mmc.local_service.max.dram.size = ${MEMCACHE_MAX_DRAM_SIZE}
    ock.mmc.local_service.storage.enabled = true
    ubsio.disk.path = ${MEMCACHE_SSD_PATHS}
    ubsio.mem.size_in_gb = ${MEMCACHE_UBS_IO_MEM_GB}
    ubsio.standalone.device_count = ${MEMCACHE_DEVICE_COUNT}
    ubsio.standalone.force_new_disk = ${MEMCACHE_FORCE_NEW_DISK}
    EOF
    ```

    The `ubsio.standalone.*` key names do not require a separate standalone service.

    Export the following in the Prefill and Decode terminals. Section 5.2 sets each host's IP and socket interfaces; clear incompatible HCCS/UB settings inherited from another experiment:

    ```bash
    unset ASCEND_ENABLE_USE_FABRIC_MEM ASCEND_GLOBAL_RESOURCE_CONFIG ASCEND_LOCAL_COMM_RES
    export HCCL_INTRA_ROCE_ENABLE=1
    ```

    **Prefill connector argument:**

    ```text
    --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_producer","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_producer","kv_port":32600,"kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":4,"pp_size":2},"decode":{"dp_size":4,"tp_size":2}}},{"kv_connector":"AscendStoreConnector","kv_role":"kv_producer","kv_connector_extra_config":{"backend":"memcache","lookup_rpc_port":"qwen38_pool_p","use_layerwise":false}}]}}'
    ```

    **Decode connector argument:**

    ```text
    --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_consumer","kv_port":32700,"kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":4,"pp_size":2},"decode":{"dp_size":4,"tp_size":2}}},{"kv_connector":"AscendStoreConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"backend":"memcache","lookup_rpc_port":"qwen38_pool_d","use_layerwise":false}}]}}'
    ```

    **Start the services:**

    Start MetaService on the prefill host with the library exports above, unless the pool is already running:

    ```bash
    export MMC_META_CONFIG_PATH="${MEMCACHE_CONFIG_DIR}/mmc-meta.conf"
    python -c "from memcache_hybrid import MetaService; MetaService.main()"
    ```

    After its configured ports are listening (`ss -lntp`), run the matching Section 5.2 Prefill and Decode commands in separate terminals, replacing only `--kv-transfer-config` with the arguments above. Keep all other model parameters unchanged. Complete the Section 5.2 API checks, start its Proxy, then perform the cache-reuse checks below.

=== "A3 series"

    Keep the A3 model configuration from Section 5.2. Use the endpoints and resource values set above to write the HCCS configuration:

    ```bash
    : "${MEMCACHE_CONFIG_DIR:?Set the personal configuration directory in Section 4.3}"

    cat > "${MEMCACHE_CONFIG_DIR}/mmc-local.conf" <<EOF
    ock.mmc.meta_service_url = ${MEMCACHE_META_URL}
    ock.mmc.local_service.config_store_url = ${MEMCACHE_CONFIG_STORE_URL}
    ock.mmc.log_level = info
    ock.mmc.local_service.world_size = ${MEMCACHE_WORLD_SIZE}
    ock.mmc.local_service.protocol = device_sdma
    ock.mmc.local_service.dram.size = ${MEMCACHE_DRAM_SIZE}
    ock.mmc.local_service.max.dram.size = ${MEMCACHE_MAX_DRAM_SIZE}
    ock.mmc.local_service.storage.enabled = true
    ubsio.disk.path = ${MEMCACHE_SSD_PATHS}
    ubsio.mem.size_in_gb = ${MEMCACHE_UBS_IO_MEM_GB}
    ubsio.standalone.device_count = ${MEMCACHE_DEVICE_COUNT}
    ubsio.standalone.force_new_disk = ${MEMCACHE_FORCE_NEW_DISK}
    EOF
    ```

    The `ubsio.standalone.*` key names do not require a separate standalone service.

    Export the following in the Prefill and Decode terminals. Clear incompatible RoCE/UB settings inherited from another experiment:

    ```bash
    unset HCCL_INTRA_ROCE_ENABLE ASCEND_GLOBAL_RESOURCE_CONFIG ASCEND_LOCAL_COMM_RES
    export ACL_OP_INIT_MODE=1
    export ASCEND_ENABLE_USE_FABRIC_MEM=1
    ```

    **Prefill connector argument:**

    ```text
    --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_producer","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_producer","kv_port":32600,"kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":4,"pp_size":2},"decode":{"dp_size":4,"tp_size":2}}},{"kv_connector":"AscendStoreConnector","kv_role":"kv_producer","kv_connector_extra_config":{"backend":"memcache","lookup_rpc_port":"qwen38_pool_p","use_layerwise":false}}]}}'
    ```

    **Decode connector argument:**

    ```text
    --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_consumer","kv_port":32700,"kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":4,"pp_size":2},"decode":{"dp_size":4,"tp_size":2}}},{"kv_connector":"AscendStoreConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"backend":"memcache","lookup_rpc_port":"qwen38_pool_d","use_layerwise":false}}]}}'
    ```

    **Start the services:**

    Start MetaService in the serving container with the library exports above, unless the pool is already running:

    ```bash
    export MMC_META_CONFIG_PATH="${MEMCACHE_CONFIG_DIR}/mmc-meta.conf"
    python -c "from memcache_hybrid import MetaService; MetaService.main()"
    ```

    After its configured ports are listening (`ss -lntp`), run the matching Section 5.2 Prefill and Decode commands in separate terminals, replacing only `--kv-transfer-config` with the arguments above. Keep all other model parameters unchanged. Complete the Section 5.2 API checks, start its Proxy, then perform the cache-reuse checks below.

=== "Ascend 950 DT"

    This single-host example uses TP2 × DP2 per role, with Prefill on devices 0–3 and Decode on devices 4–7. Use the prepared runtime from Section 4.3 and the same W4A8-MXFP checkpoint for both roles.

    **Prepare the configuration files:**

    In the configuration-editing terminal, set the actual pool communication IP and mapped SSD paths. The reference uses a dedicated 500 GiB SSD-backed loop device; take its device path from Section 4.3. For an existing pool, retain its MetaService configuration and endpoints and skip MetaService creation/startup.

    ```bash
    local_ip="<POOL_COMMUNICATION_IP>"
    export MEMCACHE_META_URL="tcp://${local_ip}:25250"
    export MEMCACHE_CONFIG_STORE_URL="tcp://${local_ip}:25251"
    export MEMCACHE_METRICS_URL="http://${local_ip}:25252"
    export MEMCACHE_EVICT_HIGH=70
    export MEMCACHE_EVICT_LOW=60
    export MEMCACHE_SSD_PATHS="<MAPPED_SSD_DEVICE_PATHS_SEPARATED_BY_COLONS>"
    cp -n "${MEMCACHE_TEMPLATE_DIR}/mmc-meta.conf" "${MEMCACHE_CONFIG_DIR}/mmc-meta.conf"
    cp -n "${MEMCACHE_TEMPLATE_DIR}/mmc-local.conf" "${MEMCACHE_CONFIG_DIR}/mmc-local-p.conf"
    cp -n "${MEMCACHE_TEMPLATE_DIR}/mmc-local.conf" "${MEMCACHE_CONFIG_DIR}/mmc-local-d.conf"

    cat > "${MEMCACHE_CONFIG_DIR}/mmc-meta.conf" <<EOF
    ock.mmc.meta_service_url = ${MEMCACHE_META_URL}
    ock.mmc.meta_service.config_store_url = ${MEMCACHE_CONFIG_STORE_URL}
    ock.mmc.meta_service.metrics_url = ${MEMCACHE_METRICS_URL}
    ock.mmc.log_level = info
    ock.mmc.log_output_target = screen
    ock.mmc.evict_threshold_high = ${MEMCACHE_EVICT_HIGH}
    ock.mmc.evict_threshold_low = ${MEMCACHE_EVICT_LOW}
    EOF
    ```

    Write the P/D LocalService files with the same endpoints and SSD mapping:

    ```bash
    export MEMCACHE_WORLD_SIZE=256
    export MEMCACHE_DRAM_SIZE=10GB
    export MEMCACHE_MAX_DRAM_SIZE=1024GB
    export MEMCACHE_UBS_IO_MEM_GB=10
    export MEMCACHE_DEVICE_COUNT=8
    export MEMCACHE_FORCE_NEW_DISK=true

    for role in p d; do
      cat > "${MEMCACHE_CONFIG_DIR}/mmc-local-${role}.conf" <<EOF
    ock.mmc.meta_service_url = ${MEMCACHE_META_URL}
    ock.mmc.local_service.config_store_url = ${MEMCACHE_CONFIG_STORE_URL}
    ock.mmc.log_level = info
    ock.mmc.local_service.world_size = ${MEMCACHE_WORLD_SIZE}
    ock.mmc.local_service.protocol = device_urma
    ock.mmc.local_service.dram.size = ${MEMCACHE_DRAM_SIZE}
    ock.mmc.local_service.max.dram.size = ${MEMCACHE_MAX_DRAM_SIZE}
    ock.mmc.local_service.storage.enabled = true
    ubsio.disk.path = ${MEMCACHE_SSD_PATHS}
    ubsio.mem.size_in_gb = ${MEMCACHE_UBS_IO_MEM_GB}
    ubsio.standalone.device_count = ${MEMCACHE_DEVICE_COUNT}
    ubsio.standalone.force_new_disk = ${MEMCACHE_FORCE_NEW_DISK}
    EOF
    done
    ```

    The eight embedded LocalServices each contribute `10GB` of pool DRAM and reserve `10` GB for UBS IO; no separate storage process is needed. `force_new_disk=true` recreates the SSD cache at stack startup, so keep the stack running during cache-reuse tests.

    **Set the service environment:**

    Open separate MetaService, P, D, and Proxy terminals in the same container. Initialize the runtime as described in Section 4.3, apply the library exports at the beginning of this section, and run the following in each terminal. Use the same communication IP as the configuration files and its matching UB interface:

    ```bash
    local_ip="<POOL_COMMUNICATION_IP>"
    nic_name="<POOL_NETWORK_INTERFACE>"

    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY ALL_PROXY all_proxy
    unset HCCL_INTRA_ROCE_ENABLE ASCEND_ENABLE_USE_FABRIC_MEM ASCEND_GLOBAL_RESOURCE_CONFIG
    unset ASCEND_RT_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES MMC_LOCAL_CONFIG_PATH MMC_META_CONFIG_PATH
    export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
    export VLLM_HOST_IP="${local_ip}"
    export HCCL_IF_IP="${local_ip}"
    export GLOO_SOCKET_IFNAME="${nic_name}"
    export TP_SOCKET_IFNAME="${nic_name}"
    export HCCL_SOCKET_IFNAME="${nic_name}"
    export NO_PROXY="localhost,127.0.0.1,${local_ip}"
    export no_proxy="${NO_PROXY}"
    export PYTHONHASHSEED=0
    export VLLM_USE_MODELSCOPE=True
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export VLLM_ENGINE_READY_TIMEOUT_S=2400
    export HCCL_BUFFSIZE=512
    ulimit -l unlimited
    ulimit -s 65536
    ```

    **Start MetaService:**

    ```bash
    export MMC_META_CONFIG_PATH="${MEMCACHE_CONFIG_DIR}/mmc-meta.conf"
    python -c "from memcache_hybrid import MetaService; MetaService.main()"
    ```

    Wait until the configured ports `25250`, `25251`, and `25252` are listening (`ss -lntp`), then start Prefill.

    **Prefill:**

    ```bash
    export MMC_LOCAL_CONFIG_PATH="${MEMCACHE_CONFIG_DIR}/mmc-local-p.conf"
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
    export HCCL_NPU_SOCKET_PORT_RANGE=61400-61499

    vllm serve Eco-Tech/Qwen3.8-Flash-Next-w4a8-mxfp \
      --host 0.0.0.0 \
      --port 8247 \
      --served-model-name qwen38-megamoe-sp-pd \
      --trust-remote-code \
      --quantization ascend \
      --tensor-parallel-size 2 \
      --data-parallel-size 2 \
      --max-model-len 262144 \
      --max-num-seqs 64 \
      --max-num-batched-tokens 4096 \
      --gpu-memory-utilization 0.9 \
      --enable-prefix-caching \
      --async-scheduling \
      --enable-expert-parallel \
      --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
      --compilation-config '{"cudagraph_mode":"NONE"}' \
      --additional-config '{"enable_cpu_binding":true,"enable_shared_expert_dp":true,"enable_fused_mc2":0,"enable_flashcomm1":true,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"fuse_norm_quant":false,"enable_npugraph_ex":true,"enable_static_kernel":false,"enable_super_kernel":false}}' \
      --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' \
      --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_producer","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_producer","kv_port":"32470","kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":2,"pp_size":1},"decode":{"dp_size":2,"tp_size":2,"pp_size":1}}},{"kv_connector":"AscendStoreConnector","kv_role":"kv_producer","kv_connector_extra_config":{"backend":"memcache","lookup_rpc_port":"qwen38_pool_p","use_layerwise":false,"load_async":false,"consumer_is_to_load":false,"consumer_is_to_put":false,"save_decode_cache":false}}]}}' \
      --enable-auto-tool-choice \
      --tool-call-parser qwen3_coder \
      --reasoning-parser qwen3 \
      --data-parallel-address "${VLLM_HOST_IP}" \
      --data-parallel-rpc-port 29770
    ```

    Wait for the Prefill API on port `8247` to become ready, then start Decode in a new terminal.

    **Decode:**

    ```bash
    export MMC_LOCAL_CONFIG_PATH="${MEMCACHE_CONFIG_DIR}/mmc-local-d.conf"
    export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
    export HCCL_NPU_SOCKET_PORT_RANGE=61500-61599

    vllm serve Eco-Tech/Qwen3.8-Flash-Next-w4a8-mxfp \
      --host 0.0.0.0 \
      --port 8248 \
      --served-model-name qwen38-megamoe-sp-pd \
      --trust-remote-code \
      --quantization ascend \
      --tensor-parallel-size 2 \
      --data-parallel-size 2 \
      --max-model-len 262144 \
      --max-num-seqs 64 \
      --max-num-batched-tokens 4096 \
      --gpu-memory-utilization 0.9 \
      --enable-prefix-caching \
      --async-scheduling \
      --enable-expert-parallel \
      --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --additional-config '{"enable_flashcomm1":true,"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"scheduler_config":{"recompute_scheduler_enable":true},"ascend_compilation_config":{"fuse_norm_quant":false,"enable_npugraph_ex":true,"enable_static_kernel":true,"enable_super_kernel":false}}' \
      --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":false}' \
      --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_consumer","kv_port":"32480","kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":2,"pp_size":1},"decode":{"dp_size":2,"tp_size":2,"pp_size":1}}},{"kv_connector":"AscendStoreConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"backend":"memcache","lookup_rpc_port":"qwen38_pool_d","use_layerwise":false,"load_async":false,"consumer_is_to_load":false,"consumer_is_to_put":false,"save_decode_cache":false}}]}}' \
      --enable-auto-tool-choice \
      --tool-call-parser qwen3_coder \
      --reasoning-parser qwen3 \
      --data-parallel-address "${VLLM_HOST_IP}" \
      --data-parallel-rpc-port 29780
    ```

    After the Decode API on port `8248` is ready, start Proxy from the vLLM-Ascend repository root in another terminal.

    **Proxy:**

    ```bash
    python examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
        --host 0.0.0.0 --port 8249 \
        --prefiller-hosts "${VLLM_HOST_IP}" --prefiller-ports 8247 \
        --decoder-hosts "${VLLM_HOST_IP}" --decoder-ports 8248 \
        --workers 1
    ```

    **Service verification:**

    ```bash
    curl --fail "http://127.0.0.1:8249/v1/models"
    curl --fail "http://127.0.0.1:8249/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"qwen38-megamoe-sp-pd","messages":[{"role":"user","content":"Who are you?"}],"max_tokens":1024,"temperature":0}'
    ```

    Prefill uses synchronous external cache lookup/loading and writes. Decode receives KV/state through Mooncake; its external-cache loading and saving options remain disabled. Both roles retain the explicit connector flags shown above.

**Verify three-tier cache reuse:**

1. **Service:** Run the Section 5.2 chat-completion request for A2/A3, or the 950 DT request above, through the matching proxy and verify correct output and P-to-D transfer.
2. **External cache:** Send requests with shared prefixes and different suffixes; check MemCache lookup, put, get, and hit activity. A local prefix-cache hit alone does not prove external cache reuse.
3. **SSD restore:** Fill DRAM past the eviction watermark, confirm SSD eviction, then request an evicted prefix again and verify SSD read/restore activity and correct output. Keep MetaService and P/D running; SSD writes alone do not prove restoration.

Use the MemCache, MemFabric, and UBS IO logs or available metrics to check these results; counter names depend on the installed version.

**Key parameters:**

- Keep `PYTHONHASHSEED=0` consistent across workers. `lookup_rpc_port` identifies a local IPC endpoint with a DP-index suffix; use distinct P/D identifiers and avoid collisions with other pools.
- Retain `use_layerwise=false`: Prefill performs external prefix lookup and cache writes, while Decode receives KV/state through the P-to-D connector.

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
| Ascend 950 DT three-tier pooling PD proxy | `http://127.0.0.1:8249` | `qwen38-megamoe-sp-pd` |

For PD, complete the proxy check in [Section 5.2](#52-basic-pd-separation-deployment), or [Section 5.3](#53-image-and-text-10k-pd-separation-deployment) for the image-and-text 10K case. For three-tier pooling, use the startup and proxy checks for the matching hardware in [Section 5.4](#54-pd-separation-with-memcache-three-tier-kv-cache-pool), then complete its SSD eviction and restore checks.

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
| A2 series | Prefill | 0–7 on prefill host | 4 × 2 × 1 | 141312 | 32 | 8192 |
| A3 series | Prefill | 0–7 | 4 × 2 × 1 | 141312 | 32 | 8192 |
| Ascend 950 DT | Prefill | 0–3 | 2 × 1 × 2 | 141312 | 64 | 4096 |
| A2 series | Decode | 0–7 on decode host | 2 × 1 × 4 | 141312 | 32 | 8192 |
| A3 series | Decode | 8–15 | 2 × 1 × 4 | 141312 | 32 | 8192 |
| Ascend 950 DT | Decode | 4–7 | 2 × 1 × 2 | 141312 | 64 | 4096 |
| Ascend 950 DT | Prefill (image-and-text 10K) | 0–3 | 2 × 2 × 1 | 143360 | 64 | 8192 |
| Ascend 950 DT | Decode (image-and-text 10K) | 4–7 | 2 × 1 × 2 | 143360 | 64 | 8192 |
| Ascend 950 DT | Prefill (three-tier pooling) | 0–3 | 2 × 1 × 2 | 262144 | 64 | 4096 |
| Ascend 950 DT | Decode (three-tier pooling) | 4–7 | 2 × 1 × 2 | 262144 | 64 | 4096 |

For A2 series and A3 series PD deployment, use the Prefill and Decode rows together: each role occupies eight devices, for 16 devices in total. A2 uses two eight-device hosts; A3 uses one 16-device host. Prefill uses eager execution with TP4 × PP2 × DP1; Decode enables static kernels with TP2 × DP4.

For A2/A3 MemCache three-tier pooling, retain the basic PD token budgets above. For 950 DT, use the dedicated three-tier pooling rows and the complete commands in [Section 5.4](#54-pd-separation-with-memcache-three-tier-kv-cache-pool); both roles use `HCCL_BUFFSIZE=512`. Pool and UBS IO memory must be budgeted separately from Engram CPU offload.

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
