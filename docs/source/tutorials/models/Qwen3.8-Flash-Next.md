# Qwen3.8-Flash-Next

## 1 Introduction

!!! note

    The "PD Separation Deployment" mentioned in this document is provided only as an example/reference design to demonstrate the basic process of deploying Prefill and Decode in separated mode. It is not recommended for use in the production environment.

    **For the production environment on the customer side, [MindIE Motor](https://www.hiascend.com/document/detail/en/mindie/latest/quickstart/docs/en/user_guide/quick_start.md) + vLLM-Ascend is recommended.**

Qwen3.8-Flash-Next is a multimodal Mixture-of-Experts (MoE) model and an experimental preview of the architecture that will underpin Qwen4. Its language model combines Gated DeltaNet (GDN) and Qwen Sparse Attention (QSA), gated residual connections, Position Learning Enhancement (PLE), and a native Multi-Token Prediction (MTP) head.

This tutorial covers preparation, deployment, verification, and evaluation on A2 series, A3 series, and 950DT products, including colocated, PD separation, and MemCache three-tier pooling deployment.

Qwen3.8-Flash-Next (experimental) was first supported in `vllm-ascend:v0.26.0rc`. The examples use W4A8 checkpoints (W4A8-MXFP on 950DT products). For the best performance, it is recommended to use **vLLM-Ascend v0.32.0rc** (`releases/v0.32.0rc`).

!!! warning

    Automatic prefix caching is experimental. For A2 series and A3 series colocated deployment, enable it with `--enable-prefix-caching --mamba-cache-mode align` and revalidate NPU memory capacity for the target context length and concurrency.

## 2 Supported Features

See [supported features](../../user_guide/support_matrix/supported_features.md) for feature availability and the [Feature Guide](../../user_guide/feature_guide/index.md) for configuration details.

## 3 Prerequisites

### 3.1 Model Weight

| Weight Version | Hardware Requirements | Download Links |
| --- | --- | --- |
| `Qwen3.8-Flash-Next-w8a8-mtp` (quantized version) | A3 series | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/Qwen3.8-Flash-Next-w8a8-mtp) |
| `Qwen3.8-Flash-Next-w4a8` (quantized version) | A2 series / A3 series | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/Qwen3.8-Flash-Next-w4a8) |
| `Qwen3.8-Flash-Next-w4a8-mxfp` (quantized version) | 950DT products | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/Qwen3.8-Flash-Next-w4a8-mxfp) |

The examples use ModelScope model IDs with `VLLM_USE_MODELSCOPE=True`. Both PD roles must load the same checkpoint.

## 4 Installation

### 4.1 Docker Image Installation

Refer to [using docker](../../getting_started/installation.md#installation-prebuilt-image) for the complete installation guide.

Select the image for your target hardware.

!!! note

    Run Docker on the host. `IMAGE` selects the image; `NAME` is the container name used by `docker exec`. Image versions follow the documentation release; replace the version template when copying raw Markdown.

    The image tag template follows the documentation build. For the recommended vLLM-Ascend 0.32 RC runtime, select the matching 0.32 RC image tag for your hardware if the rendered tag is from another release. This also applies to the pooling images in Section 4.3.

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

=== "950DT products"

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

For source installation on A2 series, A3 series, or 950DT products, follow the [installation guide](../../getting_started/installation.md#installation-existing-cann-install).

Initialize the CANN and ATB environments in the serving shell as described in the installation guide. For multi-node deployment, set up the environment on each node.

### 4.3 Docker for Three-Tier PD Pooling

Use these containers for [MemCache three-tier PD pooling](#55-pd-separation-with-memcache-three-tier-kv-cache-pool). Other deployment modes use Section 4.1. Container privileges follow the [MemCache Docker example](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#step-1-prerequisite-checks).

Prepare storage on the **host**, then create one container per host using the matching hardware tab. Start the services inside the containers as described in Section 5.5.

For A2, prepare a separate SSD and configuration directory on each host. Run Prefill, MetaService, and Proxy on the prefill host, and Decode on the decode host, using shared MetaService and ConfigStore endpoints.

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

Set `SSD_DEVICE_PATH` to the device returned by `losetup`, not the backing-file path. For the A3 two-device example, prepare a second device with a different backing file before continuing. Skip loop-device creation when using dedicated disks.

When reusing an existing pool, keep its devices and backing files; skip new-device creation and initialization.

**Prepare the personal configuration directory and device mapping:**

In the host terminal used for `docker run`, set `MEMCACHE_CONFIG_DIR` to your personal configuration directory and `SSD_DEVICE_PATH` to the dedicated or loop device:

```bash
export MEMCACHE_CONFIG_DIR="<ABSOLUTE_PERSONAL_CONFIG_DIR>"
export SSD_DEVICE_PATH="<ACTUAL_SSD_DEVICE_PATH>"
mkdir -p "${MEMCACHE_CONFIG_DIR}"
test -b "${SSD_DEVICE_PATH}" || exit 1
lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS "${SSD_DEVICE_PATH}"
sudo blkid "${SSD_DEVICE_PATH}"
```

For a new pool, proceed only after confirming that each selected device has no filesystem signature or mount point; `blkid` normally prints nothing and returns a nonzero status for a blank device. The A2 and 950DT products commands below map one device; the A3 example maps two. Prepare all devices before creating the container, and separate their matching `ubsio.disk.path` entries with `:`.

!!! warning

    In Section 5.5, `MEMCACHE_FORCE_NEW_DISK` controls `ubsio.standalone.force_new_disk`. Setting it to `true` reinitializes cache storage at startup. Select its value for the target deployment and use only dedicated cache storage.

!!! note

    Set `MEMCACHE_CONFIG_DIR` before starting Docker. `-e` passes its value into the container, and `-v` mounts the directory at the same path. Section 5.5 copies image templates into this directory and uses those personal files.

In Section 5.5, set `MEMCACHE_SSD_PATHS` to the device paths mapped by `--device`.

Verify that all host device and file paths exist before running Docker.

!!! note "950DT products communication prerequisites"

    Prepare the communication environment following the [950DT container requirements](../features/pd_disaggregation_mooncake_multi_node.md#atlas-850850e950-superpod-950pr950dt-container-requirements) and [MemCache environment preparation](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#scenario-2-memcache-backend). Add communication-file mounts required by the selected runtime and transport; the container requirements describe when `/lib/route.conf` can be omitted.

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

    The A3 example in Section 5.5 uses two SSD-backed devices. Set `SSD_DEVICE_PATH` to the first device and `SSD_DEVICE_PATH_2` to the second device prepared above; check both before creating the container. The example device names are `/dev/loop0` and `/dev/loop1`, but use the actual names returned on your host.

    ```bash
    export SSD_DEVICE_PATH_2="<SECOND_ACTUAL_SSD_DEVICE_PATH>"
    test -b "${SSD_DEVICE_PATH_2}" || exit 1
    lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS "${SSD_DEVICE_PATH_2}"
    sudo blkid "${SSD_DEVICE_PATH_2}"

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
      --device "${SSD_DEVICE_PATH_2}:${SSD_DEVICE_PATH_2}" \
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

=== "950DT products"

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

Keep the container running. Open additional service terminals from the host with:

```bash
docker exec -it vllm-ascend-qwen38-pool bash
```

Use the container name set by `NAME`. New shells inherit `MEMCACHE_CONFIG_DIR`; repeat the Section 5.5 library and service exports in each shell. Keep the directory path consistent with the container's environment and mount.

After entering the container, check that the model and storage dependencies are available:

```bash
python -c "import vllm, vllm_ascend, mooncake, memcache_hybrid, memfabric_hybrid; print('Model and pool dependencies are available')"
```

Check MemCache dependency versions in Section 5.5. Budget pool DRAM and UBS IO memory separately from the container's `/dev/shm` limit (`--shm-size`).

!!! note "950 DT pooling runtime"

    Use a runtime that supports Qwen3.8-Flash-Next and MemCache three-tier pooling. Initialize CANN/ATB as described in Section 4.1. If the image does not initialize its repository, custom operators, and model libraries automatically, apply the following in each service shell; use actual container paths, with library directories separated by `:`.

    ```bash
    export ASCEND_REPO="<VLLM_ASCEND_REPOSITORY_IN_CONTAINER>"
    export MODEL_RUNTIME_LIB_DIRS="<MOONCAKE_TORCH_AND_TORCH_NPU_LIBRARY_DIRECTORIES>"
    export PYTHONPATH="${ASCEND_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
    source "${ASCEND_REPO}/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash"
    export LD_LIBRARY_PATH="${MODEL_RUNTIME_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
    ```

    Use the active environment's Mooncake, Torch, and torch_npu library directories. MemCache/MemFabric library paths are set in Section 5.5.

## 5 Online Service Deployment {: #5-online-service-deployment }

### 5.1 Single-Node Online Deployment

Prefill and decode run in the same service. Select the hardware tab below.

Before starting:

- Ensure the serving environment can access ModelScope to download the checkpoint.
- Verify available NPU memory before changing context length, concurrency, or memory utilization.

**Common settings:**

- All three examples enable Model Runner V2, async scheduling, expert parallelism, prefix caching, decode graph capture (`FULL_DECODE_ONLY`), and three-token MTP. MTP explicitly sets `enforce_eager=false` to allow graph execution; `true` forces eager execution for MTP.
- `cpu_offload=true` places Engram embedding tables in host memory; `dp_shared_memory=false` disables sharing for DP1. Use a checkpoint with supported n-gram layers and sufficient host memory.
- CPU binding, FlashComm1, shared expert DP, and shared expert overlap are enabled. Shared expert DP requires expert parallelism and TP > 1. `enable_fused_mc2=0` disables fused MC2; norm/quantization fusion is disabled.
- `--mamba-cache-mode align` aligns GDN and PLE state checkpoints for prefix caching. Tool calling and reasoning parsing are covered in [Section 6](#6-functional-verification).
- Multimodal requests require a vision encoder; add `--language-model-only` for text-only serving.

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
        --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":false}' \
        --additional-config '{"enable_cpu_binding":true,"enable_shared_expert_dp":true,"enable_fused_mc2":0,"enable_flashcomm1":true,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"fuse_norm_quant":false,"enable_npugraph_ex":true,"enable_static_kernel":false,"enable_super_kernel":false}}'
    ```

    **Key parameters:**

    - TP8 × DP1 uses eight devices. NPU graph extension is enabled; static kernels and super kernels are disabled.

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
        --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":false}' \
        --additional-config '{"enable_cpu_binding":true,"enable_shared_expert_dp":true,"enable_fused_mc2":0,"enable_flashcomm1":true,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"fuse_norm_quant":false,"enable_npugraph_ex":true,"enable_static_kernel":true,"enable_super_kernel":false}}'
    ```

    **Key parameters:**

    - TP8 × DP1 uses eight devices. NPU graph extension and static kernels are enabled; super kernels are disabled.

=== "950DT products"

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
        --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":false}'
    ```

    **Key parameters:**

    - TP4 × DP1 uses devices 0–3 and a W4A8-MXFP checkpoint. NPU graph extension and static kernels are enabled; super kernels are disabled.

**Service Verification:**

Wait for model loading and graph capture to complete, then check the service from the serving host. Use port `8088` for A2 series and A3 series or `8087` for 950 DT:

```bash
curl --fail "http://127.0.0.1:8088/v1/models"
```

The response should list the configured model name. After startup, use [Section 6](#6-functional-verification) to verify generation.

### 5.2 Basic PD Separation Deployment

Prepare the runtime from Section 4 and follow the [Mooncake installation guide](../features/pd_disaggregation_mooncake_single_node.md#install-mooncake). Run P/D commands in separate terminals, then start the proxy after their APIs are ready.

For 950DT products, basic PD, the image-and-text 10K case, and the long-context case use the same Docker container setup from [Section 4.1](#41-docker-image-installation), under the `950DT products` tab.

!!! note

    Set the IP and interface variables in each terminal; `nic_name` must own `local_ip`. The communication exports use these values, and `no_proxy`/`NO_PROXY` bypass HTTP proxies for the service addresses.

=== "A2 series"

    This W4A8 deployment uses two eight-device A2 series hosts. Prefill runs on `PREFILL_HOST` and Decode runs on `DECODE_HOST`, each using devices 0–7. Prefill uses TP4 × PP2 × DP1, and Decode uses TP2 × DP4. Both roles load the same checkpoint; the proxy routes requests through one API endpoint per role (1P1D).

    Before starting:

    - Expose the required devices, mount `/etc/hccn.conf`, and use `--shm-size=128g`. Provide sufficient host memory for Engram CPU offload and DP shared memory.
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

    After both checks succeed, start Proxy in a separate terminal and verify the complete PD path:

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

=== "A3 series"

    This W4A8 PD deployment uses one 16-device A3 series host specified by `local_ip`. Prefill uses TP2 × PP4 × DP1 (EP2) on devices 0–7. Decode uses TP2 × DP4 (EP8) on devices 8–15. Both roles load the same checkpoint. The proxy routes requests through one Prefill API endpoint and one Decode API endpoint (1P1D); Decode runs four DP replicas behind its endpoint.

    Before starting:

    - Expose the required devices, mount `/etc/hccn.conf`, and use `--shm-size=128g`. Provide sufficient host memory for Engram CPU offload and DP shared memory.
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
        --tensor-parallel-size 2 \
        --pipeline-parallel-size 4 \
        --data-parallel-size 1 \
        --max-model-len 262144 \
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
        --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_producer","kv_port":32600,"kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":2,"pp_size":4},"decode":{"dp_size":4,"tp_size":2}}}'
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
        --max-model-len 262144 \
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
        --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_consumer","kv_port":32700,"kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":2,"pp_size":4},"decode":{"dp_size":4,"tp_size":2}}}'
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

    Check the startup logs for TP2 × PP4 × DP1 (EP2) on Prefill and TP2 × DP4 (EP8) with `enable_static_kernel=true` on Decode. From the serving host, check that both APIs list `qwen38-flash-next-a3-pd`:

    ```bash
    curl --fail "http://127.0.0.1:25568/v1/models"
    curl --fail "http://127.0.0.1:25569/v1/models"
    ```

    After both checks succeed, start Proxy in a separate terminal and verify the complete PD path:

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

=== "950DT products"

    This W4A8-MXFP deployment uses eight devices on one 950DT products host. Prefill and Decode each use TP2 × DP2 (EP4), with one command per role managing both DP replicas behind a single API endpoint. MooncakeConnectorV2 transfers KV cache between the two roles.

    | Setting | Prefill | Decode |
    | --- | --- | --- |
    | Visible devices | 0,1,2,3 | 4,5,6,7 |
    | TP × PP × DP | 2 × 1 × 2 | 2 × 1 × 2 |
    | EP | 4 | 4 |
    | API port | 8088 | 8092 |
    | DP RPC port | 13388 | 13389 |
    | Mooncake role / KV base port | `kv_producer` / 30000 | `kv_consumer` / 30100 |
    | Maximum context / sequences / batched tokens | 143360 / 64 / 4096 | 143360 / 64 / 4096 |

    Both roles load `Eco-Tech/Qwen3.8-Flash-Next-w4a8-mxfp`. Set `local_ip` to the serving host address and `nic_name` to its network interface in each command block. Provide sufficient host memory for Engram CPU offload and shared memory.

    Run Prefill and Decode in separate terminals. Start the proxy after both APIs are ready. Port 8087 is also used by the colocated example, so stop that service before starting this proxy.

    These commands retain `VLLM_VERSION=0.32.0` from the validated runtime. This variable overrides vLLM version detection; it does not install or upgrade vLLM. Initialize CANN/ATB as described in Section 4.1.

    **Prefill:**

    ```bash
    local_ip="<LOCAL_SERVICE_IP>"
    nic_name="<SERVICE_NETWORK_INTERFACE>"

    unset CPLUS_INCLUDE_PATH CPATH C_INCLUDE_PATH
    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY

    export VLLM_HOST_IP="${local_ip}"
    export HCCL_IF_IP="${local_ip}"
    export GLOO_SOCKET_IFNAME="${nic_name}"
    export TP_SOCKET_IFNAME="${nic_name}"
    export HCCL_SOCKET_IFNAME="${nic_name}"
    export VLLM_VERSION=0.32.0
    export VLLM_RPC_TIMEOUT=3600000
    export HCCL_EXEC_TIMEOUT=204
    export HCCL_CONNECT_TIMEOUT=120
    export VLLM_ASCEND_ENABLE_MLAPO=1
    export VLLM_USE_V1=1
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
    export HCCL_BUFFSIZE=8192
    export HCCL_OP_EXPANSION_MODE=AIV
    export ASCEND_BUFFER_POOL=0:0
    export HCCL_IF_BASE_PORT=62000
    export HCCL_NPU_SOCKET_PORT_RANGE=61400-61499
    export no_proxy="localhost,127.0.0.1,${local_ip}"
    export NO_PROXY="${no_proxy}"
    export LD_LIBRARY_PATH="$(python -c 'import mooncake; print(next(iter(mooncake.__path__)))'):${LD_LIBRARY_PATH:-}"

    vllm serve Eco-Tech/Qwen3.8-Flash-Next-w4a8-mxfp \
        --api-server-count 1 \
        --host 0.0.0.0 \
        --port 8088 \
        --served-model-name qwen38-flash-next-950-dt-pd \
        --trust-remote-code \
        --quantization ascend \
        --tensor-parallel-size 2 \
        --data-parallel-size 2 \
        --data-parallel-address "${local_ip}" \
        --data-parallel-rpc-port 13388 \
        --max-model-len 143360 \
        --max-num-seqs 64 \
        --max-num-batched-tokens 4096 \
        --gpu-memory-utilization 0.90 \
        --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
        --enable-prefix-caching \
        --mamba-cache-mode align \
        --mamba-ssm-cache-dtype bfloat16 \
        --async-scheduling \
        --enable-expert-parallel \
        --compilation-config '{"cudagraph_mode":"NONE"}' \
        --additional-config '{"enable_cpu_binding":true,"enable_shared_expert_dp":true,"enable_fused_mc2":0,"enable_flashcomm1":true,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"fuse_norm_quant":false,"enable_npugraph_ex":true,"enable_static_kernel":false,"enable_super_kernel":false}}' \
        --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' \
        --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_producer","kv_port":30000,"kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":2},"decode":{"dp_size":2,"tp_size":2}}}'
    ```

    **Decode:**

    ```bash
    local_ip="<LOCAL_SERVICE_IP>"
    nic_name="<SERVICE_NETWORK_INTERFACE>"

    unset CPLUS_INCLUDE_PATH CPATH C_INCLUDE_PATH
    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY

    export VLLM_HOST_IP="${local_ip}"
    export HCCL_IF_IP="${local_ip}"
    export GLOO_SOCKET_IFNAME="${nic_name}"
    export TP_SOCKET_IFNAME="${nic_name}"
    export HCCL_SOCKET_IFNAME="${nic_name}"
    export VLLM_VERSION=0.32.0
    export VLLM_RPC_TIMEOUT=3600000
    export HCCL_EXEC_TIMEOUT=204
    export HCCL_CONNECT_TIMEOUT=120
    export VLLM_ASCEND_ENABLE_MLAPO=1
    export VLLM_USE_V1=1
    export VLLM_USE_MODELSCOPE=True
    export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export VLLM_ENGINE_READY_TIMEOUT_S=4800
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export TASK_QUEUE_ENABLE=1
    export HCCL_BUFFSIZE=8192
    export HCCL_OP_EXPANSION_MODE=AIV
    export ASCEND_BUFFER_POOL=0:0
    export HCCL_IF_BASE_PORT=63000
    export HCCL_NPU_SOCKET_PORT_RANGE=61500-61599
    export no_proxy="localhost,127.0.0.1,${local_ip}"
    export NO_PROXY="${no_proxy}"
    export LD_LIBRARY_PATH="$(python -c 'import mooncake; print(next(iter(mooncake.__path__)))'):${LD_LIBRARY_PATH:-}"

    vllm serve Eco-Tech/Qwen3.8-Flash-Next-w4a8-mxfp \
        --api-server-count 1 \
        --host 0.0.0.0 \
        --port 8092 \
        --served-model-name qwen38-flash-next-950-dt-pd \
        --trust-remote-code \
        --quantization ascend \
        --tensor-parallel-size 2 \
        --data-parallel-size 2 \
        --data-parallel-address "${local_ip}" \
        --data-parallel-rpc-port 13389 \
        --max-model-len 143360 \
        --max-num-seqs 64 \
        --max-num-batched-tokens 4096 \
        --gpu-memory-utilization 0.90 \
        --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
        --enable-prefix-caching \
        --mamba-cache-mode align \
        --mamba-ssm-cache-dtype bfloat16 \
        --async-scheduling \
        --enable-expert-parallel \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --additional-config '{"enable_cpu_binding":true,"enable_shared_expert_dp":true,"enable_fused_mc2":0,"enable_flashcomm1":true,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"fuse_norm_quant":false,"enable_npugraph_ex":true,"enable_static_kernel":true,"enable_super_kernel":false},"scheduler_config":{"recompute_scheduler_enable":true}}' \
        --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":false}' \
        --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_consumer","kv_port":30100,"kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":2},"decode":{"dp_size":2,"tp_size":2}}}'
    ```

    **Proxy:**

    Run the following command from the vLLM-Ascend repository root after both model APIs are ready:

    ```bash
    local_ip="<LOCAL_SERVICE_IP>"

    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY

    python examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
        --host "${local_ip}" \
        --port 8087 \
        --prefiller-hosts "${local_ip}" \
        --prefiller-ports 8088 \
        --decoder-hosts "${local_ip}" \
        --decoder-ports 8092
    ```

    **Service Verification:**

    Check the startup logs for TP2 × DP2 (EP4) on each role, `cudagraph_mode=NONE` on Prefill, and `FULL_DECODE_ONLY` with static kernels on Decode. Check that both APIs list `qwen38-flash-next-950-dt-pd` before starting the proxy:

    ```bash
    local_ip="<LOCAL_SERVICE_IP>"

    curl --fail "http://${local_ip}:8088/v1/models"
    curl --fail "http://${local_ip}:8092/v1/models"
    ```

    Once the proxy is listening, verify the complete PD path:

    ```bash
    local_ip="<LOCAL_SERVICE_IP>"

    curl --fail "http://${local_ip}:8087/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"qwen38-flash-next-950-dt-pd","messages":[{"role":"user","content":"Who are you?"}],"max_tokens":1024,"temperature":0}'
    ```

**A2/A3 key parameters:**

- A2 Prefill uses TP4 × PP2 × DP1 (EP4), and A3 Prefill uses TP2 × PP4 × DP1 (EP2). Both run in eager mode with `enable_prefill_mc2=true` and fused MC2 disabled. Check MC2 token limits before increasing the batch budget.
- A2 and A3 Decode use TP2 × DP4 (EP8), `FULL_DECODE_ONLY`, NPU graph extension, static kernels, and recompute scheduling; super kernels are disabled.
- Both roles use three MTP draft tokens, with `enforce_eager=true` on P and `false` on D. Batch tokens are 8192; P/D HCCL buffers are 8192/512.
- Engram CPU offload requires supported n-gram layers and sufficient host memory. P uses `dp_shared_memory=false`; D uses `true`. Shared-memory DP replicas must share a host and shared-memory namespace.
- Both roles set prefix caching, Mamba cache mode `align`, and Mamba state dtype `bfloat16`. Keep Mooncake TP/PP/DP declarations consistent with the model commands; clear `VLLM_PP_LAYER_PARTITION` to use the default partition.
- Prefill sets `enable_fused_mc2=0` while enabling shared expert overlap.

**950DT products key parameters:**

- Each command starts TP2 × DP2 (EP4) on four devices. `--api-server-count 1` exposes one API endpoint per role. Use distinct DP RPC ports for Prefill and Decode.
- Model Runner V2, async scheduling, and expert parallelism are enabled.
- Engram uses CPU offload and DP shared memory. DP replicas must share the same host and shared-memory namespace.
- Both roles enable prefix caching with Mamba cache mode `align` and Mamba SSM cache dtype `bfloat16`.
- Prefill uses `cudagraph_mode=NONE` with static kernels disabled.
- Decode uses `FULL_DECODE_ONLY`, static kernels, and recompute scheduling. MTP sets `enforce_eager=false` to allow graph execution.
- Both roles enable CPU binding, shared expert DP, FlashComm1, shared expert multistream overlap, and NPU graph extension. Fused MC2, norm/quantization fusion, and super kernels are disabled.
- Both roles use three MTP draft tokens and `VLLM_ENGINE_READY_TIMEOUT_S=4800` for engine startup; model execution uses `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000`.
- Both roles use `HCCL_BUFFSIZE=8192` and `ASCEND_BUFFER_POOL=0:0`. The validated configuration separates Prefill and Decode HCCL base ports (62000/63000) and NPU socket ranges (61400–61499/61500–61599).
- Keep Mooncake DP2/TP2 declarations consistent with the services. The proxy connects to Prefill on port 8088 and Decode on port 8092. Keep the configured KV base ports at 30000 and 30100; the connector applies rank offsets internally.

### 5.3 Image-and-Text 10K PD Separation Deployment

The 10K target is the processed input length, including text and image tokens. Keep the input plus requested output within the model length configured for your deployment.

For A2 and A3, make sure Mooncake is importable in both serving containers before running the environment setup. If needed, follow the [Mooncake installation guide](../features/pd_disaggregation_mooncake_single_node.md).

=== "A2 series"

    This W4A8 example uses two eight-device A2 series hosts. Prefill uses TP4 × PP2 × DP1 (EP4) on the Prefill host, and Decode uses TP2 × DP4 (EP8) on the Decode host. Each host uses devices 0–7. The proxy runs on the Prefill host and routes to one Prefill API and four Decode APIs, one per DP rank.

    On both hosts, initialize the A2 container and runtime as described in [Section 4.1](#41-docker-image-installation). Use the same W4A8 checkpoint with the vision encoder and image processor on both hosts.

    In each engine terminal, set `PREFILL_HOST` and `DECODE_HOST` to the two hosts' service IPs and `nic_name` to the local interface. Set `local_ip="${PREFILL_HOST}"` on Prefill and `local_ip="${DECODE_HOST}"` on Decode before applying the exports below.

    ```bash
    PREFILL_HOST="<PREFILL_LOCAL_IP>"
    DECODE_HOST="<DECODE_LOCAL_IP>"
    local_ip="<THIS_HOST_SERVICE_IP>"
    nic_name="<SERVICE_NETWORK_INTERFACE>"

    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY all_proxy ALL_PROXY
    export VLLM_HOST_IP="${local_ip}"
    export HCCL_IF_IP="${local_ip}"
    export GLOO_SOCKET_IFNAME="${nic_name}"
    export TP_SOCKET_IFNAME="${nic_name}"
    export HCCL_SOCKET_IFNAME="${nic_name}"
    export no_proxy="localhost,127.0.0.1,${PREFILL_HOST},${DECODE_HOST}"
    export NO_PROXY="${no_proxy}"
    export LD_LIBRARY_PATH="$(python -c 'import mooncake; print(next(iter(mooncake.__path__)))'):${LD_LIBRARY_PATH:-}"
    export VLLM_USE_MODELSCOPE=True
    export VLLM_SERVER_DEV_MODE=1
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export VLLM_ENGINE_READY_TIMEOUT_S=2400
    export VLLM_ASCEND_PLE_BACKEND=torch
    ```

    **Prefill:**

    ```bash
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export HCCL_NPU_SOCKET_PORT_RANGE=61400-61499
    export HCCL_BUFFSIZE=4096
    export VLLM_PP_LAYER_PARTITION=24,24

    vllm serve Eco-Tech/Qwen3.8-Flash-Next-w4a8 \
        --host 0.0.0.0 \
        --port 25568 \
        --served-model-name qwen38-flash-next-a2-pd \
        --trust-remote-code \
        --mm-encoder-tp-mode data \
        --quantization ascend \
        --tensor-parallel-size 4 \
        --pipeline-parallel-size 2 \
        --data-parallel-size 1 \
        --data-parallel-size-local 1 \
        --max-model-len 262144 \
        --max-num-seqs 32 \
        --max-num-batched-tokens 8192 \
        --gpu-memory-utilization 0.90 \
        --no-enable-prefix-caching \
        --async-scheduling \
        --enable-expert-parallel \
        --compilation-config '{"cudagraph_mode":"NONE"}' \
        --engram-config '{"cpu_offload":true,"dp_shared_memory":false}' \
        --additional-config '{"enable_shared_expert_dp":false,"enable_flashcomm1":true,"enable_prefill_mc2":true,"enable_fused_mc2":0,"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"fuse_norm_quant":false}}' \
        --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}' \
        --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_producer","kv_port":"34600","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":4,"pp_size":2,"pp_layer_partition":"24,24"},"decode":{"dp_size":4,"tp_size":2}}}'
    ```

    **Decode:**

    Start all four ranks in the same Decode container so they share the host memory namespace. Keep this terminal open while the ranks run.

    | DP rank | Visible devices | API port |
    | --- | --- | --- |
    | 0 | 0,1 | 25569 |
    | 1 | 2,3 | 25579 |
    | 2 | 4,5 | 25589 |
    | 3 | 6,7 | 25599 |

    ```bash
    unset VLLM_PP_LAYER_PARTITION
    export HCCL_NPU_SOCKET_PORT_RANGE=61500-61599
    export HCCL_BUFFSIZE=512

    for rank in 0 1 2 3; do
        first_device=$((rank * 2))
        api_port=$((25569 + rank * 10))
        kv_port=$((32700 + rank * 20))

        ASCEND_RT_VISIBLE_DEVICES="${first_device},$((first_device + 1))" \
        vllm serve Eco-Tech/Qwen3.8-Flash-Next-w4a8 \
            --host 0.0.0.0 \
            --port "${api_port}" \
            --served-model-name qwen38-flash-next-a2-pd \
            --trust-remote-code \
            --quantization ascend \
            --tensor-parallel-size 2 \
            --data-parallel-size 4 \
            --data-parallel-size-local 1 \
            --data-parallel-rank "${rank}" \
            --data-parallel-address "${DECODE_HOST}" \
            --data-parallel-rpc-port 29569 \
            --max-model-len 262144 \
            --max-num-seqs 32 \
            --max-num-batched-tokens 4096 \
            --gpu-memory-utilization 0.90 \
            --no-enable-prefix-caching \
            --async-scheduling \
            --enable-expert-parallel \
            --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --additional-config '{"enable_flashcomm1":true,"enable_cpu_binding":true,"enable_shared_expert_dp":true,"multistream_overlap_shared_expert":true,"scheduler_config":{"recompute_scheduler_enable":true},"ascend_compilation_config":{"fuse_norm_quant":false,"enable_npugraph_ex":true,"enable_static_kernel":true,"enable_super_kernel":false}}' \
            --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' \
            --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_consumer","kv_port":"'"$kv_port"'","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":4,"pp_size":2,"pp_layer_partition":"24,24"},"decode":{"dp_size":4,"tp_size":2}}}' &
    done
    wait
    ```

    **Proxy:**

    On the Prefill host, check that all five model endpoints respond, then start the proxy from the vLLM-Ascend repository root.

    ```bash
    PREFILL_HOST="<PREFILL_LOCAL_IP>"
    DECODE_HOST="<DECODE_LOCAL_IP>"

    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY all_proxy ALL_PROXY
    export no_proxy="localhost,127.0.0.1,${PREFILL_HOST},${DECODE_HOST}"
    export NO_PROXY="${no_proxy}"

    curl --fail --noproxy '*' "http://${PREFILL_HOST}:25568/v1/models" || exit 1
    for port in 25569 25579 25589 25599; do
        curl --fail --noproxy '*' "http://${DECODE_HOST}:${port}/v1/models" || exit 1
    done

    python examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
        --host 0.0.0.0 --port 25570 \
        --prefiller-hosts "${PREFILL_HOST}" --prefiller-ports 25568 \
        --decoder-hosts "${DECODE_HOST}" "${DECODE_HOST}" "${DECODE_HOST}" "${DECODE_HOST}" \
        --decoder-ports 25569 25579 25589 25599
    ```

    **Service Verification:**

    Send a 10K sample through the proxy. Replace `<IMAGE_URL>` with an image URL accessible to Prefill and Decode, and replace `<TEXT_PROMPT>` with the corresponding text. For local image files, mount the dataset at the same container path on both hosts and set `--allowed-local-media-path` to that directory in both engine commands.

    ```bash
    curl --fail "http://127.0.0.1:25570/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"qwen38-flash-next-a2-pd","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"<IMAGE_URL>"}},{"type":"text","text":"<TEXT_PROMPT>"}]}],"max_tokens":1024,"temperature":0}'
    ```

    Expect HTTP 200 and an answer addressing the image and text. Check `usage.prompt_tokens` against the 10K target and confirm there are no image-processing, KV-transfer, or out-of-memory errors.

    **Key parameters:**

    - Prefill uses a `24,24` layer partition. `--mm-encoder-tp-mode data` distributes complete images across the TP ranks for the vision encoder. The language model uses TP4 × PP2.
    - Decode enables static kernels and uses an API endpoint per DP rank.
    - This case disables prefix caching on both roles. The Prefill/Decode batch token limits are 8192/4096, and both roles use `--max-model-len 262144`.

=== "A3 series"

    This single-host W4A8 example uses TP4 × PP2 × DP1 (EP4) on devices 0–7 for Prefill and TP2 × DP4 (EP8) on devices 8–15 for Decode. The proxy routes to one Prefill API and four Decode APIs, one per DP rank.

    Initialize the A3 container and runtime as described in [Section 4.1](#41-docker-image-installation). Use a W4A8 checkpoint with the vision encoder and image processor. Run the following environment setup in both the Prefill and Decode terminals. Set `local_ip` and `nic_name` to the serving host's address and interface.

    ```bash
    local_ip="<LOCAL_SERVICE_IP>"
    nic_name="<SERVICE_NETWORK_INTERFACE>"

    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY all_proxy ALL_PROXY
    export VLLM_HOST_IP="${local_ip}"
    export HCCL_IF_IP="${local_ip}"
    export GLOO_SOCKET_IFNAME="${nic_name}"
    export TP_SOCKET_IFNAME="${nic_name}"
    export HCCL_SOCKET_IFNAME="${nic_name}"
    export no_proxy="localhost,127.0.0.1,${local_ip}"
    export NO_PROXY="${no_proxy}"
    export LD_LIBRARY_PATH="$(python -c 'import mooncake; print(next(iter(mooncake.__path__)))'):${LD_LIBRARY_PATH:-}"
    export VLLM_USE_MODELSCOPE=True
    export VLLM_SERVER_DEV_MODE=1
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export VLLM_ENGINE_READY_TIMEOUT_S=2400
    export VLLM_ASCEND_PLE_BACKEND=torch
    ```

    **Prefill:**

    ```bash
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export HCCL_NPU_SOCKET_PORT_RANGE=61400-61499
    export HCCL_BUFFSIZE=4096
    export VLLM_PP_LAYER_PARTITION=24,24

    vllm serve Eco-Tech/Qwen3.8-Flash-Next-w4a8 \
        --host 0.0.0.0 \
        --port 25568 \
        --served-model-name qwen38-flash-next-a3-pd \
        --trust-remote-code \
        --mm-encoder-tp-mode data \
        --quantization ascend \
        --tensor-parallel-size 4 \
        --pipeline-parallel-size 2 \
        --data-parallel-size 1 \
        --data-parallel-size-local 1 \
        --max-model-len 262144 \
        --max-num-seqs 32 \
        --max-num-batched-tokens 8192 \
        --gpu-memory-utilization 0.90 \
        --no-enable-prefix-caching \
        --async-scheduling \
        --enable-expert-parallel \
        --compilation-config '{"cudagraph_mode":"NONE"}' \
        --engram-config '{"cpu_offload":true,"dp_shared_memory":false}' \
        --additional-config '{"enable_shared_expert_dp":false,"enable_flashcomm1":true,"enable_prefill_mc2":true,"enable_fused_mc2":0,"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"fuse_norm_quant":false}}' \
        --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}' \
        --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_producer","kv_port":"34600","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":4,"pp_size":2,"pp_layer_partition":"24,24"},"decode":{"dp_size":4,"tp_size":2}}}'
    ```

    **Decode:**

    Start all four ranks in the same container so they share the host memory namespace. Keep this terminal open while the ranks run.

    | DP rank | Visible devices | API port |
    | --- | --- | --- |
    | 0 | 8,9 | 25569 |
    | 1 | 10,11 | 25579 |
    | 2 | 12,13 | 25589 |
    | 3 | 14,15 | 25599 |

    ```bash
    unset VLLM_PP_LAYER_PARTITION
    export HCCL_NPU_SOCKET_PORT_RANGE=61500-61599
    export HCCL_BUFFSIZE=512

    for rank in 0 1 2 3; do
        first_device=$((8 + rank * 2))
        api_port=$((25569 + rank * 10))
        kv_port=$((32700 + rank * 20))

        ASCEND_RT_VISIBLE_DEVICES="${first_device},$((first_device + 1))" \
        vllm serve Eco-Tech/Qwen3.8-Flash-Next-w4a8 \
            --host 0.0.0.0 \
            --port "${api_port}" \
            --served-model-name qwen38-flash-next-a3-pd \
            --trust-remote-code \
            --quantization ascend \
            --tensor-parallel-size 2 \
            --data-parallel-size 4 \
            --data-parallel-size-local 1 \
            --data-parallel-rank "${rank}" \
            --data-parallel-address "${local_ip}" \
            --data-parallel-rpc-port 29569 \
            --max-model-len 262144 \
            --max-num-seqs 32 \
            --max-num-batched-tokens 4096 \
            --gpu-memory-utilization 0.90 \
            --no-enable-prefix-caching \
            --async-scheduling \
            --enable-expert-parallel \
            --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
            --additional-config '{"enable_flashcomm1":true,"enable_cpu_binding":true,"enable_shared_expert_dp":true,"multistream_overlap_shared_expert":true,"scheduler_config":{"recompute_scheduler_enable":true},"ascend_compilation_config":{"fuse_norm_quant":false,"enable_npugraph_ex":true,"enable_static_kernel":true,"enable_super_kernel":false}}' \
            --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' \
            --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_consumer","kv_port":"'"$kv_port"'","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":4,"pp_size":2,"pp_layer_partition":"24,24"},"decode":{"dp_size":4,"tp_size":2}}}' &
    done
    wait
    ```

    **Proxy:**

    Check that all five model endpoints respond, then start the proxy from the vLLM-Ascend repository root.

    ```bash
    local_ip="<LOCAL_SERVICE_IP>"

    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY all_proxy ALL_PROXY
    export no_proxy="localhost,127.0.0.1,${local_ip}"
    export NO_PROXY="${no_proxy}"

    for port in 25568 25569 25579 25589 25599; do
        curl --fail --noproxy '*' "http://${local_ip}:${port}/v1/models" || exit 1
    done

    python examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
        --host 0.0.0.0 --port 25570 \
        --prefiller-hosts "${local_ip}" --prefiller-ports 25568 \
        --decoder-hosts "${local_ip}" "${local_ip}" "${local_ip}" "${local_ip}" \
        --decoder-ports 25569 25579 25589 25599
    ```

    **Service Verification:**

    Send a 10K sample through the proxy. Replace `<IMAGE_URL>` with an image URL accessible to Prefill and Decode, and replace `<TEXT_PROMPT>` with the corresponding text. For local image files, mount the dataset in the serving container and set `--allowed-local-media-path` to the mounted directory in both engine commands.

    ```bash
    curl --fail "http://127.0.0.1:25570/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"qwen38-flash-next-a3-pd","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"<IMAGE_URL>"}},{"type":"text","text":"<TEXT_PROMPT>"}]}],"max_tokens":1024,"temperature":0}'
    ```

    Expect HTTP 200 and an answer addressing the image and text. Check `usage.prompt_tokens` against the 10K target and confirm there are no image-processing, KV-transfer, or out-of-memory errors.

    **Key parameters:**

    - Prefill uses a `24,24` layer partition. `--mm-encoder-tp-mode data` distributes complete images across the TP ranks for the vision encoder. The language model uses TP4 × PP2.
    - Decode enables static kernels and uses an API endpoint per DP rank. The 32-way concurrency in the MMMU-Pro benchmark is a client-side setting.
    - This case disables prefix caching on both roles. The Prefill/Decode batch token limits are 8192/4096, and both roles use `--max-model-len 262144`.

=== "950DT products"

    This case uses the same eight-device 950DT products host as the basic PD example in [Section 5.2](#52-basic-pd-separation-deployment). For image-and-text 10K validation, prefill uses TP2 × PP2 × DP1 with a `25,23` layer partition, and decode uses TP2 × DP2 behind one API endpoint. Both roles use 8192 batch tokens and `HCCL_BUFFSIZE=512`. Both roles enable prefix caching; decode also enables static kernels.

    The 10K target is the processed input length, including text and image tokens. Both roles retain `--max-model-len 143360`; keep the input plus requested output within this limit. Use the W4A8-MXFP checkpoint from [Section 3.1](#31-model-weight) with its vision encoder and image processor.

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
        --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
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

    Check P TP2/PP2/DP1, D TP2/DP2 with static kernels, and `max_model_len=143360` in the startup logs. Send a 10K sample through the proxy, replacing `<IMAGE_URL>` and `<TEXT_PROMPT>` with an accessible image and its text:

    ```bash
    curl --fail "http://127.0.0.1:8247/v1/models"
    curl --fail "http://127.0.0.1:8248/v1/models"
    curl --fail "http://127.0.0.1:8246/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"qwen38-megamoe-sp-pd","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"<IMAGE_URL>"}},{"type":"text","text":"<TEXT_PROMPT>"}]}],"max_tokens":1024,"temperature":0}'
    ```

    Expect HTTP 200 and an answer addressing the image and text. Check `usage.prompt_tokens` against the 10K target and confirm there are no image-processing, KV-transfer, or out-of-memory errors.

    **Key parameters:**

    - Prefill partitions the layers across two pipeline stages with `VLLM_PP_LAYER_PARTITION=25,23`. Keep `pp_size=2` and `pp_layer_partition="25,23"` in both Mooncake configurations consistent with this partition.
    - Decode uses TP2 × DP2 behind one API endpoint on port `8248`; the proxy routes to that endpoint rather than separate DP-rank APIs.
    - Both roles enable prefix caching. Prefill disables graph capture and sets MTP `enforce_eager=true`. Decode uses `FULL_DECODE_ONLY`, sets MTP `enforce_eager=false`, enables static kernels, and disables super kernels.
    - Both roles enable Engram CPU offload. Decode enables DP shared memory for its two replicas. Both roles use `HCCL_BUFFSIZE=512` and set `VLLM_ENGINE_READY_TIMEOUT_S=4800`. Provide sufficient host memory and container shared memory for the deployment.
    - `VLLM_ASCEND_PLE_BACKEND` only applies to runtimes that implement it; an unknown-variable warning means it has no effect.

### 5.4 Long-Context PD Separation Deployment

This section extends the model context for PD separation on A2 series, A3 series, and 950DT products.

!!! note "Extending the native 256K context to approximately 1M"

    The checkpoint's native context window is 256K (262144 tokens). This long-context case requires all three settings on **both Prefill and Decode**: the complete YaRN `--hf-overrides` shown below, with `factor=4.0` and `original_max_position_embeddings=262144`; `export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`; and `--max-model-len 1010000`. Keep the YaRN configuration and maximum model length identical on both roles; none of these three settings can be omitted.

    Each request must satisfy `input token count + maximum output token count <= max-model-len`. With this configuration, the total must not exceed 1,010,000 tokens.

    The runtime must also size the Qwen4Exp MTP draft's position cache for the configured context length. The 0.32 RC branch applies the Qwen4Exp-specific fix in [PR #21347](https://github.com/vllm-project/vllm-ascend/pull/21347). With main, use a compatible vLLM build with the upstream draft position-cache expansion, such as vLLM v0.33.0, and verify the long-context request on your target image.

On 950DT products, this case reuses ports 8247, 8248, and 8246 from the image-and-text case. Stop that deployment before starting this case.

=== "A2 series"

    This example uses two eight-device A2 series hosts. Prefill uses TP2 × PP4 × DP1 (EP2) on `PREFILL_HOST`, and Decode uses TP2 × DP4 (EP8) on `DECODE_HOST`. Each host uses devices 0–7. Set the two service addresses and the local network interface in the commands.

    Prepare the A2 containers as in [Section 5.2](#52-basic-pd-separation-deployment) and run each command on its designated host. Stop any deployment already using ports 25568, 25569, or 25570 on these hosts. After both model APIs are ready, run the proxy on the Prefill host from the vLLM-Ascend repository root.

    **Prefill:**

    ```bash
    PREFILL_HOST="<PREFILL_LOCAL_IP>"
    DECODE_HOST="<DECODE_LOCAL_IP>"
    local_ip="${PREFILL_HOST}"
    nic_name="<SERVICE_NETWORK_INTERFACE>"

    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY

    unset VLLM_PP_LAYER_PARTITION
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
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
        --pipeline-parallel-size 4 \
        --data-parallel-size 1 \
        --max-model-len 1010000 \
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
        --hf-overrides '{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":4.0,"original_max_position_embeddings":262144}}}' \
        --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_producer","kv_port":32600,"kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":2,"pp_size":4},"decode":{"dp_size":4,"tp_size":2}}}'
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
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
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
        --max-model-len 1010000 \
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
        --hf-overrides '{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":4.0,"original_max_position_embeddings":262144}}}' \
        --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_consumer","kv_port":32700,"kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":2,"pp_size":4},"decode":{"dp_size":4,"tp_size":2}}}'
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

    Confirm `max_model_len=1010000` in both startup logs. Check the model APIs from the Prefill host.

    ```bash
    PREFILL_HOST="<PREFILL_LOCAL_IP>"
    DECODE_HOST="<DECODE_LOCAL_IP>"

    curl --fail --noproxy '*' "http://${PREFILL_HOST}:25568/v1/models"
    curl --fail --noproxy '*' "http://${DECODE_HOST}:25569/v1/models"
    ```

    Start Proxy after both checks succeed. Prepare `long-context-request.json` with `model` set to `qwen38-flash-next-a2-pd`, the input in `messages`, and the requested output budget in `max_tokens`. Keep the input token count plus the maximum output token count within 1,010,000 tokens.

    ```bash
    curl --fail "http://127.0.0.1:25570/v1/chat/completions" \
        -H "Content-Type: application/json" \
        --data-binary @long-context-request.json
    ```

    Verify the response and compare `usage.prompt_tokens` with the test input.

=== "A3 series"

    This example uses one 16-device A3 series host. Prefill uses TP2 × PP4 × DP1 (EP2) on devices 0–7. Decode uses TP2 × DP4 (EP8) on devices 8–15. Set `local_ip` and `nic_name` in the commands for the serving host.

    Prepare the container as in [Section 5.2](#52-basic-pd-separation-deployment) and run Prefill and Decode in separate terminals. Stop any A3 deployment already using ports 25568, 25569, or 25570. After both model APIs are ready, run Proxy from the vLLM-Ascend repository root.

    **Prefill:**

    ```bash
    local_ip="<LOCAL_SERVICE_IP>"
    nic_name="<SERVICE_NETWORK_INTERFACE>"

    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY

    unset VLLM_PP_LAYER_PARTITION
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
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
        --tensor-parallel-size 2 \
        --pipeline-parallel-size 4 \
        --data-parallel-size 1 \
        --max-model-len 1010000 \
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
        --hf-overrides '{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":4.0,"original_max_position_embeddings":262144}}}' \
        --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_producer","kv_port":32600,"kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":2,"pp_size":4},"decode":{"dp_size":4,"tp_size":2}}}'
    ```

    **Decode:**

    ```bash
    local_ip="<LOCAL_SERVICE_IP>"
    nic_name="<SERVICE_NETWORK_INTERFACE>"

    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY

    unset VLLM_PP_LAYER_PARTITION
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
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
        --max-model-len 1010000 \
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
        --hf-overrides '{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":4.0,"original_max_position_embeddings":262144}}}' \
        --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_consumer","kv_port":32700,"kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":2,"pp_size":4},"decode":{"dp_size":4,"tp_size":2}}}'
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

    Check that both model APIs are ready and confirm `max_model_len=1010000` in the startup logs.

    ```bash
    curl --fail "http://127.0.0.1:25568/v1/models"
    curl --fail "http://127.0.0.1:25569/v1/models"
    ```

    Start Proxy after both checks succeed. Prepare `long-context-request.json` with `model` set to `qwen38-flash-next-a3-pd`, the input in `messages`, and the requested output budget in `max_tokens`. Keep the input token count plus the maximum output token count within 1,010,000 tokens.

    ```bash
    curl --fail "http://127.0.0.1:25570/v1/chat/completions" \
        -H "Content-Type: application/json" \
        --data-binary @long-context-request.json
    ```

    Verify the response and compare `usage.prompt_tokens` with the test input.

=== "950DT products"

    This case uses eight devices on a single 950DT products host, with TP4 × DP1 for both Prefill and Decode. Both roles set the maximum combined input and output length to 1,010,000 tokens, with up to 16 active sequences and 4096 batch tokens. Initialize the environment as described in [Section 4](#4-installation) and run each role in a separate terminal.

    **Prefill:**

    ```bash
    unset CPLUS_INCLUDE_PATH CPATH C_INCLUDE_PATH
    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY

    export VLLM_USE_MODELSCOPE=True
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export VLLM_ENGINE_READY_TIMEOUT_S=2400
    export VLLM_ASCEND_PLE_BACKEND=torch
    export HCCL_NPU_SOCKET_PORT_RANGE=61400-61499
    export HCCL_BUFFSIZE=512
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

    vllm serve Eco-Tech/Qwen3.8-Flash-Next-w4a8-mxfp \
        --host 0.0.0.0 \
        --port 8247 \
        --served-model-name qwen38-megamoe-sp-pd \
        --trust-remote-code \
        --quantization ascend \
        --tensor-parallel-size 4 \
        --pipeline-parallel-size 1 \
        --data-parallel-size 1 \
        --max-model-len 1010000 \
        --max-num-seqs 16 \
        --max-num-batched-tokens 4096 \
        --gpu-memory-utilization 0.9 \
        --enable-prefix-caching \
        --mamba-cache-mode align \
        --mamba-ssm-cache-dtype bfloat16 \
        --async-scheduling \
        --enable-expert-parallel \
        --engram-config '{"cpu_offload":true,"dp_shared_memory":false}' \
        --compilation-config '{"cudagraph_mode":"NONE"}' \
        --additional-config '{"enable_shared_expert_dp":true,"enable_flashcomm1":true,"enable_cpu_binding":true,"ascend_compilation_config":{"fuse_norm_quant":false}}' \
        --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}' \
        --hf-overrides '{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":4.0,"original_max_position_embeddings":262144}}}' \
        --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_producer","kv_port":"32470","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":4},"decode":{"dp_size":1,"tp_size":4}}}'
    ```

    **Decode:**

    ```bash
    unset CPLUS_INCLUDE_PATH CPATH C_INCLUDE_PATH
    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY

    export VLLM_USE_MODELSCOPE=True
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
    export VLLM_ENGINE_READY_TIMEOUT_S=4800
    export VLLM_ASCEND_PLE_BACKEND=torch
    export HCCL_NPU_SOCKET_PORT_RANGE=61500-61599
    export HCCL_BUFFSIZE=512
    export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7

    vllm serve Eco-Tech/Qwen3.8-Flash-Next-w4a8-mxfp \
        --host 0.0.0.0 \
        --port 8248 \
        --served-model-name qwen38-megamoe-sp-pd \
        --trust-remote-code \
        --quantization ascend \
        --tensor-parallel-size 4 \
        --data-parallel-size 1 \
        --max-model-len 1010000 \
        --max-num-seqs 16 \
        --max-num-batched-tokens 4096 \
        --gpu-memory-utilization 0.9 \
        --enable-prefix-caching \
        --mamba-cache-mode align \
        --mamba-ssm-cache-dtype bfloat16 \
        --async-scheduling \
        --enable-expert-parallel \
        --engram-config '{"cpu_offload":true,"dp_shared_memory":false}' \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --additional-config '{"enable_flashcomm1":true,"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"scheduler_config":{"recompute_scheduler_enable":true},"ascend_compilation_config":{"fuse_norm_quant":false,"enable_npugraph_ex":true,"enable_static_kernel":true,"enable_super_kernel":false}}' \
        --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":false}' \
        --hf-overrides '{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":4.0,"original_max_position_embeddings":262144}}}' \
        --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_consumer","kv_port":"32480","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":4},"decode":{"dp_size":1,"tp_size":4}}}'
    ```

    **Proxy:**

    After both model APIs are ready, run the proxy from the vLLM-Ascend repository root:

    ```bash
    unset ftp_proxy FTP_PROXY https_proxy HTTPS_PROXY http_proxy HTTP_PROXY

    python examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
        --host 0.0.0.0 \
        --port 8246 \
        --prefiller-hosts 127.0.0.1 \
        --prefiller-ports 8247 \
        --decoder-hosts 127.0.0.1 \
        --decoder-ports 8248
    ```

    **Service Verification:**

    Check the startup logs for TP4 × DP1 on both roles and `max_model_len=1010000`. Prepare `long-context-request.json` as a chat-completion request with `model` set to `qwen38-megamoe-sp-pd`, the long input in `messages`, and `max_tokens` set to the required output budget. Keep the processed input plus output budget within 1,010,000 tokens, then run:

    ```bash
    curl --fail "http://127.0.0.1:8247/v1/models"
    curl --fail "http://127.0.0.1:8248/v1/models"
    curl --fail "http://127.0.0.1:8246/v1/chat/completions" \
        -H "Content-Type: application/json" \
        --data-binary @long-context-request.json
    ```

    Verify the response content and `usage.prompt_tokens` against the test input. Successful API startup alone does not verify long-context inference.

    **Key parameters:**

    - Both roles use TP4 × DP1 without pipeline parallelism. The Mooncake declarations use `tp_size=4` and `dp_size=1` for both roles.
    - Both roles use Mamba cache mode `align` and Mamba SSM cache dtype `bfloat16`.
    - Prefill disables graph capture and sets MTP `enforce_eager=true`. Decode uses `FULL_DECODE_ONLY`, sets MTP `enforce_eager=false`, and enables static kernels and recompute scheduling.
    - Both roles enable Engram CPU offload with `dp_shared_memory=false` and use `HCCL_BUFFSIZE=512`. The supplied configuration sets the engine-ready timeout to 2400 seconds on Prefill and 4800 seconds on Decode; the model-execution timeout is 3000 seconds on both roles.
    - `VLLM_ASCEND_PLE_BACKEND` only applies to runtimes that implement it; an unknown-variable warning means it has no effect.

### 5.5 PD Separation with MemCache Three-Tier KV Cache Pool

`MultiConnector` combines Mooncake P-to-D transport with MemCache-backed KV caching in HBM, DRAM, and SSD. Each hardware tab contains its complete pooling Prefill and Decode commands. A2 uses the basic PD topology from [Section 5.2](#52-basic-pd-separation-deployment). A3 uses a dedicated Prefill topology for the 128K/1K multi-prefix workload. This section does not cover colocated deployment, the image-and-text 10K case, or the long-context case.

**Before starting:**

Prepare the container, SSD mapping, configuration directory, and A2 HDK requirement in [Section 4.3](#43-docker-for-three-tier-pd-pooling). Run the following inside the container. A3 HCCS also requires LingQu Computing Network 1.5 or later; see the [MemCache backend guide](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#scenario-2-memcache-backend) for runtime dependencies.

Use the same vLLM-Ascend release and compatible MemCache and MemFabric package versions across the pool. Verify cache reuse and SSD restore on the runtime you deploy.

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

Use the templates shipped inside the selected image: `/usr/local/python3.11.10/lib/python3.11/site-packages/memcache_hybrid/config/` is an example, and the actual path varies with the Python version and installation location. Set `MEMCACHE_TEMPLATE_DIR` to that directory and copy its files into your personal mounted `MEMCACHE_CONFIG_DIR`. Edit only the keys listed below in those copies, preserving the other template entries; `cp -n` leaves existing personal files unchanged. In the INI examples, replace `<MEMCACHE_...>` with the actual value selected in the matching hardware tab; configuration files do not expand shell variables.

**Choose a new or existing pool before editing configuration:**

- **New pool:** follow **Prepare a new pool** in the matching hardware tab, using only the devices mapped in Section 4.3. Choose all endpoint and resource values before editing the INI copies.
- **Existing pool:** skip the new-pool exports, template copying, INI edits, and MetaService startup below. Use the existing pool's MetaService/ConfigStore endpoints and approved LocalService files in `MEMCACHE_CONFIG_DIR` (A2/A3: `mmc-local.conf`; 950DT products: `mmc-local-p.conf` and `mmc-local-d.conf`). Preserve its storage initialization policy; do not apply the new-pool `force_new_disk=true` example. Check the existing service's actual endpoints, then continue with the matching service environment and P/D commands.

`dram.size` and `ubsio.mem.size_in_gb` apply per DRAM-enabled LocalService; budget their total alongside Engram, vLLM, and operating-system memory. `world_size` is pool capacity; `device_count` counts the host's DRAM-enabled LocalServices, not SSDs. LocalServices run inside P/D workers; the `ubsio.standalone.*` keys do not require a separate process.

!!! note "Check host huge pages if pooling stalls"

    If a P or D worker stalls during MemCache DRAM initialization and the host kernel repeatedly reports `alloc_contig_range ... PFNs busy`, check the reserved 1 GiB HugeTLB pages on each affected host. The following example assumes NUMA nodes 0–7.

    ```bash
    for n in 0 1 2 3 4 5 6 7; do
        printf 'node%s: ' "$n"
        cat "/sys/devices/system/node/node${n}/hugepages/hugepages-1048576kB/nr_hugepages"
    done
    ```

    To test with preallocated pages, run the following as root on the host before starting Prefill and Decode. Adjust the node IDs and page count for the available host memory. Writing `32` to `nr_hugepages` sets the target total to 32 pages per NUMA node (32 GiB per node, 256 GiB across eight nodes). Check the existing count before changing it and leave enough memory for MemCache, Engram, and vLLM.

    ```bash
    for n in 0 1 2 3 4 5 6 7; do
        echo 32 > "/sys/devices/system/node/node${n}/hugepages/hugepages-1048576kB/nr_hugepages"
    done
    ```

    Repeat the first command and confirm that each selected node reports the requested count. Recheck after a host reboot. Use the P/D startup logs and API checks to verify that the services are ready. Watchdog dumps and shared-memory broadcast wait messages can also occur while static kernels compile. Check the worker stack and kernel log before attributing the wait to huge-page allocation.

=== "A2 series"

    Keep the A2 model configuration from Section 5.2: TP4 × PP2 × DP1 for Prefill and TP2 × DP4 for Decode, on separate eight-device hosts.

    **Prepare a new pool:**

    Prepare LocalService files on both hosts and MetaService only on the prefill host. Set the same reachable MetaService/ConfigStore endpoints and pool `world_size` on both hosts; choose storage values for each host separately.

    ```bash
    cp -n "${MEMCACHE_TEMPLATE_DIR}/mmc-local.conf" "${MEMCACHE_CONFIG_DIR}/mmc-local.conf"
    export MEMCACHE_META_URL="tcp://<META_SERVICE_IP>:<META_SERVICE_PORT>"
    export MEMCACHE_CONFIG_STORE_URL="tcp://<CONFIG_STORE_IP>:<CONFIG_STORE_PORT>"
    ```

    On the prefill host only, prepare MetaService:

    ```bash
    cp -n "${MEMCACHE_TEMPLATE_DIR}/mmc-meta.conf" "${MEMCACHE_CONFIG_DIR}/mmc-meta.conf"
    export MEMCACHE_METRICS_URL="http://<METRICS_IP>:<METRICS_PORT>"
    export MEMCACHE_EVICT_HIGH="<HIGH_EVICTION_THRESHOLD_PERCENT>"
    export MEMCACHE_EVICT_LOW="<LOW_EVICTION_THRESHOLD_PERCENT>"
    ```

    In the copied `${MEMCACHE_CONFIG_DIR}/mmc-meta.conf`, edit the following keys:

    ```ini
    ock.mmc.meta_service_url = <MEMCACHE_META_URL>
    ock.mmc.meta_service.config_store_url = <MEMCACHE_CONFIG_STORE_URL>
    ock.mmc.meta_service.metrics_url = <MEMCACHE_METRICS_URL>
    ock.mmc.log_level = info
    ock.mmc.evict_threshold_high = <MEMCACHE_EVICT_HIGH>
    ock.mmc.evict_threshold_low = <MEMCACHE_EVICT_LOW>
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

    In the copied `${MEMCACHE_CONFIG_DIR}/mmc-local.conf`, edit the following keys:

    ```ini
    ock.mmc.meta_service_url = <MEMCACHE_META_URL>
    ock.mmc.local_service.config_store_url = <MEMCACHE_CONFIG_STORE_URL>
    ock.mmc.log_level = info
    ock.mmc.local_service.world_size = <MEMCACHE_WORLD_SIZE>
    ock.mmc.local_service.protocol = device_rdma
    ock.mmc.local_service.dram.size = <MEMCACHE_DRAM_SIZE>
    ock.mmc.local_service.max.dram.size = <MEMCACHE_MAX_DRAM_SIZE>
    ock.mmc.local_service.storage.enabled = true
    ubsio.disk.path = <MEMCACHE_SSD_PATHS>
    ubsio.mem.size_in_gb = <MEMCACHE_UBS_IO_MEM_GB>
    ubsio.standalone.device_count = <MEMCACHE_DEVICE_COUNT>
    ubsio.standalone.force_new_disk = <MEMCACHE_FORCE_NEW_DISK>
    ```

    **Start MetaService:**

    Start MetaService on the prefill host with the library exports above, unless the pool is already running:

    ```bash
    export MMC_META_CONFIG_PATH="${MEMCACHE_CONFIG_DIR}/mmc-meta.conf"
    python -c "from memcache_hybrid import MetaService; MetaService.main()"
    ```

    After its configured ports are listening (`ss -lntp`), initialize the model runtime from Section 4 and run the complete Prefill and Decode commands below in separate terminals. Set the library directories in each terminal as described above; `MEMCACHE_CONFIG_DIR` is the mounted personal configuration directory from Section 4.3.

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

    export MEMCACHE_LIB_DIR="<MEMCACHE_LIBRARY_DIRECTORY>"
    export MEMFABRIC_LIB_DIR="<MEMFABRIC_LIBRARY_DIRECTORY>"
    export PYTHON_LIB_DIR="$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
    export LD_LIBRARY_PATH="${MEMCACHE_LIB_DIR}:${MEMFABRIC_LIB_DIR}:${PYTHON_LIB_DIR}:${LD_LIBRARY_PATH:-}"
    export PYTHONHASHSEED=0
    export MMC_LOCAL_CONFIG_PATH="${MEMCACHE_CONFIG_DIR}/mmc-local.conf"
    unset ASCEND_ENABLE_USE_FABRIC_MEM ASCEND_GLOBAL_RESOURCE_CONFIG ASCEND_LOCAL_COMM_RES
    export HCCL_INTRA_ROCE_ENABLE=1

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
        --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_producer","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_producer","kv_port":32600,"kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":4,"pp_size":2},"decode":{"dp_size":4,"tp_size":2}}},{"kv_connector":"AscendStoreConnector","kv_role":"kv_producer","kv_connector_extra_config":{"backend":"memcache","lookup_rpc_port":"qwen38_pool_p","use_layerwise":false}}]}}'
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

    export MEMCACHE_LIB_DIR="<MEMCACHE_LIBRARY_DIRECTORY>"
    export MEMFABRIC_LIB_DIR="<MEMFABRIC_LIBRARY_DIRECTORY>"
    export PYTHON_LIB_DIR="$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
    export LD_LIBRARY_PATH="${MEMCACHE_LIB_DIR}:${MEMFABRIC_LIB_DIR}:${PYTHON_LIB_DIR}:${LD_LIBRARY_PATH:-}"
    export PYTHONHASHSEED=0
    export MMC_LOCAL_CONFIG_PATH="${MEMCACHE_CONFIG_DIR}/mmc-local.conf"
    unset ASCEND_ENABLE_USE_FABRIC_MEM ASCEND_GLOBAL_RESOURCE_CONFIG ASCEND_LOCAL_COMM_RES
    export HCCL_INTRA_ROCE_ENABLE=1

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
        --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_consumer","kv_port":32700,"kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":4,"pp_size":2},"decode":{"dp_size":4,"tp_size":2}}},{"kv_connector":"AscendStoreConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"backend":"memcache","lookup_rpc_port":"qwen38_pool_d","use_layerwise":false}}]}}'
    ```

    Complete the matching Section 5.2 API checks and start its Proxy, then perform the cache-reuse checks below.

=== "A3 series"

    The 128K/1K multi-prefix pooling example uses 16 devices, eight per role. Prefill uses TP2 × PP2 × DP2 (EP4) on devices 0–7, and Decode uses TP2 × DP4 (EP8) on devices 8–15. Both Mooncake configurations declare this topology. Both roles enable Engram CPU offload and DP shared memory.

    **Prepare a new pool:**

    The following values come from the single-host A3 configuration: MetaService and P/D share the host network, with 16 DRAM-enabled LocalServices and two SSD-backed devices already mapped in Section 4.3. Select these values before editing either INI file below. For an existing pool, skip this preparation block as described above.

    ```bash
    export MEMCACHE_META_URL="tcp://127.0.0.1:15000"
    export MEMCACHE_CONFIG_STORE_URL="tcp://127.0.0.1:16000"
    export MEMCACHE_METRICS_URL="http://127.0.0.1:18000"
    export MEMCACHE_EVICT_HIGH=70
    export MEMCACHE_EVICT_LOW=60
    export MEMCACHE_SSD_PATHS="/dev/loop0:/dev/loop1"
    export MEMCACHE_WORLD_SIZE=256
    export MEMCACHE_DRAM_SIZE=10GB
    export MEMCACHE_MAX_DRAM_SIZE=1024GB
    export MEMCACHE_UBS_IO_MEM_GB=10
    export MEMCACHE_DEVICE_COUNT=16
    export MEMCACHE_FORCE_NEW_DISK=true
    ```

    Replace `/dev/loop0:/dev/loop1` with the two device paths already mapped in Section 4.3. These new-pool values allocate `10GB` of DRAM and `10` GB for UBS IO per LocalService; `force_new_disk=true` reinitializes SSD cache storage at startup.

    After a host reboot, confirm that the backing filesystem is mounted and check the loop mappings with `losetup -l`. Identify any dedicated NVMe device by its serial number or WWN. If a device path changes, update both the Docker mapping and `ubsio.disk.path`. Run `blockdev --getsize64` on each configured device inside the container before starting P and D. If MemCache initialization fails after model loading, check these devices even if MetaService is listening.

    ```bash
    cp -n "${MEMCACHE_TEMPLATE_DIR}/mmc-meta.conf" "${MEMCACHE_CONFIG_DIR}/mmc-meta.conf"
    cp -n "${MEMCACHE_TEMPLATE_DIR}/mmc-local.conf" "${MEMCACHE_CONFIG_DIR}/mmc-local.conf"
    ```

    In the copied `${MEMCACHE_CONFIG_DIR}/mmc-meta.conf`, edit the following keys:

    ```ini
    ock.mmc.meta_service_url = <MEMCACHE_META_URL>
    ock.mmc.meta_service.config_store_url = <MEMCACHE_CONFIG_STORE_URL>
    ock.mmc.meta_service.metrics_url = <MEMCACHE_METRICS_URL>
    ock.mmc.log_level = info
    ock.mmc.evict_threshold_high = <MEMCACHE_EVICT_HIGH>
    ock.mmc.evict_threshold_low = <MEMCACHE_EVICT_LOW>
    ```

    Fill both INI files with the selected values; running `export` later does not update either file.

    In the copied `${MEMCACHE_CONFIG_DIR}/mmc-local.conf`, edit the following keys:

    ```ini
    ock.mmc.meta_service_url = <MEMCACHE_META_URL>
    ock.mmc.local_service.config_store_url = <MEMCACHE_CONFIG_STORE_URL>
    ock.mmc.log_level = info
    ock.mmc.local_service.world_size = <MEMCACHE_WORLD_SIZE>
    ock.mmc.local_service.protocol = device_sdma
    ock.mmc.local_service.dram.size = <MEMCACHE_DRAM_SIZE>
    ock.mmc.local_service.max.dram.size = <MEMCACHE_MAX_DRAM_SIZE>
    ock.mmc.local_service.storage.enabled = true
    ubsio.disk.path = <MEMCACHE_SSD_PATHS>
    ubsio.mem.size_in_gb = <MEMCACHE_UBS_IO_MEM_GB>
    ubsio.standalone.device_count = <MEMCACHE_DEVICE_COUNT>
    ubsio.standalone.force_new_disk = <MEMCACHE_FORCE_NEW_DISK>
    ```

    **Start MetaService:**

    Start MetaService in the serving container with the library exports above, unless the pool is already running:

    ```bash
    export MMC_META_CONFIG_PATH="${MEMCACHE_CONFIG_DIR}/mmc-meta.conf"
    python -c "from memcache_hybrid import MetaService; MetaService.main()"
    ```

    After its configured ports are listening (`ss -lntp`), initialize the model runtime from Section 4 and run the complete Prefill and Decode commands below in separate terminals. Set the library directories in each terminal as described above; `MEMCACHE_CONFIG_DIR` is the mounted personal configuration directory from Section 4.3.

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

    export MEMCACHE_LIB_DIR="<MEMCACHE_LIBRARY_DIRECTORY>"
    export MEMFABRIC_LIB_DIR="<MEMFABRIC_LIBRARY_DIRECTORY>"
    export PYTHON_LIB_DIR="$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
    export LD_LIBRARY_PATH="${MEMCACHE_LIB_DIR}:${MEMFABRIC_LIB_DIR}:${PYTHON_LIB_DIR}:${LD_LIBRARY_PATH:-}"
    export PYTHONHASHSEED=0
    export MMC_LOCAL_CONFIG_PATH="${MEMCACHE_CONFIG_DIR}/mmc-local.conf"
    unset HCCL_INTRA_ROCE_ENABLE ASCEND_GLOBAL_RESOURCE_CONFIG ASCEND_LOCAL_COMM_RES
    export ACL_OP_INIT_MODE=1
    export ASCEND_ENABLE_USE_FABRIC_MEM=1

    vllm serve Eco-Tech/Qwen3.8-Flash-Next-w4a8 \
        --host 0.0.0.0 \
        --port 25568 \
        --served-model-name qwen38-flash-next-a3-pd \
        --trust-remote-code \
        --quantization ascend \
        --tensor-parallel-size 2 \
        --pipeline-parallel-size 2 \
        --data-parallel-size 2 \
        --max-model-len 262144 \
        --max-num-seqs 32 \
        --max-num-batched-tokens 8192 \
        --gpu-memory-utilization 0.90 \
        --enable-prefix-caching \
        --async-scheduling \
        --enable-expert-parallel \
        --mamba-cache-mode align \
        --mamba-ssm-cache-dtype bfloat16 \
        --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
        --enforce-eager \
        --additional-config '{"enable_shared_expert_dp":true,"enable_flashcomm1":true,"enable_prefill_mc2":true,"enable_fused_mc2":0,"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"fuse_norm_quant":false}}' \
        --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}' \
        --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_producer","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_producer","kv_port":32600,"kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":2,"pp_size":2},"decode":{"dp_size":4,"tp_size":2}}},{"kv_connector":"AscendStoreConnector","kv_role":"kv_producer","kv_connector_extra_config":{"backend":"memcache","lookup_rpc_port":"qwen38_pool_p","use_layerwise":false}}]}}'
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

    export MEMCACHE_LIB_DIR="<MEMCACHE_LIBRARY_DIRECTORY>"
    export MEMFABRIC_LIB_DIR="<MEMFABRIC_LIBRARY_DIRECTORY>"
    export PYTHON_LIB_DIR="$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
    export LD_LIBRARY_PATH="${MEMCACHE_LIB_DIR}:${MEMFABRIC_LIB_DIR}:${PYTHON_LIB_DIR}:${LD_LIBRARY_PATH:-}"
    export PYTHONHASHSEED=0
    export MMC_LOCAL_CONFIG_PATH="${MEMCACHE_CONFIG_DIR}/mmc-local.conf"
    unset HCCL_INTRA_ROCE_ENABLE ASCEND_GLOBAL_RESOURCE_CONFIG ASCEND_LOCAL_COMM_RES
    export ACL_OP_INIT_MODE=1
    export ASCEND_ENABLE_USE_FABRIC_MEM=1

    vllm serve Eco-Tech/Qwen3.8-Flash-Next-w4a8 \
        --host 0.0.0.0 \
        --port 25569 \
        --served-model-name qwen38-flash-next-a3-pd \
        --trust-remote-code \
        --quantization ascend \
        --tensor-parallel-size 2 \
        --data-parallel-size 4 \
        --max-model-len 262144 \
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
        --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_consumer","kv_port":32700,"kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":2,"pp_size":2},"decode":{"dp_size":4,"tp_size":2}}},{"kv_connector":"AscendStoreConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"backend":"memcache","lookup_rpc_port":"qwen38_pool_d","use_layerwise":false}}]}}'
    ```

    Check the startup logs for the Prefill and Decode topologies above. Use the A3 API checks and Proxy command from Section 5.2, then perform the cache-reuse checks below.

=== "950DT products"

    This single-host example uses TP2 × DP2 (EP4) per role, with Prefill on devices 0–3 and Decode on devices 4–7. Use the prepared runtime from Section 4.3 and the same W4A8-MXFP checkpoint for both roles. Both roles use Mamba cache mode `align` and Mamba SSM cache dtype `bfloat16`.

    **Prepare a new pool:**

    For a new pool, use the following values from the supplied 102-server configuration. Its MetaService and ConfigStore use `172.27.24.102`, and its dedicated 500 GiB SSD-backed device is `/dev/loop0`. Replace the address and device for the target deployment, keeping the device consistent with Section 4.3. For an existing pool, skip this preparation block and continue at **Set the service environment** with the existing configuration files.

    ```bash
    local_ip="172.27.24.102"
    export MEMCACHE_META_URL="tcp://${local_ip}:25250"
    export MEMCACHE_CONFIG_STORE_URL="tcp://${local_ip}:25251"
    export MEMCACHE_METRICS_URL="http://${local_ip}:25252"
    export MEMCACHE_EVICT_HIGH=70
    export MEMCACHE_EVICT_LOW=60
    export MEMCACHE_SSD_PATHS="/dev/loop0"
    cp -n "${MEMCACHE_TEMPLATE_DIR}/mmc-meta.conf" "${MEMCACHE_CONFIG_DIR}/mmc-meta.conf"
    cp -n "${MEMCACHE_TEMPLATE_DIR}/mmc-local.conf" "${MEMCACHE_CONFIG_DIR}/mmc-local-p.conf"
    cp -n "${MEMCACHE_TEMPLATE_DIR}/mmc-local.conf" "${MEMCACHE_CONFIG_DIR}/mmc-local-d.conf"
    ```

    In the copied `${MEMCACHE_CONFIG_DIR}/mmc-meta.conf`, edit the following keys:

    ```ini
    ock.mmc.meta_service_url = <MEMCACHE_META_URL>
    ock.mmc.meta_service.config_store_url = <MEMCACHE_CONFIG_STORE_URL>
    ock.mmc.meta_service.metrics_url = <MEMCACHE_METRICS_URL>
    ock.mmc.log_level = info
    ock.mmc.log_output_target = screen
    ock.mmc.evict_threshold_high = <MEMCACHE_EVICT_HIGH>
    ock.mmc.evict_threshold_low = <MEMCACHE_EVICT_LOW>
    ```

    Set the resource values for both P/D LocalService copies:

    ```bash
    export MEMCACHE_WORLD_SIZE=256
    export MEMCACHE_DRAM_SIZE=10GB
    export MEMCACHE_MAX_DRAM_SIZE=1024GB
    export MEMCACHE_UBS_IO_MEM_GB=10
    export MEMCACHE_DEVICE_COUNT=8
    export MEMCACHE_FORCE_NEW_DISK=true
    ```

    In the copied `${MEMCACHE_CONFIG_DIR}/mmc-local-p.conf` and `${MEMCACHE_CONFIG_DIR}/mmc-local-d.conf`, edit the following keys:

    ```ini
    ock.mmc.meta_service_url = <MEMCACHE_META_URL>
    ock.mmc.local_service.config_store_url = <MEMCACHE_CONFIG_STORE_URL>
    ock.mmc.log_level = info
    ock.mmc.local_service.world_size = <MEMCACHE_WORLD_SIZE>
    ock.mmc.local_service.protocol = device_urma
    ock.mmc.local_service.dram.size = <MEMCACHE_DRAM_SIZE>
    ock.mmc.local_service.max.dram.size = <MEMCACHE_MAX_DRAM_SIZE>
    ock.mmc.local_service.storage.enabled = true
    ubsio.disk.path = <MEMCACHE_SSD_PATHS>
    ubsio.mem.size_in_gb = <MEMCACHE_UBS_IO_MEM_GB>
    ubsio.standalone.device_count = <MEMCACHE_DEVICE_COUNT>
    ubsio.standalone.force_new_disk = <MEMCACHE_FORCE_NEW_DISK>
    ```

    The eight embedded LocalServices each contribute `10GB` of pool DRAM and reserve `10` GB for UBS IO; no separate storage process is needed. `force_new_disk=true` recreates the SSD cache at stack startup, so keep the stack running during cache-reuse tests.

    **Set the service environment:**

    Open separate P, D, and Proxy terminals in the same container, plus a MetaService terminal only when creating a new pool. Initialize the runtime as described in Section 4.3, apply the library exports at the beginning of this section, and run the following in each terminal.

    `local_ip` is the model service IP; `nic_name` is the socket network interface that owns that IP. MetaService/ConfigStore URLs are separate TCP control endpoints and need only be reachable from all clients; loopback is suitable when all processes share the host network. `device_urma` selects the UB data path and does not make the socket interface a UB interface. The supplied example uses the same host IP for both purposes, but this is not a general requirement.

    ```bash
    local_ip="<LOCAL_SERVICE_IP>"
    nic_name="<SERVICE_NETWORK_INTERFACE>"

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
    export VLLM_ENGINE_READY_TIMEOUT_S=4800
    export HCCL_BUFFSIZE=512
    ulimit -l unlimited
    ulimit -s 65536
    ```

    **Start MetaService (new pool only):**

    Skip this command when reusing a running pool.

    ```bash
    export MMC_META_CONFIG_PATH="${MEMCACHE_CONFIG_DIR}/mmc-meta.conf"
    python -c "from memcache_hybrid import MetaService; MetaService.main()"
    ```

    For a new pool, wait until the ports configured in `mmc-meta.conf` are listening (`ss -lntp`); the example uses `25250`, `25251`, and `25252`. For an existing pool, check its actual configured endpoints instead. Then start Prefill.

    **Prefill:**

    In the Prefill terminal, first apply the library exports at the beginning of Section 5.5 and the **Set the service environment** block above, then run the following commands.

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
      --mamba-cache-mode align \
      --mamba-ssm-cache-dtype bfloat16 \
      --async-scheduling \
      --enable-expert-parallel \
      --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
      --compilation-config '{"cudagraph_mode":"NONE"}' \
      --additional-config '{"enable_cpu_binding":true,"enable_shared_expert_dp":true,"enable_fused_mc2":0,"enable_flashcomm1":true,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"fuse_norm_quant":false,"enable_npugraph_ex":true,"enable_static_kernel":false,"enable_super_kernel":false}}' \
      --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' \
      --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_producer","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_producer","kv_port":"32470","kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":2},"decode":{"dp_size":2,"tp_size":2}}},{"kv_connector":"AscendStoreConnector","kv_role":"kv_producer","kv_connector_extra_config":{"backend":"memcache","lookup_rpc_port":"0","use_layerwise":false}}]}}' \
      --enable-auto-tool-choice \
      --tool-call-parser qwen3_coder \
      --reasoning-parser qwen3 \
      --data-parallel-address "${VLLM_HOST_IP}" \
      --data-parallel-rpc-port 29770
    ```

    Wait for the Prefill API on port `8247` to become ready, then start Decode in a new terminal.

    **Decode:**

    In the Decode terminal, first apply the library exports at the beginning of Section 5.5 and the **Set the service environment** block above, then run the following commands.

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
      --mamba-cache-mode align \
      --mamba-ssm-cache-dtype bfloat16 \
      --async-scheduling \
      --enable-expert-parallel \
      --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
      --additional-config '{"enable_flashcomm1":true,"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"scheduler_config":{"recompute_scheduler_enable":true},"ascend_compilation_config":{"fuse_norm_quant":false,"enable_npugraph_ex":true,"enable_static_kernel":true,"enable_super_kernel":false}}' \
      --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":false}' \
      --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_consumer","kv_port":"32480","kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":2},"decode":{"dp_size":2,"tp_size":2}}},{"kv_connector":"AscendStoreConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"backend":"memcache","lookup_rpc_port":"1","use_layerwise":false}}]}}' \
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

    The tested configuration sets `lookup_rpc_port` to `"0"` for Prefill and `"1"` for Decode.

**Verify three-tier cache reuse:**

1. **Service:** Run the Section 5.2 chat-completion request for A2/A3, or the 950 DT request above, through the matching proxy and verify correct output and P-to-D transfer.
2. **External cache:** Send requests with shared prefixes and different suffixes; check MemCache lookup, put, get, and hit activity. A local prefix-cache hit alone does not prove external cache reuse.
3. **SSD restore:** Fill DRAM past the eviction watermark, confirm SSD eviction, then request an evicted prefix again and verify SSD read/restore activity and correct output. Keep MetaService and P/D running; SSD writes alone do not prove restoration.

Use the MemCache, MemFabric, and UBS IO logs or available metrics to check these results; counter names depend on the installed version.

**Key parameters:**

- Keep `PYTHONHASHSEED=0` consistent across workers. `lookup_rpc_port` identifies a local IPC endpoint with a DP-index suffix; use distinct P/D identifiers and avoid collisions with other pools.
- Retain `use_layerwise=false`: Prefill performs external prefix lookup and cache writes, while Decode receives KV/state through the P-to-D connector.

## 6 Functional Verification

Verify generation through `/v1/chat/completions`, using the API address and model name for your deployment:

| Deployment | API address | Model name |
| --- | --- | --- |
| A2 series colocated | `http://127.0.0.1:8088` | `qwen3.8-flash-next` |
| A3 series colocated | `http://127.0.0.1:8088` | `qwen3.8-flash-next` |
| 950DT products colocated | `http://127.0.0.1:8087` | `qwen3.8-flash-next-950-dt` |
| A2 series PD proxy | `http://127.0.0.1:25570` | `qwen38-flash-next-a2-pd` |
| A3 series PD proxy | `http://127.0.0.1:25570` | `qwen38-flash-next-a3-pd` |
| A2 series long-context PD proxy | `http://127.0.0.1:25570` | `qwen38-flash-next-a2-pd` |
| A3 series long-context PD proxy | `http://127.0.0.1:25570` | `qwen38-flash-next-a3-pd` |
| 950DT products PD proxy | `http://<LOCAL_SERVICE_IP>:8087` | `qwen38-flash-next-950-dt-pd` |
| 950DT products image-and-text 10K PD proxy | `http://127.0.0.1:8246` | `qwen38-megamoe-sp-pd` |
| 950DT products long-context PD proxy | `http://127.0.0.1:8246` | `qwen38-megamoe-sp-pd` |
| 950DT products three-tier pooling PD proxy | `http://127.0.0.1:8249` | `qwen38-megamoe-sp-pd` |

For PD, first complete the matching startup and proxy checks in Sections [5.2](#52-basic-pd-separation-deployment), [5.3](#53-image-and-text-10k-pd-separation-deployment), [5.4](#54-long-context-pd-separation-deployment), or [5.5](#55-pd-separation-with-memcache-three-tier-kv-cache-pool). Pooling also requires the SSD eviction/restore checks in Section 5.5.

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

To enable Function Calling, use the following options in the model commands for colocated deployment ([Section 5.1](#51-single-node-online-deployment)) or basic PD separation ([Section 5.2](#52-basic-pd-separation-deployment)). For PD separation, add them to both Prefill and Decode commands:

```text
--enable-auto-tool-choice --tool-call-parser qwen3_xml
```

The 950DT products pooling commands in [Section 5.5](#55-pd-separation-with-memcache-three-tier-kv-cache-pool) already enable Function Calling with `qwen3_coder`; retain that parser when verifying this deployment.

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

Reasoning parsing requires the following startup option. The colocated examples in [Section 5.1](#51-single-node-online-deployment) already include it; for basic PD separation in [Section 5.2](#52-basic-pd-separation-deployment), add it to both Prefill and Decode commands:

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

Follow [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for evaluation. The results below are for reference. The A3 result is from the W8A8 TP8 × DP1 configuration; validate accuracy for the checkpoint and deployment configuration you use.

| Hardware | Dataset | Metric | Score |
| --- | --- | --- | --- |
| A2 series | GPQA Diamond | Accuracy | 90.4 |
| A3 series | GPQA Diamond | Accuracy | 90.91 |
| 950DT products | GPQA Diamond | Accuracy | 91.2 |

## 8 Performance Evaluation

Use the endpoint and served model name for the deployment under test. Record request throughput, output token throughput, time to first token (TTFT), and time per output token (TPOT). Compare configurations using the same input/output lengths and request concurrency.

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### 8.2 Using vLLM Benchmark

After the service is ready, run this basic text benchmark in a separate terminal. The example targets the A2/A3 colocated service in Section 5.1; for another deployment, use its API address and served model name from [Section 6](#6-functional-verification) and its matching checkpoint for `--model`.

```bash
export VLLM_USE_MODELSCOPE=True
vllm bench serve \
    --backend openai-chat \
    --base-url http://127.0.0.1:8088 \
    --endpoint /v1/chat/completions \
    --model Eco-Tech/Qwen3.8-Flash-Next-w4a8 \
    --served-model-name qwen3.8-flash-next \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 128 \
    --num-prompts 100 \
    --max-concurrency 8 \
    --ignore-eos
```

Confirm that all requests succeed, then record throughput, TTFT, and TPOT from the benchmark summary. These request lengths and concurrency are a starting example; use the same workload when comparing deployments. Refer to [vLLM benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

## 9 Performance Tuning

### 9.1 Recommended Configurations

Use the configuration for your hardware and deployment mode. The table summarizes the deployment configurations in [Section 5](#5-online-service-deployment); tune context length, concurrency, and memory utilization for the target workload.

| Hardware | Deployment / role | Devices | TP × PP × DP | Max model length | Max sequences | Batch tokens |
| --- | --- | --- | --- | --- | --- | --- |
| A2 series | Colocated | 0–7 | 8 × 1 × 1 | 131072 | 8 | 4096 |
| A3 series | Colocated | 0–7 | 8 × 1 × 1 | 131072 | 8 | 4096 |
| 950DT products | Colocated | 0–3 | 4 × 1 × 1 | 141312 | 32 | 4096 |
| A2 series | Prefill | 0–7 on prefill host | 4 × 2 × 1 | 141312 | 32 | 8192 |
| A3 series | Prefill (basic PD) | 0–7 | 2 × 4 × 1 (EP2) | 262144 | 32 | 8192 |
| 950DT products | Prefill (basic PD) | 0–3 | 2 × 1 × 2 (EP4) | 143360 | 64 | 4096 |
| A2 series | Decode | 0–7 on decode host | 2 × 1 × 4 | 141312 | 32 | 8192 |
| A3 series | Decode (basic PD) | 8–15 | 2 × 1 × 4 (EP8) | 262144 | 32 | 8192 |
| 950DT products | Decode (basic PD) | 4–7 | 2 × 1 × 2 (EP4) | 143360 | 64 | 4096 |
| A2 series | Prefill (image-and-text 10K) | 0–7 on prefill host | 4 × 2 × 1 (EP4) | 262144 | 32 | 8192 |
| A3 series | Prefill (image-and-text 10K) | 0–7 | 4 × 2 × 1 (EP4) | 262144 | 32 | 8192 |
| 950DT products | Prefill (image-and-text 10K) | 0–3 | 2 × 2 × 1 | 143360 | 64 | 8192 |
| A2 series | Decode (image-and-text 10K) | 0–7 on decode host | 2 × 1 × 4 (EP8) | 262144 | 32 | 4096 |
| A3 series | Decode (image-and-text 10K) | 8–15 | 2 × 1 × 4 (EP8) | 262144 | 32 | 4096 |
| 950DT products | Decode (image-and-text 10K) | 4–7 | 2 × 1 × 2 | 143360 | 64 | 8192 |
| A2 series | Prefill (long-context) | 0–7 on prefill host | 2 × 4 × 1 (EP2) | 1010000 | 32 | 8192 |
| A3 series | Prefill (long-context) | 0–7 | 2 × 4 × 1 (EP2) | 1010000 | 32 | 8192 |
| 950DT products | Prefill (long-context) | 0–3 | 4 × 1 × 1 | 1010000 | 16 | 4096 |
| A2 series | Decode (long-context) | 0–7 on decode host | 2 × 1 × 4 (EP8) | 1010000 | 32 | 8192 |
| A3 series | Decode (long-context) | 8–15 | 2 × 1 × 4 (EP8) | 1010000 | 32 | 8192 |
| 950DT products | Decode (long-context) | 4–7 | 4 × 1 × 1 | 1010000 | 16 | 4096 |
| A3 series | Prefill (128K/1K three-tier pooling) | 0–7 | 2 × 2 × 2 (EP4) | 262144 | 32 | 8192 |
| A3 series | Decode (128K/1K three-tier pooling) | 8–15 | 2 × 1 × 4 (EP8) | 262144 | 32 | 8192 |
| 950DT products | Prefill (three-tier pooling) | 0–3 | 2 × 1 × 2 (EP4) | 262144 | 64 | 4096 |
| 950DT products | Decode (three-tier pooling) | 4–7 | 2 × 1 × 2 (EP4) | 262144 | 64 | 4096 |

Pair the P/D rows for your hardware. A2 uses two eight-device hosts; A3 uses one 16-device host. A2/A3 Prefill sets `--enforce-eager`; 950DT products Prefill sets `cudagraph_mode=NONE`. All PD Decode examples use `FULL_DECODE_ONLY` with static kernels enabled.

For MemCache three-tier pooling, A2 uses the basic PD token budgets above. Use the dedicated pooling rows for A3 and 950DT products and the corresponding commands in [Section 5.5](#55-pd-separation-with-memcache-three-tier-kv-cache-pool). Both 950DT products roles use `HCCL_BUFFSIZE=512`. Pool and UBS IO memory must be budgeted separately from Engram CPU offload.

### 9.2 Tuning Guidelines

Refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for tuning methods and the [Feature Matrix](../../user_guide/support_matrix/feature_matrix.md) for feature descriptions.

#### 9.2.1 Tool-Assisted Tuning with msmodeling

MindStudio-Modeling (msmodeling) tunes serving and benchmark parameters using measured results and its knowledge base.

For instructions, see [Optix-for-PD-disaggregation](https://gitcode.com/Ascend/msmodeling/blob/master/docs/en/quick_start/optix_agent_pd_disaggregation.md).

### 9.3 Model Runner V2 Decode Submission

Model Runner V2 automatically reduces CPU synchronization during decode when the target and MTP attention implementations support it; no extra configuration switch is required. Exact token counts and state remain on the device. Unsupported configurations retain exact CPU synchronization; runtime compatibility checks select the behavior.

## 10 FAQ

For common environment, installation, and parameter issues, refer to the [Public FAQs](../../faqs.md).
