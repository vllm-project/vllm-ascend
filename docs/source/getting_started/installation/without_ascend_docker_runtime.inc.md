The commands below select the same devices as [Quick Start](quick_start.md#quick-start-installation): one NPU, which is `/dev/davinci0`, or `/dev/davinci0` and `/dev/davinci1` on Atlas A3. They also pass the management devices and driver mounts that Ascend Docker Runtime would otherwise provide. Add one `--device /dev/davinciN` line for each extra device. They are a reference for manual mapping, not a replacement for a model tutorial's container command: keep the image, the weight and cache mounts, the ports, the HCCL and network settings, and any extra devices from the tutorial page.

=== "Atlas A2 inference products"

    Quick Start selects one NPU, `/dev/davinci0`. A machine with eight NPUs exposes `/dev/davinci0` to `/dev/davinci7`; add one `--device` line for each extra NPU.

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
    export CONTAINER_NAME=vllm-ascend

    docker run --rm \
        --name ${CONTAINER_NAME} \
        --shm-size=1g \
        --net=host \
        --device /dev/davinci0 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /root/.cache:/root/.cache \
        -it $IMAGE bash
    ```

    For a multi-node deployment, add `-v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool`.

=== "Atlas A3 inference products"

    Quick Start selects one NPU. A3 uses a dual-DIE design, so that NPU is `/dev/davinci0` and `/dev/davinci1`. A machine with eight NPUs exposes `/dev/davinci0` to `/dev/davinci15`; NPU 0 and NPU 1 are `/dev/davinci0` to `/dev/davinci3`, and so on.

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3
    export CONTAINER_NAME=vllm-ascend

    docker run --rm \
        --name ${CONTAINER_NAME} \
        --shm-size=1g \
        --net=host \
        --device /dev/davinci0 \
        --device /dev/davinci1 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /root/.cache:/root/.cache \
        -it $IMAGE bash
    ```

    For a multi-node deployment, add `-v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool`.

=== "950DT products"

    Quick Start selects one NPU, `/dev/davinci0`. A machine with eight NPUs exposes `/dev/davinci0` to `/dev/davinci7`. The UB device nodes `/dev/ummu` and `/dev/uburma` are not part of the driver library set, so pass them explicitly as well.

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a5
    export CONTAINER_NAME=vllm-ascend

    docker run --rm \
        --name ${CONTAINER_NAME} \
        --shm-size=1g \
        --net=host \
        --device /dev/davinci0 \
        --device /dev/davinci_manager \
        --device /dev/hisi_hdc \
        --device /dev/ummu \
        --device /dev/uburma \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v /usr/local/sbin:/usr/local/sbin \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /root/.cache:/root/.cache \
        -it $IMAGE bash
    ```

    A PD deployment also mounts `/etc/hccl_rootinfo.json`, `/etc/hixlep/`, and `/lib/route.conf`; keep those mounts from the tutorial page.

=== "Atlas 300I Duo"

    Quick Start selects one chip, `/dev/davinci0`. The card exposes two chips, so a machine with two cards has `/dev/davinci0` to `/dev/davinci3`; add one `--device` line for each extra chip.

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-310p
    export CONTAINER_NAME=vllm-ascend

    docker run --rm \
        --name ${CONTAINER_NAME} \
        --shm-size=1g \
        --device /dev/davinci0 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /root/.cache:/root/.cache \
        -p 8000:8000 \
        -it $IMAGE bash
    ```

=== "Atlas 200I Pro"

    Atlas 200I Pro needs a privileged container plus the device nodes and host files below.

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-310p
    export MODEL_CACHE="${HOME}/.cache"

    docker run --rm \
        --privileged \
        --name vllm-ascend \
        --shm-size=10g \
        --device=/dev/davinci0:/dev/davinci0 \
        --device=/dev/davinci_manager \
        --device=/dev/ascend_manager \
        --device=/dev/user_config \
        -v /etc/sys_version.conf:/etc/sys_version.conf \
        -v /etc/ld.so.conf.d/mind_so.conf:/etc/ld.so.conf.d/mind_so.conf \
        -v /etc/hdcBasic.cfg:/etc/hdcBasic.cfg \
        -v /var/dmp_daemon:/var/dmp_daemon \
        -v /usr/lib64/libmmpa.so:/usr/lib64/libmmpa.so \
        -v /usr/lib64/libcrypto.so.1.1:/usr/lib64/libcrypto.so.1.1 \
        -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
        -v /usr/lib64/libstackcore.so:/usr/lib64/libstackcore.so \
        -v /usr/lib/aarch64-linux-gnu/libyaml-0.so.2:/usr/lib64/libyaml-0.so.2 \
        -v /etc/slog.conf:/etc/slog.conf \
        -v /var/slogd:/var/slogd \
        -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
        -v /usr/lib64/libtensorflow.so:/usr/lib64/libtensorflow.so \
        -v "$MODEL_CACHE:/root/.cache" \
        -p 8000:8000 \
        -it "$IMAGE" bash
    ```

    On an openEuler host, mount `/usr/lib64/libsemanage.so.2` and use the file that libyaml builds as the source of the `libyaml-0.so.2` mount instead.

Every `--device` line needs its node to exist on the host: `docker run` fails with `no such file or directory` when a device is missing, so drop that line rather than keeping a node the machine does not have. The same applies to the driver and host-file mounts listed above.
