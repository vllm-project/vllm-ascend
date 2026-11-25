These are the `docker run` commands the tutorials used before Ascend Docker Runtime, kept for hosts that cannot install it or choose not to. The runtime reads `ASCEND_VISIBLE_DEVICES` and binds the NPU nodes and driver paths at container start; these commands pass the same nodes with `--device` and the same paths with `-v`, so they also start on a host without the runtime.

For example, [Quick Start](quick_start.md#quick-start-installation) starts its container with one device: `/dev/davinci0`, or `/dev/davinci0` and `/dev/davinci1` on Atlas A3, where an NPU is two DIE devices. The tabs below do the same, and add one `--device /dev/davinciN` line for each further device.

Everything else stays as the tutorial you are adapting writes it: the weight and cache mounts, the ports, the HCCL and network settings, and any extra devices.

=== "Atlas A2 inference products"

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

=== "Atlas A3 inference products"

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

=== "950DT products"

    The UB device nodes `/dev/ummu` and `/dev/uburma` are not part of the driver library set, so pass them explicitly as well.

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

=== "Atlas 300I Duo"

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

    Atlas 200I Pro needs the device nodes and host files below. Make sure the container user can access the mapped devices.

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-310p
    export MODEL_CACHE="${HOME}/.cache"

    docker run --rm \
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

    On an openEuler host, use the `-310p-openeuler` image, mount `-v /usr/lib64/libsemanage.so.2:/usr/lib64/libsemanage.so.2`, and replace the `libyaml` source with `-v /usr/lib64/libyaml-0.so.2.0.9:/usr/lib64/libyaml-0.so.2`.

Every `--device` line needs its node to exist on the host: `docker run` fails with `no such file or directory` when a device is missing, so drop that line rather than keeping a node the machine does not have. The same applies to the driver and host-file mounts listed above.
