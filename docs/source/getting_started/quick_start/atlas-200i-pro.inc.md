=== "Atlas 200I Pro"

    **This path uses**

    - Model: `Qwen/Qwen3.5-2B`
    - Precision: FP16
    - Parallelism: Quickstart uses TP=1 with one visible NPU
    - Mode: online serving
    - Model reference: [Qwen3.5 Dense](../tutorials/models/Qwen3.5-Dense.md)

    !!! important "Atlas 200I Pro container requirements"

        Atlas 200I Pro requires additional device nodes, driver libraries, and host configuration files.
        Before starting the container, confirm that the host paths mounted by the commands below exist.

    ### 3.1 Start the Container {: #quickstart-atlas-200i-pro-container }

    === "Ubuntu 24.04"

        ```bash
        export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-310p
        export MODEL_CACHE="${HOME}/.cache"

        mkdir -p "$MODEL_CACHE"
        docker pull "$IMAGE"

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

    === "openEuler 24.03"

        ```bash
        export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-310p-openeuler
        export MODEL_CACHE="${HOME}/.cache"

        mkdir -p "$MODEL_CACHE"
        docker pull "$IMAGE"

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
            -v /usr/lib64/libsemanage.so.2:/usr/lib64/libsemanage.so.2 \
            -v /usr/lib64/libmmpa.so:/usr/lib64/libmmpa.so \
            -v /usr/lib64/libcrypto.so.1.1:/usr/lib64/libcrypto.so.1.1 \
            -v /usr/lib64/libyaml-0.so.2.0.9:/usr/lib64/libyaml-0.so.2 \
            -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
            -v /usr/lib64/libstackcore.so:/usr/lib64/libstackcore.so \
            -v /etc/slog.conf:/etc/slog.conf \
            -v /var/slogd:/var/slogd \
            -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
            -v /usr/lib64/libtensorflow.so:/usr/lib64/libtensorflow.so \
            -v "$MODEL_CACHE:/root/.cache" \
            -p 8000:8000 \
            -it "$IMAGE" bash
        ```

    <a id="quickstart-atlas-200i-pro-verify"></a>

{% include "getting_started/quick_start/container_verification.inc.md" %}

    <a id="quickstart-atlas-200i-pro-inference"></a>

{% include "getting_started/quick_start/qwen35_serving.inc.md" %}
