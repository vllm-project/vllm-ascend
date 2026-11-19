=== "Atlas 200I Pro"

    #### Pull the image

{% filter indent(4, true) %}{% include "getting_started/quick_start/ascend_image/image_download_mirror.inc.md" %}{% endfilter %}

    === "Ubuntu"

        <span id="quick-start-atlas-200i-pro-ubuntu"></span>

        ```bash
        export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-310p
        docker pull "$IMAGE"
        ```

    === "openEuler"

        <span id="quick-start-atlas-200i-pro-openeuler"></span>

        ```bash
        export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-310p-openeuler
        docker pull "$IMAGE"
        ```

    #### Start the container {: #quick-start-atlas-200i-pro-container }

    ???+ warning "Atlas 200I Pro container startup requirements"

        Atlas 200I Pro requires additional device nodes, driver libraries, and host configuration files. Before starting the container, make sure that all host paths mounted by the command below exist.

    === "Ubuntu"

        ```bash
        export MODEL_CACHE="${HOME}/.cache"

        mkdir -p "$MODEL_CACHE"

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

    === "openEuler"

        ```bash
        export MODEL_CACHE="${HOME}/.cache"

        mkdir -p "$MODEL_CACHE"

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
