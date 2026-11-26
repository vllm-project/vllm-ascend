=== "Atlas 200I Pro"

    #### Pull the image

    === "Ubuntu"

        ```bash
        export IMAGE=quay.io/ascend/cann:{{ release_cann_version }}-310p-ubuntu22.04-py3.12
        docker pull "$IMAGE"
        ```

    === "openEuler"

        ```bash
        export IMAGE=quay.io/ascend/cann:{{ release_cann_version }}-310p-openeuler24.03-py3.12
        docker pull "$IMAGE"
        ```

    #### Start the container

    ???+ warning "Atlas 200I Pro container startup requirements"

        Atlas 200I Pro selects its NPU with `ASCEND_VISIBLE_DEVICES`, so the container needs Ascend Docker Runtime on the host. The runtime mounts the driver libraries, but its default mount list looks for `npu-smi` at `/usr/local/bin/npu-smi`, while this product installs it at `/usr/local/sbin/npu-smi`, so the command below mounts it explicitly. The command also passes the `ascend_manager` and `user_config` devices and the Atlas 300I Duo host libraries. Make sure every host path it mounts exists.

    === "Ubuntu"

        ```bash
        export MODEL_CACHE="${HOME}/.cache"

        mkdir -p "$MODEL_CACHE"

        docker run --rm \
            --name vllm-ascend-cann \
            --shm-size=10g \
            -e ASCEND_VISIBLE_DEVICES=0 \
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
            --name vllm-ascend-cann \
            --shm-size=10g \
            -e ASCEND_VISIBLE_DEVICES=0 \
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
            -v /usr/lib64/libtensorflow.so:/usr/lib64/libtensorflow.so \
            -v "$MODEL_CACHE:/root/.cache" \
            -p 8000:8000 \
            -it "$IMAGE" bash
        ```
