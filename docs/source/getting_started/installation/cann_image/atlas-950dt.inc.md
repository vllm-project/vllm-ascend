=== "950DT"

    #### Pull the image

    === "Ubuntu"

        ```bash
        export IMAGE=quay.io/ascend/cann:{{ release_cann_version }}-950-ubuntu22.04-py3.12
        docker pull "$IMAGE"
        ```

    === "openEuler"

        ```bash
        export IMAGE=quay.io/ascend/cann:{{ release_cann_version }}-950-openeuler24.03-py3.12
        docker pull "$IMAGE"
        ```

    #### Start the container

    ???+ note "Host URMA provider libraries"

        If `/usr/local/Ascend/driver/lib64/driver/urma` links to `/usr/lib64/urma` on the host, ensure that the link target is available inside the container.
        When it is absent, add `-v /usr/lib64/urma:/usr/lib64/urma:ro` to the `docker run` command below.
        See [950DT URMA provider troubleshooting](../faqs.md#atlas-950dt-urma-provider) for checks and verification.

    === "Ubuntu"

        ```bash
        export MODEL_CACHE="${HOME}/.cache"

        mkdir -p "$MODEL_CACHE"

        docker run --rm \
            --name vllm-ascend-cann \
            --net=host \
            --shm-size=1g \
            --device /dev/davinci0 \
            --device /dev/davinci_manager \
            --device /dev/ummu \
            --device /dev/uburma \
            --device /dev/hisi_hdc \
            -v /usr/local/dcmi:/usr/local/dcmi \
            -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
            -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
            -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
            -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
            -v /etc/ascend_install.info:/etc/ascend_install.info \
            -v "$MODEL_CACHE:/root/.cache" \
            -it "$IMAGE" bash
        ```

    === "openEuler"

        ```bash
        export MODEL_CACHE="${HOME}/.cache"

        mkdir -p "$MODEL_CACHE"

        docker run --rm \
            --name vllm-ascend-cann \
            --net=host \
            --shm-size=1g \
            --device /dev/davinci0 \
            --device /dev/davinci_manager \
            --device /dev/ummu \
            --device /dev/uburma \
            --device /dev/hisi_hdc \
            -v /usr/local/dcmi:/usr/local/dcmi \
            -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
            -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
            -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
            -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
            -v /etc/ascend_install.info:/etc/ascend_install.info \
            -v "$MODEL_CACHE:/root/.cache" \
            -it "$IMAGE" bash
        ```
