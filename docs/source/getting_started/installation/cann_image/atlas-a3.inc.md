=== "A3"

    #### Pull the image

    === "Ubuntu"

        ```bash
        export IMAGE=quay.io/ascend/cann:{{ release_cann_version }}-a3-ubuntu22.04-py3.12
        docker pull "$IMAGE"
        ```

    === "openEuler"

        ```bash
        export IMAGE=quay.io/ascend/cann:{{ release_cann_version }}-a3-openeuler24.03-py3.12
        docker pull "$IMAGE"
        ```

    #### Start the container

    ???+ warning "A3 container startup requirements"

        A3 uses a dual-DIE design and requires two Ascend device nodes, such as `/dev/davinci0` and `/dev/davinci1`.

    === "Ubuntu"

        ```bash
        export MODEL_CACHE="${HOME}/.cache"

        mkdir -p "$MODEL_CACHE"

        docker run --rm \
            --name vllm-ascend-cann \
            --shm-size=1g \
            -e ASCEND_VISIBLE_DEVICES=0-1 \
            -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
            -v /etc/ascend_install.info:/etc/ascend_install.info \
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
            --shm-size=1g \
            -e ASCEND_VISIBLE_DEVICES=0-1 \
            -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
            -v /etc/ascend_install.info:/etc/ascend_install.info \
            -v "$MODEL_CACHE:/root/.cache" \
            -p 8000:8000 \
            -it "$IMAGE" bash
        ```
