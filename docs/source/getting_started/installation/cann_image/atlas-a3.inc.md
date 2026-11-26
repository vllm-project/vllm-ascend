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

        A3 uses a dual-DIE design: one NPU appears as two DIE devices, so `ASCEND_VISIBLE_DEVICES=0-1` selects one NPU.

    === "Ubuntu"

        ```bash
        export MODEL_CACHE="${HOME}/.cache"

        mkdir -p "$MODEL_CACHE"

        docker run --rm \
            --name vllm-ascend-cann \
            --shm-size=1g \
            -e ASCEND_VISIBLE_DEVICES=0-1 \
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
            -v "$MODEL_CACHE:/root/.cache" \
            -p 8000:8000 \
            -it "$IMAGE" bash
        ```
