=== "A2"

    #### Pull the image

    === "Ubuntu"

        <!-- doctest: installation-cann-image-a2-ubuntu -->
        ```bash
        export IMAGE=quay.io/ascend/cann:{{ release_cann_version }}-910b-ubuntu22.04-py3.12
        docker pull "$IMAGE"
        ```

    === "openEuler"

        <!-- doctest: installation-cann-image-a2-openeuler -->
        ```bash
        export IMAGE=quay.io/ascend/cann:{{ release_cann_version }}-910b-openeuler24.03-py3.12
        docker pull "$IMAGE"
        ```

    #### Start the container

    === "Ubuntu"

        ```bash
        export MODEL_CACHE="${HOME}/.cache"

        mkdir -p "$MODEL_CACHE"

        docker run --rm \
            --name vllm-ascend-cann \
            --shm-size=1g \
            -e ASCEND_VISIBLE_DEVICES=0 \
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
            -e ASCEND_VISIBLE_DEVICES=0 \
            -v "$MODEL_CACHE:/root/.cache" \
            -p 8000:8000 \
            -it "$IMAGE" bash
        ```
