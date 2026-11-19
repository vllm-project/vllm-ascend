=== "A2"

    #### Pull the image

{% filter indent(4, true) %}{% include "getting_started/quick_start/ascend_image/image_download_mirror.inc.md" %}{% endfilter %}

    === "Ubuntu"

        <span id="quick-start-atlas-a2-ubuntu"></span>

        <!-- doctest: quickstart-image-a2-ubuntu -->
        ```bash
        export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
        docker pull "$IMAGE"
        ```

    === "openEuler"

        <span id="quick-start-atlas-a2-openeuler"></span>

        <!-- doctest: quickstart-image-a2-openeuler -->
        ```bash
        export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-openeuler
        docker pull "$IMAGE"
        ```

    #### Start the container {: #quick-start-atlas-a2-container }

    === "Ubuntu"

        ```bash
        export MODEL_CACHE="${HOME}/.cache"

        mkdir -p "$MODEL_CACHE"

        docker run --rm \
            --name vllm-ascend \
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
            --name vllm-ascend \
            --shm-size=1g \
            -e ASCEND_VISIBLE_DEVICES=0 \
            -v "$MODEL_CACHE:/root/.cache" \
            -p 8000:8000 \
            -it "$IMAGE" bash
        ```
