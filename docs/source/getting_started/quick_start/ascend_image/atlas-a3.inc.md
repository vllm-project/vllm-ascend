=== "A3"

    #### Pull the image

{% filter indent(4, true) %}{% include "getting_started/quick_start/ascend_image/image_download_mirror.inc.md" %}{% endfilter %}

    === "Ubuntu"

        <span id="quick-start-atlas-a3-ubuntu"></span>

        ```bash
        export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3
        docker pull "$IMAGE"
        ```

    === "openEuler"

        <span id="quick-start-atlas-a3-openeuler"></span>

        ```bash
        export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3-openeuler
        docker pull "$IMAGE"
        ```

    #### Start the container {: #quick-start-atlas-a3-container }

    ???+ warning "A3 container startup requirements"

        A3 uses a dual-DIE design: one NPU appears as two DIE devices, so `ASCEND_VISIBLE_DEVICES=0-1` selects one NPU.

    === "Ubuntu"

        ```bash
        export MODEL_CACHE="${HOME}/.cache"

        mkdir -p "$MODEL_CACHE"

        docker run --rm \
            --name vllm-ascend \
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
            --name vllm-ascend \
            --shm-size=1g \
            -e ASCEND_VISIBLE_DEVICES=0-1 \
            -v "$MODEL_CACHE:/root/.cache" \
            -p 8000:8000 \
            -it "$IMAGE" bash
        ```
