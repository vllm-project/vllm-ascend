=== "950DT Products"

    #### Pull the image

{% filter indent(4, true) %}{% include "getting_started/quick_start/ascend_image/image_download_mirror.inc.md" %}{% endfilter %}

    === "Ubuntu"

        <span id="quick-start-atlas-950dt-ubuntu"></span>

        ```bash
        export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a5
        docker pull "$IMAGE"
        ```

    === "openEuler"

        <span id="quick-start-atlas-950dt-openeuler"></span>

        ```bash
        export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a5-openeuler
        docker pull "$IMAGE"
        ```

    #### Start the container {: #quick-start-atlas-950dt-container }

    === "Ubuntu"

        ```bash
        export MODEL_CACHE="${HOME}/.cache"

        mkdir -p "$MODEL_CACHE"

        docker run --rm \
            --name vllm-ascend \
            --net=host \
            --shm-size=1g \
            -e ASCEND_VISIBLE_DEVICES=0 \
            -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
            -v "$MODEL_CACHE:/root/.cache" \
            -it "$IMAGE" bash
        ```

    === "openEuler"

        ```bash
        export MODEL_CACHE="${HOME}/.cache"

        mkdir -p "$MODEL_CACHE"

        docker run --rm \
            --name vllm-ascend \
            --net=host \
            --shm-size=1g \
            -e ASCEND_VISIBLE_DEVICES=0 \
            -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
            -v "$MODEL_CACHE:/root/.cache" \
            -it "$IMAGE" bash
        ```
