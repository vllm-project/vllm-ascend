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

    !!! note "External URMA provider mount on some hosts"
        If `/usr/local/Ascend/driver/lib64/driver/urma` on the host is an absolute
        symlink to a directory outside the driver tree (typically
        `/usr/lib64/urma`), mounting the driver directory alone does not expose the
        link target inside the container. `npu-smi info` then hangs even though the
        host reports every device healthy. Mount the provider directory read-only as
        well:

        ```bash
        -v /usr/lib64/urma:/usr/lib64/urma:ro
        ```

        Check the host with `readlink /usr/local/Ascend/driver/lib64/driver/urma`.
        Provider directories that already live inside
        `/usr/local/Ascend/driver/lib64/` need no extra mount. See
        [the 950DT URMA FAQ](../../../faqs.md#faq-atlas-950dt-urma-provider-mount) for
        diagnosis and verification steps.

    === "Ubuntu"

        ```bash
        export MODEL_CACHE="${HOME}/.cache"

        mkdir -p "$MODEL_CACHE"

        # Add the line below only when the host driver links the URMA provider
        # directory outside the driver tree (see the note above):
        #   -v /usr/lib64/urma:/usr/lib64/urma:ro \
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

        # Add the line below only when the host driver links the URMA provider
        # directory outside the driver tree (see the note above):
        #   -v /usr/lib64/urma:/usr/lib64/urma:ro \
        docker run --rm \
            --name vllm-ascend \
            --net=host \
            --shm-size=1g \
            -e ASCEND_VISIBLE_DEVICES=0 \
            -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
            -v "$MODEL_CACHE:/root/.cache" \
            -it "$IMAGE" bash
        ```
