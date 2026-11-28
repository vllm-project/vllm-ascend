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

    !!! note "External URMA provider mount on some hosts"
        If `/usr/local/Ascend/driver/lib64/driver/urma` on the host is an absolute
        symlink to a directory outside the driver tree (typically
        `/usr/lib64/urma`), mounting the driver directory alone does not expose the
        link target inside the container, and `npu-smi info` hangs inside the
        container even though the host reports every device healthy. Mount the
        provider directory read-only as well:

        ```bash
        -v /usr/lib64/urma:/usr/lib64/urma:ro
        ```

        Check the host with `readlink /usr/local/Ascend/driver/lib64/driver/urma`.
        Provider directories that already live inside
        `/usr/local/Ascend/driver/lib64/` need no extra mount. See
        [the 950DT URMA FAQ](../../../faqs.md#faq-atlas-950dt-urma-provider-mount)
        for diagnosis and verification steps.

    === "Ubuntu"

        ```bash
        export MODEL_CACHE="${HOME}/.cache"

        mkdir -p "$MODEL_CACHE"

        # Add the line below only when the host driver links the URMA provider
        # directory outside the driver tree (see the note above):
        #   -v /usr/lib64/urma:/usr/lib64/urma:ro \
        docker run --rm \
            --name vllm-ascend-cann \
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
            --name vllm-ascend-cann \
            --net=host \
            --shm-size=1g \
            -e ASCEND_VISIBLE_DEVICES=0 \
            -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
            -v "$MODEL_CACHE:/root/.cache" \
            -it "$IMAGE" bash
        ```
