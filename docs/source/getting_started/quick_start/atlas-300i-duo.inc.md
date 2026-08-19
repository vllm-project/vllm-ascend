=== "Atlas 300I DUO"

    **This path uses**

    - Model: `Qwen/Qwen3.5-2B`
    - Precision: FP16
    - Mode: online serving
    - Model reference: [Qwen3.5 Dense](../tutorials/models/Qwen3.5-Dense.md)

    !!! important "Runtime requirements for this hardware path"

        - Use the `-310p` image suffix.
        - The Qwen3.5 example uses `float16`.
        - Disable `enable_npugraph_ex`.
        - Atlas 300I DUO does not use `triton` / `triton-ascend`.

    ### 3.1 Start the Container {: #quickstart-atlas-300i-duo-container }

    === "Ubuntu"

        ```bash
        export DEVICE=/dev/davinci0
        export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-310p
        export MODEL_CACHE="${HOME}/.cache"

        mkdir -p "$MODEL_CACHE"
        docker pull "$IMAGE"

        docker run --rm \
            --name vllm-ascend \
            --shm-size=1g \
            --device "$DEVICE" \
            --device /dev/davinci_manager \
            --device /dev/devmm_svm \
            --device /dev/hisi_hdc \
            -v /usr/local/dcmi:/usr/local/dcmi \
            -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
            -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
            -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
            -v /etc/ascend_install.info:/etc/ascend_install.info \
            -v "$MODEL_CACHE:/root/.cache" \
            -p 8000:8000 \
            -it "$IMAGE" bash
        ```

    === "openEuler"

        ```bash
        export DEVICE=/dev/davinci0
        export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-310p-openeuler
        export MODEL_CACHE="${HOME}/.cache"

        mkdir -p "$MODEL_CACHE"
        docker pull "$IMAGE"

        docker run --rm \
            --name vllm-ascend \
            --shm-size=1g \
            --device "$DEVICE" \
            --device /dev/davinci_manager \
            --device /dev/devmm_svm \
            --device /dev/hisi_hdc \
            -v /usr/local/dcmi:/usr/local/dcmi \
            -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
            -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
            -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
            -v /etc/ascend_install.info:/etc/ascend_install.info \
            -v "$MODEL_CACHE:/root/.cache" \
            -p 8000:8000 \
            -it "$IMAGE" bash
        ```

    <a id="quickstart-atlas-300i-duo-verify"></a>

{% include "getting_started/quick_start/container_verification.inc.md" %}

    <a id="quickstart-atlas-300i-duo-inference"></a>

{% include "getting_started/quick_start/qwen35_serving.inc.md" %}
