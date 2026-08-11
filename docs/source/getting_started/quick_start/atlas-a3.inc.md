=== "Atlas A3"

    **This path uses**

    - Model: `Qwen/Qwen3-0.6B`
    - Mode: offline batch inference / online OpenAI-compatible serving
    - Model reference: [Qwen3 Dense](../tutorials/models/Qwen3-Dense.md)

    !!! note "A3 device mapping"

        Atlas A3 uses a dual-DIE design. This Quickstart exposes two Ascend device nodes for one selected A3 device, such as `/dev/davinci0` and `/dev/davinci1`.

        These two device nodes do not mean that Quickstart automatically uses
        `tensor_parallel_size=2`. Whether to enable parallelism depends on the specific model and deployment method.

    ### 3.1 Start the Container {: #quickstart-atlas-a3-container }

    === "Ubuntu"

        ```bash
        export DEVICE0=/dev/davinci0
        export DEVICE1=/dev/davinci1
        export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3
        export MODEL_CACHE="${HOME}/.cache"

        mkdir -p "$MODEL_CACHE"
        docker pull "$IMAGE"

        docker run --rm \
            --name vllm-ascend \
            --shm-size=1g \
            --device "$DEVICE0" \
            --device "$DEVICE1" \
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
        export DEVICE0=/dev/davinci0
        export DEVICE1=/dev/davinci1
        export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3-openeuler
        export MODEL_CACHE="${HOME}/.cache"

        mkdir -p "$MODEL_CACHE"
        docker pull "$IMAGE"

        docker run --rm \
            --name vllm-ascend \
            --shm-size=1g \
            --device "$DEVICE0" \
            --device "$DEVICE1" \
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

    <a id="quickstart-atlas-a3-verify"></a>

{% include "getting_started/quick_start/container_verification.inc.md" %}

    <a id="quickstart-atlas-a3-inference"></a>

{% include "getting_started/quick_start/qwen3_inference.inc.md" %}
