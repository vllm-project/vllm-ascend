=== "Atlas 950DT"

    **This path uses**

    - Image: a prebuilt image specifically for Atlas 950DT
    - Environment verification: the NPU, PyTorch, and vLLM Ascend plugin inside the container
    - Model verification reference: [DeepSeek-V4-Flash](../tutorials/models/DeepSeek-V4-Flash.md)

    ### 3.1 Start the Container {: #quickstart-atlas-950dt-container }

    The commands below expose eight NPUs on an Atlas 950DT node. Before running a command, confirm that `/dev/davinci0` through `/dev/davinci7` exist on the host and adjust the command for the actual device topology.

    === "Ubuntu"

        ```bash
        export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-950dt
        export MODEL_CACHE="${HOME}/.cache"

        mkdir -p "$MODEL_CACHE"
        docker pull "$IMAGE"

        docker run --rm \
            --name vllm-ascend \
            --net=host \
            --shm-size=1g \
            --device /dev/davinci0 \
            --device /dev/davinci1 \
            --device /dev/davinci2 \
            --device /dev/davinci3 \
            --device /dev/davinci4 \
            --device /dev/davinci5 \
            --device /dev/davinci6 \
            --device /dev/davinci7 \
            --device /dev/davinci_manager \
            --device /dev/devmm_svm \
            --device /dev/hisi_hdc \
            -v /usr/local/dcmi:/usr/local/dcmi \
            -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
            -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
            -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
            -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
            -v /etc/ascend_install.info:/etc/ascend_install.info \
            -v "$MODEL_CACHE:/root/.cache" \
            -it "$IMAGE" bash
        ```

    === "openEuler"

        ```bash
        export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-950dt-openeuler
        export MODEL_CACHE="${HOME}/.cache"

        mkdir -p "$MODEL_CACHE"
        docker pull "$IMAGE"

        docker run --rm \
            --name vllm-ascend \
            --net=host \
            --shm-size=1g \
            --device /dev/davinci0 \
            --device /dev/davinci1 \
            --device /dev/davinci2 \
            --device /dev/davinci3 \
            --device /dev/davinci4 \
            --device /dev/davinci5 \
            --device /dev/davinci6 \
            --device /dev/davinci7 \
            --device /dev/davinci_manager \
            --device /dev/devmm_svm \
            --device /dev/hisi_hdc \
            -v /usr/local/dcmi:/usr/local/dcmi \
            -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
            -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
            -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
            -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
            -v /etc/ascend_install.info:/etc/ascend_install.info \
            -v "$MODEL_CACHE:/root/.cache" \
            -it "$IMAGE" bash
        ```

    <a id="quickstart-atlas-950dt-verify"></a>

{% include "getting_started/quick_start/container_verification.inc.md" %}

    ### 3.3 Verify a Model {: #quickstart-atlas-950dt-model }

    Model deployment on Atlas 950DT might require a specialized device topology, parallelism, communication, quantization, NPU memory, and runtime parameters. After verifying the container environment above, continue in the following order:

    1. Check the current support status of the model on Atlas 950DT in [Supported Models](../user_guide/support_matrix/supported_models.md).
    2. Follow [DeepSeek-V4-Flash](../tutorials/models/DeepSeek-V4-Flash.md) to prepare the model weights and model-specific parameters.
    3. Adjust the deployment parameters for the actual 950DT topology, and then verify the model.

    This page does not repeat the general A2/A3 model commands, to avoid giving the impression that a command that only verifies the environment also verifies a model on 950DT.
