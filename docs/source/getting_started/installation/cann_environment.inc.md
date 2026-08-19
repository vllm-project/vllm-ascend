=== "Existing CANN environment"

    ## 4.2 Install in an Existing CANN Environment {: #existing-cann-environment }

    This path covers two starting points: an official CANN base image, or CANN already installed on the host or in an existing container. Both approaches use the same subsequent installation steps for vLLM and vLLM Ascend.

    ### 4.2.1 Prepare the CANN Environment {: #prepare-existing-cann }

    === "CANN base image"

        Select a CANN image that matches the target hardware and the release baseline on this page:

        | Hardware | Recommended Ubuntu CANN base image |
        | --- | --- |
        | Atlas A2 | `quay.io/ascend/cann:{{ release_cann_version }}-910b-ubuntu22.04-py3.12` |
        | Atlas A3 | `quay.io/ascend/cann:{{ release_cann_version }}-a3-ubuntu22.04-py3.12` |
        | Atlas 300I DUO / Atlas 200I Pro | The official archive for `{{ release_cann_version }}` does not currently confirm a corresponding 310P base image tag. Use a pre-built vLLM Ascend image, or manually install the matching packages. |
        | Atlas 950DT | `quay.io/ascend/cann:{{ release_cann_version }}-950-ubuntu22.04-py3.12` |

        For other operating systems and available tags, refer to the
        [CANN Container Image Overview](https://github.com/Ascend/cann-container-image/blob/main/OVERVIEW.md).
        The CANN base image already includes the Toolkit, hardware operator package, and NNAL, so you do not need to reinstall CANN inside the container.

        The following example starts an Atlas A2 CANN base image:

        ```bash
        export DEVICE=/dev/davinci0
        export IMAGE=quay.io/ascend/cann:{{ release_cann_version }}-910b-ubuntu22.04-py3.12
        export MODEL_CACHE="${HOME}/.cache"

        mkdir -p "$MODEL_CACHE"
        docker pull "$IMAGE"

        docker run --rm \
            --name vllm-ascend-cann \
            --shm-size=4g \
            --net=host \
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
            -it "$IMAGE" bash
        ```

        Other hardware must use the corresponding device nodes and host driver mounts. For exact examples, see:

        - [Atlas A2 container mappings](quick_start.md#quickstart-atlas-a2-container)
        - [Atlas A3 container mappings](quick_start.md#quickstart-atlas-a3-container)
        - [Atlas 300I DUO container mappings](quick_start.md#quickstart-atlas-300i-duo-container)
        - [Atlas 200I Pro container mappings](quick_start.md#quickstart-atlas-200i-pro-container)
        - [Atlas 950DT container mappings](quick_start.md#quickstart-atlas-950dt-container)

    === "CANN already installed"

        If a compatible CANN version is already installed on the host or in an existing container, load the default installation paths:

        ```bash
        source /usr/local/Ascend/ascend-toolkit/set_env.sh

        if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
            source /usr/local/Ascend/nnal/atb/set_env.sh
        fi

        export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"

        npu-smi info
        ```

        If CANN is installed in another location, use the corresponding `set_env.sh`. Before continuing, verify that `/usr/local/Ascend/nnal/atb/set_env.sh` and `libatb.so` are available. If either is missing, first install the NNAL version that matches the Toolkit and hardware operator package.

    ### 4.2.2 Install vLLM and vLLM Ascend {: #existing-cann-install-vllm }

{% include "getting_started/installation/install_vllm_ascend.inc.md" %}

    ### 4.2.3 Verify the Installation {: #existing-cann-verify }

{% include "getting_started/installation/verify_installation.inc.md" %}

    ### 4.2.4 Completion Criteria {: #existing-cann-complete }

    This path is complete after the NPU tensor operation, vLLM Ascend plugin registration, and core dependency checks all pass.
