=== "Base environment"

    ## 4.3 Build from a Base Environment {: #base-environment }

    This path is intended for advanced users who need to manage the userspace software stack themselves.

    You can:

    - install on an existing Linux host;
    - or start with a minimal Linux container, then install CANN and vLLM Ascend.

    ### 4.3.1 Prepare the Base System {: #base-environment-prepare }

    For example, use Ubuntu 22.04:

    ```bash
    export BASE_IMAGE=ubuntu:22.04

    docker run --rm \
        --name vllm-ascend-base \
        --shm-size=4g \
        --net=host \
        -it "$BASE_IMAGE" bash
    ```

    !!! important "Device and driver mounts"

        For a minimal container to access the NPU, you must still map the device nodes and host driver files required by the target hardware into the container.

        Device mappings differ between hardware products, so this base environment example does not present a single `docker run` command as universally applicable.

        If you need to install CANN in the container, add the appropriate device and driver mounts according to the official CANN/container documentation for the target hardware.

    ### 4.3.2 Install CANN {: #base-environment-install-cann }

{% include "getting_started/installation/install_cann.inc.md" %}

    ### 4.3.3 Install vLLM and vLLM Ascend {: #base-environment-install-vllm }

{% include "getting_started/installation/install_vllm_ascend.inc.md" %}

    ### 4.3.4 Verify the Installation {: #base-environment-verify }

{% include "getting_started/installation/verify_installation.inc.md" %}

    ### 4.3.5 Completion Criteria {: #base-environment-complete }

    This path is complete after the CANN, NPU tensor operation, vLLM Ascend plugin registration, and core dependency checks all pass.
