=== "Base environment"

    <span id="installation-base-environment"></span>

    This path is intended for advanced users who need to manage the user-space software stack themselves. You can install on an existing Linux host or start from a minimal Linux container.

    #### Install CANN manually {: #installation-base-environment-install-cann }

    Please refer to [CANN Installation Resources](https://www.hiascend.com/cann/download) or the following code to complete the installation.

    The commands below use the default CANN and NNAL installation paths. If you install either component in a non-default directory, source the corresponding `set_env.sh` from the actual installation directory.

    ??? code

        ```bash
        # Create a virtual environment.
        python -m venv vllm-ascend-env
        source vllm-ascend-env/bin/activate

        # Install required Python packages.
        python -m pip install --upgrade pip
        pip3 install attrs numpy decorator sympy cffi pyyaml pathlib2 psutil protobuf scipy requests absl-py wheel typing_extensions

        # Download and install the CANN package.
        wget --header="Referer: https://www.hiascend.com/" https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%20{{ release_cann_version }}/Ascend-cann-toolkit_{{ release_cann_version }}_linux-"$(uname -i)".run
        chmod +x ./Ascend-cann-toolkit_{{ release_cann_version }}_linux-"$(uname -i)".run
        ./Ascend-cann-toolkit_{{ release_cann_version }}_linux-"$(uname -i)".run --full
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
        export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"

        wget --header="Referer: https://www.hiascend.com/" https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%20{{ release_cann_version }}/Ascend-cann-910b-ops_{{ release_cann_version }}_linux-"$(uname -i)".run
        chmod +x ./Ascend-cann-910b-ops_{{ release_cann_version }}_linux-"$(uname -i)".run
        ./Ascend-cann-910b-ops_{{ release_cann_version }}_linux-"$(uname -i)".run --install

        wget --header="Referer: https://www.hiascend.com/" https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%20{{ release_nnal_version }}/Ascend-cann-nnal_{{ release_nnal_version }}_linux-"$(uname -i)".run
        chmod +x ./Ascend-cann-nnal_{{ release_nnal_version }}_linux-"$(uname -i)".run
        ./Ascend-cann-nnal_{{ release_nnal_version }}_linux-"$(uname -i)".run --install

        source /usr/local/Ascend/nnal/atb/set_env.sh
        ```

    #### Install vLLM and vLLM Ascend {: #installation-base-environment-install }

{% filter indent(4, true) %}{% include "getting_started/installation/install_vllm_ascend.inc.md" %}{% endfilter %}
