===+ "CANN environment"

    <span id="installation-existing-cann"></span>

    This path applies to an official CANN base image or to CANN that is already installed on the host or in an existing container.

    #### Prepare the CANN environment {: #installation-existing-cann-prepare }

    === "Use a CANN base image"

        ??? note "Recommended CANN base images"

            | Hardware | Ubuntu | openEuler |
            | --- | --- | --- |
            | Ascend A2 series products | `quay.io/ascend/cann:{{ release_cann_version }}-910b-ubuntu22.04-py3.12` | `quay.io/ascend/cann:{{ release_cann_version }}-910b-openeuler24.03-py3.12` |
            | Ascend A3 series products | `quay.io/ascend/cann:{{ release_cann_version }}-a3-ubuntu22.04-py3.12` | `quay.io/ascend/cann:{{ release_cann_version }}-a3-openeuler24.03-py3.12` |
            | Atlas 300I DUO | `quay.io/ascend/cann:{{ release_cann_version }}-310p-ubuntu22.04-py3.12` | `quay.io/ascend/cann:{{ release_cann_version }}-310p-openeuler24.03-py3.12` |
            | Atlas 200I Pro | `quay.io/ascend/cann:{{ release_cann_version }}-310p-ubuntu22.04-py3.12` | `quay.io/ascend/cann:{{ release_cann_version }}-310p-openeuler24.03-py3.12` |
            | Ascend 950DT series products | `quay.io/ascend/cann:{{ release_cann_version }}-950-ubuntu22.04-py3.12` | `quay.io/ascend/cann:{{ release_cann_version }}-950-openeuler24.03-py3.12` |

        The CANN base image already includes the Toolkit, the operator package for the target hardware, and NNAL. You do not need to reinstall CANN in the container. For other operating systems and tags, see the [CANN Container Images Overview](https://github.com/Ascend/cann-container-image/blob/main/OVERVIEW.md).

        Select your hardware and operating system, then start the CANN container:

{% filter indent(8, true) %}{% include "getting_started/installation/cann_image/atlas-a2.inc.md" %}{% endfilter %}

{% filter indent(8, true) %}{% include "getting_started/installation/cann_image/atlas-a3.inc.md" %}{% endfilter %}

{% filter indent(8, true) %}{% include "getting_started/installation/cann_image/atlas-300i-duo.inc.md" %}{% endfilter %}

{% filter indent(8, true) %}{% include "getting_started/installation/cann_image/atlas-200i-pro.inc.md" %}{% endfilter %}

{% filter indent(8, true) %}{% include "getting_started/installation/cann_image/atlas-950dt.inc.md" %}{% endfilter %}

    === "Use an existing CANN installation"

        ???+ warning "Verify the NNAL environment"

            Confirm that `/usr/local/Ascend/nnal/atb/set_env.sh` and `libatb.so` are available. If CANN is installed elsewhere, source the corresponding `set_env.sh`. If a "libatb.so not found" error occurs at runtime, make sure that the manual installation steps installed NNAL correctly.

        ```bash
        source /usr/local/Ascend/ascend-toolkit/set_env.sh

        if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
            source /usr/local/Ascend/nnal/atb/set_env.sh
        fi

        export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
        npu-smi info
        ```

    #### Install vLLM and vLLM Ascend {: #installation-existing-cann-install }

{% filter indent(4, true) %}{% include "getting_started/installation/install_vllm_ascend.inc.md" %}{% endfilter %}
