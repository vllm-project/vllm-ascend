# Quick Start

This guide uses Qwen3-0.6B as an example to help you run your first offline inference workload or deploy an online service on a prepared Ascend host using a prebuilt vLLM Ascend container.

## Requirements {: #quick-start-requirements }

- Operating system: Linux
- Python: {{ release_python_version }}
- Docker
- Supported hardware:

{% include "getting_started/installation/supported_hardware.inc.md" %}

??? note "Software stack included in the vLLM Ascend image"

    The prebuilt image includes a validated Python and Ascend user-space software stack, including CANN, NNAL, PyTorch, TorchNPU, vLLM, and vLLM Ascend.

    A2, A3, and 950DT images also include the matching Triton Ascend runtime. Atlas 300I DUO and Atlas 200I Pro do not use Triton Ascend.

    For the exact validated versions, see [Installation Guide > Hardware and software stack](installation.md#installation-hardware-software-stack).

## Installation {: #quick-start-installation }

Before using containers, make sure Docker is installed on your system. If Docker is not installed, please refer to the [Docker installation guide](https://docs.docker.com/get-started/get-docker/) for installation instructions.

{% include "getting_started/quick_start/ascend_image/atlas-a2.inc.md" %}

{% filter indent(4, true) %}{% include "getting_started/quick_start/ascend_image/verify_container.inc.md" %}{% endfilter %}

{% include "getting_started/quick_start/ascend_image/atlas-a3.inc.md" %}

{% filter indent(4, true) %}{% include "getting_started/quick_start/ascend_image/verify_container.inc.md" %}{% endfilter %}

{% include "getting_started/quick_start/ascend_image/atlas-300i-duo.inc.md" %}

{% filter indent(4, true) %}{% include "getting_started/quick_start/ascend_image/verify_container.inc.md" %}{% endfilter %}

{% include "getting_started/quick_start/ascend_image/atlas-200i-pro.inc.md" %}

{% filter indent(4, true) %}{% include "getting_started/quick_start/ascend_image/verify_container.inc.md" %}{% endfilter %}

{% include "getting_started/quick_start/ascend_image/atlas-950dt.inc.md" %}

{% filter indent(4, true) %}{% include "getting_started/quick_start/ascend_image/verify_container.inc.md" %}{% endfilter %}

## Inference {: #quick-start-inference }

The following sections provide offline inference and online serving examples. Choose the method you need to get started.

??? tip "If Hugging Face access is restricted"

    If your environment cannot reliably access Hugging Face, model downloads may fail due to connection timeouts, DNS errors, or other network issues. You can switch to ModelScope:

    ```bash
    export VLLM_USE_MODELSCOPE=True
    pip install "modelscope>=1.18.1,<1.38"
    ```

    If the model has already been downloaded locally, replace the model ID in the examples below with the local directory. You do not need to set this environment variable.

### Offline inference {: #quick-start-offline-inference }

=== "A2 / A3 / 950DT"

    <span id="quick-start-atlas-a2-offline"></span>
    <span id="quick-start-atlas-a3-offline"></span>
    <span id="quick-start-atlas-950dt-offline"></span>

{% filter indent(4, true) %}{% include "getting_started/quick_start/offline/qwen3-0.6b.inc.md" %}{% endfilter %}

=== "Atlas 300I DUO / Atlas 200I Pro"

    <span id="quick-start-atlas-300i-duo-offline"></span>
    <span id="quick-start-atlas-200i-pro-offline"></span>

{% filter indent(4, true) %}{% include "getting_started/quick_start/offline/qwen3-0.6b-310p.inc.md" %}{% endfilter %}

### Online serving {: #quick-start-online-serving }

=== "A2 / A3 / 950DT"

    <span id="quick-start-atlas-a2-online"></span>
    <span id="quick-start-atlas-a3-online"></span>
    <span id="quick-start-atlas-950dt-online"></span>

{% filter indent(4, true) %}{% include "getting_started/quick_start/online/qwen3-0.6b.inc.md" %}{% endfilter %}

=== "Atlas 300I DUO / Atlas 200I Pro"

    <span id="quick-start-atlas-300i-duo-online"></span>
    <span id="quick-start-atlas-200i-pro-online"></span>

{% filter indent(4, true) %}{% include "getting_started/quick_start/online/qwen3-0.6b-310p.inc.md" %}{% endfilter %}

## Next steps {: #quick-start-next-steps }

- See [Supported Models](../user_guide/support_matrix/supported_models.md) to choose another model.
- See [Model Tutorials](../tutorials/models/index.md) for deployment instructions for specific models.
- See [Installation Guide > Set up the software environment](installation.md#installation-software-environment) for pip, CANN, and source installation methods.
- See [Feature Tutorials](../tutorials/features/index.md) for distributed deployment and advanced features.
- See [FAQ](../faqs.md) to troubleshoot common deployment issues.
