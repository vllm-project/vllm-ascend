# Quickstart

This guide helps you quickly run your first inference or online serving workload with a prebuilt vLLM Ascend container on a prepared Ascend host.

If you have not yet installed the Ascend driver, firmware, or Docker, or if you want to prepare the runtime environment with pip, an existing CANN environment, or a source build, first read
[Installation Guide > 2. Check the Host Environment](installation.md#installation-host).

## 1. Verify the Environment {: #quickstart-prerequisites }

Quickstart performs only the minimum environment checks. Run the following commands on the **host**:

```bash
npu-smi info
docker info >/dev/null && echo "Docker access: OK"
```

You can continue when both of the following conditions are met:

- `npu-smi info` lists the Ascend NPU that you plan to use.
- The Docker daemon is accessible and the command prints `Docker access: OK`.

If either check fails, follow
[Installation Guide > 2.1 Troubleshoot Host Check Failures](installation.md#installation-host-troubleshooting)
to prepare and verify the host environment before continuing.

If Docker is not yet installed, go directly to the
[official Docker Engine installation guide](https://docs.docker.com/engine/install/), select the Linux distribution that matches your host, and return to this page after the installation is complete.

!!! important "Confirm hardware support"
    If you are unsure whether your product supports vLLM Ascend, see [Installation Guide > 1. Identify Your Hardware](installation.md#installation-hardware).

## 2. Confirm Version Compatibility {: #quickstart-version-compatibility }

This Quickstart uses a single release baseline by default. It does not maintain separate software version matrices for different hardware products.

### 2.1 Current Release Stack {: #quickstart-release-stack }

{% include "getting_started/software_stack.inc.md" %}

!!! important "Keep the entire version stack compatible"

    Upgrading vLLM, vLLM Ascend, CANN, PyTorch/TorchNPU, or Triton Ascend individually in a prebuilt image is not recommended.
    If you need other versions, use a complete compatible combination instead of replacing only one component.

    For other releases, see the project-maintained
    [Release compatibility matrix](../community/versioning_policy.md#release-compatibility-matrix),
    and switch the entire stack together. For more installation options, see
    [Installation Guide > 3. Confirm Software Compatibility](installation.md#installation-release-stack).

## 3. Select Your Hardware and Run the Example {: #quickstart-hardware }

Select a path based on the actual product name of your device. An image tag might contain a chip-series suffix, but you should still select the path based on the product name of your server or accelerator card.

| Your hardware | Example on this page | Path provided on this page |
| --- | --- | --- |
| [Atlas A2](#quickstart-atlas-a2-container) | `Qwen/Qwen3-0.6B` | Container, offline inference, and online serving |
| [Atlas A3](#quickstart-atlas-a3-container) | `Qwen/Qwen3-0.6B` | Container, offline inference, and online serving |
| [Atlas 300I DUO](#quickstart-atlas-300i-duo-container) | `Qwen/Qwen3.5-2B` | Container, offline inference status, and online serving |
| [Atlas 200I Pro](#quickstart-atlas-200i-pro-container) | `Qwen/Qwen3.5-2B` | Container, offline inference status, and online serving |
| [Atlas 950DT](#quickstart-atlas-950dt-container) | DeepSeek-V4-Flash reference path | Container, environment verification, and model-specific documentation |

Select your hardware below and follow the steps in order.

{% include "getting_started/quick_start/atlas-a2.inc.md" %}
{% include "getting_started/quick_start/atlas-a3.inc.md" %}
{% include "getting_started/quick_start/atlas-300i-duo.inc.md" %}
{% include "getting_started/quick_start/atlas-200i-pro.inc.md" %}
{% include "getting_started/quick_start/atlas-950dt.inc.md" %}

## 4. Troubleshooting {: #quickstart-troubleshooting }

This section contains only a few entry points for issues that directly affect Quickstart. For comprehensive environment troubleshooting, see the [Installation Guide](installation.md)
and [FAQs](../faqs.md).

??? tip "The image pull is slow or fails"

    `docker pull` is initiated by the Docker daemon. Proxy variables in the shell do not automatically configure the Docker daemon.

    Use a registry mirror or Docker daemon proxy permitted by your organization.

??? tip "The model download fails"

    If the model is available on ModelScope, you can use:

    ```bash
    export VLLM_USE_MODELSCOPE=True
    pip install "modelscope>=1.18.1,<1.38"
    ```

??? tip "The service does not become ready"

    Check the `vllm serve` logs first. Common causes include:

    - The model download failed.
    - NPU memory is insufficient.
    - The host driver does not match the image.
    - The device mapping is incorrect.
    - Model-specific parameters are incorrect.

## 5. Next Steps {: #quickstart-next-steps }

- See [Supported Models](../user_guide/support_matrix/supported_models.md) to select more models.
- See [Model Tutorials](../tutorials/models/index.md) for model-specific deployment steps.
- See [Installation Guide > 4. Choose an Installation Method](installation.md#installation-methods)
  to learn about pip, CANN, and source installations.
- See [Feature Tutorials](../tutorials/features/index.md) to learn about distributed and advanced features.
- See [FAQs](../faqs.md) to troubleshoot common deployment issues.
