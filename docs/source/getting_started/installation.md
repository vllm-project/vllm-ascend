# Installation

This guide prepares the host and software environment for the Quickstart.

If you have already set up the Ascend driver/firmware, Docker, and a pre-built image environment and only want to run your first model, go directly to the [Quickstart](quick_start.md).

## 1. Identify Your Hardware {: #installation-hardware }

You do not need to understand internal chip codenames during installation. Choose the documentation path below based primarily on the actual product name.

| Installation/Quickstart path | Typical products |
| --- | --- |
| Atlas A2 | Atlas 800T A2, Atlas 900 A2 PoD, Atlas 200T A2 Box16, Atlas 300T A2, Atlas 800I A2, and others |
| Atlas A3 | Atlas 800T A3, Atlas 900 A3 SuperPoD, Atlas 9000 A3 SuperPoD, Atlas 800I A3, and others |
| Atlas 300I DUO | Atlas 300I DUO inference card |
| Atlas 200I Pro | Atlas 200I Pro inference card |
| Atlas 950DT | Atlas 950DT inference products |

??? tip "Not sure which hardware category to choose?"

    Do not determine the product type solely from the number of `/dev/davinci*` devices or the chip codename.

    Check the following first:

    - the server or accelerator card nameplate;
    - procurement records and device model;
    - official product pages and hardware documentation.

    For reference, see:

    - [Ascend product overview](https://www.hiascend.com/document/detail/en/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html)
    - [Atlas 800I A2](https://e.huawei.com/cn/products/computing/ascend/atlas-800i-a2)
    - [Atlas 300I DUO](https://e.huawei.com/cn/products/computing/ascend/atlas-300i-duo)
    - [Ascend hardware](https://www.hiascend.com/hardware/ai-server/)

## 2. Check the Host Environment {: #installation-host }

Whether you use a pre-built image, an existing CANN environment, or a source installation, the host must first have a working Ascend driver/firmware.

First, run:

```bash
uname -s
npu-smi info
```

Verify that:

- the operating system is Linux;
- `npu-smi info` lists the target NPU;
- the target NPU reports a healthy status.

If you plan to use Docker, also run:

```bash
docker --version
docker info >/dev/null && echo "Docker access: OK"
```

The command should display the Docker version and print:

```text
Docker access: OK
```

### 2.1 Troubleshoot Host Check Failures {: #installation-host-troubleshooting }

??? tip "`npu-smi` is missing or cannot detect the device"

    Check the management utility, driver information, and device nodes on the host:

    ```bash
    command -v npu-smi
    cat /usr/local/Ascend/driver/version.info
    ls -l /dev/davinci* /dev/davinci_manager /dev/devmm_svm
    ```

    If `npu-smi`, the driver information, or device nodes are missing, fix the host driver/firmware before installing vLLM Ascend.

    Refer to the official Huawei installation documentation for the driver/firmware installation and upgrade sequence.

??? tip "Docker is inaccessible"

    If Docker is not installed, follow the
    [official Docker Engine installation guide](https://docs.docker.com/engine/install/)
    for your Linux distribution.

    If Docker is already installed, first verify that the Docker daemon is running:

    ```bash
    systemctl status docker
    ```

    If the current user does not have permission to access the daemon, configure Docker user permissions according to your system security policy, or use an administrator-approved method to run Docker.

??? tip "Docker images cannot be downloaded"

    `docker pull` is initiated by the Docker daemon. The daemon does not necessarily use the shell's `HTTP_PROXY` / `HTTPS_PROXY` settings.

    Configure an organization-approved registry mirror or Docker daemon proxy.

## 3. Confirm Software Compatibility {: #installation-release-stack }

This installation page and the Quickstart use the same release baseline:

{% include "getting_started/software_stack.inc.md" %}

!!! important "Choose the complete version set before installation"

    CANN, PyTorch/TorchNPU, Triton Ascend, vLLM, and vLLM Ascend must be used in a compatible combination.
    Before installing a different release, consult the project's
    [release compatibility matrix](../community/versioning_policy.md#release-compatibility-matrix),
    rather than upgrading only one component.

## 4. Choose an Installation Method {: #installation-methods }

Choose one complete path based on your current environment. Do not mix individual steps from different paths.

| Starting point | Recommended method | Intended audience |
| --- | --- | --- |
| Need a working environment as quickly as possible | **Pre-built vLLM Ascend image** | Most users; recommended |
| Already have a compatible CANN environment | **Install in an existing CANN environment** | Users familiar with Python/CANN |
| Want to manage the complete userspace software stack | **Build from a base environment** | Advanced users / developers |

{% include "getting_started/installation/prebuilt_image.inc.md" %}

{% include "getting_started/installation/cann_environment.inc.md" %}

{% include "getting_started/installation/base_environment.inc.md" %}

## 5. Advanced Topics {: #installation-advanced }

The following sections cover build verification, multi-node environment preparation, and installation troubleshooting.

{% include "getting_started/installation/cpu_only_build.inc.md" %}

{% include "getting_started/installation/multi_node.inc.md" %}

{% include "getting_started/installation/troubleshooting.inc.md" %}

## 6. Installation Complete {: #installation-complete }

After the host, NPU backend, and vLLM Ascend plugin pass verification, the environment preparation stage is complete.

Next, proceed to the [Quickstart](quick_start.md), select your hardware, and run your first model inference or start an online service.
