First, install the system dependencies and configure the pip mirror.

```bash
# Using apt-get with mirror
sed -i 's|ports.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list
apt-get update -y && apt-get install -y gcc g++ cmake ninja-build libnuma-dev wget git curl jq
# Or using yum
# yum update -y && yum install -y gcc g++ cmake ninja-build numactl-devel wget git curl jq
# Config pip mirror, only versions 0.11.0 and earlier are supported, if using a version later than 0.11.0, do not execute this command
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

Optional: If you are working on an x86 machine or using a TorchNPU development version, configure pip's `extra-index`:

```bash
# For TorchNPU dev version or x86 machine
pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/"
```

Choose one of the following methods to install `vllm` and `vllm-ascend`. PyTorch and TorchNPU are installed automatically as dependencies during this step using the compatible versions for the selected release.

??? note "Installation method comparison"

    | Method | Package type | Hardware handling | Recommended for |
    | --- | --- | --- | --- |
    | **pip** | Prebuilt wheel | Installs the standard `vllm-ascend` wheel, which currently targets A2 | A2 users who want the simplest installation |
    | **uv** | Prebuilt wheel | Uses WheelNext wheel variants to select a compatible build for the detected hardware | Users who want a prebuilt wheel selected for their hardware |
    | **Source** | Source code | Builds and installs vLLM Ascend for the current environment | Development, customization, or environments without a suitable prebuilt wheel |

=== "pip installation"

    This installation method currently supports only **A2** builds and does not automatically support A3, Atlas 300I DUO, Atlas 200I Pro, or 950DT. For other hardware, use a prebuilt image, WheelNext, or a source installation.

    ```bash
    pip install "vllm=={{ release_vllm_version }}"
    pip install \
        --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
        "vllm-ascend=={{ release_vllm_ascend_version }}"
    ```

    ??? tip "If pip cannot verify the Huawei Cloud mirror certificate"

        The Huawei Cloud package index uses HTTPS, so `--trusted-host` is not normally required. If pip reports an SSL certificate verification or untrusted-host error for `mirrors.huaweicloud.com`, first update pip and the operating system CA certificates, or configure the CA bundle required by your network.

        As a temporary workaround on a trusted network, retry the vLLM Ascend installation command with `--trusted-host mirrors.huaweicloud.com`.

        This option tells pip to trust the host even when HTTPS validation fails, which weakens protection against man-in-the-middle attacks. Use it only when you trust the network and cannot fix the certificate configuration.

    Check the device build type:

    ```bash
    python - <<'PY'
    from vllm_ascend._build_info import __device_type__

    print("vLLM Ascend wheel device type:", __device_type__)
    assert __device_type__ == "A2", __device_type__
    PY
    ```

=== "uv installation"

    WheelNext selects a vLLM Ascend wheel that matches the hardware from the variant index. First, install and verify `uv`:

    ```bash
    # install uv-wheelnext
    curl -LsSf https://astral.sh/uv/install.sh | sed 's/verify_checksum "$_file"/true/' | INSTALLER_DOWNLOAD_URL=https://wheelnext.astral.sh sh
    source $HOME/.local/bin/env
    ```

    ```bash
    # Install vllm-project/vllm. The newest supported version is {{ vllm_version }}.
    pip install vllm=={{ release_vllm_version }}

    # Install vllm-project/vllm-ascend from wheelnext index.
    uv pip install --system \
        --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/variant \
        --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
        --find-links https://mirrors.huaweicloud.com/ascend/repos/pypi/triton-ascend/ \
        vllm-ascend=={{ release_vllm_ascend_version }}
    ```

    ??? tip "Clear the cache if uv installation fails"

        If `uv pip install` fails because of a corrupted cache or stale package data, clear the uv cache before running the installation command again:

            uv cache clean

    Check the device build type:

    ```bash
    python - <<'PY'
    from vllm_ascend._build_info import __device_type__
    print("vLLM Ascend wheel device type:", __device_type__)
    PY
    ```

=== "source installation"

    ???+ warning "A3 source installation"

        When installing vLLM Ascend from source for A3, run `git submodule update --init --recursive` manually or make sure that the environment has internet access so the required submodules can be downloaded automatically.

    Install vLLM first, then install vLLM Ascend:

    ```bash
    # Install vLLM.
    git clone --depth 1 --branch {{ vllm_version }} https://github.com/vllm-project/vllm
    cd vllm
    VLLM_TARGET_DEVICE=empty pip install -e .
    cd ..

    # Install vLLM Ascend.
    git clone --depth 1 --branch {{ vllm_ascend_version }} https://github.com/vllm-project/vllm-ascend.git
    cd vllm-ascend
    # git submodule update --init --recursive
    export ASCEND_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi
    pip install -e . --extra-index-url "${ASCEND_INDEX_URL}"
    cd ..
    ```

Finally, handle `triton` and `triton-ascend` according to the hardware:

=== "A2 / A3 / 950DT"

    To ensure that Triton Ascend matches the current vLLM Ascend version, remove any existing `triton` and `triton-ascend` packages before installing the corresponding version of `triton-ascend`:

    ```bash
    pip uninstall -y triton triton-ascend

    pip install triton-ascend=={{ release_triton_ascend_version }} \
        --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
    ```

    ??? note "Can community Triton and Triton Ascend coexist?"

        For Triton Ascend 3.2.0 and earlier, uninstall community Triton before installing Triton Ascend.

        For Triton Ascend 3.2.1 and later, installing Triton Ascend first installs the required community Triton package and then provides its Ascend-specific implementation over the shared installation paths. If Triton is reinstalled or upgraded afterward, it may overwrite the Triton Ascend files. In that case, uninstall both packages and reinstall Triton Ascend.

        For more Triton Ascend installation guidance, see [Triton-Ascend > FAQ](https://github.com/Ascend/triton-ascend/blob/main/docs/en/FAQ.md#1-installation-and-environment-configuration).

    When installing another version of `vllm-ascend`, check the repository's `requirements.txt` for the corresponding `triton-ascend` version.

=== "Atlas 300I DUO / Atlas 200I Pro"

    Atlas 300I DUO and Atlas 200I Pro do not support `triton` or `triton-ascend`. If either package is already installed in the environment, uninstall it and do not reinstall `triton-ascend`:

    ```bash
    pip uninstall -y triton triton-ascend
    ```
