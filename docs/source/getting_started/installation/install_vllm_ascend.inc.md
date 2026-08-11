    Choose one method.

    === "Standard pip wheel"

        !!! warning "Standard pip installation currently supports Atlas A2 only"

            The current standard PyPI `vllm-ascend` wheel is built for **Atlas A2**. It does not automatically support Atlas A3, Atlas 300I DUO, Atlas 200I Pro, or Atlas 950DT.

            For non-A2 hardware, use a pre-built image, WheelNext, or a source installation.

        ```bash
        pip install "vllm=={{ pip_vllm_version }}"
        pip install "vllm-ascend=={{ pip_vllm_ascend_version }}"
        ```

        Check the installed device build:

        ```bash
        python - <<'PY'
        from vllm_ascend._build_info import __device_type__

        print("vLLM Ascend wheel device type:", __device_type__)
        assert __device_type__ == "A2", __device_type__
        PY
        ```

    === "uv-wheelnext"

        WheelNext selects a vLLM Ascend wheel that matches the hardware from a variant repository.

        First, install and verify `uv`:

        ```bash
        case "$(uname -m)" in
            aarch64) UV_TARGET=uv-aarch64-unknown-linux-gnu ;;
            x86_64) UV_TARGET=uv-x86_64-unknown-linux-gnu ;;
            *) echo "Unsupported CPU architecture: $(uname -m)"; exit 1 ;;
        esac

        curl -LO "https://wheelnext.astral.sh/${UV_TARGET}.tar.gz"
        curl -LO "https://wheelnext.astral.sh/${UV_TARGET}.tar.gz.sha256"
        sha256sum -c "${UV_TARGET}.tar.gz.sha256"
        tar -xzf "${UV_TARGET}.tar.gz"
        install -m 0755 "${UV_TARGET}/uv" /usr/local/bin/uv
        uv --version
        ```

        Install vLLM:

        ```bash
        pip install "vllm=={{ pip_vllm_version }}"
        ```

        Install the vLLM Ascend variant:

        ```bash
        uv pip install --system \
            --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
            --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/variant \
            "vllm-ascend=={{ pip_vllm_ascend_version }}"
        ```

        Check the device build:

        ```bash
        python - <<'PY'
        from vllm_ascend._build_info import __device_type__

        print("vLLM Ascend wheel device type:", __device_type__)
        PY
        ```

        ??? warning "Known dependency metadata conflict"

            The current release wheel combination may declare inconsistent PyTorch versions for vLLM and vLLM Ascend, causing the final `pip check` to report a conflict.

            If you need a fully consistent and reproducible dependency environment, use a **pre-built vLLM Ascend image** or a **source installation**.

    === "Source installation"

        This method is intended for users who need to modify Python/C++/kernel code or build the environment in release dependency order.

        Install the base tools:

        === "Ubuntu / Debian"

            ```bash
            apt-get update -y
            apt-get install -y gcc g++ cmake ninja-build libnuma-dev git
            ```

        === "openEuler / RHEL-like"

            ```bash
            yum install -y gcc gcc-c++ cmake ninja-build numactl-devel git
            ```

        Clone the matching versions:

        ```bash
        git clone --depth 1 --branch {{ vllm_ascend_version }} \
            https://github.com/vllm-project/vllm-ascend.git
        cd vllm-ascend
        git submodule update --init --recursive --depth 1

        git clone --depth 1 --branch {{ vllm_version }} \
            https://github.com/vllm-project/vllm.git ../vllm
        ```

        Install the vLLM Ascend requirements first, then install vLLM:

        ```bash
        export PIP_EXTRA_INDEX_URL="https://mirrors.huaweicloud.com/ascend/repos/pypi"

        pip install -r requirements.txt
        pip install \
            "setuptools>=77,<81" \
            "setuptools-rust>=1.9" \
            "setuptools-scm>=8" \
            ninja

        VLLM_TARGET_DEVICE=empty pip install \
            --no-build-isolation \
            --extra-index-url https://download.pytorch.org/whl/cpu/ \
            -e ../vllm

        pip install --no-deps --no-build-isolation -e .
        ```

        In environments that support Triton Ascend, install the version specified in `requirements.txt` last:

        ```bash
        export TRITON_ASCEND_REQUIREMENT="$(
            grep -m1 -E '^triton-ascend==' requirements.txt
        )"

        test -n "$TRITON_ASCEND_REQUIREMENT"

        pip uninstall -y triton triton-ascend

        pip install \
            --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
            "$TRITON_ASCEND_REQUIREMENT"
        ```

        !!! note "Atlas 300I DUO / Atlas 200I Pro"

            For paths that do not support Triton/Triton Ascend, do not install the Triton combination above:

            ```bash
            pip uninstall -y triton triton-ascend
            ```

        After installation, run:

        ```bash
        PYTHONPATH= pip check
        ```

        Resolve any additional conflicts involving vLLM, vLLM Ascend, PyTorch, TorchNPU, or Triton Ascend before proceeding to the Quickstart.
