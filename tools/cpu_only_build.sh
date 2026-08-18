#!/usr/bin/env bash
# Build and verify vllm-ascend in an environment without an NPU or CANN.
#
# The Dockerfile.cpu and the CPU-only CI job both use this script so that
# dependency installation and validation have one executable source of truth.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTORCH_CPU_INDEX_URL="${PYTORCH_CPU_INDEX_URL:-https://download.pytorch.org/whl/cpu/}"
ASCEND_EXTRA_INDEX_URL="${ASCEND_EXTRA_INDEX_URL:-https://mirrors.huaweicloud.com/ascend/repos/pypi}"
SOC_VERSION="${SOC_VERSION:-ascend910b1}"

export COMPILE_CUSTOM_KERNELS=0
export SOC_VERSION
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PIP_DISABLE_PIP_VERSION_CHECK=1

install_deps() {
    python3 -m pip install --upgrade \
        pip "setuptools>=64" "setuptools-scm>=8" wheel \
        attrs googleapis-common-protos "cmake>=3.26" ninja tomli

    # Install the CPU wheels before the Ascend packages.  The later
    # requirements install then reuses these exact distributions instead of
    # resolving a CUDA-enabled PyTorch wheel from the default index.
    python3 -m pip install \
        --index-url "${PYTORCH_CPU_INDEX_URL}" \
        torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0

    # Resolve the complete build-system contract as well as the runtime
    # requirements.  The non-isolated build below intentionally reuses this
    # environment, but must not hide a newly added or un-installable
    # pyproject.toml build requirement.
    python3 - "${ASCEND_EXTRA_INDEX_URL}" <<'PY'
import subprocess
import sys

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

with open("pyproject.toml", "rb") as project:
    build_system_requires = tomllib.load(project)["build-system"]["requires"]
subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "--extra-index-url",
    sys.argv[1],
    *build_system_requires,
])
PY

    # Keep the explicitly documented Ascend wheel installation visible and
    # deterministic even when those packages are already build requirements.
    python3 -m pip install \
        --extra-index-url "${ASCEND_EXTRA_INDEX_URL}" \
        torch-npu==2.10.0.post4 triton-ascend==3.2.2

    # This is deliberately a normal requirements install.  Do not filter or
    # otherwise special-case individual packages: a new un-installable
    # dependency must fail this verification.
    python3 -m pip install \
        --extra-index-url "${ASCEND_EXTRA_INDEX_URL}" \
        -r requirements.txt
}

build_package() {
    python3 -m pip install \
        --no-build-isolation \
        --no-deps \
        -e .
}

verify_cpu_only() {
    if ls /dev/davinci* >/dev/null 2>&1 || [[ -e /dev/davinci_manager ]]; then
        echo "An Ascend NPU device is visible in the CPU-only environment." >&2
        return 1
    fi
    if [[ -d /usr/local/Ascend ]]; then
        echo "CANN is installed in the CPU-only environment." >&2
        return 1
    fi

    python3 - <<'PY'
import importlib.metadata
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
assert not torch.cuda.is_available(), "a CUDA device is visible in CPU-only CI"
print("vllm-ascend:", importlib.metadata.version("vllm-ascend"))
PY
    python3 -m pip check
}

case "${1:-all}" in
    install-deps)
        install_deps
        ;;
    build)
        build_package
        ;;
    verify)
        verify_cpu_only
        ;;
    all)
        install_deps
        build_package
        verify_cpu_only
        ;;
    *)
        echo "Usage: $0 [install-deps|build|verify|all]" >&2
        exit 2
        ;;
esac
