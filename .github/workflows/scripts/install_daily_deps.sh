#!/bin/bash
set -x
set -euo pipefail

# Install daily packages in Dockerfiles.
# DAILY_DEPS_MODE: "full" (all packages) or "torch_npu_only" (310p)
# Required env vars per mode:
#   full:            MEMFABRIC_VERSION/DATE, MEMCACHE_VERSION/DATE,
#                    TRITON_ASCEND_VERSION/PACKAGE_VERSION, TORCH_NPU_VERSION/DATE
#   torch_npu_only:  TORCH_NPU_VERSION/DATE

if [ "$BUILD_TYPE" != "daily" ]; then
    echo "Building release version without daily packages"
    exit 0
fi

echo "Building daily version with extra packages..."
ARCH=$(uname -m)

if [ "$DAILY_DEPS_MODE" = "torch_npu_only" ]; then
    # ---- torch-npu only ----
    echo "Download, extract and install torch-npu..."
    mkdir -p /tmp/torch_npu
    wget -q --retry-connrefused --tries=5 --timeout=30 --waitretry=10 \
        -O /tmp/torch_npu/torch_npu.tar.gz \
        "https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/Daily/v2.10.0-${TORCH_NPU_VERSION}/${TORCH_NPU_DATE}/pytorch_v2.10.0-${TORCH_NPU_VERSION}_py312.tar.gz"
    tar -xzf /tmp/torch_npu/torch_npu.tar.gz -C /tmp/torch_npu
    python3 -m pip install /tmp/torch_npu/torch_npu-2.10.0*_"${ARCH}".whl --force-reinstall --extra-index-url https://download.pytorch.org/whl/cpu/
    echo "Clean up temporary files..."
    rm -rf /tmp/torch_npu
    echo "Daily packages installation complete (torch_npu_only)."
    exit 0
fi

# ---- full mode ----
# ---- memfabric_hybrid ----
echo "Install memfabric_hybrid based on architecture..."
if [ "$ARCH" = "x86_64" ]; then
    MEMFABRIC_URL="https://obs-memfabric-hybrid.obs.cn-north-4.myhuaweicloud.com/mf/v1.2.0/${MEMFABRIC_DATE}/memfabric_hybrid-${MEMFABRIC_VERSION}-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
else
    MEMFABRIC_URL="https://obs-memfabric-hybrid.obs.cn-north-4.myhuaweicloud.com/mf/v1.2.0/${MEMFABRIC_DATE}/memfabric_hybrid-${MEMFABRIC_VERSION}-cp312-cp312-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl"
fi
python3 -m pip install "$MEMFABRIC_URL" --force-reinstall --no-deps

# ---- memcache_hybrid ----
echo "Install memcache_hybrid based on architecture..."
if [ "$ARCH" = "x86_64" ]; then
    MEMCACHE_URL="https://obs-memfabric-hybrid.obs.cn-north-4.myhuaweicloud.com/memcache/v1.2.0/${MEMCACHE_DATE}/memcache_hybrid-${MEMCACHE_VERSION}-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
else
    MEMCACHE_URL="https://obs-memfabric-hybrid.obs.cn-north-4.myhuaweicloud.com/memcache/v1.2.0/${MEMCACHE_DATE}/memcache_hybrid-${MEMCACHE_VERSION}-cp312-cp312-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl"
fi
python3 -m pip install "$MEMCACHE_URL" --force-reinstall --no-deps

# ---- triton-ascend ----
# Controlled by INSTALL_TRITON_ASCEND env var (default: false).
# Set INSTALL_TRITON_ASCEND=true to enable triton-ascend daily installation.
if [ "${INSTALL_TRITON_ASCEND:-false}" = "true" ]; then
    echo "Install triton-ascend..."
    TRITON_ASCEND_URL="https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/Triton_Innersource/B_Version/Triton%20Performance%20Optimization%20${TRITON_ASCEND_VERSION}/triton_ascend-${TRITON_ASCEND_PACKAGE_VERSION}-cp312-cp312-manylinux_2_27_${ARCH}.manylinux_2_28_${ARCH}.whl"
    python3 -m pip install "$TRITON_ASCEND_URL" --force-reinstall
else
    echo "Skipping triton-ascend (set INSTALL_TRITON_ASCEND=true to enable)"
fi

# ---- torch-npu ----
echo "Download, extract and install torch-npu..."
mkdir -p /tmp/torch_npu
wget -q --retry-connrefused --tries=5 --timeout=30 --waitretry=10 \
    -O /tmp/torch_npu/torch_npu.tar.gz \
    "https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/Daily/v2.10.0-${TORCH_NPU_VERSION}/${TORCH_NPU_DATE}/pytorch_v2.10.0-${TORCH_NPU_VERSION}_py312.tar.gz"
tar -xzf /tmp/torch_npu/torch_npu.tar.gz -C /tmp/torch_npu
python3 -m pip install /tmp/torch_npu/torch_npu-2.10.0*_"${ARCH}".whl --force-reinstall --extra-index-url https://download.pytorch.org/whl/cpu/
echo "Clean up temporary files..."
rm -rf /tmp/torch_npu

echo "Daily packages installation complete."