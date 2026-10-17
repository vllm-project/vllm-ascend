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
    python3 -m pip install /tmp/torch_npu/torch_npu-2.10.0*_"${ARCH}".whl --extra-index-url https://download.pytorch.org/whl/cpu/
    echo "Clean up temporary files..."
    rm -rf /tmp/torch_npu
    echo "Daily packages installation complete (torch_npu_only)."
    exit 0
fi

# ---- full mode ----
# ---- memfabric_hybrid ----
echo "Resolving memfabric_hybrid wheel url..."
DOWNLOAD_URL_MF="https://obs-memfabric-hybrid.obs.cn-north-4.myhuaweicloud.com/mf/mf_version.json"
if [ ! -f "mf_version.json" ]; then
    echo "ERROR: download mf_version.json failed" >&2
    exit 1
fi

LINE=$(grep 'mf' mf_version.json | tail -n1)
MF_FULL=$(echo "${LINE}" | sed -n 's/.*"mf\/\([^"]*\)":.*/\1/p')
MF_VAL=$(echo "${LINE}" | sed -n 's/.*"mf[^"]*": "\([^"]*\)".*/\1/p')

MF_RELEASE_PART=$(echo "${MF_FULL}" | awk -F'/' '{print $1 "/" $2}')
MEMFABRIC_DATE=$(echo "${MF_FULL}" | awk -F'/' '{print $3}')
MEMFABRIC_VERSION=${MF_VAL}

if [ -z "${MF_FULL}" ] || [ -z "${MF_VAL}" ] || [ -z "${MF_RELEASE_PART}" ] || [ -z "${MEMFABRIC_DATE}" ];then
    echo "ERROR: parse mf info failed" >&2
    exit 1
fi

if [ "$ARCH" = "x86_64" ]; then
    MEMFABRIC_URL="https://obs-memfabric-hybrid.obs.cn-north-4.myhuaweicloud.com/mf/${MF_RELEASE_PART}/${MEMFABRIC_DATE}/memfabric_hybrid-${MEMFABRIC_VERSION}-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
else
    MEMFABRIC_URL="https://obs-memfabric-hybrid.obs.cn-north-4.myhuaweicloud.com/mf/${MF_RELEASE_PART}/${MEMFABRIC_DATE}/memfabric_hybrid-${MEMFABRIC_VERSION}-cp312-cp312-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl"
fi
python3 -m pip install "$MEMFABRIC_URL" --force-reinstall --no-deps

# ---- memcache_hybrid ----
echo "Resolving memcache_hybrid wheel url..."
DOWNLOAD_URL_MC="https://obs-memfabric-hybrid.obs.cn-north-4.myhuaweicloud.com/memcache/mc_version.json"
wget -q --no-check-certificate --tries=3 --timeout=30 "${DOWNLOAD_URL_MC}" -O mc_version.json
if [ ! -f "mc_version.json" ]; then
    echo "ERROR: download mc_version.json failed" >&2
    exit 1
fi

LINE_MEMCACHE=$(grep 'memcache' mc_version.json | tail -n1)
MC_FULL=$(echo "${LINE_MEMCACHE}" | sed -n 's/.*"memcache\/\([^"]*\)":.*/\1/p')
MC_VAL=$(echo "${LINE_MEMCACHE}" | sed -n 's/.*"memcache[^"]*": "\([^"]*\)".*/\1/p')

MC_RELEASE_PART=$(echo "${MC_FULL}" | awk -F'/' '{print $1 "/" $2}')
MEMCACHE_DATE=$(echo "${MC_FULL}" | awk -F'/' '{print $3}')
MEMCACHE_VERSION=${MC_VAL}

if [ -z "${MC_FULL}" ] || [ -z "${MC_VAL}" ] || [ -z "${MC_RELEASE_PART}" ] || [ -z "${MEMCACHE_DATE}" ];then
    echo "ERROR: parse memcache info failed" >&2
    exit 1
fi

if [ "$ARCH" = "x86_64" ]; then
    MEMCACHE_URL="https://obs-memfabric-hybrid.obs.cn-north-4.myhuaweicloud.com/memcache/${MC_RELEASE_PART}/${MEMCACHE_DATE}/memcache_hybrid-${MEMCACHE_VERSION}-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
else
    MEMCACHE_URL="https://obs-memfabric-hybrid.obs.cn-north-4.myhuaweicloud.com/memcache/${MC_RELEASE_PART}/${MEMCACHE_DATE}/memcache_hybrid-${MEMCACHE_VERSION}-cp312-cp312-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl"
fi

echo "MEMCACHE_URL=${MEMCACHE_URL}"
echo "Install memcache_hybrid: ${MEMCACHE_URL}"
python3 -m pip install "$MEMCACHE_URL" --force-reinstall --no-deps

# ---- mfcli kernel install ----
# Install the memfabric kernel for the target SoC. Only A5 and A3 require it;
# Disable torch device backend autoload so `import torch` does not pull in
# torch_npu (unusable on the CPU build host), which breaks mfcli's torch ABI
# check on A3/A5.
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
case "${SOC_VERSION:-}" in
    ascend950dt_9582)
        echo "Install memfabric kernel for A5..."
        mfcli kernel install --soc-version A5
        ;;
    ascend910_9391)
        echo "Install memfabric kernel for A3..."
        mfcli kernel install --soc-version A3
        ;;
    *)
        echo "Skip memfabric kernel install (SOC_VERSION=${SOC_VERSION:-})."
        ;;
esac

# ---- triton-ascend ----
# Controlled by INSTALL_TRITON_ASCEND env var (default: false).
# Set INSTALL_TRITON_ASCEND=true to enable triton-ascend daily installation.
echo "Install triton-ascend..."
TRITON_ASCEND_URL="https://ascend-triton-open.obs.cn-north-4.myhuaweicloud.com/triton-ascend/${TRITON_ASCEND_VERSION}/triton_ascend-${TRITON_ASCEND_PACKAGE_VERSION}-cp312-cp312-manylinux_2_27_${ARCH}.manylinux_2_28_${ARCH}.whl"
python3 -m pip install "$TRITON_ASCEND_URL"

# ---- torch-npu ----
echo "Download, extract and install torch-npu..."
mkdir -p /tmp/torch_npu
wget -q --retry-connrefused --tries=5 --timeout=30 --waitretry=10 \
    -O /tmp/torch_npu/torch_npu.tar.gz \
    "https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/Daily/v2.10.0-${TORCH_NPU_VERSION}/${TORCH_NPU_DATE}/pytorch_v2.10.0-${TORCH_NPU_VERSION}_py312.tar.gz"
tar -xzf /tmp/torch_npu/torch_npu.tar.gz -C /tmp/torch_npu
python3 -m pip install /tmp/torch_npu/torch_npu-2.10.0*_"${ARCH}".whl --extra-index-url https://download.pytorch.org/whl/cpu/
echo "Clean up temporary files..."
rm -rf /tmp/torch_npu

echo "Daily packages installation complete."
