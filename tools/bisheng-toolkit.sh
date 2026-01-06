#!/bin/bash
set -e

# Install Ascend BiSheng Toolkit for triton_ascend
ARCH=$(uname -m)
BISHENG_DIR="/vllm-workspace/Ascend-BiSheng"
BISHENG_NAME="Ascend-BiSheng-toolkit_${ARCH}_20260105.run"
BISHENG_URL="https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/${BISHENG_NAME}"
BISHENG_INSTALLER_PATH="${BISHENG_DIR}/${BISHENG_NAME}"
BISHENG_EVN_PATH="/usr/local/Ascend/tools/bishengir/bin"
mkdir -p "${BISHENG_DIR}"
wget -P "${BISHENG_DIR}" "${BISHENG_URL}"
chmod a+x "${BISHENG_INSTALLER_PATH}" 
"${BISHENG_INSTALLER_PATH}" --install
rm  -rf "${BISHENG_DIR}"
echo "export PATH=${BISHENG_EVN_PATH}:\$PATH" >> ~/.bashrc