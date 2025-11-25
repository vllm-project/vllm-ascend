#!/bin/bash

BISHENG_DIR="/vllm-workspace/Ascend-BiSheng"
BISHENG_PACKAGE="Ascend-BiSheng-toolkit_aarch64.run"
BISHENG_URL="https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/${BISHENG_PACKAGE}"
BISHENG_INSTALLER_PATH="${BISHENG_DIR}/${BISHENG_PACKAGE}"
BISHENG_ENV_FILE="/usr/local/Ascend/8.3.RC1/bisheng_toolkit/set_env.sh"

mkdir -p "${BISHENG_DIR}"
wget -P "${BISHENG_DIR}" "${BISHENG_URL}"
chmod a+x "${BISHENG_INSTALLER_PATH}"
"${BISHENG_INSTALLER_PATH}" --install
echo "source ${BISHENG_ENV_FILE}" >> ~/.bashrc