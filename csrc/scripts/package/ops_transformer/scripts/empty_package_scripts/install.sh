#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# error number and description

if [ "$(id -u)" != "0" ]; then
  _LOG_PATH=$(echo "${HOME}")"/var/log/ascend_seclog"
  _INSTALL_LOG_FILE="${_LOG_PATH}/ascend_install.log"
else
  _LOG_PATH="/var/log/ascend_seclog"
  _INSTALL_LOG_FILE="${_LOG_PATH}/ascend_install.log"
fi

# log functions
getdate() {
  _cur_date=$(date +"%Y-%m-%d %H:%M:%S")
  echo "${_cur_date}"
}

logandprint() {
  is_error_level=$(echo $1 | grep -E 'ERROR|WARN|INFO')
  if [ "${is_quiet}" != "y" ] || [ "${is_error_level}" != "" ]; then
    echo "[OpsTransformer] [$(getdate)] ""$1"
  fi
  echo "[OpsTransformer] [$(getdate)] ""$1" >>"${_INSTALL_LOG_FILE}"
}
logandprint "[INFO]: Opp package installed successfully! The new version takes effect immediately."

exit 0
