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

CURPATH=$(dirname $(readlink -f "$0"))
USERNAME=$(id -un)
USERGROUP=$(id -gn)
common_func_path="${CURPATH}/common_func.inc"

. "$common_func_path"

INSTALL_PATH=""
IS_UPGRADE="n"

set_comm_log "Latest_manager" "$COMM_LOGFILE"

while true
do
    case "$1" in
    --install-path=*)
        INSTALL_PATH="$(echo "$1" | cut -d"=" -f2-)"
        shift
        ;;
    --upgrade)
        IS_UPGRADE="y"
        shift
        ;;
    -*)
        comm_log "ERROR" "Unsupported parameters : $1"
        exit 1
        ;;
    *)
        break
        ;;
    esac
done

if [ "$INSTALL_PATH" = "" ]; then
    comm_log "ERROR" "--install-path parameter is required!"
    exit 1
fi

if ! sh "$CURPATH/install_common_parser.sh" --package="latest_manager" --install --username="$USERNAME" --usergroup="$USERGROUP" \
    --simple-install "full" "$INSTALL_PATH" "$CURPATH/filelist.csv" "all"; then
    comm_log "ERROR" "install failed!"
    exit 1
fi

if [ "$IS_UPGRADE" = "y" ] && ! "$INSTALL_PATH/manager.sh" "migrate_latest_data"; then
    comm_log "ERROR" "migrate latest data failed!"
    exit 1
fi
