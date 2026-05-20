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
VARPATH="$(dirname "$CURPATH")"
USERNAME=$(id -un)
USERGROUP=$(id -gn)
common_func_path="$CURPATH/common_func.inc"
manager_func_path="$CURPATH/manager_func.sh"

. "$common_func_path"
. "$manager_func_path"

set_comm_log "Latest_manager" "$COMM_LOGFILE"

IS_UPGRADE="n"

while true
do
    case "$1" in
    --upgrade)
        IS_UPGRADE="y"
        shift
        ;;
    *)
        break
        ;;
    esac
done

if ! sh "$CURPATH/install_common_parser.sh" --package="latest_manager" --uninstall --username="$USERNAME" --usergroup="$USERGROUP" \
    --simple-uninstall "full" "$VARPATH" "$CURPATH/filelist.csv" "all"; then
    comm_log "ERROR" "uninstall failed!"
    exit 1
fi

if [ "$IS_UPGRADE" = "n" ]; then
    remove_manager_refs "$VARPATH"
fi

if ! remove_dir_if_empty "$VARPATH"; then
    comm_log "ERROR" "uninstall failed!"
    exit 1
fi
