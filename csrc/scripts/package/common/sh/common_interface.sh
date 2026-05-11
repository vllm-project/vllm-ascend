#!/bin/sh
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

mk_custom_path() {
    if [ $(id -u) -eq 0 ]; then
        return 0
    fi
    local _custom_path_file="$1"
    while read line || [ -n "$line" ]
    do
        local _custom_path="$(echo "$line" | cut --only-delimited -d= -f2)"
        if [ -z "$_custom_path" ]; then
            continue
        fi
        eval "_custom_path=$_custom_path"
        if [ ! -d "$_custom_path" ]; then
            mkdir -p "$_custom_path"
            if [ $? -ne 0 ]; then
                cur_date="$(date +"%Y-%m-%d %H:%M:%S")"
                echo "[Common] [$cur_date] [ERROR]: create $_custom_path failed."
                return 1
            fi
        fi
    done < $_custom_path_file
    return 0
}

py_version_check(){
    local pyver_set="3.7 3.8 3.9 3.10 3.11"
    local cur_date="$(date +"%Y-%m-%d %H:%M:%S")"
    which python3 > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        local python_version="$(python3 --version 2>&1 | head -n 1)"
        local python3_version=$(echo "$python_version" | sed -n 's/.*[^\.0-9]\([0-9]\+\.[0-9]\+\).*/\1/p')
        if [ "x$python3_version" != "x" ]; then
            for ver in $pyver_set; do
                if [ "x$ver" = "x$python3_version" ]; then
                    return 0
                fi
            done
            
            echo "[Common] [$cur_date] [WARNING]: $python_version is not in Python3.7.x, Python3.8.x, Python3.9.x, Python3.10.x, Python3.11.x."
            return 1
        else
            echo "[Common] [$cur_date] [WARNING]: $python_version cannot be identified as a standard version, please check manually."
            return 1
        fi
    else
        echo "[Common] [$cur_date] [WARNING]: python3 is not found."
        return 1
    fi
}

