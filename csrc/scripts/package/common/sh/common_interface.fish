#!/usr/bin/env fish
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

function mk_custom_path
    set -l custom_file_path $argv[1]
    if test (id -u) -eq 0
        return 0
    end
    while read line
        set -l _custom_path (echo "$line" | cut --only-delimited -d= -f2)
        if test -z $_custom_path
            continue
        end
        set -l _custom_path (eval echo "$_custom_path")
        if not test -d $_custom_path
            mkdir -p "$_custom_path"
            if not test $status -eq 0
                set -l cur_date (date +"%Y-%m-%d %H:%M:%S")
                echo "[Common] [$cur_date] [ERROR]: create $_custom_path failed."
                return 1
            end
        end
    end < $custom_file_path
    return 0
end
