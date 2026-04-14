#!/bin/csh
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set func_name = "$1"
switch ( "$func_name" )
    case "mk_custom_path":
        if ( "`id -u`" == 0 ) then
            exit 0
        endif
        set file_path = "$2"
        foreach line ("` cat $file_path `")
            set custom_path = "`echo '$line' | cut --only-delimited -d= -f2`"
            if ( "$custom_path" == "" ) then
                continue
            endif
            set custom_path = "` eval echo $custom_path `"
            if ( ! -d "$custom_path" ) then
                mkdir -p "$custom_path"
                if ( $status != 0 ) then
                    set cur_date = "`date +'%Y-%m-%d %H:%M:%S'`"
                    echo "[Common] [$cur_date] [ERROR]: create $custom_path failed."
                    exit 1
                endif
            endif
        end
        breaksw
    default:
        breaksw
endsw
