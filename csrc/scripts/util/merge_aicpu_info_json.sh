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

echo $@
top_dir=$1
output_json=$2
first_json=$3
shift 3

merge_json_tool="${top_dir}/scripts/util/insert_op_info.py"
>$output_json

if [[ -f "$first_json" ]]
then
    cp -f $first_json $output_json
else
    echo "[ERROR] ${first_json} is not a file"
    exit 1
fi

for single_json in "$@"
do
    if [[ -f "${single_json}" ]]
    then
        python3 ${top_dir}/scripts/util/insert_op_info.py ${single_json} ${output_json}
    else
        echo "[ERROR] ${single_json} is not a file"
    fi
done
