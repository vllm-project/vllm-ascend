#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

set -e

CURRENT_DIR=$(dirname $(readlink -f ${BASH_SOURCE[0]}))

# Distinguish whether it is fa or fag based on the operator name parameters passed in by cmake.
case_name=$1
test_program=$2

echo "=========================================== Run $case_name ====================================="

echo "=========================================== Execute $case_name sample start ===================="

echo "=========================================== Generating golden data ============================="
# generate golden data
python3 $CURRENT_DIR/gen_data.py
if [ $? -ne 0 ]; then
    echo "Error: Execute gen_data.py failed."
    exit 1
fi

# execute test program
${test_program}
if [ $? -ne 0 ];then
    echo "Error: Execute ${test_program} failed."
    exit 1
fi

# compare output
python3 $CURRENT_DIR/verify_result.py
if [ $? -ne 0 ]; then
    echo "Error: Execute verify_result.py failed."
    exit 1
fi
echo "=========================================== Execute $case_name sample end ======================"

echo "=========================================== Run $case_name success ============================="
exit 0
