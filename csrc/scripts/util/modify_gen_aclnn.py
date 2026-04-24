#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import sys
import os
from pathlib import Path
import subprocess
import logging as log


def shell_exec(cmd, shell=False):
    try:
        ps = subprocess.Popen(cmd, shell)
        ps.communicate(timeout=180)
    except BaseException as e:
        log.error(f"shell_exec error: {e}")
        sys.exit(1)


def search_file(aclnn_cpp):
    op_type = None
    index = 0
    with open(aclnn_cpp, 'r') as f:
        for line in f.readlines():
            index = index + 1
            if "_op_resource.h" in line:
                op_type = line.replace("_op_resource.h\"", "").replace("#include \"", "").strip()
            if "EXTERN_OP_RESOURCE" in line or "namespace op {" in line:
                break
    return (op_type, index)


def modify_gen_aclnn(build_path):
    auto_gen_cpps = Path(os.path.join(build_path, "autogen")).rglob("aclnn*.cpp")
    for aclnn_cpp in auto_gen_cpps:
        (op_type, index) = search_file(aclnn_cpp)
        if op_type:
            shell_exec(["bash", "-c", f"""sed -i 's/{op_type}_op_resource.h/op_resource.h/g' {aclnn_cpp}"""],
                       shell=False)
            shell_exec(
                ["bash", "-c", f"""sed -i 's/{op_type}_RESOURCES/AUTO_GEN_OP_RESOURCE({op_type})/g' {aclnn_cpp}"""],
                shell=False)
            shell_exec(["bash", "-c", f"sed -i '{index}i\\EXTERN_OP_RESOURCE({op_type})' {aclnn_cpp}"], shell=False)
    return

if __name__ == '__main__':
    modify_gen_aclnn(sys.argv[1])
