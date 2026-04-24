#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import json
import os
import sys
import stat
import const_var


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(sys.argv)
        print('argv error, inert_op_info.py your_op_file lib_op_file')
        sys.exit(2)

    with open(sys.argv[1], 'r') as load_f:
        insert_operator = json.load(load_f)

    all_operators = {}
    if os.path.exists(sys.argv[2]):
        if os.path.getsize(sys.argv[2]) != 0:
            with open(sys.argv[2], 'r') as load_f:
                all_operators = json.load(load_f)

    for k in insert_operator.keys():
        if k in all_operators.keys():
            print('replace op:[', k, '] success')
        else:
            print('insert op:[', k, '] success')
        all_operators[k] = insert_operator[k]

    with os.fdopen(os.open(sys.argv[2], const_var.WFLAGS, const_var.WMODES), 'w') as json_file:
        json_file.write(json.dumps(all_operators, indent=4))
