#!/usr/bin/python
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import fileinput
import sys


def replace_paths_in_test_file(test_file_path, path):
    """
    替换test_quant_lightning_indexer_v2_batch.py中的占位符为实际路径
    :param test_file_path: test_quant_lightning_indexer_v2_batch.py的路径
    :param path1: 实际路径
    """
    try:
        # 逐行替换占位符
        with fileinput.FileInput(test_file_path, inplace=True, backup=".bak") as f:
            for line in f:
                # 替换__PATH__为实际路径
                line = line.replace("__PATH__", path)
                # 输出替换后的行（inplace=True会自动写回文件）
                print(line, end="")
        print(f" 已成功替换 {test_file_path} 中的路径")
    except Exception as e:
        print(f" 替换路径失败：{e}")
        sys.exit(1)


if __name__ == "__main__":
    test_file = sys.argv[1]
    path = sys.argv[2]
    replace_paths_in_test_file(test_file, path)
