#!/usr/bin/env python3
# coding: utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
test_config.yaml 格式转换

转换成 ops-nn 仓的格式, 方便ci读取
"""

import logging
import sys
import os
import yaml


def load_test_config(test_config_path: str):
    """读取并解析test_config.yaml文件"""
    try:
        with open(test_config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    except Exception as e:
        logging.error(f"Failed to read test_config.yaml file: {e}")
        return None


def extract_src_and_exclude(data):
    """从数据中提取所有有options的算子的src和exclude路径"""
    src_paths = set()
    exclude_paths = set()

    def extract_from_dict(obj, current_key=None):
        if isinstance(obj, dict):
            if 'src' in obj and isinstance(obj['src'], list):
                for path in obj['src']:
                    src_paths.add(path)
            if 'exclude' in obj and isinstance(obj['exclude'], list):
                for path in obj['exclude']:
                    exclude_paths.add(path)
            if 'ut_cov_exclude' in obj and isinstance(obj['ut_cov_exclude'], list):
                for path in obj['ut_cov_exclude']:
                    exclude_paths.add(f'"{path}"' if path.startswith('*') else path)

            # 递归处理所有值
            for key, value in obj.items():
                extract_from_dict(value, key)
        elif isinstance(obj, list):
            for item in obj:
                extract_from_dict(item, current_key)

    extract_from_dict(data)

    return sorted(src_paths), sorted(exclude_paths)


def write_new_format(new_file_path: str, src_paths: list, exclude_paths: list):
    """以新格式写入文件"""
    try:
        with open(new_file_path, 'w', encoding='utf-8') as file:
            file.write("ops-transformer:\n")
            file.write("  src:\n")

            file.write("    release:\n")
            for path in src_paths:
                file.write(f"      - {path}\n")

            file.write("    unrelease:\n")
            for path in exclude_paths:
                file.write(f"      - {path}\n")

        return True

    except Exception as e:
        logging.error(f"Failed to write file: {e}")
        return False


def main(test_config_path: str, output_path: str):
    """主函数"""
    # 检查文件是否存在
    if not os.path.exists(test_config_path):
        logging.error(f"File does not exist: {test_config_path}")
        return

    # 读取test_config文件
    data = load_test_config(test_config_path)
    if data is None:
        return

    # 提取所有有options的算子的src和exclude路径
    src_paths, exclude_paths = extract_src_and_exclude(data)

    logging.info(f"Found {len(src_paths)} src paths")
    logging.info(f"Found {len(exclude_paths)} exclude paths")

    # 以新格式写回
    if write_new_format(output_path, src_paths, exclude_paths):
        logging.info("File conversion completed")
    else:
        logging.error("File conversion failed")

if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

    if len(sys.argv) == 1:
        main("test_config.yaml", "test_config.yaml")
    elif len(sys.argv) == 2:
        main(sys.argv[1], sys.argv[1])
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        logging.error("usage: convert_yaml.py test_config_path [output_path]")
        exit(1)