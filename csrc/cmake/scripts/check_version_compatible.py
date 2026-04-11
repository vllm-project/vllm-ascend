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
版本兼容性检查

检查当前代码仓与基础 CANN 包间的兼容性.
"""

import argparse
import logging
from pathlib import Path
from typing import NoReturn


class VersionChecker:

    @classmethod
    def main(cls) -> NoReturn:
        parser = argparse.ArgumentParser(description="Check Version Compatible", epilog="Best Regards!")
        sub_parser = parser.add_subparsers(help="Sub-Command")
        # 参数注册
        parser.add_argument("--cann_path", required=True, nargs=1, type=str,
                            help="CANN install path")
        parser.add_argument("--cann_package_name", required=True, nargs=1, type=str,
                            help="CANN package name")
        # 子命令行(Check)
        p_chk = sub_parser.add_parser('check_code_compatible', help="Check Version Compatible.")
        p_chk.add_argument("--code_version_info_file", required=True, nargs=1, type=str,
                           help="Code version info file path")
        p_chk.set_defaults(func=VersionChecker._check_compatible)
        # 子命令行(Get)
        p_get = sub_parser.add_parser('get_package_version', help="Get Package Version")
        p_get.set_defaults(func=VersionChecker._get_package_version)
        # 参数处理
        args = parser.parse_args()
        # 基本合法性检查, 版本号获取
        cann_version_info_file = Path(args.cann_path[0], args.cann_package_name[0], 'version.info').absolute()
        if not cann_version_info_file.exists():
            raise ValueError(f"CANN version info file({cann_version_info_file}) not exist.")
        ret, cann_version = cls._get_version_str(file=cann_version_info_file)
        if not ret:
            raise ValueError(f"Can't get version from CANN version info file({cann_version_info_file}).")
        rst = args.func(cann_version, args)
        return rst

    @classmethod
    def _check_compatible(cls, cann_version: str, args) -> str:
        code_version_info_file = Path(args.code_version_info_file[0]).absolute()
        if not code_version_info_file.exists():
            raise ValueError(f"Code version info file({code_version_info_file}) not exist.")
        ret, code_version = cls._get_version_str(file=code_version_info_file)
        if not ret:
            raise ValueError(f"Can't get version from Code version info file({code_version_info_file}).")
        # 兼容性检查
        cann_sub_version = cann_version.rsplit('.', 1)[0]
        code_sub_version = code_version.rsplit('.', 1)[0]
        if cann_sub_version != code_sub_version:
            raise ValueError(f"The version number of the current code is {code_sub_version}, "
                             f"and the version number of the cann package used is {cann_sub_version}. "
                             f"Please install version {code_sub_version} of the cann package.")
        return cann_sub_version

    @classmethod
    def _get_package_version(cls, cann_version: str, args) -> str:
        cann_sub_version = cann_version.rsplit('.', 1)[0]
        return cann_sub_version

    @classmethod
    def _get_version_str(cls, file: Path):
        with open(file, 'r') as fh:
            lines = fh.readlines()
            for line in lines:
                if not line.startswith('Version='):
                    continue
                version = line[8:].replace('\r', '').replace('\n', '')
                return True, version
        return False, ''


if __name__ == "__main__":
    logging.basicConfig(format='%(filename)s:%(lineno)d [%(levelname)s] %(message)s', level=logging.INFO)
    try:
        print(VersionChecker.main())
    except Exception as e:
        logging.error(e)
        exit(1)
