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
获取 opapi 二进制绝对路径

Examples 场景下, 用于 built-in 包与 custom 包共存场景下获取正确的 opapi 动态库绝对路径.
"""
import argparse
import os
import logging
import subprocess
from pathlib import Path
from typing import List, Optional


class OpApiMgr:

    @staticmethod
    def has_symbol(file_path: Path, sym: str) -> bool:
        cmd = f"nm -D {file_path}".split()
        ret = subprocess.run(cmd, capture_output=True, check=True, encoding='utf-8')
        ret.check_returncode()
        return sym in ret.stdout

    @staticmethod
    def get_environ_custom_lib_paths() -> List[Path]:
        paths: List[Path] = []
        env = os.getenv('ASCEND_CUSTOM_OPP_PATH')
        if env is None:
            logging.debug("ASCEND_CUSTOM_OPP_PATH is none.")
            return paths
        str_paths = str(env).split(sep=':')
        if len(str_paths) == 0:
            return paths
        for s in str_paths:
            if len(s) == 0:
                continue
            p = Path(s, 'op_api/lib/libcust_opapi.so').resolve(strict=False)
            if not p.exists():
                logging.warning("Skip not exist path(%s)", p)
                continue
            paths.append(p)
        return paths

    @staticmethod
    def get_default_custom_lib_paths() -> List[Path]:
        paths: List[Path] = []
        env = os.getenv('ASCEND_OPP_PATH')
        if env is None:
            logging.warning("ASCEND_OPP_PATH is none.")
            return paths
        path_env = Path(env).resolve(strict=False)
        if not path_env.exists():
            logging.warning("ASCEND_CUSTOM_OPP_PATH(%s) not exist.", path_env)
            return paths
        cfg_file = Path(path_env, "vendors/config.ini").resolve(strict=False)
        if not cfg_file.exists():
            logging.debug("Config file(%s) not exist.", cfg_file)
            return paths
        # 手工解析 ini 文件
        with open(cfg_file, 'r') as fh:
            lines = fh.readlines()
        for line in lines:
            if not line.startswith('load_priority='):
                continue
            sub_str = line[14:]
            sub_str = sub_str.split(sep='#')[0]
            sub_str = sub_str.replace('\r', '').replace('\n', '').replace(' ', '')
            if len(sub_str) == 0:
                continue
            vendors = sub_str.split(sep=',')
            for v in vendors:
                if len(v) == 0:
                    continue
                p = Path(path_env, 'vendors', v, 'op_api/lib/libcust_opapi.so')
                if not p.exists():
                    logging.warning("Skip not exist path(%s)", p)
                    continue
                paths.append(p)
        return paths

    @staticmethod
    def get_default_builtin_lib_paths() -> List[Path]:
        paths: List[Path] = []
        env = os.getenv('ASCEND_OPP_PATH')
        if env is None:
            logging.warning("ASCEND_OPP_PATH is none.")
            return paths
        path_env = Path(env).resolve(strict=False)
        if not path_env.exists():
            logging.warning("ASCEND_CUSTOM_OPP_PATH(%s) not exist.", path_env)
            return paths
        shared = Path(path_env, 'lib64/libopapi.so').resolve(strict=False)
        if not shared.exists():
            logging.error("Can't get built-in libopapi.so(%s)", shared)
            return paths
        paths.append(shared)
        return paths

    @staticmethod
    def judge_lib_path(sym: str) -> Optional[Path]:
        path = None
        environ_custom_lib_paths = OpApiMgr.get_environ_custom_lib_paths()
        default_custom_lib_paths = OpApiMgr.get_default_custom_lib_paths()
        default_builtin_lib_paths = OpApiMgr.get_default_builtin_lib_paths()
        path_list = environ_custom_lib_paths + default_custom_lib_paths + default_builtin_lib_paths
        for p in path_list:
            if OpApiMgr.has_symbol(file_path=p, sym=sym):
                path = p
                break
        return path

    @staticmethod
    def main() -> str:
        ps = argparse.ArgumentParser(description="Get opapi path", epilog="Best Regards!")
        ps.add_argument("-f", "--func", required=True, nargs=1, type=str, help="Func name")
        args = ps.parse_args()
        sym = args.func[0]
        if sym is None or len(sym) == 0:
            return ""
        lib = OpApiMgr.judge_lib_path(sym=sym)
        if lib is None:
            return ""
        else:
            return str(lib)


if __name__ == "__main__":
    logging.basicConfig(format='%(filename)s:%(lineno)d [%(levelname)s] %(message)s', level=logging.INFO)
    print(OpApiMgr.main(), end='')
