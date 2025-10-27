#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import yaml


class Module:
    def __init__(self, name):
        self.name: str = name
        self.src_files: List[Path] = []
        self.tests_ut_ops_test_src_files: List[Path] = []
        self.tests_ut_ops_test_options: List[str] = []

    @staticmethod
    def _add_str_cfg(src, dst: List[str]):
        if isinstance(src, str):
            src = [src]
        for s in src:
            if s not in dst:
                dst.append(s)
        return True

    def update_classify_cfg(self, desc: Dict[str, Any]) -> bool:
        if not self._update_src(desc=desc):
            return False
        if not self._update_tests(desc=desc):
            return False
        if not self._check():
            return False
        return True

    def get_test_ut_ops_test_options(self, f: Path) -> List[str]:
        for s in self.src_files + self.tests_ut_ops_test_src_files:
            try:
                if f.relative_to(s):
                    return self.tests_ut_ops_test_options
            except ValueError:
                continue
        return []

    def print(self):
        dbg_str = (f"Name={self.name} SrcLen={len(self.src_files)} "
                   f"TestUtOpsTestSrcLen={len(self.tests_ut_ops_test_src_files)} "
                   f"TestUtOpsTestOptions={self.tests_ut_ops_test_options}")
        logging.debug(dbg_str)

    def _add_rel_path(self, src, dst: List[Path]):
        if isinstance(src, str) or isinstance(src, Path):
            src = [src]
        for p in src:
            p = Path(p)
            if p.is_absolute():
                logging.error("[%s]'s Path[%s] is absolute path.", self.name, p)
                return False
            if p not in dst:
                dst.append(p)
        return True

    def _update_src(self, desc: Dict[str, Any]) -> bool:
        src_paths = desc.get('src', [])
        return self._add_rel_path(src=src_paths, dst=self.src_files)

    def _update_tests(self, desc: Dict[str, Any]) -> bool:
        tests_desc = desc.get('tests', {})
        for sub_name, sub_desc in tests_desc.items():
            if sub_name == 'ut':
                if not self._update_ut(desc=sub_desc):
                    return False
        return True

    def _update_ut(self, desc: Dict[str, Any]) -> bool:
        for sub_name, sub_desc in desc.items():
            if sub_name == "ops_test":
                src_files = sub_desc.get('src', [])
                if not self._add_rel_path(src=src_files, dst=self.tests_ut_ops_test_src_files):
                    return False
                options = sub_desc.get('options', [])
                if not self._add_str_cfg(src=options, dst=self.tests_ut_ops_test_options):
                    return False
        return True

    def _check(self) -> bool:
        if len(self.src_files) != 0 or len(self.tests_ut_ops_test_src_files) != 0:
            return True
        logging.error('[%s] don\'t set any sources.', self.name)
        return False


class Parser:

    def __init__(self):
        self.modules: List[Module] = []

    def parse_file(self, file: Path):
        with open(file, 'r', encoding='utf-8') as f:
            desc: Dict[str, Any] = yaml.safe_load(f)
        for name, sub_desc in desc.items():
            if not self.parse_desc(name=name, desc=sub_desc):
                return False
        return True

    def parse_desc(self, name: str, desc: Optional[Dict[str, Any]] = None):
        if desc is None:
            logging.error("[%s]'s desc is None.", name)
            return False
        if desc.get('module', False):
            mod = Module(name=name)
            rst = mod.update_classify_cfg(desc=desc)
            if rst:
                self.modules.append(mod)
            return rst
        for k, sub_desc in desc.items():
            if not self.parse_desc(name=name + '/' + k, desc=sub_desc):
                return False
        return True

    def get_tests_ut_ops_test(self, file: Path):
        file = Path(file).resolve()
        if not file.exists():
            logging.error("Change files desc file(%s) not exist.", file)
            return ""
        with open(file, "r") as fh:
            lines = fh.readlines()

        ops_test_option_lst: List[str] = []
        for cur_line in lines:
            cur_line = cur_line.strip()
            f = Path(cur_line)
            if f.is_absolute():
                logging.error("%s is absolute path.", f)
                return ""
            for m in self.modules:
                new_options = m.get_test_ut_ops_test_options(f=f)
                for opt in new_options:
                    if opt not in ops_test_option_lst:
                        ops_test_option_lst.append(opt)
                        logging.info("TESTS_UT_OPS_TEST [%s] is trigger!", opt)
        if len(ops_test_option_lst) == 0:
            logging.info("Don't trigger any target.")
            return ""
        ops_test_ut_str: str = ""
        if "all" in ops_test_option_lst:
            ops_test_ut_str = "all"
        else:
            for opt in ops_test_option_lst:
                ops_test_ut_str += f"{opt};"
        ops_test_ut_str = f"{ops_test_ut_str}"
        return ops_test_ut_str

    def print(self):
        for m in self.modules:
            m.print()


class Process:

    @staticmethod
    def get_related_ut(args) -> str:
        logging.debug(args)
        parser: Parser = Parser()
        if not parser.parse_file(file=Path(args.classify[0])):
            return ""
        parser.print()
        tests_ut_ops_test = parser.get_tests_ut_ops_test(file=args.file[0])
        return tests_ut_ops_test

    @staticmethod
    def main() -> str:
        # 参数注册
        ps = argparse.ArgumentParser(description="Parse changed files", epilog="Best Regards!")
        ps.add_argument("-c", "--classify", required=True, nargs=1, type=Path, help="classify_rule.yaml")
        ps.add_argument("-f", "--file", required=True, nargs=1, type=Path, help="changed files desc file.")
        # 子命令行
        sub_ps = ps.add_subparsers(help="Sub-Command")
        p_ut = sub_ps.add_parser('get_related_ut', help="Get related ut.")
        p_ut.set_defaults(func=Process.get_related_ut)
        # 处理
        args = ps.parse_args()
        rst = args.func(args)
        return rst


if __name__ == '__main__':
    logging.basicConfig(format='%(filename)s:%(lineno)d [%(levelname)s] %(message)s', level=logging.INFO)
    print(Process.main())
