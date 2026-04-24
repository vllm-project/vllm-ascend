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
获取修改文件应触发的测试范围.

当前仅支持对应触发的 UTest 用例进行分析, 切仅支持 ops_test 这个 UTest 目标.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import yaml


class Module:
    def __init__(self, name):
        self.name: str = name
        self.src_files: List[Path] = []
        self.src_exclude_files: List[Path] = []
        self.tests_ut_ops_test_src_files: List[Path] = []
        self.tests_ut_ops_test_src_exclude_files: List[Path] = []
        self.tests_ut_ops_test_options: List[str] = []
        self.options: List[str] = []
        self.test_excludes: List[str] = []

    @staticmethod
    def _add_str_cfg(src, dst: List[str]):
        if isinstance(src, str):
            src = [src]
        for s in src:
            if s not in dst:
                dst.append(s)
        return True

    @staticmethod
    def _add_test_excludes(test_options, dst: List[str]):
        if isinstance(test_options, Dict):
            if 'examples' in test_options and not test_options['examples']:
                dst.append('examples')
            if 'ut' in test_options and not test_options['ut']:
                dst.append('ut')
        return True

    def update_classify_cfg(self, desc: Dict[str, Any]) -> bool:
        if not self._update_src(desc=desc):
            return False
        if not self._update_exclude_src(desc=desc):
            return False
        if not self._update_test_excludes(desc=desc):
            return False
        if not self._update_options(desc=desc):
            return False
        return True

    def get_test_options(self, f: Path) -> List[str]:

        def is_excluded(e_f: Path):
            for e in self.src_exclude_files:
                try:
                    e_f.relative_to(e)
                    return True
                except ValueError:
                    continue
            return False

        related_options: List[str] = []
        for s in self.src_files:
            if is_excluded(e_f=f):
                continue
            try:
                if f.relative_to(s):
                    # 当同一个修改文件需要触发多个 Options 时, 需要把这些 Options 全部添加
                    related_options.extend(self.options)
            except ValueError:
                continue
        # 关联 Options 去重
        related_options = list(set(related_options))
        return related_options

    def get_test_example_ops_test_options(self, f: Path) -> List[str]:
        return self.get_test_options(f)

    def print_details(self):
        dbg_str = (f"Name={self.name} SrcLen={len(self.src_files)} "
                   f"TestUtOpsTestSrcLen={len(self.tests_ut_ops_test_src_files)} "
                   f"TestUtOpsTestOptions={self.options} "
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

    def _update_exclude_src(self, desc: Dict[str, Any]) -> bool:
        src_paths = desc.get('exclude', [])
        return self._add_rel_path(src=src_paths, dst=self.src_exclude_files)

    def _update_test_excludes(self, desc: Dict[str, Any]) -> bool:
        test_options = desc.get('test', [])
        return self._add_test_excludes(test_options=test_options, dst=self.test_excludes)

    def _update_options(self, desc: Dict[str, Any]) -> bool:
        options = desc.get('options', [])
        return self._add_str_cfg(src=options, dst=self.options)


class Parser:
    """
    规则文件、修改文件列表文件解析.
    """

    _Modules: List[Module] = []         # 保存规则文件(tests/test_config.yaml)内设置的模块列表
    _ChangedPaths: List[Path] = []      # 修改文件列表文件(changed_file)内设置的修改文件列表
    _UTExcludes: List[str] = []
    _ExamplesExcludes: List[str] = []

    @classmethod
    def print_details(cls):
        for m in cls._Modules:
            m.print_details()
        for p in cls._ChangedPaths:
            logging.debug(p)

    @classmethod
    def parse_classify_file(cls, file: Path) -> bool:
        file = Path(file).resolve()
        if not file.exists():
            logging.error("Classify file(%s) not exist.", file)
            return False
        with open(file, 'r', encoding='utf-8') as f:
            desc: Dict[str, Any] = yaml.load(f, Loader=yaml.SafeLoader)

        def extract_from_dict(obj, current_key='root') -> bool:
            # 只看 dict 类型
            if not isinstance(obj, dict):
                return True

            # 递归到 module 时说明到达最后一层
            if 'module' in obj:
                return cls._parse_classify_item(current_key, desc)

            # 递归处理其他值
            for key, value in obj.items():
                if not extract_from_dict(value, key):
                    return False
            return True

        return extract_from_dict(desc)

    @classmethod
    def parse_changed_file(cls, file: Path) -> bool:
        file = Path(file).resolve()
        if not file.exists():
            logging.error("Change files desc file(%s) not exist.", file)
            return False
        with open(file, "r") as fh:
            lines = fh.readlines()
        for cur_line in lines:
            cur_line = cur_line.strip()
            f = Path(cur_line)
            if f.is_absolute():
                logging.error("%s is absolute path.", f)
                return False
            cls._ChangedPaths.append(f)
        return True

    @classmethod
    def get_related_ut(cls):
        ops_test_option_lst: List[str] = []
        for p in cls._ChangedPaths:
            for m in cls._Modules:
                new_options = m.get_test_options(f=p)
                for opt in new_options:
                    if opt not in ops_test_option_lst:
                        ops_test_option_lst.append(opt)
        if len(ops_test_option_lst) == 0:
            logging.info("Don't trigger any UT.")
            return ""
        ops_test_ut_str: str = ""
        if "all" in ops_test_option_lst:
            ops_test_ut_str = "all"
        else:
            for opt in ops_test_option_lst:
                if opt not in cls._UTExcludes:
                    ops_test_ut_str += f"{opt};"
        ops_test_ut_str = f"{ops_test_ut_str}"
        logging.info(f"Trigger UT: {ops_test_ut_str}")
        return ops_test_ut_str

    @classmethod
    def get_ops_test_option_lst(cls) -> List[str]:
        ops_test_option_lst: List[str] = []
        for p in cls._ChangedPaths:
            for m in cls._Modules:
                new_options = m.get_test_example_ops_test_options(f=p)
                for opt in new_options:
                    if opt not in ops_test_option_lst:
                        ops_test_option_lst.append(opt)
        return ops_test_option_lst

    @classmethod
    def get_related_examples(cls) -> str:
        ops_test_option_lst = cls.get_ops_test_option_lst()
        if len(ops_test_option_lst) == 0:
            logging.info("Don't trigger any examples.")
            return ""
        ops_test_examples_str: str = ""
        if "all" in ops_test_option_lst:
            ops_test_examples_str = "all"
        else:
            for opt in ops_test_option_lst:
                if opt not in cls._ExamplesExcludes:
                    ops_test_examples_str += f"{opt};"
        ops_test_examples_str = f"{ops_test_examples_str}"
        logging.info(f"Trigger examples: {ops_test_examples_str}")
        return ops_test_examples_str

    @classmethod
    def _parse_classify_item(cls, name: str, desc: Optional[Dict[str, Any]] = None) -> bool:
        if desc is None:
            logging.error("[%s]'s desc is None.", name)
            return False
        if desc.get('module', False):
            mod = Module(name=name)
            rst = mod.update_classify_cfg(desc=desc)
            if rst:
                cls._Modules.append(mod)
                short_name = name.split('/')[-1]
                if 'examples' in mod.test_excludes:
                    cls._ExamplesExcludes.append(short_name)
                if 'ut' in mod.test_excludes:
                    cls._UTExcludes.append(short_name)
            return rst
        for k, sub_desc in desc.items():
            if not cls._parse_classify_item(name=name + '/' + k, desc=sub_desc):
                return False
        return True

    @staticmethod
    def main() -> str:
        # 参数注册
        ps = argparse.ArgumentParser(description="Parse changed files", epilog="Best Regards!")
        ps.add_argument("-c", "--classify", required=True, nargs=1, type=Path, help="tests/test_config.yaml")
        ps.add_argument("-f", "--file", required=True, nargs=1, type=Path, help="changed files desc file.")
        # 子命令行
        sub_ps = ps.add_subparsers(help="Sub-Command")
        p_ut = sub_ps.add_parser('get_related_ut', help="Get related ut.")
        p_ut.set_defaults(func=Parser.get_related_ut)
        p_examples = sub_ps.add_parser('get_related_examples', help="Get related examples.")
        p_examples.set_defaults(func=Parser.get_related_examples)
        # 处理
        args = ps.parse_args()
        logging.debug(args)
        if not Parser.parse_classify_file(file=Path(args.classify[0])):
            return ""
        if not Parser.parse_changed_file(file=Path(args.file[0])):
            return ""
        Parser.print_details()
        rst = args.func()
        return rst


if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s][%(filename)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    print(Parser.main())
