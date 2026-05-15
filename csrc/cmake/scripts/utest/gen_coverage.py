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
生成覆盖率
"""
import os
import argparse
import dataclasses
import logging
import subprocess
from pathlib import Path
from typing import Optional, List
import yaml


class GenCoverage:

    @dataclasses.dataclass
    class Param:
        source_dir: Optional[Path] = None
        data_dir: Optional[Path] = None
        info_file: Optional[Path] = None
        info_file_filtered: Optional[Path] = None
        html_report_dir: Optional[Path] = None
        filter_str: str = ''

        @staticmethod
        def get_exclude_paths_from_yaml(yaml_path: Path):
            with open(yaml_path, 'r', encoding='utf-8') as file:
                data = yaml.safe_load(file)

            exclude_paths = set()

            def extract_from_dict(obj, current_key=None):
                if isinstance(obj, dict):
                    if 'exclude' in obj and isinstance(obj['exclude'], list):
                        for path in obj['exclude']:
                            exclude_paths.add(path)
                    if 'ut_cov_exclude' in obj and isinstance(obj['ut_cov_exclude'], list):
                        for path in obj['ut_cov_exclude']:
                            exclude_paths.add(path)

                    # 递归处理所有值
                    for key, value in obj.items():
                        extract_from_dict(value, key)
                elif isinstance(obj, list):
                    for item in obj:
                        extract_from_dict(item, current_key)

            extract_from_dict(data)

            return exclude_paths

        def init_filter_str(self, fs: Optional[List[List[str]]]):
            if not fs:
                return
            for fl in fs:
                self.filter_str += f"{fl[0]} "

        def init_filter_str_from_yaml(self, source_dir: Path, yaml_path: Path):
            exclude_paths = self.get_exclude_paths_from_yaml(yaml_path)

            for path in sorted(exclude_paths):
                if path.startswith('*'):
                    lcov_path = path
                else:
                    full_path = source_dir / Path(path)
                    if full_path.is_dir():
                        lcov_path = f"{full_path}/*"
                    else:
                        lcov_path = f"{full_path}"

                self.filter_str += f"{lcov_path} "

    @classmethod
    def main(cls):
        # 参数注册
        parser = argparse.ArgumentParser(description="Generate Coverage", epilog="Best Regards!")
        parser.add_argument("-s", "--source_base_dir", required=True,
                            nargs=1, type=Path, help="Explicitly specify the source base directory.")
        parser.add_argument("-c", "--coverage_data_dir", required=True,
                            nargs=1, type=Path, help="Explicitly specify the *.da's base directory.")
        parser.add_argument("-i", "--info_file", required=False,
                            nargs=1, type=Path, help="Explicitly specify coverage info file path.")
        # 考虑最低支持 Python 版本为 3.7, 此处用 append 而非 extend
        parser.add_argument("-f", "--filter", required=False, action='append',
                            nargs='*', type=str, help="Explicitly specify filter file/dir in coverage info.")
        parser.add_argument("-y", "--yaml", required=False,
                            nargs=1, type=Path, help="Explicitly specify filter file/dir from tests/test_config.yaml.")
        parser.add_argument("--html_report", required=False,
                            nargs=1, type=Path, help="Explicitly specify coverage html report dir.")
        # 参数解析, 默认值处理
        p = cls.Param()
        args = parser.parse_args()
        p.source_dir = Path(args.source_base_dir[0]).absolute()
        p.data_dir = Path(args.coverage_data_dir[0]).absolute()
        if args.info_file:
            p.info_file = Path(args.info_file[0]).absolute()
            p.info_file_filtered = Path(p.info_file.parent, f"{p.info_file.stem}_filtered{p.info_file.suffix}")
        else:
            p.info_file = Path(p.data_dir, 'cov_result/coverage.info')
            p.info_file_filtered = p.info_file
        p.html_report_dir = args.html_report[0] if args.html_report else Path(p.info_file.parent, "html_report")
        p.html_report_dir = Path(p.html_report_dir).absolute()
        p.init_filter_str(fs=args.filter)
        p.init_filter_str_from_yaml(source_dir=p.source_dir, yaml_path=args.yaml[0]) if args.yaml else None
        logging.debug("[DEBUG] filter_str=%s", p.filter_str)
        # 参数检查
        if not p.data_dir.exists():
            logging.error("[ERROR] The dir(%s) required to find the .da files not exist.", p.data_dir)
            exit(1)
        if not p.info_file.exists():
            p.info_file.parent.mkdir(parents=True, exist_ok=True)
        if not p.html_report_dir.exists():
            p.html_report_dir.mkdir(parents=True, exist_ok=True)
        # 环境检查
        if not cls._chk_env():
            exit(1)
        # 生成覆盖率数据
        cls._gen_cov(param=p)

    @classmethod
    def _chk_env(cls):
        try:
            ret = subprocess.run('lcov --version'.split(), capture_output=True, check=True, encoding='utf-8')
            ret.check_returncode()
        except FileNotFoundError:
            logging.error("[ERROR] lcov is required to generate coverage data, please install.")
            return False
        try:
            ret = subprocess.run('genhtml --version'.split(), capture_output=True, check=True, encoding='utf-8')
            ret.check_returncode()
        except FileNotFoundError:
            logging.error("[ERROR] genhtml is required to generate coverage html report, please install.")
            return False
        return True

    @classmethod
    def _gen_cov(cls, param: Param):
        """
        使用 lcov 生成覆盖率
        """
        # 当 log 等级小于 INFO 时，lcov 不带 -q 标签
        lcov_log_tag = "" if logging.getLogger().level <= logging.INFO else "-q"
        logging.critical("================================================================================")
        logging.critical("Coverage Report")
        logging.critical("================================================================================")

        # 生成覆盖率
        cmd = f"lcov -c -d {param.data_dir} -o {param.info_file} {lcov_log_tag}"
        logging.debug("[DEBUG] Generate origin coverage file, cmd=`%s`", cmd)
        ret = subprocess.run(cmd.split(), capture_output=False, check=True, encoding='utf-8')
        ret.check_returncode()
        if param.info_file.stat().st_size == 0:
            logging.critical("No file found in origin coverage file.")
            return
        logging.debug("[DEBUG] Generated origin coverage file %s", param.info_file)
        # 滤掉某些文件/路径的覆盖率信息
        cmd = f"lcov --remove {param.info_file} {param.filter_str} -o {param.info_file_filtered} {lcov_log_tag}"
        logging.debug("[DEBUG] Generate filtered coverage file, cmd=`%s`", cmd)
        ret = subprocess.run(cmd.split(), capture_output=False, check=True, encoding='utf-8')
        ret.check_returncode()
        logging.debug("[DEBUG] Generated filtered coverage file %s", param.info_file_filtered)
        logging.info("[INFO] Generated coverage result in %s", os.path.dirname(param.info_file))

        if param.info_file_filtered.stat().st_size == 0:
            logging.critical("No file found in filtered coverage file.")
            return
        # 生成 html 报告
        sub_cmd_prefix = f"-p {param.source_dir}" if param.source_dir else ""
        cmd = f'genhtml {param.info_file_filtered} {sub_cmd_prefix} -o {param.html_report_dir} {lcov_log_tag}'
        logging.debug("[DEBUG] Generate filtered coverage html report, cmd=`%s`", cmd)
        ret = subprocess.run(cmd.split(), capture_output=False, check=True, encoding='utf-8')
        ret.check_returncode()
        logging.info("[INFO] Generated filtered coverage html report. %s", param.html_report_dir)
        # 输出覆盖率数据到终端
        cmd = f"lcov --list {param.info_file_filtered}"
        ret = subprocess.run(cmd.split(), capture_output=False, check=True, encoding='utf-8')
        logging.critical("================================================================================")
        ret.check_returncode()


if __name__ == "__main__":
    # 将环境变量中的 ASCEND_GLOBAL_LOG_LEVEL 换算成 python 的 log 等级
    log_level = (int(os.getenv("ASCEND_GLOBAL_LOG_LEVEL", "3")) + 1) * 10
    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=log_level)
    GenCoverage.main()
