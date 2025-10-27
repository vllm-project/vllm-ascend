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
"""
生成覆盖率
"""
import argparse
import dataclasses
import logging
import subprocess
from pathlib import Path
from typing import Optional, List


class GenCoverage:

    @dataclasses.dataclass
    class Param:
        source_dir: Optional[Path] = None
        data_dir: Optional[Path] = None
        info_file: Optional[Path] = None
        info_file_filtered: Optional[Path] = None
        html_report_dir: Optional[Path] = None
        filter_str: str = ''

        def init_filter_str(self, fs: Optional[List[List[Path]]]):
            if not fs:
                return
            for fl in fs:
                f = fl[0]
                if not f.exists():
                    continue
                if f.is_dir():
                    self.filter_str += f" {f}/*"
                else:
                    self.filter_str += f" {f}"

    @classmethod
    def main(cls):
        # 参数注册
        parser = argparse.ArgumentParser(description="Generate Coverage", epilog="Best Regards!")
        parser.add_argument("-s", "--source_base_dir", required=False,
                            nargs=1, type=Path, help="Explicitly specify the source base directory.")
        parser.add_argument("-c", "--coverage_data_dir", required=True,
                            nargs=1, type=Path, help="Explicitly specify the *.da's base directory.")
        parser.add_argument("-i", "--info_file", required=False,
                            nargs=1, type=Path, help="Explicitly specify coverage info file path.")
        # 考虑最低支持 Python 版本为 3.7, 此处用 append 而非 extend
        parser.add_argument("-f", "--filter", required=False, action='append',
                            nargs='*', type=Path, help="Explicitly specify filter file/dir in coverage info.")
        parser.add_argument("--html_report", required=False,
                            nargs=1, type=Path, help="Explicitly specify coverage html report dir.")
        # 参数解析, 默认值处理
        p = cls.Param()
        args = parser.parse_args()
        p.source_dir = Path(args.source_base_dir[0]).absolute() if args.source_base_dir else None
        p.data_dir = Path(args.coverage_data_dir[0]).absolute()
        p.info_file = args.info_file[0] if args.info_file else Path(p.data_dir, 'cov_result/coverage.info')
        p.info_file = Path(p.info_file).absolute()
        p.info_file_filtered = Path(p.info_file.parent, f"{p.info_file.stem}_filtered{p.info_file.suffix}")
        p.html_report_dir = args.html_report[0] if args.html_report else Path(p.info_file.parent, "html_report")
        p.html_report_dir = Path(p.html_report_dir).absolute()
        p.init_filter_str(fs=args.filter)
        logging.debug("filter_str=%s", p.filter_str)
        # 参数检查
        if not p.data_dir.exists():
            logging.error("The dir(%s) required to find the .da files not exist.", p.data_dir)
            return
        if not p.info_file.exists():
            p.info_file.parent.mkdir(parents=True, exist_ok=True)
        if not p.html_report_dir.exists():
            p.html_report_dir.mkdir(parents=True, exist_ok=True)
        # 环境检查
        if not cls._chk_env():
            return
        # 生成覆盖率数据
        cls._gen_cov(param=p)

    @classmethod
    def _chk_env(cls):
        try:
            ret = subprocess.run('lcov --version'.split(), capture_output=True, check=True, encoding='utf-8')
            ret.check_returncode()
        except FileNotFoundError:
            logging.error("lcov is required to generate coverage data, please install.")
            return False
        try:
            ret = subprocess.run('genhtml --version'.split(), capture_output=True, check=True, encoding='utf-8')
            ret.check_returncode()
        except FileNotFoundError:
            logging.error("genhtml is required to generate coverage html report, please install.")
            return False
        return True

    @classmethod
    def _gen_cov(cls, param: Param):
        """
        使用 lcov 生成覆盖率
        """
        # 生成覆盖率
        cmd = f"lcov -c -d {param.data_dir} -o {param.info_file}"
        logging.info("[BGN] generated origin coverage file, cmd=%s", cmd)
        ret = subprocess.run(cmd.split(), capture_output=False, check=True, encoding='utf-8')
        ret.check_returncode()
        logging.info("[END] generated origin coverage file %s", param.info_file)
        # 滤掉某些文件/路径的覆盖率信息
        cmd = f"lcov --remove {param.info_file} {param.filter_str} -o {param.info_file_filtered}"
        logging.info("[BGN] generated filtered coverage file, cmd=%s", cmd)
        ret = subprocess.run(cmd.split(), capture_output=False, check=True, encoding='utf-8')
        ret.check_returncode()
        logging.info("[END] generated filtered coverage file %s", param.info_file_filtered)
        # 生成 html 报告
        sub_cmd_prefix = f"-p {param.source_dir}" if param.source_dir else ""
        cmd = f'genhtml {param.info_file_filtered} {sub_cmd_prefix} -o {param.html_report_dir}'
        logging.info("[BGN] generated filtered coverage html report, cmd=%s", cmd)
        ret = subprocess.run(cmd.split(), capture_output=False, check=True, encoding='utf-8')
        ret.check_returncode()
        logging.info("[END] generated filtered coverage html report. %s", param.html_report_dir)


if __name__ == "__main__":
    logging.basicConfig(format='%(filename)s:%(lineno)d [%(levelname)s] %(message)s', level=logging.INFO)
    GenCoverage.main()
