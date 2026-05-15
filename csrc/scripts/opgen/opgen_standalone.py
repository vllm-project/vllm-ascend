# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import argparse
import os
import shutil
import sys
import re
import logging


class OpGenerator:
    """算子工程生成器"""

    def __init__(self, op_type, op_name, output_path):
        self.op_type = op_type
        self.op_name = op_name
        self.output_path = output_path
        self.template_name = "add_example"

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.template_dir = os.path.abspath(os.path.join(self.script_dir, 'template', 'add'))
        self.dest_dir = os.path.abspath(os.path.join(self.output_path, self.op_type, self.op_name))

    def run(self):
        """执行生成流程"""
        self._validate_inputs()
        self._copy_template()
        self._rename_files()
        self._replace_content()
        logging.info(f"成功为 {self.op_type}/{self.op_name} 创建算子工程！")
        logging.info(f"工程路径: {self.dest_dir}")
        logging.info(f"Create the initial directory for {self.op_name} under {self.op_type} success")
        

    def _validate_inputs(self):
        """校验输入参数的有效性和安全性"""
        if not self.op_type or not self.op_name:
            raise ValueError("算子类型和算子名称均不能为空。")

        if not re.match(r"^[a-zA-Z0-9_]+$", self.op_type):
            raise ValueError(f"算子类型 '{self.op_type}' 包含无效字符。只允许字母、数字和下划线。")

        if not re.match(r"^[a-zA-Z0-9_]+$", self.op_name):
            raise ValueError(f"算子名称 '{self.op_name}' 包含无效字符。只允许字母、数字和下划线。")
        
        if os.path.exists(self.dest_dir):
            raise FileExistsError(f"目标目录 '{self.dest_dir}' 已存在。")

    def _copy_template(self):
        """复制模板文件到目标目录"""
        logging.info(f"使用模板在 '{self.dest_dir}' 创建算子工程...")
        if not os.path.exists(self.template_dir):
            raise FileNotFoundError(f"找不到模板目录 '{self.template_dir}'。请确保 'template/add' 目录存在。")
        
        try:
            shutil.copytree(self.template_dir, self.dest_dir)
            if not os.path.isfile(os.path.join(os.path.dirname(self.dest_dir), "CMakeLists.txt")):
                cmake_src = os.path.join(os.path.dirname(self.template_dir), "CMakeLists.txt")
                cmake_dest = os.path.join(os.path.dirname(self.dest_dir), "CMakeLists.txt")
                shutil.copy2(cmake_src, cmake_dest)
        except OSError as e:
            raise OSError(f"复制模板文件失败: {e}") from e

    def _rename_files(self):
        """重命名文件和目录中的占位符"""
        for root, dirs, files in os.walk(self.dest_dir, topdown=False):
            for name in files + dirs:
                if self.template_name not in name:
                    continue

                old_path = os.path.join(root, name)
                new_name = name.replace(self.template_name, self.op_name)
                new_path = os.path.join(root, new_name)
                try:
                    os.rename(old_path, new_path)
                except OSError as e:
                    raise OSError(f"重命名 '{old_path}' 到 '{new_path}' 失败: {e}") from e

    def _replace_content_in_file(self, file_path, replacements):
        """Helper to replace content in a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except (IOError, OSError) as e:
            logging.warning(f"读取文件 '{file_path}' 失败: {e}")
            return

        original_content = content
        for old, new in replacements.items():
            content = content.replace(old, new)

        if content == original_content:
            return

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except (IOError, OSError) as e:
            logging.warning(f"写入文件 '{file_path}' 失败: {e}")

    def _replace_content(self):
        """替换文件内容中的占位符"""
        op_name_capitalized = ''.join(word.capitalize() for word in self.op_name.split('_'))
        template_name_capitalized = ''.join(word.capitalize() for word in self.template_name.split('_'))

        replacements = {
            self.template_name: self.op_name,
            self.template_name.upper(): self.op_name.upper(),
            template_name_capitalized: op_name_capitalized,
            "add_example": self.op_name,
        }
        for root, _, files in os.walk(self.dest_dir):
            for file in files:
                if file.endswith(('.pyc', '.pyo')):
                    continue
                
                file_path = os.path.join(root, file)
                self._replace_content_in_file(file_path, replacements)


def execute(args):
    """根据命令行参数执行算子生成"""
    generator = OpGenerator(
        op_type=args.op_type,
        op_name=args.op_name,
        output_path=args.output_path
    )
    generator.run()


def register_parser(subparsers):
    """为 opgen 命令注册解析器。"""
    parser_opgen = subparsers.add_parser('opgen', help='生成项目骨架')
    parser_opgen.add_argument('--op_type', '-t', required=True, help='算子分类，例如 math')
    parser_opgen.add_argument('--op_name', '-n', required=True, help='新算子的名称，例如 asinh')
    parser_opgen.add_argument('--output_path', '-p', default='.', help='生成工程的根路径')
    parser_opgen.set_defaults(func=execute)


def main():
    """主函数，用于独立执行"""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', stream=sys.stdout)
    parser = argparse.ArgumentParser(description="生成项目骨架")
    
    parser.add_argument('--op_type', '-t', required=True, help='算子分类，例如 math')
    parser.add_argument('--op_name', '-n', required=True, help='新算子的名称，例如 asinh')
    parser.add_argument('--output_path', '-p', default='.', help='生成工程的根路径')
    
    args = parser.parse_args()

    try:
        execute(args)
    except Exception as e:
        logging.error(f"发生非预期的错误，退出。错误信息: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()