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
生成 TilingData 桩

用于 UTest 场景下, 生成 Struct 表示的 TilingData 相关头文件.
"""

import argparse
import datetime
import logging
import os
import re
import stat
from pathlib import Path
from typing import List


def check_if_new_tiling_file_path_existed(ori_file: Path) -> Path:
    current_path = ori_file
    target_dir = ori_file / "op_kernel"
    while True:
        if target_dir.is_dir():
            break
        parent_path = current_path.parent
        if parent_path == current_path:
            return ori_file, False
        current_path = parent_path
        target_dir = current_path / "op_kernel"
    tiling_files = list(target_dir.glob("*tiling_data.h"))
    if not tiling_files:
        return ori_file, False
    new_file = tiling_files[0]
    new_path = target_dir / new_file.name
    return new_path, True

def process_class_fields(fields_str, class_name):
    result = []
    seen = set()
    arr_pattern = re.compile(
        r'^\s*'
        r'(?P<type>(?:[\w:<>]+\s+)*[\w:<>* &]+?)\s+'   # 类型（含修饰符/指针/引用）
        r'(?P<name>\w+)\s*'                            # 变量名
        r'\[(?P<len>\d+)\]'                            # 数组长度
        r'(?:\s*=\s*\{\s*[^\}]*\s*})?'                 # 可选初始化，支持 {} 或 {0, ...}
        r'\s*;\s*'                                     # 以 ; 结尾
        r'(?:[ \t]*(?://[^\n]*)?)?$',                  # 行尾可有 // 注释
        re.MULTILINE
    )
    var_pattern = re.compile(
        r'^\s*(?P<type>[\w:<>]+)\s+(?P<name>\w+)\s*(?:=\s*[^;]*)?;',
        re.MULTILINE
    )
    # 逐行处理，保持顺序
    for line in fields_str.splitlines():
        # 跳空行或纯注释行
        if not line.strip() or line.strip().startswith('//'):
            continue

        m = arr_pattern.match(line)
        if m:
            t, n, ln = m.group('type'), m.group('name'), m.group('len')
            result.append(('array', t, n, ln))
            seen.add(n)
            continue

        m = var_pattern.match(line)
        if m:
            t, n = m.group('type'), m.group('name')
            if n not in seen:
                result.append(('normal', t, n))
                seen.add(n)
            continue
    return result


def find_classes(content):
    out = []
    class_re = re.compile(r'\bclass\s+(\w+)\s*{')
    for m in class_re.finditer(content):
        class_name = m.group(1)
        start = m.end()
        idx = start
        braces = 1
        while idx < len(content):
            c = content[idx]
            if c == '{':
                braces += 1
            elif c == '}':
                braces -= 1
                if braces == 0:
                    out.append((class_name, content[start:idx].strip()))
                    break
            idx += 1
    return out

def convert_template_tilingkey(ori_file: Path):
    with open(ori_file, 'r') as f:
        content = f.read()

    classes = find_classes(content)
    output = []
    for class_name, fields_str in classes:
        fields = process_class_fields(fields_str, class_name)
        output.append(f"BEGIN_TILING_DATA_DEF({class_name})")
        for entry in fields:
            if entry[0] == 'normal':
                _, field_type, field_name = entry
                if field_type in ['uint32_t', 'int32_t', 'uint8_t', 'uint16_t', 'float', 'uint64_t', 'int64_t', 'double']:
                    output.append(f"TILING_DATA_FIELD_DEF({field_type}, {field_name});")
                else:
                    output.append(f"TILING_DATA_FIELD_DEF_STRUCT({field_type}, {field_name});")
            elif entry[0] == 'array':
                _, field_type, field_name, field_len = entry
                output.append(f"TILING_DATA_FIELD_DEF_ARR({field_type}, {field_len}, {field_name});")
        output.append("END_TILING_DATA_DEF;")
        output.append(f"REGISTER_TILING_DATA_CLASS({class_name}Op, {class_name})\n")
    result_code = '\n'.join(output)

    return result_code


def process_fields(fields_str, struct_name):
    field_pattern = re.compile(r"(\w+)\s+(\w+)(?:\s*=\d+)?;")
    fields = field_pattern.findall(fields_str)
    return fields


def convert_to_old_tiling_struct_style(redirected_file_path):
    with open(redirected_file_path, 'r') as f:
        content = f.read()
    struct_pattern = re.compile(r"struct (\w+) {([^}]*)}", re.DOTALL)
    structs = struct_pattern.findall(content)
    output = []
    for struct_name, fields_str in structs:
        fields = process_fields(fields_str, struct_name)
        output.append(f"BEGIN_TILING_DATA_DEF({struct_name})")
        for field_type, field_name in fields:
            if field_type in ['uint32_t', 'uint8_t', 'uint16_t']:
                output.append(f"TILING_DATA_FIELD_DEF({field_type}, {field_name});")
            else:
                output.append(f"TILING_DATA_FIELD_DEF_STRUCT({field_type}, {field_name});")
        output.append("END_TILING_DATA_DEF;")
        output.append(f"REGISTER_TILING_DATA_CLASS({struct_name}Op, {struct_name})\n")
    result_code = '\n'.join(output)
    return result_code


class Process:
    _WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    _WRITE_MODES = stat.S_IWUSR | stat.S_IRUSR

    @classmethod
    def _write_file(cls, file: Path, src: str):
        with os.fdopen(os.open(file, cls._WRITE_FLAGS, cls._WRITE_MODES), 'w') as fh:
            fh.write(src)

    @classmethod
    def _get_begin_source(cls, ori_file: Path, gen_file: Path) -> str:
        bgn_src: str = \
            ("/**\n"
             " * This program is free software, you can redistribute it and/or modify.\n"
             " * Copyright (c) {year} Huawei Technologies Co., Ltd.\n"
             " * This file is a part of the CANN Open Software.\n"
             " * Licensed under CANN Open Software License Agreement Version 2.0 (the \"License\").\n"
             " * Please refer to the License for details. "
             "You may not use this file except in compliance with the License.\n"
             " * THIS SOFTWARE IS PROVIDED ON AN \"AS IS\" BASIS, "
             "WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, "
             "INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.\n"
             " * See LICENSE in the root of the software repository for the full text of the License.\n"
             " */\n").format(year=datetime.datetime.today().year)
        bgn_src += "\n"
        bgn_src += \
            ("/*!\n"
             " * \\file {gen_file_name}\n"
             " * \\brief Generate {ori_file_name}\n"
             " */\n").format(gen_file_name=gen_file.name, ori_file_name=ori_file.name)
        bgn_src += "\n"
        bgn_src += "#pragma once\n"
        bgn_src += "\n"
        return bgn_src

    @classmethod
    def _get_tiling_source(cls, ori_file: Path, isTemplateTilingKey: bool = False) -> str:
        """
        获取 TilingData 定义源码

        :param ori_file: 原始文件
        :return: 生成文件内容
        """
        rst_source = \
            ("#include <cstdint>\n"
             "#include <cstring>\n"
             "#include <securec.h>\n"
             "#include <kernel_tiling/kernel_tiling.h>\n"
             "\n")
        pattern = re.compile(r'[(](.*)[)]', re.S)
        if isTemplateTilingKey:
            lines = convert_template_tilingkey(ori_file)
            lines = lines.splitlines()
        else: 
            ori_file, existed_flag = check_if_new_tiling_file_path_existed(ori_file)
            if existed_flag:
                lines = convert_to_old_tiling_struct_style(ori_file)
                lines = lines.splitlines()
            else:
                with open(ori_file, 'r') as fd:
                    lines = fd.readlines()
        for line in lines:
                line = line.strip()
                struct_src = ""
                if line.startswith('BEGIN_TILING_DATA_DEF'):
                    struct_name = re.findall(pattern, line)[0]
                    struct_src += \
                        ("#pragma pack(1)\n"
                         "struct {}\n").format(struct_name)
                    struct_src += "{\n"
                    struct_offset = 0
                elif line.startswith('TILING_DATA_FIELD_DEF_ARR'):
                    field_params = re.findall(pattern, line)[0]
                    fds = field_params.split(',')
                    fds_dtype = fds[0].strip()
                    fds_num = int(fds[1].strip())
                    fds_name = fds[2].strip()
                    tmp_src, tmp_offset = cls._get_tmp_src(offset=struct_offset, dtype=fds_dtype,
                                                           name=fds_name, num=fds_num)
                    struct_src += tmp_src
                    struct_offset += tmp_offset
                elif line.startswith('TILING_DATA_FIELD_DEF_STRUCT'):
                    field_params = re.findall(pattern, line)[0]
                    fds = field_params.split(',')
                    struct_src += "  {} {};\n".format(fds[0].strip(), fds[1].strip())
                elif line.startswith('TILING_DATA_FIELD_DEF'):
                    field_params = re.findall(pattern, line)[0]
                    fds = field_params.split(',')
                    fds_dtype = fds[0].strip()
                    fds_num = int(1)
                    fds_name = fds[1].strip()
                    tmp_src, tmp_offset = cls._get_tmp_src(offset=struct_offset, dtype=fds_dtype,
                                                           name=fds_name, num=fds_num)
                    struct_src += tmp_src
                    struct_offset += tmp_offset
                elif line.startswith('END_TILING_DATA_DEF'):
                    # 要求结构体满足 8 字节对齐
                    if struct_offset % 8 != 0:
                        pad_num = 8 - (struct_offset % 8)
                        struct_src += "  uint8_t {}_PH[{}] = {{}};\n".format(struct_name, pad_num)
                        struct_offset += pad_num
                    struct_src += "};"
                    struct_src += "\n"
                    struct_src += "#pragma pack()\n"
                    struct_src += "\n"
                    struct_src += "inline void Init{struct_name}(" \
                                  "uint8_t* tiling, {struct_name}* const_data)\n".format(struct_name=struct_name)
                    struct_src += "{\n"
                    struct_src += "  (void)memcpy_s(const_data, sizeof({struct_name}), " \
                                  "tiling, sizeof({struct_name}));\n".format(struct_name=struct_name)
                    struct_src += "}\n"
                    struct_src += "\n"
                rst_source += struct_src
        rst_source += \
            (""
             "#undef GET_TILING_DATA\n"
             "#define GET_TILING_DATA(tiling_data, tiling_arg) \\\n"
             "{struct_name} tiling_data;                       \\\n"
             "Init{struct_name}(tiling_arg, &tiling_data)\n"
             "\n").format(struct_name=struct_name)
        return rst_source
    
    @classmethod
    def _get_tiling_whole(cls, ori_file: Path, isTemplateTilingKey: bool = False) -> str:
        with open(ori_file, 'r') as f:
            content = f.read()
        return content

    @classmethod
    def _gen_tiling_h(cls, ori_file: Path, gen_dir: Path):
        gen_file = Path(gen_dir, "_gen_" + ori_file.name)
        flag = "op_kernel" in [part for part in ori_file.parts]
        if not gen_file.exists():
            if not flag:
                bgn_src = cls._get_begin_source(ori_file=ori_file, gen_file=gen_file)
                def_src = cls._get_tiling_source(ori_file=ori_file, isTemplateTilingKey=flag)
                source = bgn_src + def_src
            else:
                source = "\n"
            cls._write_file(file=gen_file, src=source)
            logging.info("Generate TilingDefFile:  %s", gen_file)
        return gen_file

    @classmethod
    def _get_type_size(cls, dtype: str):
        mp = {
            "int8_t": 1,
            "int16_t": 2,
            "int32_t": 4,
            "int64_t": 8,
            "uint8_t": 1,
            "uint16_t": 2,
            "uint32_t": 4,
            "uint64_t": 8,
            "float": 4,
        }
        d_len = mp.get(dtype, None)
        if d_len is None:
            raise ValueError(f"Unknown dtype({dtype})")
        return d_len

    @classmethod
    def _get_tmp_src(cls, offset: int, dtype: str, name: str, num: int):
        source = ""
        result = 0
        dtype_size = cls._get_type_size(dtype=dtype)

        if offset % dtype_size != 0:
            pad_num = dtype_size - (offset % dtype_size)
            source += "  uint8_t {}_PH[{}] = {{}};\n".format(name, pad_num)
            result += pad_num

        if num == 1:
            source += "  {} {} = 0;\n".format(dtype, name)
        else:
            source += "  {} {}[{}] = {{}};\n".format(dtype, name, num)
        result += cls._get_type_size(dtype=dtype) * num
        return source, result

    @classmethod
    def gen_tiling_h(cls, ori_files: List[Path], gen_dir: Path):
        gen_files: List[Path] = []
        gen_dir.mkdir(parents=True, exist_ok=True)
        for ori_file in ori_files:
            if not ori_file.exists():
                raise ValueError(f"Origin file({ori_file}) not exist.")
            gen_file = cls._gen_tiling_h(ori_file=ori_file, gen_dir=gen_dir)
            gen_files.append(gen_file)
        return gen_files

    @classmethod
    def gen_tiling_data_h(cls, op: str, gen_files: List[Path], data_file: Path):
        if not data_file.exists():
            bgn_src = cls._get_begin_source(ori_file=data_file, gen_file=data_file)
            def_src = ""
            for gen_f in gen_files:
                def_src += "#include \"tiling/{op}/{file_name}\"\n".format(op=op, file_name=gen_f.name)
            source = bgn_src + def_src
            cls._write_file(file=data_file, src=source)
            logging.info("Generate TilingDataFile: %s", data_file)
        return data_file

    @classmethod
    def gen_tiling_stub_h(cls, data_file: Path, stub_file: Path):
        if not stub_file.exists():
            bgn_src = cls._get_begin_source(ori_file=stub_file, gen_file=stub_file)
            def_src = ""
            def_src += "#include \"{}\"\n".format(data_file.name)
            def_src += \
                ("\n"
                 "#undef GET_TILING_DATA_WITH_STRUCT\n"
                 "#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \\\n"
                 "tiling_struct tiling_data;                                                  \\\n"
                 "(void)memcpy_s(&tiling_data, sizeof(tiling_struct), tiling_arg, sizeof(tiling_struct));\n"
                 "\n")
            def_src += \
                ("\n"
                 "#undef GET_TILING_DATA_MEMBER\n"
                 "#define GET_TILING_DATA_MEMBER(tiling_type, member, var, tiling)                      \\\n"
                 "decltype(tiling_type::member) var;                                                    \\\n"
                 "size_t offset##var = (size_t)(&((tiling_type *)0)->member);                                \\\n"
                 "(void)memcpy_s(&var, sizeof(decltype(var)), tiling + offset##var, sizeof(decltype(var)));  \n"
                )
            source = bgn_src + def_src
            cls._write_file(file=stub_file, src=source)
            logging.info("Generate TilingStubFile: %s", stub_file)
        return stub_file

    @classmethod
    def main(cls):
        # 参数注册
        parser = argparse.ArgumentParser(description="TilingData Generator", epilog="Best Regards!")
        parser.add_argument("-o", "--operator", required=True,
                            nargs=1, type=str, help="Target operator.")
        parser.add_argument("-s", "--srcs", required=True,
                            action='append', nargs="+", type=Path, help="Origin tiling data define files(.h).")
        parser.add_argument("-d", "--dest", required=True,
                            nargs=1, type=Path, help="Generate directory.")
        # 参数解析
        result = parser.parse_args()
        op = result.operator[0].lower()
        ori_files: List[Path] = []
        for file in result.srcs:
            ori_files.append(file[0].absolute())
        gen_dir = Path(result.dest[0], "tiling/{}".format(op)).absolute()
        data_file = Path(gen_dir, "tiling_data.h".format(op))
        stub_file = Path(gen_dir, "tiling_stub.h".format(op))

        # 流程处理
        gen_files = cls.gen_tiling_h(ori_files=ori_files, gen_dir=gen_dir)
        cls.gen_tiling_data_h(op=op, gen_files=gen_files, data_file=data_file)
        cls.gen_tiling_stub_h(data_file=data_file, stub_file=stub_file)


if __name__ == "__main__":
    logging.basicConfig(format='%(filename)s:%(lineno)d [%(levelname)s] %(message)s',
                        level=logging.DEBUG)
    try:
        Process.main()
    except Exception as e:
        logging.error(e)
        raise e
