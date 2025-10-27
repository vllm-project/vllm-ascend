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
             " * Copyright (c) {year} Huawei Technologies Co., Ltd.\n"
             " * This file is a part of the CANN Open Software.\n"
             " * Licensed under CANN Open Software License Agreement Version 1.0 (the \"License\").\n"
             " * Please refer to the License for details. "
             "You may not use this file except in compliance with the License.\n"
             " * THIS SOFTWARE IS PROVIDED ON AN \"AS IS\" BASIS, "
             "WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,\n"
             " * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.\n"
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
    def _get_tiling_source(cls, ori_file: Path) -> str:
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
    def _gen_tiling_h(cls, ori_file: Path, gen_dir: Path):
        gen_file = Path(gen_dir, "_gen_" + ori_file.name)
        if not gen_file.exists():
            bgn_src = cls._get_begin_source(ori_file=ori_file, gen_file=gen_file)
            def_src = cls._get_tiling_source(ori_file=ori_file)
            source = bgn_src + def_src
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
            def_src += "#include <kernel_operator.h>\n"
            def_src += "#include <securec.h>\n"
            def_src += \
                ("\n"
                 "#undef GET_TILING_DATA_WITH_STRUCT\n"
                 "#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \\\n"
                 "tiling_struct tiling_data;                                                  \\\n"
                 "(void)memcpy_s(&tiling_data, sizeof(tiling_struct), tiling_arg, sizeof(tiling_struct));\n"
                 "\n"
                 "#undef max\n"
                 "#define max std::max\n"
                 "\n"
                 "#undef min\n"
                 "#define min std::min\n")
            def_src += \
                ("\n"
                 "#undef GET_TILING_DATA_MEMBER\n"
                 "#define GET_TILING_DATA_MEMBER(tiling_type, member, var, tiling)                      \\\n"
                 "decltype(tiling_type::member) var;                                                    \\\n"
                 "(void)memcpy_s(&var, sizeof(decltype(var)), tiling + (size_t)(&((tiling_type *)0)->member), sizeof(decltype(var)));  \n"
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
