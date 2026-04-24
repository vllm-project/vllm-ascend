#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
build_opp_kernel_static.py
"""
import concurrent.futures
import glob
import multiprocessing
import sys
import os
import re
import platform
import stat
import json
import argparse
import subprocess
import logging as log
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


class Const:
    x86 = "x86_64"
    arm = "aarch64"


def shell_exec(cmd, shell=False):
    try:
        ps = subprocess.Popen(cmd, shell)
        ps.communicate(timeout=180)
    except BaseException as e:
        log.error(f"shell_exec error: {e}")
        sys.exit(1)


def shell_checkout_key_func(symbol_file, key_str):
    process = subprocess.Popen(("cat", symbol_file), stdout=subprocess.PIPE)
    awk_out = subprocess.check_output(("awk", "{print $8}"), stdin=process.stdout)
    process.wait()
    cppfilt = subprocess.check_output(("c++filt",), input=awk_out)
    if key_str not in cppfilt.decode("utf-8"):
        return "".encode("utf-8")
    grep_out = subprocess.check_output(("grep", key_str), input=cppfilt)
    return grep_out.decode("utf-8")


def to_upper_camel_case(x) -> str:
    """转大驼峰法命名"""
    s = re.sub('_([a-zA-Z])', lambda m: (m.group(1).upper()), x.lower())
    return s[0].upper() + s[1:]


def generate_symbol(args):
    library_file = args.library_file
    symbol_file = args.symbol_file
    if not os.path.exists(library_file):
        raise FileExistsError(f"generate_symbol input library error, file <{library_file}> not exists.")
    process = subprocess.Popen(("readelf", "-Ws", library_file), stdout=subprocess.PIPE)
    output = process.communicate(timeout=180)[0].decode("utf-8")
    with open(symbol_file, "w") as fd:
        fd.write(output)


def parser_generate_symbol(subparsers):
    generate_symbol_parser = subparsers.add_parser(name='GenerateSymbol', help='Generate Symbol file of input library')
    generate_symbol_parser.add_argument('-l', '--library_file', type=str, required=False, dest="library_file",
                                        default="", help="Input the library file")
    generate_symbol_parser.add_argument('-s', '--symbol_file', type=str, required=False, dest="symbol_file", default="",
                                        help="The symbol file for output")
    generate_symbol_parser.set_defaults(func=generate_symbol)


class CompileOpStaticLib:
    def __init__(self, ops_compile_files: Dict, out_path: str, dist_index: int, arch: str):
        self.ops_compile_files = ops_compile_files
        self.out_path = out_path
        self.part_index = dist_index
        self.cpu_arch = arch
        if self.cpu_arch not in [Const.x86, Const.arm]:
            raise Exception(f"CompileOpStaticLib Error, input arch<{arch}> error...")


    def compile_link_single(self, file_path, file_o):
        (dir_path, file_name) = os.path.split(file_path)
        if self.cpu_arch == Const.x86:
            shell_exec(["bash", "-c", f"cd {dir_path} && "
                                      f"objcopy --input-target binary --output-target elf64-x86-64 "
                                      f"--binary-architecture i386 "
                                      f"{file_name} {file_o}"], shell=False)
        elif self.cpu_arch == Const.arm and platform.machine() != Const.x86:
            shell_exec(["bash", "-c", f"cd {dir_path} && "
                                      f"objcopy --input-target binary "
                                      f"--output-target elf64-littleaarch64 --binary-architecture aarch64 "
                                      f"{file_name} {file_o}"], shell=False)
        elif self.cpu_arch == Const.arm:
            shell_exec(["bash", "-c", f"cd {dir_path} && "
                                      f"aarch64-linux-gnu-objcopy --input-target binary "
                                      f"--output-target elf64-littleaarch64 --binary-architecture aarch64 "
                                      f"{file_name} {file_o}"], shell=False)


    def compile_link_o(self, out_path, file_path, is_need_path=True):
        file_pre = os.path.basename(file_path).replace('.', '_').replace('-', '_')
        path_o_prefix = os.path.join(out_path, f"data_{file_pre}_{self.cpu_arch}.o")
        # 向json文件中写入"filePath"参数
        if is_need_path and file_path.name.endswith(".json"):
            with open(file_path, 'r', encoding='UTF-8') as json_fd:
                json_dict = json.load(json_fd)
                soc = str(file_path).split("/binary/")[-1].split("/bin/")[0]
                json_dict["filePath"] = os.path.join(soc, str(file_path).split("/bin/")[-1].split("/kernel/")[-1])
                if "opp/built-in/" in str(file_path):
                    json_dict["filePath"] = str(file_path).split("/bin/")[-1].split("/kernel/")[-1].replace("/ops_transformer", "")
                file_path = os.path.join(out_path, os.path.basename(file_path))
                with open(file_path, 'w', encoding='UTF-8') as new_json_fd:
                    new_json_fd.write(json.dumps(json_dict, indent=4))
        file_path = os.path.realpath(file_path)
        self.compile_link_single(file_path, path_o_prefix)
        return

    def compile_ops_part_o(self, out_path):
        path_data_o = os.path.join(out_path, f"data_*{self.cpu_arch}.o")
        path_data_o_list = glob.glob(path_data_o)
        if not path_data_o_list:
            return
        (dir_path, ops_name) = os.path.split(out_path)
        file_part_o = f"{ops_name}_{self.cpu_arch}_part{self.part_index}.o"  # eg: floor_mod_aarch64_1.a
        path_part_o = os.path.join(dir_path, file_part_o)
        if self.cpu_arch == Const.x86 or (self.cpu_arch == Const.arm and platform.machine() != Const.x86):
            shell_exec(["bash", "-c", f"cd {out_path} && "
                                      f"ld -r {path_data_o} -o {path_part_o}"], shell=False)
        if self.cpu_arch == Const.arm and platform.machine() == Const.x86:
            shell_exec(["bash", "-c", f"cd {out_path} && "
                                      f"aarch64-linux-gnu-ld -r {path_data_o} -o {path_part_o}"], shell=False)
        return

    def exec_compile(self):
        """
        编译算子静态库
        :return:
        """
        def get_parallel_num() -> int:
            """
            获取多线程最大并发数量
            """
            num = multiprocessing.cpu_count() * 2
            if num == 0:
                num = 16
            return num

        job_num = get_parallel_num()
        for op in self.ops_compile_files:
            compile_files = self.ops_compile_files[op].kernel_files
            json_files = self.ops_compile_files[op].binary_config_files
            runtime_kb_files = self.ops_compile_files[op].runtime_kb_files
            op_out_path = os.path.join(self.out_path, op)
            if not os.path.exists(op_out_path):
                os.makedirs(op_out_path, exist_ok=True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=job_num) as executor:
                for file in compile_files:
                    executor.submit(self.compile_link_o, op_out_path, file.resolve())
                for file in json_files:
                    executor.submit(self.compile_link_o, op_out_path, file.resolve(), False)
                for file in runtime_kb_files:
                    executor.submit(self.compile_link_o, op_out_path, file.resolve(), False)

            with concurrent.futures.ThreadPoolExecutor(max_workers=job_num) as executor:
                executor.submit(self.compile_ops_part_o, op_out_path)
        return 0


def compile_static_library(args):
    index_num = args.index_num
    cpu_aarch = args.cpu_aarch

    if cpu_aarch not in [Const.x86, Const.arm]:
        raise Exception(f"Input cpu_aarch<{cpu_aarch}> Error, Please input parase.")

    ops_compile_files = GenOpResourceIni(args.soc_version, args.build_dir, args.jit).analyze_ops_files()

    csl = CompileOpStaticLib(ops_compile_files,
                            os.path.join(args.build_dir, f"bin_tmp/{args.soc_version}"), index_num, cpu_aarch)
    ret = csl.exec_compile()
    return ret


def parser_compile_static_library(subparsers):
    """ 配置静态编译参数及执行信息 """
    compile_lib_parser = subparsers.add_parser(name='StaticCompile',
                                    help='Compile static libraries(.a) on distributed server')
    compile_lib_parser.add_argument('-s', '--soc_version', type=str, required=True, dest="soc_version",
                                    help="Operator Name, eg: ascend910b, ascend310p")
    compile_lib_parser.add_argument('-b', '--build_dir', type=str, required=True, dest="build_dir",
                                    help="Input build dir for this project")
    compile_lib_parser.add_argument('-j', '--jit', action='store_true', dest="jit",
                                    help="Compile static libraries(.a) with cann package")
    compile_lib_parser.add_argument('-n', '--index_num', type=int, required=True, dest="index_num",
                                    help="Please input distributed compilation idx")
    compile_lib_parser.add_argument('-a', '--cpu_aarch', type=str, required=True, dest="cpu_aarch",
                                    help="Please input cpu aarch, eg:x86_64,aarch64")
    compile_lib_parser.set_defaults(func=compile_static_library)


@dataclass
class OpResource:
    """算子资源"""
    # tiling 注册函数
    tiling_register: str = field(default=None)
    extend_register: str = field(default_factory=list)
    # InferShape 注册函数
    infer_shape_register: str = field(default=None)
    # 知识库注册
    tuning_bank_key_register: str = field(default=None)
    tuning_bank_parse_register: str = field(default=None)
    tuning_tiling_helper: str = field(default=None)
    # 二进制配置
    binary_config_files: list = field(default_factory=list)
    # kernel 编译产出文件
    kernel_files: list = field(default_factory=list)
    # 知识库文件
    runtime_kb_files: list = field(default_factory=list)


class GenOpResourceIni:
    def __init__(self, soc_version: str, build_dir: str, build_with_package: bool):
        self._soc_version = soc_version
        self._build_dir = Path(build_dir)
        opp_path = os.environ.get('ASCEND_OPP_PATH')
        if build_with_package and opp_path:
            opp_path = Path(opp_path)
            self._binary_path = opp_path / "built-in/op_impl/ai_core/tbe/kernel"
            self._tuning_basic_path = opp_path / "built-in/data/op"
            ops_info = opp_path / "built-in/op_impl/ai_core/tbe/config" / self._soc_version
            ops_info = list(ops_info.glob(f"aic-{self._soc_version}-ops-info-transformer.json"))
            self._ops_info = ops_info[0] if len(ops_info) != 0 else None
        else:
            self._binary_path = self._build_dir / "binary" / self._soc_version / "bin"
            self._tuning_basic_path = self._build_dir / "tbe/config" / self._soc_version
            # transformer aic*.json 适配
            ops_info = self._build_dir / "custom/op_impl/ai_core/tbe/config" / self._soc_version
            ops_info = list(ops_info.glob(f"aic-{self._soc_version}-ops-info*.json"))
            self._ops_info = ops_info[0] if len(ops_info) != 0 else None        
        self._op_resource_path = self._build_dir / "autogen" / self._soc_version / "aclnnop_resource"
        self._op_res: Dict[str, OpResource] = defaultdict(OpResource)
        self._l0op_list = []


    TILING_REG_DECL_FMT = """
namespace {namespace} {{
    extern gert::OpImplRegisterV2 {func_name};
}}
"""
    EXTEND_REG_DECL_FMT = """
namespace {namespace} {{
    extern uint32_t {func_name};
}}
"""
    TILING_REG_RES_FUNC_FMT = """
void * {op_type}TilingRegisterResource() {{
    return {reference_code};
}}
"""
    INFER_SHAPE_REG_DECL_FMT = """
namespace {namespace} {{
    extern gert::OpImplRegisterV2 {func_name};
}}
"""
    INFER_SHAPE_REG_RES_FUNC_FMT = """
void * {op_type}InferShapeRegisterResource() {{
    return {reference_code};
}}
"""
    TUNING_REG_DECL_FMT = """
namespace {namespace} {{
    class {class_type};
    extern {class_type} {func_name};
}}
"""
    EXTLEND_REG_RES_FUNC_FMT = """
void * {op_type}ExtendRegisterResource() {{
    static std::vector<void *> resource = {{{reference_code}}};
    return &resource;
}}
"""
    TUNING_REG_RES_FUNC_FMT = """
void * {op_type}TuningRegisterResource() {{
    static std::vector<void *> resource = {{{tuning_bank_key}, {tuning_bank_parse}, {tuning_helper}}};
    return &resource;
}}
"""
    KERNEL_BINARY_RES_FUNC_FMT = """
const OP_BINARY_RES& {op_type}KernelResource() {{
    static const OP_BINARY_RES resource = {{
    {binary_config_ref_code}
    {kernel_files_ref_code}
    }};
    return resource;
}}
"""
    TUNING_KB_BINARY_RES_FUNC_FMT = """
const OP_RUNTIME_KB_RES& {op_type}TuningResource() {{
    static const OP_RUNTIME_KB_RES resource = {{
    {reference_code}
    }};
    return resource;
}}
"""
    OP_RESOURCE_CPP_FMT = """/******************{op_type}算子的所有资源**********************/
#include "register/op_impl_registry.h"
#include <vector>
#include <tuple>
#include <map>
#include <graph/ascend_string.h>
#include <static_space.h>

using OP_HOST_FUNC_HANDLE = std::vector<void *>;
using OP_RES = std::tuple<const uint8_t *, const uint8_t *>;
using OP_BINARY_RES = std::vector<OP_RES>;
using OP_RUNTIME_KB_RES = std::vector<OP_RES>;
using OP_RESOURCES  = std::map<ge::AscendString,
    std::tuple<OP_HOST_FUNC_HANDLE, OP_BINARY_RES, OP_RUNTIME_KB_RES>>;
namespace {op_type} {{
    auto initializer = StaticSpaceInitializer::GetInstance();;
}}

// 资源声明
// extend resource
{extend_declaration}
// Tiling
{tiling_declaration}
// InferShape
{infer_shape_declaration}
// Tuning
{tuning_bank_key_declaration}
{tuning_bank_parse_declaration}
{tuning_helper_declaration}
// kernel 二进制
{binary_config_declaration}
{kernel_files_declaration}
// kb 二进制
{tuning_kb_declaration}

namespace l0op {{
// 资源函数
// Tiling register resource func
{tiling_reg_func}
// InferShape register resource func
{infer_shape_reg_func}
// Tuning register resource func
{tuning_reg_func}
// kernel resource func
{kernel_resource}
// Tuning resource func
{tuning_kb_resource}
}}

// extend resource func
{extend_reg_func}

"""


    @staticmethod
    def _extract_op_symbol_pair(symbol_file: str, search_key: str, prefix: str, suffix: str):
        symbol_ret = shell_checkout_key_func(symbol_file, search_key)
        for symbol in symbol_ret.splitlines():
            symbol_name = symbol.split("::")[-1]
            if not (symbol_name.startswith(prefix) and symbol_name.endswith(suffix)):
                log.warning(f"symbol not satisfied with the format:{prefix}<op_type>{suffix}, skip")
                continue
            op_type = symbol_name
            if prefix:
                op_type = op_type[len(prefix):]
            if suffix:
                op_type = op_type[:-len(suffix)]
            yield op_type, symbol


    @staticmethod
    def _extract_op_symbol_pair_v2(symbol_file: str, search_key: str, prefix: str):
        symbol_ret = shell_checkout_key_func(symbol_file, search_key)
        for symbol in symbol_ret.splitlines():
            symbol_name = symbol.split("::")[-1]
            if not (symbol_name.startswith(prefix)):
                log.warning(f"symbol not satisfied with the format:{prefix}, skip")
                continue
            op_type = symbol_name
            if prefix:
                op_type = op_type[len(prefix):]
            op_type = op_type.split("_")[0]
            yield op_type, symbol


    @staticmethod
    def _extract_register_symbol(register_symbol: str):
        if not register_symbol:
            return "", "", "nullptr"

        symbol_data = register_symbol.split("::")
        namespace = "::".join(symbol_data[:-1])
        if "anonymous" in namespace:
            return "", "", "nullptr"
        func_name = symbol_data[-1]
        reference_code = f"&{register_symbol}"
        return namespace, func_name, reference_code


    @staticmethod
    def _gen_binary_res_code(files):
        declaration = ""
        reference_code = ""
        for binary_file in files:
            # static not support supperkernel
            if ("relocatable" in binary_file.name):
                continue
            binary_name = binary_file.name.replace(".", "_").replace("-", "_")
            declaration += f"""// {binary_file.name}
extern const uint8_t _binary_{binary_name}_start[];
extern const uint8_t _binary_{binary_name}_end[];
"""
            reference_code += f"{{_binary_{binary_name}_start, _binary_{binary_name}_end}},\n"
        return declaration, reference_code


    def gen_ops_ini_files(self):
        self.analyze_ops_files()
        self._analyze_symbols()
        self._analyze_ops_l0op()
        if not os.path.exists(self._op_resource_path):
            os.makedirs(self._op_resource_path)
        for op_type in self._l0op_list:
            ini_content = self.generate_op_resouce_ini(op_type)
            self._save_op_resource(op_type, ini_content)
        for op_type in self._op_res:
            if op_type in self._l0op_list:
                continue
            ini_content = self.generate_op_resouce_ini(op_type)
            self._save_op_resource(op_type, ini_content)


    def generate_op_resouce_ini(self, op_type: str) -> str:
        value_dict = {
            "op_type": op_type,
        }
        value_dict.update(self._gen_register_resouce_code(op_type))
        value_dict.update(self._gen_tuning_register_resouce_code(op_type))
        value_dict.update(self._gen_binary_resource_code(op_type))
        # 处理特殊的共用算子kernel资源的算子
        sepical_ops = {"MatMulV2": "MatMul"}
        if op_type in sepical_ops:
            value_dict["kernel_files_declaration"] = ""
            value_dict["kernel_resource"] = f"""
extern const OP_BINARY_RES& {sepical_ops[op_type]}KernelResource();
const OP_BINARY_RES& {op_type}KernelResource() {{
    return {sepical_ops[op_type]}KernelResource();
}}
"""
        return self.OP_RESOURCE_CPP_FMT.format_map(value_dict)


    def analyze_ops_files(self):
        if not self._ops_info:
            return self._op_res
        with open(self._ops_info, "r") as autogen_fd:
            ops_info_json = json.load(autogen_fd)

        for ops in ops_info_json:
            if 'opFile' in ops_info_json[ops]:
                json_file = f"{ops_info_json[ops]['opFile']['value']}.json"
            else:
                o_lists = list(Path(self._binary_path).rglob(f"{self._soc_version}/**/*{ops}*.o"))
                if len(o_lists) == 0:
                    continue
                else:
                    json_file = f"{os.path.basename(os.path.dirname(o_lists[0]))}.json"
            # json_path = self._binary_path / "config" / self._soc_version / json_file
            json_path = self._binary_path / json_file
            if "opp/built-in/" in str(self._binary_path):
                json_path = self._binary_path / "config" / self._soc_version / "ops_transformer" / json_file
            if not os.path.exists(json_path):
                continue
            with open(json_path, "r") as op_json_fd:
                op_json_content = json.load(op_json_fd)
            if "binList" not in op_json_content or len(op_json_content["binList"]) == 0:
                continue
            # 算子.json内 kernel json路径适配
            bin_json_file = self._binary_path / op_json_content["binList"][0]["binInfo"]["jsonFilePath"].split("/", 1)[1]
            if "opp/built-in/" in str(self._binary_path):
                bin_json_file = self._binary_path / self._soc_version / "ops_transformer" / op_json_content["binList"][0]["binInfo"]["jsonFilePath"].split("/", 1)[1]
            ops_path = os.path.dirname(bin_json_file)
            self._op_res[ops].binary_config_files.append(json_path)
            self._op_res[ops].kernel_files.extend(sorted(Path(ops_path).iterdir()))
        for kb_json in list(Path(self._tuning_basic_path).rglob(f"*_AiCore_*_runtime_kb.json")):
            ops = kb_json.name.split("_AiCore_")[-1].split("_runtime_kb")[0]
            self._op_res[ops].runtime_kb_files.append(kb_json)
            self._op_res[ops].runtime_kb_files.sort(key=lambda p: p.name)
        return self._op_res
    

    def _analyze_ops_l0op(self):
        opapi_symbol = self._build_dir / "opapi_transformer.txt"
        if not os.path.exists(opapi_symbol):
            return
        # infershape
        for op_type, _ in self._extract_op_symbol_pair(
                opapi_symbol, "_kernelName_Be_Defined_Multi_Times__", "", ""
            ):
            self._l0op_list.append(op_type.split("_kernelName_")[0])
        self._l0op_list.sort()
    

    def _save_op_resource(self, op_type, res_content):
        res_cpp_file = self._op_resource_path / f"{op_type}_op_resource.cpp"
        try:
            res_cpp_file.unlink()
        except FileNotFoundError:
            pass

        flags = os.O_WRONLY | os.O_CREAT
        modes = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(res_cpp_file, flags, modes), "w") as fd:
            fd.write(res_content)


    def _analyze_symbols(self):
        # ophost txt 适配
        ophost_symbol = self._build_dir / "ophost_transformer.txt"
        if not os.path.exists(ophost_symbol):
            return
        # infershape
        for op_type, symbol in self._extract_op_symbol_pair(
                ophost_symbol, "op_impl_register_infershape_", "op_impl_register_infershape_", ""
            ):
            self._op_res[op_type].infer_shape_register = symbol
        # tiling
        for op_type, symbol in self._extract_op_symbol_pair(
                ophost_symbol, "op_impl_register_optiling_", "op_impl_register_optiling_", ""
            ):
            self._op_res[op_type].tiling_register = symbol
        for op_type, symbol in self._extract_op_symbol_pair_v2(
                ophost_symbol, "op_impl_register_template_", "op_impl_register_template_"
            ):
            self._op_res[op_type].extend_register.append(symbol)
        # 知识库
        for op_type, symbol in self._extract_op_symbol_pair(
                ophost_symbol, "BankKeyRegistryInterf", "g_", "BankKeyRegistryInterf"
            ):
            self._op_res[op_type].tuning_bank_key_register = symbol
        for op_type, symbol in self._extract_op_symbol_pair(ophost_symbol, "BankParseInterf", "g_", "BankParseInterf"):
            self._op_res[op_type].tuning_bank_parse_register = symbol
        for op_type, symbol in self._extract_op_symbol_pair(
                ophost_symbol, "g_tuning_tiling_", "g_tuning_tiling_", "Helper"
            ):
            self._op_res[op_type].tuning_tiling_helper = symbol


    def _gen_register_resouce_code(self, op_type: str):
        """ 注册函数 """
        # Tiling
        namespace, func_name, reference_code = self._extract_register_symbol(
            self._op_res[op_type].tiling_register
        )
        symbol_map = {
            "op_type": op_type,
            "namespace": namespace,
            "func_name": func_name,
            "reference_code": reference_code,
        }
        tiling_declaration = self.TILING_REG_DECL_FMT.format_map(symbol_map) if func_name else ""
        tiling_reg_func = self.TILING_REG_RES_FUNC_FMT.format_map(symbol_map) if func_name else ""

        reference_code_list = []
        extend_declaration = ""
        for symbol in self._op_res[op_type].extend_register:
            namespace, func_name, reference_code = self._extract_register_symbol(symbol)
            if func_name:
                extend_declaration += self.EXTEND_REG_DECL_FMT.format(namespace = namespace, func_name = func_name)
                reference_code_list.append(reference_code)
        reference_code = ", ".join(reference_code_list)
        extend_reg_func = self.EXTLEND_REG_RES_FUNC_FMT.format(op_type = op_type, reference_code = reference_code)

        # InferShape
        namespace, func_name, reference_code = self._extract_register_symbol(self._op_res[op_type].infer_shape_register)
        symbol_map = {
            "op_type": op_type,
            "namespace": namespace,
            "func_name": func_name,
            "reference_code": reference_code,
        }
        infer_shape_declaration = self.INFER_SHAPE_REG_DECL_FMT.format_map(symbol_map) if func_name else ""
        infer_shape_reg_func = self.INFER_SHAPE_REG_RES_FUNC_FMT.format_map(symbol_map) if func_name else ""

        return {
            "tiling_declaration": tiling_declaration,
            "infer_shape_declaration": infer_shape_declaration,
            "tiling_reg_func": tiling_reg_func,
            "infer_shape_reg_func": infer_shape_reg_func,
            "extend_reg_func": extend_reg_func,
            "extend_declaration": extend_declaration
        }

    def _gen_tuning_register_resouce_code(self, op_type: str):
        """ 知识库注册函数 """
        # Tuning
        namespace, func_name, tuning_bank_key_ref_code = self._extract_register_symbol(
            self._op_res[op_type].tuning_bank_key_register
        )
        tuning_bank_key_declaration = self.TUNING_REG_DECL_FMT.format(
            namespace=namespace,
            class_type="OpBankKeyFuncRegistryV2",
            func_name=func_name,
        ) if func_name else ""
        namespace, func_name, tuning_bank_parse_ref_code = self._extract_register_symbol(
            self._op_res[op_type].tuning_bank_parse_register
        )
        tuning_bank_parse_declaration = self.TUNING_REG_DECL_FMT.format(
            namespace=namespace,
            class_type="OpBankKeyFuncRegistryV2",
            func_name=func_name,
        ) if func_name else ""
        namespace, func_name, tuning_helper_ref_code = self._extract_register_symbol(
            self._op_res[op_type].tuning_tiling_helper
        )
        tuning_helper_declaration = self.TUNING_REG_DECL_FMT.format(
            namespace=namespace,
            class_type=f"{op_type}ClassHelper",
            func_name=func_name,
        ) if func_name else ""
        tuning_reg_func = self.TUNING_REG_RES_FUNC_FMT.format(
            op_type=op_type,
            tuning_bank_key=tuning_bank_key_ref_code,
            tuning_bank_parse=tuning_bank_parse_ref_code,
            tuning_helper=tuning_helper_ref_code,
        )
        return {
            "tuning_bank_key_declaration": tuning_bank_key_declaration,
            "tuning_bank_parse_declaration": tuning_bank_parse_declaration,
            "tuning_helper_declaration": tuning_helper_declaration,
            "tuning_reg_func": tuning_reg_func,
        }
    
    def _gen_binary_resource_code(self, op_type: str) -> str:
        """二进制"""
        # kernel
        binary_config_declaration, binary_config_ref_code = self._gen_binary_res_code(
            self._op_res[op_type].binary_config_files
        )
        kernel_files_declaration, kernel_files_ref_code = self._gen_binary_res_code(self._op_res[op_type].kernel_files)
        kernel_resource = self.KERNEL_BINARY_RES_FUNC_FMT.format(
            op_type=op_type,
            binary_config_ref_code=binary_config_ref_code,
            kernel_files_ref_code=kernel_files_ref_code,
        ) if kernel_files_ref_code else ""
        # 知识库
        tuning_kb_declaration, tuning_kb_ref_code = self._gen_binary_res_code(self._op_res[op_type].runtime_kb_files)
        tuning_kb_resource = self.TUNING_KB_BINARY_RES_FUNC_FMT.format(
            op_type=op_type,
            reference_code=tuning_kb_ref_code,
        )
        return {
            "binary_config_declaration": binary_config_declaration,
            "kernel_files_declaration": kernel_files_declaration,
            "tuning_kb_declaration": tuning_kb_declaration,
            "kernel_resource": kernel_resource,
            "tuning_kb_resource": tuning_kb_resource,
        }


def generate_op_resource_h_file(args):
    soc_version: str = args.soc_version
    build_dir = args.build_dir

    gen_ini = GenOpResourceIni(soc_version, build_dir, args.jit)
    gen_ini.gen_ops_ini_files()
    return


def parser_generate_op_resource_h_file(subparsers):
    gen_resource_ini_parser = subparsers.add_parser(name='GenStaticOpResourceIni',
                                                    help='Generate xxx_op_resource.h on consolidation server')
    gen_resource_ini_parser.add_argument('-s', '--soc_version', type=str, required=True, dest="soc_version",
                                         help="Operator Name, eg: ascend910b, ascend310p")
    gen_resource_ini_parser.add_argument('-b', '--build_dir', type=str, required=True, dest="build_dir",
                                         help="Input build dir for this project")
    gen_resource_ini_parser.add_argument('-j', '--jit', action='store_true', dest="jit",
                                        help="Generate xxx_op_resource.h  with cann package")
    gen_resource_ini_parser.set_defaults(func=generate_op_resource_h_file)


def execute_argus_parse_func():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(help='Subparsers Commands')

    """ 配置静态编译参数及执行信息 """
    parser_compile_static_library(subparsers)

    """ 配置头文件生成功能参数及执行信息 """
    parser_generate_op_resource_h_file(subparsers)

    """ 生成指定库的symbol文件 """
    parser_generate_symbol(subparsers)

    """ 执行函数功能 """
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    execute_argus_parse_func()
    exit(0)
