#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import argparse
import glob
import math
import os
import shutil
import subprocess
import sys

import const_var
from tbe.tikcpp.log_utils import AscendCLogLevel, LogUtil


class PackKernel:
    def __init__(self: any, args: any):
        self.in_path = os.path.realpath(args.input_path)
        self.copy_path = os.path.realpath(args.copy_path)
        self.out_path = os.path.realpath(args.output_path)
        self.op_soc_ver = args.compute_unit.split("-")
        self.vendor_name = args.vendor_name
        self.framework_type = args.framework_type
        self.platform = args.platform
        self.op_info = {}
        self.file_info = {}
        if os.path.exists(self.copy_path):
            try:
                shutil.rmtree(self.copy_path)
            except OSError as e:
                LogUtil.print_compile_log(
                    "",
                    f"remove {self.copy_path} error!",
                    AscendCLogLevel.LOG_ERROR,
                    LogUtil.Option.NON_SOC,
                )
        if os.path.exists(self.out_path):
            try:
                shutil.rmtree(self.out_path)
            except OSError as e:
                LogUtil.print_compile_log(
                    "",
                    f"remove {self.out_path} error!",
                    AscendCLogLevel.LOG_ERROR,
                    LogUtil.Option.NON_SOC,
                )
        try:
            os.makedirs(self.copy_path, exist_ok=True)
        except Exception as e:
            LogUtil.print_compile_log(
                "",
                f"make {self.copy_path} error: {e}!",
                AscendCLogLevel.LOG_ERROR,
                LogUtil.Option.NON_SOC,
            )
        try:
            os.makedirs(self.out_path, exist_ok=True)
        except Exception as e:
            LogUtil.print_compile_log(
                "",
                f"make {self.out_path} error: {e}!",
                AscendCLogLevel.LOG_ERROR,
                LogUtil.Option.NON_SOC,
            )

    def get_symbol(self: any, name: str):
        name = name.replace("/", "_")
        name = name.replace("-", "_")
        return name.replace(".", "_")

    def ascendc_gen_object(self: any, in_file: str, path: str):
        sym = self.get_symbol("_binary_" + in_file)
        out_file = os.path.join(self.out_path, sym + ".o")
        # ascend610lite only support aarch64
        if path.find("ascend610lite") != -1:
            try:
                subprocess.run(
                    [
                        "llvm-objcopy",
                        "--input-target",
                        "binary",
                        "--output-target",
                        "elf64-littleaarch64",
                        "--binary-architecture",
                        "aarch64",
                        in_file,
                        out_file,
                    ]
                )
            except Exception as e:
                LogUtil.print_compile_log(
                    "",
                    " ascend610lite execute objcopy fail!",
                    AscendCLogLevel.LOG_ERROR,
                    LogUtil.Option.NON_SOC,
                )
                return None
            return [sym + "_start", sym + "_end"]

        uname = os.popen("uname -m").read().strip()
        if self.platform is not None:
            target_platform = self.platform
        else:
            target_platform = uname
        try:
            if target_platform == "x86_64":
                subprocess.run(
                    [
                        "llvm-objcopy",
                        "--input-target",
                        "binary",
                        "--output-target",
                        "elf64-x86-64",
                        "--binary-architecture",
                        "i386",
                        in_file,
                        out_file,
                    ]
                )
            elif target_platform == "aarch64":
                subprocess.run(
                    [
                        "llvm-objcopy",
                        "--input-target",
                        "binary",
                        "--output-target",
                        "elf64-littleaarch64",
                        "--binary-architecture",
                        "aarch64",
                        in_file,
                        out_file,
                    ]
                )
            else:
                subprocess.run(["echo", "unsupported environment!"])
        except Exception as e:
            LogUtil.print_compile_log(
                "",
                f"{target_platform} execute objcopy error: {e}!",
                AscendCLogLevel.LOG_ERROR,
                LogUtil.Option.NON_SOC,
            )
            return None
        return [sym + "_start", sym + "_end"]

    def ascendc_get_config(self: any):
        os.chdir(self.copy_path)
        current_directory = os.getcwd()
        catalog_file = os.listdir(current_directory)
        for catalog in catalog_file:
            if catalog == "op_impl" or catalog == "framework":
                files_dict = {}
                for root, _, files in os.walk(catalog):
                    for file in files:
                        if (
                            file.endswith(".json")
                            or file.endswith(".so")
                            or file.endswith(".cpp")
                            or file.endswith(".py")
                            or file.endswith(".o")
                        ):
                            file_path = os.path.join(root, file)
                            file_name = os.path.basename(file_path)
                            files_dict[file_name] = file_path
                self.file_info[catalog] = files_dict

    def ascendc_pack_kernel(self: any):
        op_info = {}
        for files in self.file_info.keys():
            os.chdir(self.copy_path)
            op_cfgs = self.file_info.get(files)
            for file_name in op_cfgs.keys():
                op_info[file_name] = []
                path, filename = os.path.split(op_cfgs[file_name])
                op_info[file_name].append(os.path.join(self.vendor_name, path))
                op_info[file_name].append(
                    self.ascendc_gen_object(op_cfgs[file_name], path)
                )
        self.op_info = op_info

    def ascendc_gen_header(self: any):
        socs_res = []
        var_str = ""
        macro_op = (
            "std::vector<std::tuple<ge::AscendString, ge::AscendString, "
            "const uint8_t *, const uint8_t *>> __ascendc_op_info = \n"
        )
        for file_name in self.op_info.keys():
            file_addr = self.op_info.get(file_name)
            soc_pairs = []
            op_syms = []
            soc_res = ' {{ "{}", '.format(file_name)
            soc_res += '"{}", '.format(file_addr[0])
            for pair_addr in file_addr[1]:
                op_syms.append(pair_addr)
                pair_addr1 = "&" + pair_addr
                soc_pairs.append(pair_addr1)
            soc_res += "{}, {}".format(soc_pairs[0], soc_pairs[1])
            soc_res += "}, \n"
            socs_res.append(soc_res)
            if len(op_syms) > 0:
                var_str += "".join(
                    ["extern uint8_t {};\n".format(sym) for sym in op_syms]
                )
        macro_op += "{{\n{}}}; \n".format("".join(socs_res))
        head_file = os.path.join(self.out_path, "ge_table_op_resource.h")
        try:
            with os.fdopen(
                os.open(head_file, const_var.WFLAGS, const_var.WMODES), "w"
            ) as fd:
                fd.write("#include <stdint.h>\n")
                fd.write("#include <map>\n")
                fd.write("#include <tuple>\n")
                fd.write("#include <vector>\n")
                fd.write('#include "graph/ascend_string.h"\n')
                fd.write('#include "register/op_impl_registry.h"\n\n')
                fd.write(var_str)
                fd.write("\n")
                fd.write("namespace AscendC {\n")
                fd.write(macro_op)
                fd.write("}\n")
        except Exception as e:
            LogUtil.print_compile_log(
                "",
                f"ge_table_op_resource.h create error: {e}!",
                AscendCLogLevel.LOG_ERROR,
                LogUtil.Option.NON_SOC,
            )

    def ascendc_gen_lib(self: any):
        out_lib = os.path.join(self.out_path, "libopregistry.a")
        if os.path.exists(out_lib):
            os.remove(out_lib)
        objs = glob.glob(os.path.join(self.out_path, "*.o"))
        start = 0
        batch_size = 100
        for _ in range(math.ceil(len(objs) / batch_size)):
            sub_objs = objs[start : start + batch_size]
            start += batch_size
            try:
                subprocess.run(["ar", "qc", out_lib] + sub_objs)
                subprocess.run(["ranlib", out_lib])
            except Exception as e:
                LogUtil.print_compile_log(
                    "",
                    f"execute ar/ranlib command error: {e}!",
                    AscendCLogLevel.LOG_ERROR,
                    LogUtil.Option.NON_SOC,
                )

    def ascendc_copy_dir(self: any, src_dir: str, target_dir: str):
        file_list = os.listdir(src_dir)
        for file_name in file_list:
            source_file = os.path.join(src_dir, file_name)
            target_file = os.path.join(target_dir, file_name)
            if os.path.isdir(source_file):
                try:
                    shutil.copytree(source_file, target_file)
                except Exception as e:
                    LogUtil.print_compile_log(
                        "",
                        f"copy {source_file} error: {e}!",
                        AscendCLogLevel.LOG_ERROR,
                        LogUtil.Option.NON_SOC,
                    )

    def ascendc_copy_file(self: any, src_dir: str, target_dir: str):
        file_list = os.listdir(src_dir)
        for file_name in file_list:
            source_file = os.path.join(src_dir, file_name)
            if os.path.isfile(source_file):
                try:
                    os.makedirs(target_dir, exist_ok=True)
                except Exception as e:
                    LogUtil.print_compile_log(
                        "",
                        f"make {target_dir} error: {e}!",
                        AscendCLogLevel.LOG_ERROR,
                        LogUtil.Option.NON_SOC,
                    )
                try:
                    shutil.copy(source_file, target_dir)
                except Exception as e:
                    LogUtil.print_compile_log(
                        "",
                        f"copy {source_file} error: {e}!",
                        AscendCLogLevel.LOG_ERROR,
                        LogUtil.Option.NON_SOC,
                    )

    def ascendc_copy_func(self: any):
        os.chdir(self.in_path)
        framework_catalog = os.listdir("framework")
        for catalog_file in framework_catalog:
            if (
                catalog_file == "tf_plugin"
                or catalog_file == "caffe_plugin"
                or catalog_file == "onnx_plugin"
            ):
                source_dir = "op_kernel/tbe/op_info_cfg/ai_core"
                dst_dir = os.path.join(self.copy_path, "framework", self.framework_type)
                self.ascendc_copy_file(source_dir, dst_dir)
                source_dir = os.path.join("framework", catalog_file)
                dst_dir = os.path.join(self.copy_path, "framework", self.framework_type)
                self.ascendc_copy_file(source_dir, dst_dir)
        source_dir = "op_kernel/tbe/op_info_cfg/ai_core"
        dst_dir = os.path.join(self.copy_path, "op_impl/ai_core/tbe/config")
        self.ascendc_copy_dir(source_dir, dst_dir)
        source_dir = "op_kernel/binary/dynamic"
        dst_dir = os.path.join(
            self.copy_path, "op_impl/ai_core/tbe", self.vendor_name + "_impl", "dynamic"
        )
        self.ascendc_copy_file(source_dir, dst_dir)
        for compute_unit in self.op_soc_ver:
            source_dir = os.path.join("op_kernel/binary", compute_unit)
            dst_dir = os.path.join(
                self.copy_path, "op_impl/ai_core/tbe/kernel", compute_unit
            )
            self.ascendc_copy_dir(source_dir, dst_dir)
        source_dir = "op_kernel/binary/config"
        dst_dir = os.path.join(self.copy_path, "op_impl/ai_core/tbe/kernel/config")
        self.ascendc_copy_dir(source_dir, dst_dir)
        so_file = "op_impl/ai_core/tbe/op_master_device/lib/libcust_opmaster.so"
        if os.path.exists(so_file):
            dst_dir = os.path.join(
                self.copy_path, "op_impl/ai_core/tbe/op_master_device/lib"
            )
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy(so_file, dst_dir)


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input-path", nargs="?", help="Input path of compile result."
    )
    parser.add_argument(
        "-c", "--copy-path", nargs="?", help="Copy path of compile result."
    )
    parser.add_argument(
        "-o", "--output-path", nargs="?", help="Output path of compile result."
    )
    parser.add_argument("-n", "--vendor-name", nargs="?", help="Vendor name.")
    parser.add_argument("-u", "--compute-unit", nargs="?", help="Compute unit.")
    parser.add_argument(
        "-t", "--framework-type", nargs="?", help="Framework type, eg:tensorflow."
    )
    parser.add_argument(
        "-p",
        "--platform",
        nargs="?",
        default=None,
        help="target platform is x86_64 or aarch64.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = args_parse()
    kernel_packer = PackKernel(args)
    kernel_packer.ascendc_copy_func()
    kernel_packer.ascendc_get_config()
    kernel_packer.ascendc_pack_kernel()
    kernel_packer.ascendc_gen_header()
    kernel_packer.ascendc_gen_lib()
