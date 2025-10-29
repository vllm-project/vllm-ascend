#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import argparse
import glob
import json
import math
import os
import subprocess
import sys

import ascendc_ops_config
import const_var
from tbe.tikcpp.log_utils import AscendCLogLevel, LogUtil


class PackKernel:
    def __init__(self: any, args: any):
        self.in_path = os.path.realpath(args.input_path)
        self.out_path = os.path.realpath(args.output_path)
        self.is_lib = args.enable_library
        self.platform = args.platform
        self.op_info = {}
        self.file_info = {}
        try:
            os.makedirs(self.out_path, exist_ok=True)
        except Exception as e:
            LogUtil.print_compile_log(
                "",
                f"make {self.out_path} error: {e}!",
                AscendCLogLevel.LOG_ERROR,
                LogUtil.Option.NON_SOC,
            )

    def load_json(self: any, json_file: str):
        with open(json_file, encoding="utf-8") as file:
            json_content = json.load(file)
            return json_content

    def get_symbol(self: any, name: str):
        name = name.replace("/", "_")
        return name.replace(".", "_")

    def ascendc_gen_object(self: any, in_file: str, soc: str):
        sym = self.get_symbol("_binary_" + in_file)
        out_file = os.path.join(self.out_path, sym + ".o")
        # ascend610lite only support aarch64
        if soc == "ascend610lite":
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
                subprocess.run(["echo", "unsported environment!"])
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
        os.chdir(self.in_path)
        soc_vers = os.listdir("config")
        for soc in soc_vers:
            bin_infos = glob.glob(os.path.join("config", soc, "*.json"))
            cfgs = {}
            for bin_info in bin_infos:
                if bin_info.find("binary_info_config.json") > 0:
                    continue
                jobj = self.load_json(bin_info)
                for bin_cfg in jobj.get("binList"):
                    js_cfg = bin_cfg.get("binInfo").get("jsonFilePath")
                    op_type = os.path.basename(js_cfg).split("_")[0]
                    if cfgs.get(op_type) is None:
                        op_obj = {}
                        op_obj["obj"] = []
                        op_obj["cfg"] = bin_info
                        cfgs[op_type] = op_obj
                    op_obj = cfgs.get(op_type)
                    op_obj.get("obj").append(js_cfg[:-5])
                self.file_info[soc] = cfgs

    def ascendc_pack_kernel(self: any):
        for soc in self.file_info.keys():
            os.chdir(self.in_path)
            op_cfgs = self.file_info.get(soc)
            for op_type in op_cfgs.keys():
                op_obj = op_cfgs.get(op_type)
                if self.op_info.get(op_type) is None:
                    op_info = {}
                    op_info["op_fun"] = ["nullptr", "nullptr"]
                    op_info["op_bin"] = {}
                    op_info["op_rkb"] = []
                    self.op_info[op_type] = op_info
                op_info = self.op_info.get(op_type)
                op_bin = op_info.get("op_bin")
                if op_bin.get(soc) is None:
                    op_bin[soc] = []
                    op_bin[soc].append(self.ascendc_gen_object(op_obj["cfg"], soc))
                op_soc = op_bin.get(soc)
                for objs in op_obj["obj"]:
                    op_soc.append(self.ascendc_gen_object(objs + ".json", soc))
                    op_soc.append(self.ascendc_gen_object(objs + ".o", soc))

    def ascendc_gen_header(self: any):
        for op_type in self.op_info.keys():
            op_obj = self.op_info.get(op_type)
            macro_op = (
                "#define {}_OP_RESOURCES std::make_tuple<std::vector<void *>, \\\n"
                "    std::map<ge::AscendString, std::vector<std::tuple<const uint8_t *, const uint8_t *>>>, \\\n"
                "    std::vector<std::tuple<const uint8_t *, const uint8_t *>>>({{{}}}, \\\n".format(
                    op_type, ", ".join(op_obj.get("op_fun"))
                )
            )
            op_bin = op_obj.get("op_bin")
            socs_res = []
            op_syms = []
            for soc in op_bin.keys():
                soc_res = '{{ "{}", {{'.format(soc)
                soc_syms = op_bin.get(soc)
                soc_pairs = []
                for pair_addr in soc_syms:
                    pair_addr1 = ["&" + s for s in pair_addr]
                    op_syms += pair_addr
                    soc_pairs.append(
                        "    {{ {} }} ".format(", \\\n      ".join(pair_addr1))
                    )
                soc_res += ", \\\n        ".join(soc_pairs)
                soc_res += " } }"
                socs_res.append(soc_res)
            macro_op += "    {{ {} }}, \\\n".format(", \\\n      ".join(socs_res))
            macro_op += "    {{ {} }})\n\n".format(", ".join(op_obj.get("op_rkb")))
            macro_str = '#define {}_RESOURCES {{{{"{}", {}}}}}'.format(
                op_type, op_type, "{}_OP_RESOURCES".format(op_type)
            )
            var_str = (
                "extern gert::OpImplRegisterV2 op_impl_register_optiling_{};\n".format(
                    op_type
                )
            )
            if len(op_syms) > 0:
                var_str += (
                    "extern uint8_t " + ";\nextern uint8_t ".join(op_syms) + ";\n"
                )
            head_file = os.path.join(self.out_path, "{}_op_resource.h".format(op_type))
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
                    fd.write(macro_op)
                    fd.write(macro_str)
            except Exception as e:
                LogUtil.print_compile_log(
                    "",
                    f"{op_type}_op_resource.h create error: {e}!",
                    AscendCLogLevel.LOG_ERROR,
                    LogUtil.Option.NON_SOC,
                )

    def ascendc_gen_lib(self: any):
        out_lib = os.path.join(self.out_path, "libkernels.a")
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

    def ascendc_gen_opsinfo(self: any):
        ascendc_ops_config.gen_all_soc_config(self.in_path)


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input-path", nargs="?", help="Input path of compile result."
    )
    parser.add_argument(
        "-o", "--output-path", nargs="?", help="Output path of compile result."
    )
    parser.add_argument(
        "-l",
        "--enable-library",
        nargs="?",
        default=None,
        help="Whether library is enabled.",
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
    if kernel_packer.is_lib is None:
        kernel_packer.ascendc_gen_opsinfo()
    kernel_packer.ascendc_get_config()
    kernel_packer.ascendc_pack_kernel()
    kernel_packer.ascendc_gen_header()
    kernel_packer.ascendc_gen_lib()
