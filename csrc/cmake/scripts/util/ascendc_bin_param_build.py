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

import argparse
import sys
import os
import json
import hashlib
import re
import copy
from collections import defaultdict
from typing import Dict, List, Set, Tuple, NamedTuple

import const_var
import opdesc_parser

PYF_PATH = os.path.dirname(os.path.realpath(__file__))


class ParamInfo(NamedTuple):
    dtype_list: list
    format_list: list
    dtype_for_bin_list: dict
    format_for_bin_list: dict


class BinParamBuilder(opdesc_parser.OpDesc):
    def __init__(self: any, op_type: str):
        super().__init__(op_type)
        self.soc = ''
        self.out_path = ''
        self.tiling_keys = set()
        self.op_debug_config = ''
        self.op_super_config = []

    def set_soc_version(self: any, soc: str):
        self.soc = soc

    def set_out_path(self: any, out_path: str):
        self.out_path = out_path

    def set_tiling_key(self: any, tiling_key_info: Set):
        if tiling_key_info:
            self.tiling_keys.update(tiling_key_info)

    def set_op_debug_config(self: any, op_debug_config: str):
        if op_debug_config:
            self.op_debug_config = op_debug_config

    def set_op_super_config(self: any, op_super_config: str):
        if op_super_config:
            self.op_super_config = op_super_config

    def get_full_list(self: any):
        dtype_list = []
        for dtype_in in self.input_dtype:
            dtype_list.append(dtype_in.split(','))
        for dtype_out in self.output_dtype:
            dtype_list.append(dtype_out.split(','))

        format_list = []
        for fmt_in in self.input_fmt:
            format_list.append(fmt_in.split(','))
        for fmt_out in self.output_fmt:
            format_list.append(fmt_out.split(','))

        dtype_for_bin_list = [[] for _ in range(len(self.input_dtype) + len(self.output_dtype))]
        format_for_bin_list = copy.deepcopy(dtype_for_bin_list)

        for key, value in self.input_dtype_for_bin.items():
            dtype_for_bin_list[key] = value.split(',')
        for key, value in self.output_dtype_for_bin.items():
            dtype_for_bin_list[key + len(self.input_dtype)] = value.split(',')
        for key, value in self.input_fmt_for_bin.items():
            format_for_bin_list[key] = value.split(',')
        for key, value in self.output_fmt_for_bin.items():
            format_for_bin_list[key + len(self.input_dtype)] = value.split(',')

        return ParamInfo(dtype_list, format_list, dtype_for_bin_list, format_for_bin_list)


    def gen_bin_cprs_list(self: any, param_info: ParamInfo):
        combine_dict = {}
        origin_combine_dict = {}
        for cob_idx in range(0, len(self.input_dtype[0].split(','))):
            origin_combine = ""
            combine = ""
            for param_idx in range(0, len(self.input_dtype) + len(self.output_dtype)):
                if (param_info.dtype_for_bin_list[param_idx]):
                    combine += param_info.dtype_for_bin_list[param_idx][cob_idx]
                else:
                    combine += param_info.dtype_list[param_idx][cob_idx]
                origin_combine += param_info.dtype_list[param_idx][cob_idx]
                if (param_info.format_for_bin_list[param_idx]):
                    combine += param_info.format_for_bin_list[param_idx][cob_idx]
                else:
                    combine += param_info.format_list[param_idx][cob_idx]
                origin_combine += param_info.format_list[param_idx][cob_idx]
            if (combine not in combine_dict):
                combine_dict[combine] = []
            combine_dict[combine].append(cob_idx)
            origin_combine_dict[origin_combine] = cob_idx
        for key, value in combine_dict.items():
            if (key not in origin_combine_dict):
                print(f"WARNING: ForBinQuery {key} not in origin combine")
                self.bin_save_list += value
                continue
            if len(value) == 1 and value[0] == origin_combine_dict[key]:
                self.bin_save_list += value
                continue
            self.bin_cprs_head.append(origin_combine_dict[key])
            self.bin_cprs_list.append(value)
        for index, sub_list in enumerate(self.bin_cprs_list):
            if self.bin_cprs_head[index] not in self.bin_save_list:
                continue
            sub_list.append(self.bin_cprs_head[index])
        self.bin_save_list += self.bin_cprs_head


    def gen_for_bin_list(self: any, param_info: ParamInfo):
        combine_size = len(self.input_dtype[0].split(','))
        input_size = len(self.input_dtype)
        output_size = len(self.output_dtype)

        self.input_dtype_for_bin_list = [[] for _ in range(input_size)]
        self.output_dtype_for_bin_list = [[] for _ in range(output_size)]
        for i in range(0, input_size):
            self.input_dtype_for_bin_list[i] = [[] for _ in range(combine_size)]
        for i in range(0, output_size):
            self.output_dtype_for_bin_list[i] = [[] for _ in range(combine_size)]
        self.input_fmt_for_bin_list = copy.deepcopy(self.input_dtype_for_bin_list)
        self.output_fmt_for_bin_list = copy.deepcopy(self.output_dtype_for_bin_list)

        for index, sub_list in enumerate(self.bin_cprs_list):
            head_idx = self.bin_cprs_head[index]
            for cmb_idx in sub_list:
                for i in range(0, input_size):
                    self.input_dtype_for_bin_list[i][head_idx].append(param_info.dtype_list[i][cmb_idx])
                    self.input_fmt_for_bin_list[i][head_idx].append(param_info.format_list[i][cmb_idx])
                for i in range(0, output_size):
                    self.output_dtype_for_bin_list[i][head_idx].append(param_info.dtype_list[i + input_size][cmb_idx])
                    self.output_fmt_for_bin_list[i][head_idx].append(param_info.format_list[i + input_size][cmb_idx])


    def rm_cprs_cmb(self: any, dtype_list, format_list, input_size, output_size):
        for i in range(0, input_size):
            self.input_dtype_for_bin_list[i] = [
                element for index, element in enumerate(self.input_dtype_for_bin_list[i])
                if index in self.bin_save_list
            ]
            self.input_fmt_for_bin_list[i] = [
                element for index, element in enumerate(self.input_fmt_for_bin_list[i])
                if index in self.bin_save_list
            ]
            new_dtype_list = [
                element for index, element in enumerate(dtype_list[i])
                if index in self.bin_save_list
            ]
            new_dtype_str = ""
            for dtype in new_dtype_list:
                new_dtype_str += f"{dtype},"
            self.input_dtype[i] = new_dtype_str[:-1]
            new_format_list = [
                element for index, element in enumerate(format_list[i])
                if index in self.bin_save_list
            ]
            new_format_str = ""
            for fmt in new_format_list:
                new_format_str += f"{fmt},"
            self.input_fmt[i] = new_format_str[:-1]
        for i in range(0, output_size):
            self.output_dtype_for_bin_list[i] = [
                element for index, element in enumerate(self.output_dtype_for_bin_list[i])
                if index in self.bin_save_list
            ]
            self.output_fmt_for_bin_list[i] = [
                element for index, element in enumerate(self.output_fmt_for_bin_list[i])
                if index in self.bin_save_list
            ]
            new_dtype_list = [
                element for index, element in enumerate(dtype_list[i + input_size])
                if index in self.bin_save_list
            ]
            new_dtype_str = ""
            for dtype in new_dtype_list:
                new_dtype_str += f"{dtype},"
            self.output_dtype[i] = new_dtype_str[:-1]
            new_format_list = [
                element for index, element in enumerate(format_list[i + input_size])
                if index in self.bin_save_list
            ]
            new_format_str = ""
            for fmt in new_format_list:
                new_format_str += f"{fmt},"
            self.output_fmt[i] = new_format_str[:-1]


    def is_set_for_bin_query(self: any):
        return any([
            self.input_dtype_for_bin,
            self.output_dtype_for_bin,
            self.input_fmt_for_bin,
            self.output_fmt_for_bin,
        ])


    def for_bin_list_match(self: any):
        if not self.is_set_for_bin_query():
            return
        input_size = len(self.input_dtype)
        output_size = len(self.output_dtype)
        param_info = self.get_full_list()
        self.gen_bin_cprs_list(param_info)
        self.gen_for_bin_list(param_info)
        if len(self.bin_save_list) == len(self.input_dtype[0].split(',')):
            print(f'WARNING: ForBinQuery can not compress number of bin file with this set, please check!!.')
            return
        self.rm_cprs_cmb(param_info.dtype_list, param_info.format_list, input_size, output_size)


    def gen_input_json(self: any, auto_gen_path: str):
        key_map = {}
        self.for_bin_list_match()
        if len(self.input_dtype) == 0:
            count = len(self.output_dtype[0].split(','))
        else:
            count = len(self.input_dtype[0].split(','))
        if count == 0:
            raise RuntimeError(f'Op {self.op_type} must have at least one input or output')
        required_parameters = set()
        index_value = -1

        for i in range(0, count):
            inputs = []
            outputs = []
            attrs = []
            required_parameter = []
            op_node = {}

            for idx in range(0, len(self.input_name)):
                idtypes = self.input_dtype[idx].split(',')
                ifmts = self.input_fmt[idx].split(',')
                itype = self.input_type[idx]
                para = {}
                para['name'] = self.input_name[idx][:-5]
                para['index'] = idx
                para['dtype'] = idtypes[i]
                if self.is_set_for_bin_query() and self.input_dtype_for_bin_list[idx][i]:
                    para['dtypeForBinQuery'] = self.input_dtype_for_bin_list[idx][i]
                para['format'] = ifmts[i]
                if self.is_set_for_bin_query() and self.input_fmt_for_bin_list[idx][i]:
                    para['formatForBinQuery'] = self.input_fmt_for_bin_list[idx][i]
                para['paramType'] = itype
                para['shape'] = [-2]
                para['format_match_mode'] = 'FormatAgnostic'

                input_parameter_key = (idtypes[i], ifmts[i])
                if itype == 'dynamic':
                    inputs.append([para])
                    required_parameter.append(input_parameter_key)
                elif itype == 'required':
                    inputs.append(para)
                    required_parameter.append(input_parameter_key)
                else:
                    inputs.append(para)

            for idx in range(0, len(self.output_name)):
                odtypes = self.output_dtype[idx].split(',')
                ofmts = self.output_fmt[idx].split(',')
                otype = self.output_type[idx]
                para = {}
                para['name'] = self.output_name[idx][:-5]
                para['index'] = idx
                para['dtype'] = odtypes[i]
                if self.is_set_for_bin_query() and self.output_dtype_for_bin_list[idx][i]:
                    para['dtypeForBinQuery'] = self.output_dtype_for_bin_list[idx][i]
                para['format'] = ofmts[i]
                if self.is_set_for_bin_query() and self.output_fmt_for_bin_list[idx][i]:
                    para['formatForBinQuery'] = self.output_fmt_for_bin_list[idx][i]
                para['paramType'] = otype
                para['shape'] = [-2]
                para['format_match_mode'] = 'FormatAgnostic'
                output_parameter_key = (odtypes[i], ofmts[i])
                if otype == 'dynamic':
                    outputs.append([para])
                    required_parameter.append(output_parameter_key)
                elif otype == 'required':
                    outputs.append(para)
                    required_parameter.append(output_parameter_key)
                else:
                    outputs.append(para)

            for attr in self.attr_list:
                att = {}
                att['name'] = attr
                atype = self.attr_val.get(attr).get('type').lower()
                att['dtype'] = atype
                att['value'] = const_var.ATTR_DEF_VAL.get(atype)
                attrs.append(att)

            required_parameter_tuple = tuple(required_parameter)
            if required_parameter_tuple in required_parameters:
                continue
            else:
                required_parameters.add(required_parameter_tuple)
                index_value +=1

            op_node['bin_filename'] = ''
            op_node['inputs'] = inputs
            op_node['outputs'] = outputs
            if len(attrs) > 0:
                op_node['attrs'] = attrs

            param = {}
            param['op_type'] = self.op_type
            param['op_list'] = [op_node]
            objstr = json.dumps(param, indent='  ')
            md5sum = hashlib.md5(objstr.encode('utf-8')).hexdigest()
            while key_map.get(md5sum) is not None:
                objstr += '1'
                md5sum = hashlib.md5(objstr.encode('utf-8')).hexdigest()
            key_map[md5sum] = md5sum
            bin_file = self.op_type + '_' + md5sum
            op_node['bin_filename'] = bin_file
            param_file = os.path.join(self.out_path, bin_file + '_param.json')
            param_file = os.path.realpath(param_file)

            self._write_build_json(param_file, param)
            self._write_build_cmd(param_file, bin_file, index_value, auto_gen_path)
            if self.op_super_config:
                bin_file += "_relocatable"
                op_node['bin_filename'] = bin_file
                param_file = os.path.join(self.out_path, bin_file + '_param.json')
                param_file = os.path.realpath(param_file)
                self._write_build_json(param_file, param)
                index_value += 1
                self._write_build_cmd(param_file, bin_file, index_value, auto_gen_path, True)

    def _write_build_json(self: any, param_file: str, param):
        with os.fdopen(os.open(param_file, const_var.WFLAGS, const_var.WMODES), 'w') as fd:
            json.dump(param, fd, indent='  ')

    def _generate_check_result(self: any, enable_tiling_keys: bool, bin_file: str):
        check_result = ""
        if enable_tiling_keys is False:
            check_result += "echo \"${res}\"\n"
            check_result += const_var.CHK_CMD.format(res_file=bin_file + '.json')
            check_result += const_var.CHK_CMD.format(res_file=bin_file + '.o')
        else:
            check_result += "if [ $? -eq 1 ]; then\n"
            check_result += "    if echo \"${res}\" | \
grep -q \"None of the given tiling keys are in the supported list\"; then\n"
            check_result += "        echo \"${res}\"\n"
            check_result += "    else\n"
            check_result += "        echo \"${res}\"\n"
            check_result += "        exit 1\n"
            check_result += "    fi\n"
            check_result += "else\n"
            check_result += "echo \"${res}\"\n"
            check_result += const_var.CHK_CMD.format(res_file=bin_file + '.json')
            check_result += const_var.CHK_CMD.format(res_file=bin_file + '.o')
            check_result += "fi\n"
        return check_result

    def _write_build_cmd(self: any, param_file: str, bin_file: str, index: int, auto_gen_path: str, super_mode=False):
        hard_soc = const_var.conv_soc_ver(self.soc)
        if not hard_soc:
            hard_soc = self.soc.capitalize()
        name_com = [self.op_type, self.op_file, str(index)]
        compile_file = os.path.join(self.out_path, '-'.join(name_com) + '.sh')
        compile_file = os.path.realpath(compile_file)

        bin_cmd_str = 'res=$(opc $1 --main_func={fun} --input_param={param} --soc_version={soc} \
                --output=$2 --impl_mode={impl} --simplified_key_mode=0 --op_mode=dynamic '

        build_cmd_var = "#!/bin/bash\n"
        build_cmd_var += f'echo "[{self.soc}] Generating {bin_file} ..."\n'
        plog_level = os.environ.get("ASCEND_GLOBAL_LOG_LEVEL")
        plog_stdout = os.environ.get("ASCEND_SLOG_PRINT_TO_STDOUT")
        if plog_level is None:
            build_cmd_var += const_var.SET_PLOG_LEVEL_ERROR
        if plog_stdout is None:
            build_cmd_var += const_var.SET_PLOG_STDOUT
        build_cmd_var += const_var.SRC_ENV
        if hard_soc == "Ascend610Lite":
            build_cmd_var += f'export ASCEND_CUSTOM_OPP_PATH={auto_gen_path}:$ASCEND_CUSTOM_OPP_PATH \n'
        build_cmd_var += bin_cmd_str.format(fun=self.op_intf, soc=hard_soc, param=param_file,
                                           impl='high_performance,optional')
        enable_tiling_keys = False
        if self.tiling_keys:
            tiling_keys_list = sorted(list(self.tiling_keys))
            tiling_key_str = ','.join([str(_key) for _key in tiling_keys_list])
            build_cmd_var += f' --tiling_key="{tiling_key_str}"'
            enable_tiling_keys = True

        if self.op_debug_config:
            op_debug_str = ','.join([str(_key) for _key in list(self.op_debug_config)])
            build_cmd_var += f' --op_debug_config={op_debug_str}'

        if super_mode and self.op_super_config:
            op_super_config_str = ' '.join([str(_key) for _key in list(self.op_super_config)])
            build_cmd_var += f' {op_super_config_str}'

        build_cmd_var += ")\n"
        build_cmd_var += "\n"
    
        check_result = self._generate_check_result(enable_tiling_keys, bin_file)
        build_cmd_var += check_result
        build_cmd_var += f'echo "[{self.soc}] Generating {bin_file} Done"\n'

        with os.fdopen(os.open(compile_file, const_var.WFLAGS, const_var.WMODES), 'w') as fd:
            fd.write(build_cmd_var)


def get_tiling_keys(tiling_keys: str) -> Set:
    all_tiling_keys = set()
    if not tiling_keys:
        return all_tiling_keys

    tiling_key_list = tiling_keys.split(';')
    for tiling_key_value in tiling_key_list:
        pattern = r"(?<![^\s])(\d+)-(\d+)(?![^\s])"
        results = re.findall(pattern, tiling_key_value)
        if results:
            start, end = results[0]
            if int(start) > int(end):
                continue
            for i in range(int(start), int(end) + 1):
                all_tiling_keys.add(i)
        elif tiling_key_value.isdigit():
            all_tiling_keys.add(int(tiling_key_value))
    return all_tiling_keys


def trans_soc_verion(soc_ver: str):
    low_soc_ver = soc_ver.lower()
    if low_soc_ver not in opdesc_parser.SOC_TO_SHORT_SOC_MAP:
        return low_soc_ver
    return opdesc_parser.SOC_TO_SHORT_SOC_MAP[low_soc_ver]


def parse_op_debug_confg(opc_config_file: str, soc: str) -> Dict:
    tiling_key_info = defaultdict(set)
    op_debug_config = defaultdict(set)
    if not opc_config_file:
        return tiling_key_info, op_debug_config

    if not os.path.exists(opc_config_file):
        return tiling_key_info, op_debug_config

    with open(opc_config_file, 'r') as file:
        contents = file.readlines()

    for _content in contents:
        content = _content.strip()
        opc_configs = content.split('@')
        if len(opc_configs) < 3:
            continue

        op_type = opc_configs[0]
        if not op_type:
            continue

        compute_unit = opc_configs[1]
        if compute_unit:
            compute_unit_list = compute_unit.split(';')
            soc_lists = []
            for soc_ver in compute_unit_list:
                short_soc_ver = trans_soc_verion(soc_ver)
                soc_lists.append(short_soc_ver)
            if soc not in soc_lists:
                continue

        for options in opc_configs[2:]:
            if "--tiling_key" in options:
                format_tiling_keys = get_tiling_keys(options.split('=')[1])
                if format_tiling_keys:
                    tiling_key_info[op_type].update(format_tiling_keys)
            if "--op_debug_config" in options:
                first_index = options.find('=')
                if first_index != -1:
                    debug_config = options[first_index + 1:]
                else:
                    debug_config = ""

                format_debug_config = set(debug_config.split(';'))
                for _config in format_debug_config:
                    op_debug_config[op_type].add(_config)
    return tiling_key_info, op_debug_config


def gen_bin_param_file(cfgfile: str, out_dir: str, soc: str,
                        opc_config_file: str = '', ops: list = None):
    if not os.path.exists(cfgfile):
        print(f'INFO: {cfgfile} does not exists in this project, skip generating compile commands.')
        return

    debug_config = defaultdict(set)
    super_config = defaultdict(set)
    
    op_descs = opdesc_parser.get_op_desc(cfgfile, [], [], BinParamBuilder, ops)
    tiling_key_info, op_debug_config = parse_op_debug_confg(opc_config_file, soc)
    for _op_type, _op_option in op_debug_config.items():
        for _option in _op_option:
            if (_option.startswith("--op_relocatable_kernel_binary")
                or _option.startswith("--op_super_kernel_options")):
                super_config[_op_type].add(_option)
            else:
                debug_config[_op_type].add(_option)

    auto_gen_path_dir = os.path.dirname(cfgfile)
    all_soc_key = "ALL"
    for op_desc in op_descs:
        op_desc.set_soc_version(soc)
        op_desc.set_out_path(out_dir)
        if op_desc.op_type in debug_config:
            op_desc.set_op_debug_config(debug_config[op_desc.op_type])
        if all_soc_key in debug_config:
            op_desc.set_op_debug_config(debug_config[all_soc_key])
        if op_desc.op_type in super_config:
            op_desc.set_op_super_config(super_config[op_desc.op_type])
        if op_desc.op_type in tiling_key_info:
            op_desc.set_tiling_key(tiling_key_info[op_desc.op_type])
        if all_soc_key in tiling_key_info:
            op_desc.set_tiling_key(tiling_key_info[all_soc_key])
        op_desc.gen_input_json(auto_gen_path_dir)


def parse_args(argv):
    """Command line parameter parsing"""
    parser = argparse.ArgumentParser()
    parser.add_argument('argv', nargs='+')
    parser.add_argument('--opc-config-file', nargs='?', const='', default='')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args(sys.argv)
    if len(args.argv) <= 3:
        raise RuntimeError('arguments must greater than 3')
    gen_bin_param_file(args.argv[1],
                    args.argv[2],
                    args.argv[3],
                    opc_config_file=args.opc_config_file)