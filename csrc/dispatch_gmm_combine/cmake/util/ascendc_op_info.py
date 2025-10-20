#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import sys

import opdesc_parser

PYF_PATH = os.path.dirname(os.path.realpath(__file__))


class OpInfo:
    def __init__(self: any, op_type: str, cfg_file: str):
        op_descs = opdesc_parser.get_op_desc(
            cfg_file, [], [], opdesc_parser.OpDesc, [op_type]
        )
        if op_descs is None or len(op_descs) != 1:
            raise RuntimeError("cannot get op info of {}".format(op_type))
        self.op_desc = op_descs[0]

    def get_op_file(self: any):
        return self.op_desc.op_file

    def get_op_intf(self: any):
        return self.op_desc.op_intf

    def get_inputs_name(self: any):
        return self.op_desc.input_ori_name

    def get_outputs_name(self: any):
        return self.op_desc.output_ori_name


if __name__ == "__main__":
    if len(sys.argv) <= 2:
        raise RuntimeError("arguments must greater than 2")
    op_info = OpInfo(sys.argv[1], sys.argv[2])
    print(op_info.get_op_file())
    print(op_info.get_op_intf())
    print(op_info.get_inputs_name())
    print(op_info.get_outputs_name())
