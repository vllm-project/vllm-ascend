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
获取 Soc 相关信息

Examples 场景下, 用于获取 Soc 相关信息.
"""
import argparse
import logging
import ctypes


class SocInfoMgr:

    @staticmethod
    def get_soc_name() -> str:
        acl_lib = ctypes.cdll.LoadLibrary("libascendcl.so")
        acl_lib.aclrtGetSocName.restype = ctypes.c_char_p
        rst = acl_lib.aclrtGetSocName()
        if rst:
            rst = str(rst, encoding='utf-8')
        else:
            rst = ""
        return rst

    @staticmethod
    def main() -> str:
        ps = argparse.ArgumentParser(description="Get soc info", epilog="Best Regards!")
        ps.add_argument("-i", "--info", required=True, type=str, help="SocInfo")
        args = ps.parse_args()
        rst = ""
        if args.info == "soc_name":
            rst = SocInfoMgr.get_soc_name()
        else:
            logging.error("Unknown SocInfo name %s", args.info)
        return rst


if __name__ == "__main__":
    logging.basicConfig(format='%(filename)s:%(lineno)d [%(levelname)s] %(message)s', level=logging.INFO)
    g_rst = ""
    try:
        g_rst = SocInfoMgr.main()
    except Exception as e:
        logging.error(e)
    print(g_rst, end='')
