#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""
Function:
The replay funtion entry
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""

import os
import stat


REPLAY_BATCH = 'batch'
REPLAY_ITERATE = 'iterate'
CFG_IMPL_DIR = 'impl_dir'
CFG_OUT_DIR = 'out_dir'
AUTO_GEN_DIR = 'auto_gen_dir'
WFLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
WMODES = stat.S_IWUSR | stat.S_IRUSR
SOC_MAP_EXT = {'ascend310p': 'Ascend310P3', 'ascend310b': 'Ascend310B1',
               'ascend910': 'Ascend910A', 'ascend910b': 'Ascend910B1',
               'ascend910_93': 'Ascend910_9391', 'kirinx90': 'KirinX90'}
BIN_CMD = 'asc_opc $1 --main_func={fun} --input_param={param} --soc_version={soc} \
--output=$2 --impl_mode={impl} --simplified_key_mode=0 --op_mode=dynamic\n'
SET_PLOG_LEVEL_ERROR = "export ASCEND_GLOBAL_LOG_LEVEL=3\n"
SET_PLOG_STDOUT = "export ASCEND_SLOG_PRINT_TO_STDOUT=1\n"
SRC_ENV = '''
while true; do
  case "$1" in
    --kernel-src=*)
      export BUILD_KERNEL_SRC=$(echo "$1" | cut -d"=" -f2-)
      shift
      ;;
    -*)
      shift
      ;;
    *)
      break
      ;;
  esac
done
'''
CHK_CMD = '''
if ! test -f $2/{res_file} ; then
  echo "$2/{res_file} not generated!"
  exit 1
fi
'''
ATTR_DEF_VAL = {'str' : '', 'int': 0, 'float': 0.0, 'bool': False, 'list_bool': [],
                'list_int': [], 'list_float': [], 'list_list_int': [[]]}


def conv_soc_ver(ver: str):
    return SOC_MAP_EXT.get(ver)
