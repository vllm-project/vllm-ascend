#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import time
import inspect
import logging

logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [%(pathname)s] [line:%(lineno)d] %(message)s',
                    level=logging.INFO)


class CommLog:
    @staticmethod
    def cilog_get_timestamp():
        return time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())

    @staticmethod
    def cilog_print_element(cilog_element):
        print("["+cilog_element+"]", end=' ')
        return

    @staticmethod
    def cilog_logmsg(log_level, filename, line_no, log_msg, *log_paras):
        log_timestamp = CommLog.cilog_get_timestamp()
        CommLog.cilog_print_element(log_timestamp)
        CommLog.cilog_print_element(log_level)
        CommLog.cilog_print_element(filename)
        CommLog.cilog_print_element(str(line_no))
        print(log_msg % log_paras[0])
        return

    @staticmethod
    def cilog_error(log_msg, *log_paras):
        frame = inspect.currentframe().f_back
        line_no = frame.f_lineno
        filename = frame.f_code.co_filename
        CommLog.cilog_logmsg("ERROR", filename, line_no, log_msg, log_paras)
        return

    @staticmethod
    def cilog_warning(log_msg, *log_paras):
        frame = inspect.currentframe().f_back
        line_no = frame.f_lineno
        filename = frame.f_code.co_filename
        CommLog.cilog_logmsg("WARNING", filename, line_no, log_msg, log_paras)
        return

    @staticmethod
    def cilog_info(log_msg, *log_paras):
        frame = inspect.currentframe().f_back
        line_no = frame.f_lineno
        filename = frame.f_code.co_filename
        CommLog.cilog_logmsg("INFO", filename, line_no, log_msg, log_paras)
        return
