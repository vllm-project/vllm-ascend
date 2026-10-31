#!/usr/bin/python
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import concurrent.futures
import os
from pathlib import Path

import pytest
import result_compare_method
from batch import quant_lightning_indexer_v2_pt_loadprocess
from qliv2_test_utils import (
    QliV2CaseSelector,
    QliV2ResultWriter,
    ensure_comparison_passed,
)

TEST_INPUT_PATH_ENV = os.environ.get("QLIV2_TESTCASE_DIR", "").strip()
TEST_INPUT_PATH = TEST_INPUT_PATH_ENV or "pt_path"
RESULT_PATH = Path(os.environ.get("QLIV2_RESULT_PATH", "result.xlsx").strip())
DEVICE_ID = int(os.environ.get("QLIV2_DEVICE_ID", "0"))

# 支持通过环境变量 QLIV2_TESTCASE_PATH 指定单条用例文件，实现进程级隔离执行：
#   - 设置时：仅运行该条用例（配合 batch_isolated_run.sh 每条用例拉起独立进程）
#   - 未设置：回退为原有行为，一次性加载目录下全部用例
SINGLE_CASE_PATH = os.environ.get("QLIV2_TESTCASE_PATH", "").strip()
# flag：是否处于批量隔离模式（由 batch_isolated_run.sh 设置 QLIV2_TESTCASE_PATH 触发）
IS_ISOLATED_MODE = bool(SINGLE_CASE_PATH)
# flag：运行模式 eager / graph（通过环境变量 QLIV2_RUN_MODE 或命令行参数设置，默认 eager）
RUN_MODE = os.environ.get("QLIV2_RUN_MODE", "eager").strip().lower()
# 支持通过环境变量 QLIV2_PT_FILE_LIST 指定用例文件列表（逗号分隔），用于从 Excel 筛选的 batch_exec 模式
PT_FILE_LIST = os.environ.get("QLIV2_PT_FILE_LIST", "").strip()
CASE_NAMES = os.environ.get("QLIV2_CASE_NAMES", "").strip()
CASE_INDEXES = os.environ.get("QLIV2_CASE_INDEXES", "").strip()

try:
    if SINGLE_CASE_PATH:
        TESTCASE_FILES = QliV2CaseSelector.resolve(TEST_INPUT_PATH, explicit_files=SINGLE_CASE_PATH)
        print(f"单用例隔离模式, 仅执行: {SINGLE_CASE_PATH}")
    else:
        TESTCASE_FILES = QliV2CaseSelector.resolve(
            TEST_INPUT_PATH,
            explicit_files=PT_FILE_LIST,
            case_names=CASE_NAMES,
            case_indexes=CASE_INDEXES,
        )
        print(f"找到 {len(TESTCASE_FILES)} 个测试用例文件")
except ValueError as error:
    has_explicit_selection = any(
        (
            TEST_INPUT_PATH_ENV,
            SINGLE_CASE_PATH,
            PT_FILE_LIST,
            CASE_NAMES,
            CASE_INDEXES,
        )
    )
    if has_explicit_selection:
        raise
    print(f"未配置 batch PT 用例，跳过收集: {error}")
    TESTCASE_FILES = []


def qliv2(testcase_file):
    try:
        if RUN_MODE == "graph":
            (
                cpu_result,
                npu_result,
                topk_value,
                cpu_topk_value,
                npu_topk_value,
                output_idx_offset,
                params,
            ) = quant_lightning_indexer_v2_pt_loadprocess.test_qliv2_process_graph(testcase_file, device_id=DEVICE_ID)
        else:
            (
                cpu_result,
                npu_result,
                topk_value,
                cpu_topk_value,
                npu_topk_value,
                output_idx_offset,
                params,
            ) = quant_lightning_indexer_v2_pt_loadprocess.test_qliv2_process(testcase_file, device_id=DEVICE_ID)
        if npu_result is not None:
            result, fulfill_percent = result_compare_method.check_result(
                cpu_result,
                npu_result,
                topk_value,
                output_idx_offset,
                params,
                cpu_topk_value,
                npu_topk_value,
            )
        else:
            result = "Failed"
            fulfill_percent = 0
        return_value = params[31]
        if return_value:
            result_return_value, fulfill_precent_return_value = result_compare_method.check_result_return_value(
                cpu_topk_value,
                npu_topk_value,
                params,
                cpu_result,
                npu_result,
                topk_value,
                output_idx_offset,
            )
            print(f"result_return_value: {result_return_value}")
            print(f"fulfill_precent_return_value: {fulfill_precent_return_value}")
        else:
            result_return_value = "N/A"
            fulfill_precent_return_value = 0
    except Exception as error:
        print("NPU ERROR：", error)
        result = "NPU ERROR"
        fulfill_percent = 0
        result_return_value = "N/A"
        fulfill_precent_return_value = 0
        params = [None] * 33

    row_data = QliV2ResultWriter.row(
        Path(testcase_file).stem,
        params,
        result,
        fulfill_percent,
        result_return_value,
        fulfill_precent_return_value,
    )
    QliV2ResultWriter.append(RESULT_PATH, row_data)

    case_name = Path(testcase_file).stem
    if result != "NPU ERROR":
        try:
            ensure_comparison_passed(
                case_name,
                result,
                fulfill_percent,
                result_return_value,
                fulfill_precent_return_value,
            )
        except AssertionError as error:
            return str(error)

    if result == "NPU ERROR":
        return f"用例执行失败:{Path(testcase_file).stem}"
    return None


@pytest.mark.ci
@pytest.mark.parametrize("testcase_file", TESTCASE_FILES)
def test_qliv2(testcase_file):
    if IS_ISOLATED_MODE:
        # 批量隔离模式：shell 层已通过独立 pytest 进程提供进程隔离，内部使用线程池即可
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            futures = executor.submit(qliv2, testcase_file)
            for future in concurrent.futures.as_completed([futures]):
                try:
                    result = future.result()
                    if result is not None:
                        pytest.fail(str(result))
                except Exception as e:
                    pytest.fail(f"当前用例线程执行失败：{e}")
    else:
        # 非隔离模式（直接 pytest）：使用子进程隔离，防止单条用例崩溃影响整体
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future1 = executor.submit(qliv2, testcase_file)
            for future in concurrent.futures.as_completed([future1]):
                try:
                    result = future.result()
                    if result is not None:
                        pytest.fail(str(result))
                except Exception as e:
                    pytest.fail(f"当前用例子进程执行失败：{e}")
