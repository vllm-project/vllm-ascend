#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
import pytest
import pandas as pd
from pathlib import Path
import os
import concurrent.futures
import result_compare_method
from batch import mixed_quant_sparse_flash_mla_process
import utils
import sys
import torch_npu

SMLA_PYTEST_PATH = Path(__file__).resolve().parent
if str(SMLA_PYTEST_PATH) not in sys.path:
    sys.path.append(str(SMLA_PYTEST_PATH))

from batch_consistency.config import resolve_consistency_config  # noqa: E402
from batch_consistency.model import RunResult  # noqa: E402
from batch_consistency.pytest_support import (  # noqa: E402
    consistency_fulfill_percent,
    format_consistency_summary,
    run_configured_consistency,
)

testcase_path = os.environ.get("MQSMLA_PT_DIR", "mqsmla_testcase")
batch_test_mode = int(
    os.environ.get("MQSMLA_BATCH_TEST_MODE", 0)
)  # 0:路径下全量批跑，1:按表格中case批跑
excel_path = os.environ.get(
    "MQSMLA_EXCEL_PATH",
    os.path.join(os.path.dirname(__file__), "excel", "testcase.xlsx"),
)
result_path = os.environ.get("MQSMLA_RESULT_SAVE_PATH", "./mqsmla_result.xlsx")
device_id = int(os.environ.get("MQSMLA_DEVICE_ID", 0))

locals()["testcase_files"] = []
if os.path.isdir(testcase_path):
    if batch_test_mode == 1:
        df = pd.read_excel(excel_path)
        target_names = [
            str(name)
            for name in df["Testcase_Name"].dropna().tolist()
            if str(name) != "None"
        ]
        if not target_names:
            print(f"错误: 表格中没有有效的Testcase_Name: {excel_path}")
        else:
            print(f"从表格中读取到 {len(target_names)} 个目标用例名")
            pt_files = [f for f in os.listdir(testcase_path) if f.endswith(".pt")]
            for target_name in target_names:
                matched = [f for f in pt_files if target_name in f]
                if matched:
                    for f in matched:
                        filepath = os.path.join(testcase_path, f)
                        if filepath not in locals()["testcase_files"]:
                            locals()["testcase_files"].append(filepath)
                else:
                    print(f"警告: 用例名 '{target_name}' 未匹配到任何.pt文件")
            print(f"按表格筛选后共 {len(locals()['testcase_files'])} 个测试用例文件")
    else:
        pt_files = [f for f in os.listdir(testcase_path) if f.endswith(".pt")]
        if not pt_files:
            print(f"错误: 目录中没有找到.pt文件: {testcase_path}")
        else:
            print(f"找到 {len(pt_files)} 个测试用例文件")
            for pt_file in pt_files:
                filepath = os.path.join(testcase_path, pt_file)
                locals()["testcase_files"].append(filepath)
else:
    print(f"错误: 输出目录不存在: {testcase_path}")


def mqsmla_aclgraph(testcase_files):
    test_data = torch.load(testcase_files, map_location="cpu")
    consistency_config = resolve_consistency_config(
        test_data, os.environ.get("MQSMLA_BATCH_CONSISTENCY", "auto")
    )
    if consistency_config is not None:

        def consistency_executor(data):
            values = (
                mixed_quant_sparse_flash_mla_process.test_mqsmla_quant_process_graph(
                    data, device_id=device_id
                )
            )
            return RunResult(values[0], values[2])

        report = run_configured_consistency(
            test_data,
            consistency_config,
            consistency_executor,
            result_compare_method.check_result,
            lambda: torch_npu.npu.set_device(device_id),
        )
        summary = format_consistency_summary(report)
        print(summary, flush=True)
        fulfill_percent = consistency_fulfill_percent(report)
        result = "Pass" if report["pass"] else "Failed"
        utils.save_result(
            test_data["params"], result, fulfill_percent, Path(result_path)
        )
        if not report["relations"]:
            pytest.skip(f"batch consistency has no applicable relation: {report}")
        if not report["pass"]:
            pytest.fail(summary)
        return
    npu_error_msg = None
    try:
        npu_result, cpu_quant_result, npu_lse, cpu_lse = (
            mixed_quant_sparse_flash_mla_process.test_mqsmla_quant_process_graph(
                test_data, device_id=device_id
            )
        )

        attn_result, attn_pct = result_compare_method.check_result(
            cpu_quant_result, npu_result
        )
        lse_result, lse_pct = None, 0.0
        fail_info = []
        min_fulfill = attn_pct

        if attn_result != "Pass":
            fail_info.append(f"MAIN_FAILED:{attn_result}")

        if test_data["params"].get("return_softmax_lse"):
            print("return_softmax_lse is true!!!")
            lse_result, lse_pct = result_compare_method.check_result(cpu_lse, npu_lse)
            min_fulfill = min(min_fulfill, lse_pct)
            if lse_result != "Pass":
                fail_info.append(f"LSE_FAILED:{lse_result}")

        fulfill_percent = min_fulfill
        result = "; ".join(fail_info) if fail_info else "Pass"

    except Exception as e:
        npu_error_msg = str(e)
        print("NPU ERROR：", npu_error_msg)
        result = "NPU ERROR"
        fulfill_percent = 0

    utils.save_result(test_data["params"], result, fulfill_percent, Path(result_path))

    if result == "NPU ERROR":
        pytest.fail(
            f"用例执行失败:{test_data['Testcase_Name']} NPU ERROR: {npu_error_msg}"
        )
    elif result != "Pass":
        pytest.fail(
            f"用例精度失败:{test_data['Testcase_Name']} 精度:{fulfill_percent:.2f}%"
        )


testcase_ids = [
    os.path.splitext(os.path.basename(f))[0] for f in locals()["testcase_files"]
]


@pytest.mark.graph
@pytest.mark.parametrize("testcase_files", locals()["testcase_files"], ids=testcase_ids)
def test_mixed_quant_sparse_flash_mla(testcase_files):
    # 线程池
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = executor.submit(mqsmla_aclgraph, testcase_files)
        # 等待并获取结果
        for future in concurrent.futures.as_completed([futures]):
            try:
                future.result()
            except Exception:
                pytest.fail("当前用例线程执行失败")
