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

import itertools
import torch
import result_compare_method
import check_valid_param
import mixed_quant_sparse_flash_mla_golden
from batch import mixed_quant_sparse_flash_mla_process
import pytest
import pandas as pd
from pathlib import Path
import numpy as np
import os
import multiprocessing as mp
import concurrent.futures
import utils
import sys
import torch_npu

from mixed_quant_sparse_flash_mla_paramset import ENABLED_PARAMS

SMLA_PYTEST_PATH = Path(__file__).resolve().parent
if str(SMLA_PYTEST_PATH) not in sys.path:
    sys.path.append(str(SMLA_PYTEST_PATH))

from batch_consistency.config import (  # noqa: E402
    prepare_consistency_params,
    resolve_consistency_config,
)
from batch_consistency.model import RunResult  # noqa: E402
from batch_consistency.pytest_support import (  # noqa: E402
    consistency_fulfill_percent,
    format_consistency_summary,
    run_configured_consistency,
)

pt_save_path = "mqsmla_testcase"
device_id = 0
save_pt = os.environ.get("SAVE_PT", "0") == "1"
result_path = Path(os.environ.get("MQSMLA_RESULT_SAVE_PATH", "result.xlsx"))

param_combinations = []
for params in ENABLED_PARAMS:
    param_values = {
        "Testcase_Name": params.get("Testcase_Name", [None]),
        "layout_q": params.get("layout_q"),
        "layout_kv": params.get("layout_kv"),
        "q_type": params.get("q_type"),
        "ori_kv_type": params.get("ori_kv_type"),
        "cmp_kv_type": params.get("cmp_kv_type", [None]),
        "B": params.get("B"),
        "S1": params.get("S1"),
        "S2": params.get("S2", [None]),
        "N1": params.get("N1"),
        "N2": params.get("N2"),
        "D": params.get("D"),
        "K1": params.get("K1", [None]),
        "K": params.get("K", [None]),
        "block_num1": params.get("block_num1", [None]),
        "block_num2": params.get("block_num2", [None]),
        "block_size1": params.get("block_size1"),
        "block_size2": params.get("block_size2", [None]),
        "seqused_q": params.get("seqused_q", [None]),
        "cu_seqlens_q": params.get("cu_seqlens_q", [None]),
        "seqused_ori_kv": params.get("seqused_ori_kv", [None]),
        "seqused_cmp_kv": params.get("seqused_cmp_kv", [None]),
        "cu_seqlens_ori_kv": params.get("cu_seqlens_ori_kv", [None]),
        "cu_seqlens_cmp_kv": params.get("cu_seqlens_cmp_kv", [None]),
        "cmp_residual_kv": params.get("cmp_residual_kv", [None]),
        "softmax_scale": params.get("softmax_scale"),
        "cmp_ratio": params.get("cmp_ratio", [None]),
        "ori_mask_mode": params.get("ori_mask_mode"),
        "cmp_mask_mode": params.get("cmp_mask_mode", [None]),
        "ori_win_left": params.get("ori_win_left"),
        "ori_win_right": params.get("ori_win_right"),
        "quant_mode": params.get("quant_mode"),
        "tile_size": params.get("tile_size"),
        "rope_head_dim": params.get("rope_head_dim"),
        "template_run_mode": params.get("template_run_mode"),
        "actlen_mode": params.get("actlen_mode"),
        "S1EQS2": params.get("S1EQS2", [False]),
        "return_softmax_lse": params.get("return_softmax_lse", [False]),
        "ori_kv_topk_mode": params.get("ori_kv_topk_mode", [None]),
        "cmp_kv_topk_mode": params.get("cmp_kv_topk_mode", [None]),
        "ori_sparse_indices_mode": params.get("ori_sparse_indices_mode", ["full"]),
        "cmp_sparse_indices_mode": params.get("cmp_sparse_indices_mode", ["full"]),
        "ori_topk_length": params.get("ori_topk_length", [None]),
        "cmp_topk_length": params.get("cmp_topk_length", [None]),
        "batch_consistency": params.get("batch_consistency", [None]),
        "batch_consistency_seed": params.get("batch_consistency_seed", [None]),
        "batch_consistency_order": params.get("batch_consistency_order", [None]),
        "batch_consistency_batch_split": params.get(
            "batch_consistency_batch_split", [None]
        ),
        "batch_consistency_mode_batch": params.get(
            "batch_consistency_mode_batch", [None]
        ),
        "batch_consistency_token_split": params.get(
            "batch_consistency_token_split", [None]
        ),
        "batch_consistency_shape_change": params.get(
            "batch_consistency_shape_change", [None]
        ),
    }

    param_names = list(param_values.keys())
    values_lists = [param_values[name] for name in param_names]

    for combo in itertools.product(*values_lists):
        combination = dict(zip(param_names, combo))
        param_combinations.append(combination)
case_id = 0


def mqsmla(param_combinations):
    global case_id

    # 填充None参数的默认值
    params = utils.fill_none_params(param_combinations)
    batch_consistency_policy = os.environ.get("MQSMLA_BATCH_CONSISTENCY", "auto")
    prepare_consistency_params(params, batch_consistency_policy)

    # 生成测试用例名称
    Testcase_Name = params["Testcase_Name"]
    if Testcase_Name is None:
        ops_mode = "prefill" if params["S1"] > 4 else "decode"
        q_type_str = "BF16" if params["q_type"] == torch.bfloat16 else "FP16"
        kv_type_str = (
            "FP8_AS_UINT8" if params["ori_kv_type"] == torch.uint8 else "FP8_E4M3FN"
        )
        prefix_part = (
            f"{param_combinations['tc_prefix']}_"
            if param_combinations.get("tc_prefix", "")
            else ""
        )
        Testcase_Name = f"mixedQuantSparseFlashMla_{params['template_run_mode']}_{ops_mode}_{params['layout_q']}_{q_type_str}_{params['layout_kv']}_{kv_type_str}_{params['B']}_{params['N1']}_{params['N2']}_{params['S1']}_{params['S2']}_{params['D']}_{params['K']}_{params['rope_head_dim']}_{case_id:06d}"
        params["Testcase_Name"] = Testcase_Name

    # 输入参数的合法性校验
    try:
        check_valid_param.check_valid_param(params)
    except ValueError as e:
        pytest.skip(f"输入参数校验失败:{e}")

    # 生成测试数据及golden
    test_data = mixed_quant_sparse_flash_mla_golden.gen_data(params)
    consistency_config = resolve_consistency_config(
        test_data,
        batch_consistency_policy,
        persist=save_pt,
    )
    if save_pt:
        mixed_quant_sparse_flash_mla_golden.save_test_case(test_data, pt_save_path)

    if consistency_config is not None:
        if params["layout_q"] == "TND" and test_data.get("cpu_lse") is not None:
            test_data["cpu_lse"] = test_data["cpu_lse"].transpose(0, 1).contiguous()

        def consistency_executor(data):
            values = mixed_quant_sparse_flash_mla_process.test_mqsmla_quant_process_ci(
                data, device_id=device_id
            )
            return RunResult(values[0], values[3])

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
        case_id += 1
        utils.save_result(test_data["params"], result, fulfill_percent, result_path)
        if not report["relations"]:
            pytest.skip(f"batch consistency has no applicable relation: {report}")
        if not report["pass"]:
            pytest.fail(summary)
        return

    # 获得cpu结果(真值)和算子结果（测试值）
    npu_error_msg = None
    try:
        npu_result, cpu_quant_result, cpu_lse, npu_lse = (
            mixed_quant_sparse_flash_mla_process.test_mqsmla_quant_process_ci(
                test_data, device_id=device_id
            )
        )
        attn_result, attn_percent = result_compare_method.check_result(
            cpu_quant_result, npu_result
        )
        lse_result, lse_percent = None, None
        fail_info = []
        min_fulfill = attn_percent

        if test_data["params"].get("return_softmax_lse"):
            print("return_softmax_lse is true!!!")
            lse_result, lse_percent = result_compare_method.check_result(
                cpu_lse, npu_lse
            )
            min_fulfill = min(min_fulfill, lse_percent)

        if attn_result != "Pass":
            fail_info.append(f"MAIN_FAILED:{attn_result}")
        if lse_result is not None and lse_result != "Pass":
            fail_info.append(f"LSE_FAILED:{lse_result}")

        if fail_info:
            result = "; ".join(fail_info)
            fulfill_percent = min_fulfill
        else:
            result = "Pass"
            fulfill_percent = min_fulfill
    except Exception as e:
        npu_error_msg = str(e)
        print("NPU ERROR：", npu_error_msg)
        result = "NPU ERROR"
        fulfill_percent = 0

    case_id += 1

    utils.save_result(test_data["params"], result, fulfill_percent, result_path)

    if result == "NPU ERROR":
        pytest.fail(
            f"用例执行失败:{test_data['Testcase_Name']} NPU ERROR: {npu_error_msg}"
        )
    elif result != "Pass":
        pytest.fail(
            f"用例精度失败:{test_data['Testcase_Name']} 精度:{fulfill_percent:.2f}%"
        )


def _gen_testcase_id(params, idx):
    name = params.get("Testcase_Name")
    if name is not None:
        return name
    ops_mode = "prefill" if params["S1"] > 4 else "decode"
    q_type_str = "BF16" if params["q_type"] == torch.bfloat16 else "FP16"
    kv_type_str = (
        "FP8_AS_UINT8" if params["ori_kv_type"] == torch.uint8 else "FP8_E4M3FN"
    )
    return f"{params['template_run_mode']}_{ops_mode}_{params['layout_q']}_{q_type_str}_{params['layout_kv']}_{kv_type_str}_B{params['B']}_S1{params['S1']}_S2{params['S2']}_D{params['D']}_K{params['K']}_{idx:06d}"


testcase_ids = [_gen_testcase_id(p, i) for i, p in enumerate(param_combinations)]


@pytest.mark.ci
@pytest.mark.parametrize("param_combinations", param_combinations, ids=testcase_ids)
def test_mixed_quant_sparse_flash_mla(param_combinations):
    # 线程池
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = executor.submit(mqsmla, param_combinations)
        # 等待并获取结果
        for future in concurrent.futures.as_completed([futures]):
            try:
                result = future.result()
            except Exception as e:
                pytest.fail("当前用例线程执行失败")
