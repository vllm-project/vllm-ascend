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
import check_valid_param
import mixed_quant_sparse_flash_mla_golden
import pytest
import random
import pandas as pd
from pathlib import Path
import numpy as np
import math
import os
import sys
import utils
import argparse
import concurrent.futures
import traceback

SMLA_PYTEST_PATH = Path(__file__).resolve().parents[1]
if str(SMLA_PYTEST_PATH) not in sys.path:
    sys.path.append(str(SMLA_PYTEST_PATH))

from batch_consistency.config import (  # noqa: E402
    prepare_consistency_params,
    resolve_consistency_config,
)

# 读取表格（支持通过环境变量传入）
save_path = os.environ.get("MQSMLA_PT_DIR", "mqsmla_testcase")
excel_file = os.environ.get("MQSMLA_EXCEL", "./excel/example.xlsx")
sheet_name = os.environ.get("MQSMLA_SHEET", "decode")
ENABLED_PARAMS = utils.load_excel_test_cases(excel_file, sheet_name)

param_combinations = []
for _, params in enumerate(ENABLED_PARAMS):
    normalized_params = {}
    for key, value in params.items():
        if value == "torch.bfloat16":
            value = torch.bfloat16
        elif value == "torch.float8_e4m3fn":
            value = torch.float8_e4m3fn
        elif value in ("torch.uint8", "uint8"):
            value = torch.uint8
        elif value == "FALSE":
            value = False
        elif value == "TRUE":
            value = True
        normalized_params[key] = value
    normalized_params["topk_value_mode"] = params.get("topk_value_mode", 1)
    normalized_params["return_softmax_lse"] = params.get("return_softmax_lse", False)
    normalized_params["ori_kv_topk_mode"] = params.get("ori_kv_topk_mode")
    normalized_params["cmp_kv_topk_mode"] = params.get("cmp_kv_topk_mode")
    normalized_params["ori_sparse_indices_mode"] = params.get(
        "ori_sparse_indices_mode", "full"
    )
    normalized_params["cmp_sparse_indices_mode"] = params.get(
        "cmp_sparse_indices_mode", "full"
    )
    normalized_params["ori_topk_length"] = utils.parse_list_param(
        params.get("ori_topk_length")
    )
    normalized_params["cmp_topk_length"] = utils.parse_list_param(
        params.get("cmp_topk_length")
    )
    int_keys = [
        "B",
        "S1",
        "S2",
        "N1",
        "N2",
        "D",
        "K",
        "K1",
        "block_num1",
        "block_num2",
        "block_size1",
        "block_size2",
        "cmp_ratio",
        "ori_mask_mode",
        "cmp_mask_mode",
        "ori_win_left",
        "ori_win_right",
        "quant_mode",
        "tile_size",
        "rope_head_dim",
        "topk_value_mode",
        "batch_consistency_seed",
        "batch_consistency_mode_batch",
    ]
    for ik in int_keys:
        v = normalized_params.get(ik)
        if v is not None and not isinstance(v, bool):
            normalized_params[ik] = int(v)
    for key in [
        "seqused_q",
        "cu_seqlens_q",
        "seqused_ori_kv",
        "seqused_cmp_kv",
        "cu_seqlens_ori_kv",
        "cu_seqlens_cmp_kv",
        "cmp_residual_kv",
    ]:
        normalized_params[key] = utils.parse_list_param(params.get(key))
    for key in ["q_datarange", "ori_kv_datarange", "cmp_kv_datarange"]:
        normalized_params[key] = utils.parse_datarange_param(params.get(key))
    for key in [
        "batch_consistency_order",
        "batch_consistency_batch_split",
        "batch_consistency_token_split",
        "batch_consistency_shape_change",
    ]:
        normalized_params[key] = utils.parse_list_param(params.get(key))
    template_run_mode = normalized_params["template_run_mode"]
    if isinstance(template_run_mode, list):
        template_run_mode = template_run_mode[0]

    param_names = [
        "Testcase_Name",
        "layout_q",
        "layout_kv",
        "q_type",
        "ori_kv_type",
        "cmp_kv_type",
        "B",
        "S1",
        "S2",
        "N1",
        "N2",
        "D",
        "K",
        "K1",
        "block_num1",
        "block_num2",
        "block_size1",
        "block_size2",
        "softmax_scale",
        "cmp_ratio",
        "ori_mask_mode",
        "cmp_mask_mode",
        "ori_win_left",
        "ori_win_right",
        "quant_mode",
        "tile_size",
        "rope_head_dim",
        "template_run_mode",
        "actlen_mode",
        "S1EQS2",
        "topk_value_mode",
        "return_softmax_lse",
        "ori_kv_topk_mode",
        "cmp_kv_topk_mode",
        "ori_sparse_indices_mode",
        "cmp_sparse_indices_mode",
        "ori_topk_length",
        "cmp_topk_length",
        "seqused_q",
        "cu_seqlens_q",
        "seqused_ori_kv",
        "seqused_cmp_kv",
        "cu_seqlens_ori_kv",
        "cu_seqlens_cmp_kv",
        "cmp_residual_kv",
        "q_datarange",
        "ori_kv_datarange",
        "cmp_kv_datarange",
        "batch_consistency",
        "batch_consistency_seed",
        "batch_consistency_order",
        "batch_consistency_batch_split",
        "batch_consistency_mode_batch",
        "batch_consistency_token_split",
        "batch_consistency_shape_change",
    ]

    param_values = [
        [normalized_params["Testcase_Name"]],
        [normalized_params["layout_q"]],
        [normalized_params["layout_kv"]],
        [normalized_params["q_type"]],
        [normalized_params["ori_kv_type"]],
        [normalized_params["cmp_kv_type"]],
        [normalized_params["B"]],
        [normalized_params["S1"]],
        [normalized_params["S2"]],
        [normalized_params["N1"]],
        [normalized_params["N2"]],
        [normalized_params["D"]],
        [normalized_params["K"]],
        [normalized_params.get("K1")],
        [normalized_params.get("block_num1")],
        [normalized_params.get("block_num2")],
        [normalized_params["block_size1"]],
        [normalized_params["block_size2"]],
        [normalized_params["softmax_scale"]],
        [normalized_params["cmp_ratio"]],
        [normalized_params["ori_mask_mode"]],
        [normalized_params["cmp_mask_mode"]],
        [normalized_params["ori_win_left"]],
        [normalized_params["ori_win_right"]],
        [normalized_params["quant_mode"]],
        [normalized_params["tile_size"]],
        [normalized_params["rope_head_dim"]],
        [normalized_params["template_run_mode"]],
        [normalized_params["actlen_mode"]],
        [normalized_params["S1EQS2"]],
        [normalized_params["topk_value_mode"]],
        [normalized_params["return_softmax_lse"]],
        [normalized_params["ori_kv_topk_mode"]],
        [normalized_params["cmp_kv_topk_mode"]],
        [normalized_params["ori_sparse_indices_mode"]],
        [normalized_params["cmp_sparse_indices_mode"]],
        [normalized_params["ori_topk_length"]],
        [normalized_params["cmp_topk_length"]],
        [normalized_params["seqused_q"]],
        [normalized_params["cu_seqlens_q"]],
        [normalized_params["seqused_ori_kv"]],
        [normalized_params["seqused_cmp_kv"]],
        [normalized_params["cu_seqlens_ori_kv"]],
        [normalized_params["cu_seqlens_cmp_kv"]],
        [normalized_params["cmp_residual_kv"]],
        [normalized_params["q_datarange"]],
        [normalized_params["ori_kv_datarange"]],
        [normalized_params["cmp_kv_datarange"]],
        [normalized_params.get("batch_consistency")],
        [normalized_params.get("batch_consistency_seed")],
        [normalized_params.get("batch_consistency_order")],
        [normalized_params.get("batch_consistency_batch_split")],
        [normalized_params.get("batch_consistency_mode_batch")],
        [normalized_params.get("batch_consistency_token_split")],
        [normalized_params.get("batch_consistency_shape_change")],
    ]

    # 生成所有的组合，并转换为字典列表
    for combo in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combo))
        param_combinations.append(param_dict)
    print(param_combinations)

case_id = 0
failed_cases = []
failed_record_path = Path("pt_save_failed.xlsx")


def record_failed_case(param_combinations, error_msg):
    row = {
        "case_id": case_id,
        "template_run_mode": param_combinations.get("template_run_mode"),
        "layout_q": param_combinations.get("layout_q"),
        "layout_kv": param_combinations.get("layout_kv"),
        "B": param_combinations.get("B"),
        "S1": param_combinations.get("S1"),
        "S2": param_combinations.get("S2"),
        "N1": param_combinations.get("N1"),
        "K": param_combinations.get("K"),
        "K1": param_combinations.get("K1"),
        "quant_mode": param_combinations.get("quant_mode"),
        "cmp_ratio": param_combinations.get("cmp_ratio"),
        "ori_mask_mode": param_combinations.get("ori_mask_mode"),
        "cmp_mask_mode": param_combinations.get("cmp_mask_mode"),
        "seqused_q": str(param_combinations.get("seqused_q")),
        "seqused_ori_kv": str(param_combinations.get("seqused_ori_kv")),
        "cmp_residual_kv": str(param_combinations.get("cmp_residual_kv")),
        "ori_kv_topk_mode": param_combinations.get("ori_kv_topk_mode"),
        "cmp_kv_topk_mode": param_combinations.get("cmp_kv_topk_mode"),
        "error_msg": str(error_msg),
    }
    failed_cases.append(row)
    df = pd.DataFrame(failed_cases)
    df.to_excel(failed_record_path, index=False)


def mqsmla(param_combinations):
    global case_id
    try:
        params = utils.fill_none_params(param_combinations)
        batch_consistency_policy = os.environ.get("MQSMLA_BATCH_CONSISTENCY", "auto")
        prepare_consistency_params(params, batch_consistency_policy)

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
            Testcase_Name = f"MQSMLA_{prefix_part}{params['template_run_mode']}_{ops_mode}_{params['layout_q']}_{q_type_str}_{params['layout_kv']}_{kv_type_str}_{params['B']}_{params['N1']}_{params['N2']}_{params['S1']}_{params['S2']}_{params['D']}_{params['K']}_{params['rope_head_dim']}_{case_id:06d}"
            params["Testcase_Name"] = Testcase_Name
        print("input_params:", params)

        # 输入参数的合法性校验
        try:
            check_valid_param.check_valid_param(params)
        except ValueError as e:
            pytest.skip(f"输入参数校验失败:{e}")

        # 生成测试数据
        input_data = mixed_quant_sparse_flash_mla_golden.gen_data(params)
        config = resolve_consistency_config(
            input_data,
            batch_consistency_policy,
            persist=True,
        )
        if config is not None:
            print("batch consistency config:", config)
        mixed_quant_sparse_flash_mla_golden.save_test_case(input_data, save_path)
    except Exception as e:
        record_failed_case(param_combinations, e)
        pytest.fail(
            f"[FAILED CASE RECORDED] case_id={case_id}\n{traceback.format_exc()}"
        )
    finally:
        case_id += 1


@pytest.mark.ci
@pytest.mark.parametrize("param_combinations", param_combinations)
def test_mixed_quant_sparse_flash_mla(param_combinations):  # 初始化参数和tensor
    # 线程池
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = executor.submit(mqsmla, param_combinations)
        # 等待并获取结果
        for future in concurrent.futures.as_completed([futures]):
            try:
                result = future.result()
            except Exception as e:
                pytest.fail("当前用例线程执行失败")
