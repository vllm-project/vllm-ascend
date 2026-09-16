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

import pytest
import random
import pandas as pd
import numpy as np
import os
import math


def normalize_topk_mode(value):
    """Accept the case-insensitive ``fullK`` spelling used by Excel cases."""
    if isinstance(value, str) and value.casefold() == "fullk":
        return "fullK"
    return value


def infer_template_run_mode(
    K,
    cmp_ratio,
    ori_kv_type,
    cmp_kv_type,
    cmp_mask_mode,
    block_size1,
    block_size2,
    block_num2,
):
    """
    根据K和cmp_ratio推断template_run_mode并设置默认值

    参数:
        K: sparse block count (可为None)
        cmp_ratio: 压缩率 (可为None)
        ori_kv_type: ori_kv数据类型
        cmp_kv_type: cmp_kv数据类型 (可为None)
        cmp_mask_mode: cmp mask模式 (可为None)
        block_size1: ori block大小
        block_size2: cmp block大小 (可为None)
        block_num2: cmp block总数 (可为None)

    返回:
        dict: 包含template_run_mode及相关参数的字典
    """
    template_run_mode = "SWA"

    if K is None or K == ["None"] or K == 0:
        if cmp_ratio is None or cmp_ratio == ["None"]:
            template_run_mode = "SWA"
            cmp_ratio = 1
            cmp_kv_type = ori_kv_type
            cmp_mask_mode = 0
            K = 0
            block_size2 = block_size1 if block_size2 is None else block_size2
            block_num2 = 0 if block_num2 is None else block_num2
        else:
            template_run_mode = "HCA"
            cmp_ratio = int(cmp_ratio)
            cmp_kv_type = ori_kv_type if cmp_kv_type is None else cmp_kv_type
            K = 0
            block_size2 = int(block_size2) if block_size2 is not None else block_size1
            cmp_mask_mode = int(cmp_mask_mode) if cmp_mask_mode is not None else 3
            block_num2 = block_num2  # 后续计算
    else:
        template_run_mode = "CSA"
        K = int(K)
        cmp_ratio = int(cmp_ratio)
        cmp_kv_type = ori_kv_type if cmp_kv_type is None else cmp_kv_type
        block_size2 = int(block_size2) if block_size2 is not None else block_size1
        cmp_mask_mode = int(cmp_mask_mode) if cmp_mask_mode is not None else 3
        block_num2 = block_num2  # 后续计算

    return {
        "template_run_mode": template_run_mode,
        "K": K,
        "cmp_ratio": cmp_ratio,
        "cmp_kv_type": cmp_kv_type,
        "cmp_mask_mode": cmp_mask_mode,
        "block_size2": block_size2,
        "block_num2": block_num2,
    }


def gen_seqused_cmp_kv(seqused_ori_kv, cmp_ratio):
    """
    生成seqused_cmp_kv

    参数:
        seqused_ori_kv: ori_kv真实长度列表
        cmp_ratio: 压缩率

    返回:
        list: seqused_cmp_kv列表
    """
    return [math.floor(s / cmp_ratio) for s in seqused_ori_kv]


def parse_list_param(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return [value]
    if isinstance(value, float) and not pd.isna(value):
        return [int(value)]
    if isinstance(value, str):
        s = value.strip()
        if s.lower() == "none" or s == "":
            return None
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        try:
            return [
                (int(x.strip()) if "." not in s else float(x.strip()))
                for x in s.split(",")
                if x.strip() != ""
            ]
        except ValueError:
            return None
    return None


def parse_datarange_param(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s.lower() == "none" or s == "":
            return None
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        try:
            return [float(x.strip()) for x in s.split(",") if x.strip() != ""]
        except ValueError:
            return None
    return None


def gen_cu_seqlens_from_seqused(seqused):
    """
    从seqused生成cu_seqlens累加和

    参数:
        seqused: 真实长度列表

    返回:
        list: cu_seqlens累加和列表（长度B+1）
    """
    cu_seqlens = [0]
    for s in seqused:
        cu_seqlens.append(cu_seqlens[-1] + s)
    return cu_seqlens


def gen_cu_seqlens_cmp_kv(seqused_cmp_kv):
    """Generate CMP prefixes from CMP lengths without consulting ORI lengths."""
    return gen_cu_seqlens_from_seqused(seqused_cmp_kv)


def calc_block_num(seqused_ori_kv, seqused_cmp_kv, block_size1, block_size2):
    """
    计算block_num1和block_num2

    参数:
        seqused_ori_kv: ori_kv真实长度列表
        seqused_cmp_kv: cmp_kv真实长度列表
        block_size1: ori block大小
        block_size2: cmp block大小
    返回:
        tuple: (block_num1, block_num2)
    """
    ori_block_num_sum = 0
    cmp_block_num_sum = 0

    for cur_ori_act_kv in seqused_ori_kv:
        cur_ori_kv_block_num = math.ceil(cur_ori_act_kv / block_size1)
        ori_block_num_sum += cur_ori_kv_block_num

    for cur_cmp_act_kv in seqused_cmp_kv:
        cur_cmp_kv_block_num = math.ceil(cur_cmp_act_kv / block_size2)
        cmp_block_num_sum += cur_cmp_kv_block_num

    return ori_block_num_sum, cmp_block_num_sum


def fill_none_params(params_dict):
    """
    填充None参数的默认值（完整封装）

    参数:
        params_dict: 参数字典

    返回:
        dict: 填充后的完整参数字典
    """
    layout_q = params_dict["layout_q"]
    B = params_dict["B"]
    S1 = params_dict["S1"]
    S2 = params_dict["S2"]
    ori_kv_type = params_dict["ori_kv_type"]
    block_size1 = params_dict["block_size1"]
    actlen_mode = params_dict.get("actlen_mode", "full")
    S1EQS2 = params_dict.get("S1EQS2", False)

    # 1. 获取template_run_mode（优先从params_dict获取）
    template_run_mode = params_dict.get("template_run_mode")

    if template_run_mode is None:
        # 如果未提供template_run_mode，则推断（向后兼容）
        infer_result = infer_template_run_mode(
            params_dict.get("K"),
            params_dict.get("cmp_ratio"),
            ori_kv_type,
            params_dict.get("cmp_kv_type"),
            params_dict.get("cmp_mask_mode"),
            block_size1,
            params_dict.get("block_size2"),
            params_dict.get("block_num2"),
        )
        K = infer_result["K"]
        cmp_ratio = infer_result["cmp_ratio"]
        cmp_kv_type = infer_result["cmp_kv_type"]
        cmp_mask_mode = infer_result["cmp_mask_mode"]
        block_size2 = infer_result["block_size2"]
        block_num2 = infer_result["block_num2"]
        template_run_mode = infer_result["template_run_mode"]
    else:
        # 如果提供了template_run_mode，直接使用params_dict中的参数
        K = params_dict.get("K")
        cmp_ratio = params_dict.get("cmp_ratio")
        cmp_kv_type = params_dict.get("cmp_kv_type")
        cmp_mask_mode = params_dict.get("cmp_mask_mode")
        block_size2 = params_dict.get("block_size2")
        block_num2 = params_dict.get("block_num2")

        # 处理None参数的默认值（保持一致性）
        if cmp_ratio is None:
            cmp_ratio = 1
        if cmp_kv_type is None:
            cmp_kv_type = ori_kv_type
        if cmp_mask_mode is None:
            cmp_mask_mode = 3 if cmp_ratio != 1 else 0
        if block_size2 is None:
            block_size2 = block_size1
        if K is None:
            K = 0

    # 2. 获取需要填充的参数
    seqused_q = params_dict.get("seqused_q")
    cu_seqlens_q = params_dict.get("cu_seqlens_q")
    seqused_ori_kv = params_dict.get("seqused_ori_kv")
    cu_seqlens_ori_kv = params_dict.get("cu_seqlens_ori_kv")
    seqused_cmp_kv = params_dict.get("seqused_cmp_kv")
    cu_seqlens_cmp_kv = params_dict.get("cu_seqlens_cmp_kv")
    cmp_residual_kv = params_dict.get("cmp_residual_kv")

    if seqused_q is not None and len(seqused_q) == 1:
        seqused_q = seqused_q * B
    if seqused_ori_kv is not None and len(seqused_ori_kv) == 1:
        seqused_ori_kv = seqused_ori_kv * B
    if seqused_cmp_kv is not None and len(seqused_cmp_kv) == 1:
        seqused_cmp_kv = seqused_cmp_kv * B
    if cmp_residual_kv is not None and len(cmp_residual_kv) == 1:
        cmp_residual_kv = cmp_residual_kv * B

    need_fill_q_ori = any(
        v is None for v in [seqused_q, cu_seqlens_q, seqused_ori_kv, cu_seqlens_ori_kv]
    )

    if need_fill_q_ori:
        # 从cu推导seqused
        if seqused_q is None and cu_seqlens_q is not None:
            seqused_q = [
                cu_seqlens_q[i + 1] - cu_seqlens_q[i]
                for i in range(len(cu_seqlens_q) - 1)
            ]
        if seqused_ori_kv is None and cu_seqlens_ori_kv is not None:
            seqused_ori_kv = [
                cu_seqlens_ori_kv[i + 1] - cu_seqlens_ori_kv[i]
                for i in range(len(cu_seqlens_ori_kv) - 1)
            ]

        # S1EQS2联动: seqused_q与seqused_ori_kv
        if S1EQS2:
            if seqused_q is None and seqused_ori_kv is not None:
                seqused_q = list(seqused_ori_kv)
            elif seqused_ori_kv is None and seqused_q is not None:
                seqused_ori_kv = list(seqused_q)

        # 独立生成seqused_q
        if seqused_q is None:
            if layout_q == "TND" and actlen_mode == "random":
                seqused_q = [S1 - random.randint(0, S1 - 1) for _ in range(B)]
            else:
                seqused_q = [S1 for _ in range(B)]

        # 再次S1EQS2联动 (seqused_q现已生成)
        if S1EQS2 and seqused_ori_kv is None:
            seqused_ori_kv = list(seqused_q)

        # 独立生成seqused_ori_kv
        if seqused_ori_kv is None:
            kv_full_len = S2 if S2 is not None else S1
            if actlen_mode == "random":
                seqused_ori_kv = [
                    kv_full_len - random.randint(0, math.ceil(kv_full_len / 2) - 1)
                    for _ in range(B)
                ]
            else:
                seqused_ori_kv = [kv_full_len for _ in range(B)]

    # 填充cu_seqlens_q和cu_seqlens_ori_kv (S1EQS2联动优先)
    if cu_seqlens_q is None:
        if S1EQS2 and cu_seqlens_ori_kv is not None:
            cu_seqlens_q = list(cu_seqlens_ori_kv)
        else:
            cu_seqlens_q = gen_cu_seqlens_from_seqused(seqused_q)

    if cu_seqlens_ori_kv is None:
        if S1EQS2 and cu_seqlens_q is not None:
            cu_seqlens_ori_kv = list(cu_seqlens_q)
        else:
            cu_seqlens_ori_kv = gen_cu_seqlens_from_seqused(seqused_ori_kv)

    for i in range(B):
        slot_len = cu_seqlens_q[i + 1] - cu_seqlens_q[i]
        if slot_len < seqused_q[i]:
            raise ValueError(
                f"cu_seqlens_q slot ({slot_len}) at batch {i} must >= seqused_q ({seqused_q[i]})"
            )

    for i in range(B):
        slot_len = cu_seqlens_ori_kv[i + 1] - cu_seqlens_ori_kv[i]
        if slot_len < seqused_ori_kv[i]:
            raise ValueError(
                f"cu_seqlens_ori_kv slot ({slot_len}) at batch {i} must >= seqused_ori_kv ({seqused_ori_kv[i]})"
            )

    # 3. 填充cmp参数 (SWA/ORI_SPARSE场景为None, 其他场景自动生成)
    if template_run_mode in ("SWA", "ORI_SPARSE"):
        seqused_cmp_kv = None
        cu_seqlens_cmp_kv = None
        cmp_residual_kv = None
    else:
        if seqused_cmp_kv is None:
            seqused_cmp_kv = gen_seqused_cmp_kv(seqused_ori_kv, cmp_ratio)
        if cu_seqlens_cmp_kv is None:
            cu_seqlens_cmp_kv = gen_cu_seqlens_cmp_kv(seqused_cmp_kv)
        if cmp_residual_kv is None:
            cmp_residual_kv = [0] * B
        for i in range(B):
            slot_len = cu_seqlens_cmp_kv[i + 1] - cu_seqlens_cmp_kv[i]
            if slot_len < seqused_cmp_kv[i]:
                raise ValueError(
                    f"cu_seqlens_cmp_kv slot ({slot_len}) at batch {i} must >= seqused_cmp_kv ({seqused_cmp_kv[i]})"
                )

    # 6. 计算block_num
    block_num1 = params_dict.get("block_num1")
    block_num2 = params_dict.get("block_num2")

    if block_num1 is None:
        ori_block_num, _ = calc_block_num(
            seqused_ori_kv,
            seqused_cmp_kv if seqused_cmp_kv is not None else [0] * B,
            block_size1,
            block_size2,
        )
        block_num1 = ori_block_num
    if block_num2 is None:
        if template_run_mode in ("SWA", "ORI_SPARSE"):
            block_num2 = 0
        else:
            _, cmp_block_num = calc_block_num(
                seqused_ori_kv, seqused_cmp_kv, block_size1, block_size2
            )
            block_num2 = cmp_block_num

    # 7. 计算T1
    T1 = cu_seqlens_q[-1] if layout_q == "TND" else None

    ori_mask_mode = params_dict.get("ori_mask_mode")
    ori_kv_topk_mode = normalize_topk_mode(params_dict.get("ori_kv_topk_mode"))
    if ori_kv_topk_mode is None:
        ori_kv_topk_mode = "fullK" if ori_mask_mode == 0 else "no"
    cmp_kv_topk_mode = normalize_topk_mode(params_dict.get("cmp_kv_topk_mode"))
    if cmp_kv_topk_mode is None:
        cmp_kv_topk_mode = "fullK" if cmp_mask_mode == 0 else "no"

    # 构建完整参数字典
    filled_params = {
        "Testcase_Name": params_dict.get("Testcase_Name"),
        "layout_q": layout_q,
        "layout_kv": params_dict["layout_kv"],
        "q_type": params_dict["q_type"],
        "ori_kv_type": ori_kv_type,
        "cmp_kv_type": cmp_kv_type,
        "B": B,
        "S1": S1,
        "S2": S2,
        "T1": T1,
        "N1": params_dict["N1"],
        "N2": params_dict["N2"],
        "D": params_dict["D"],
        "K1": params_dict.get("K1"),
        "K": K,
        "block_num1": block_num1,
        "block_num2": block_num2,
        "block_size1": block_size1,
        "block_size2": block_size2,
        "seqused_q": seqused_q,
        "cu_seqlens_q": cu_seqlens_q,
        "seqused_ori_kv": seqused_ori_kv,
        "seqused_cmp_kv": seqused_cmp_kv,
        "cu_seqlens_ori_kv": cu_seqlens_ori_kv,
        "cu_seqlens_cmp_kv": cu_seqlens_cmp_kv,
        "cmp_residual_kv": cmp_residual_kv,
        "softmax_scale": params_dict["softmax_scale"],
        "cmp_ratio": cmp_ratio,
        "ori_mask_mode": params_dict["ori_mask_mode"],
        "cmp_mask_mode": cmp_mask_mode,
        "ori_win_left": params_dict["ori_win_left"],
        "ori_win_right": params_dict["ori_win_right"],
        "quant_mode": params_dict["quant_mode"],
        "tile_size": params_dict["tile_size"],
        "rope_head_dim": params_dict["rope_head_dim"],
        "template_run_mode": template_run_mode,
        "actlen_mode": actlen_mode,
        "S1EQS2": S1EQS2,
        "topk_value_mode": params_dict.get("topk_value_mode", 1),
        "return_softmax_lse": params_dict.get("return_softmax_lse", False),
        "q_datarange": params_dict.get("q_datarange"),
        "ori_kv_datarange": params_dict.get("ori_kv_datarange"),
        "cmp_kv_datarange": params_dict.get("cmp_kv_datarange"),
        "ori_kv_topk_mode": ori_kv_topk_mode,
        "cmp_kv_topk_mode": cmp_kv_topk_mode,
        "ori_sparse_indices_mode": params_dict.get("ori_sparse_indices_mode", "full"),
        "cmp_sparse_indices_mode": params_dict.get("cmp_sparse_indices_mode", "full"),
        "ori_topk_length": params_dict.get("ori_topk_length", None),
        "cmp_topk_length": params_dict.get("cmp_topk_length", None),
        "batch_consistency": params_dict.get("batch_consistency"),
        "batch_consistency_seed": params_dict.get("batch_consistency_seed"),
        "batch_consistency_order": params_dict.get("batch_consistency_order"),
        "batch_consistency_batch_split": params_dict.get(
            "batch_consistency_batch_split"
        ),
        "batch_consistency_mode_batch": params_dict.get("batch_consistency_mode_batch"),
        "batch_consistency_token_split": params_dict.get(
            "batch_consistency_token_split"
        ),
        "batch_consistency_shape_change": params_dict.get(
            "batch_consistency_shape_change"
        ),
    }

    return filled_params


def load_excel_test_cases(excel_file_path: str, sheetname: str):
    """
    从 Excel 文件加载测试用例。

    参数:
        excel_file_path (str): Excel 文件的路径。
        sheetname (str, optional): 工作表名称。若未提供，则默认 'Sheet1'。

    返回:
        list[]: 测试用例字典列表
    """
    if sheetname is None:
        sheetname = "Sheet1"

    # 检查文件是否存在
    if not os.path.exists(excel_file_path):
        pytest.skip(f"Excel file not found: {excel_file_path}", allow_module_level=True)

    try:
        # 读取 Excel 文件的指定 sheet
        df = pd.read_excel(excel_file_path, sheet_name=sheetname)
        df = df.replace({np.nan: None, pd.NA: None})

        # 定义必需的列名
        required_columns = [
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
            "ori_kv_topk_mode",
            "cmp_kv_topk_mode",
            "ori_sparse_indices_mode",
            "cmp_sparse_indices_mode",
        ]
        optional_columns = [
            "seqused_q",
            "cu_seqlens_q",
            "seqused_ori_kv",
            "seqused_cmp_kv",
            "cu_seqlens_ori_kv",
            "cu_seqlens_cmp_kv",
            "cmp_residual_kv",
            "ori_topk_length",
            "cmp_topk_length",
        ]
        datarange_columns = [
            "q_datarange",
            "ori_kv_datarange",
            "cmp_kv_datarange",
        ]
        default_value_columns = {
            "return_softmax_lse": False,
        }
        # 检查是否缺少必要列
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            pytest.skip(
                f"Missing required columns in Excel: {missing_cols}",
                allow_module_level=True,
            )

        # 构建测试用例列表
        test_cases = []
        i = 0
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            for col in optional_columns:
                if col not in row_dict:
                    row_dict[col] = None
                else:
                    row_dict[col] = parse_list_param(row_dict[col])
            for col in datarange_columns:
                if col not in row_dict:
                    row_dict[col] = None
                else:
                    row_dict[col] = parse_datarange_param(row_dict[col])
            for col, default_val in default_value_columns.items():
                if col not in row_dict or row_dict[col] is None:
                    row_dict[col] = default_val
            test_cases.append(row_dict)
            i = i + 1

        return test_cases

    except Exception as e:
        pytest.skip(f"Failed to read Excel file: {e}", allow_module_level=True)
        return None


def save_result(params, result, fulfill_percent, result_path):
    """
    存储测试结果到表格
    """
    row_data = {
        "Testcase_Name": params.get("Testcase_Name"),
        "layout_q": params.get("layout_q"),
        "layout_kv": params.get("layout_kv"),
        "q_type": params.get("q_type"),
        "ori_kv_type": params.get("ori_kv_type"),
        "cmp_kv_type": params.get("cmp_kv_type"),
        "B": params.get("B"),
        "S1": params.get("S1"),
        "S2": params.get("S2"),
        "T1": params.get("T1"),
        "N1": params.get("N1"),
        "N2": params.get("N2"),
        "D": params.get("D"),
        "K": params.get("K"),
        "block_num1": params.get("block_num1"),
        "block_num2": params.get("block_num2"),
        "block_size1": params.get("block_size1"),
        "block_size2": params.get("block_size2"),
        "seqused_q": params.get("seqused_q"),
        "cu_seqlens_q": params.get("cu_seqlens_q"),
        "seqused_ori_kv": params.get("seqused_ori_kv"),
        "seqused_cmp_kv": params.get("seqused_cmp_kv"),
        "cu_seqlens_ori_kv": params.get("cu_seqlens_ori_kv"),
        "cu_seqlens_cmp_kv": params.get("cu_seqlens_cmp_kv"),
        "cmp_residual_kv": params.get("cmp_residual_kv"),
        "softmax_scale": params.get("softmax_scale"),
        "cmp_ratio": params.get("cmp_ratio"),
        "ori_mask_mode": params.get("ori_mask_mode"),
        "cmp_mask_mode": params.get("cmp_mask_mode"),
        "ori_win_left": params.get("ori_win_left"),
        "template_run_mode": params.get("template_run_mode"),
        "actlen_mode": params.get("actlen_mode"),
        "S1EQS2": params.get("S1EQS2"),
        "tc_prefix": params.get("tc_prefix", " "),
        "ori_win_right": params.get("ori_win_right"),
        "quant_mode": params.get("quant_mode"),
        "tile_size": params.get("tile_size"),
        "rope_head_dim": params.get("rope_head_dim"),
        "topk_value_mode": params.get("topk_value_mode"),
        "return_softmax_lse": params.get("return_softmax_lse"),
        "q_datarange": params.get("q_datarange"),
        "ori_kv_datarange": params.get("ori_kv_datarange"),
        "cmp_kv_datarange": params.get("cmp_kv_datarange"),
        "ori_kv_topk_mode": params.get("ori_kv_topk_mode"),
        "cmp_kv_topk_mode": params.get("cmp_kv_topk_mode"),
        "ori_sparse_indices_mode": params.get("ori_sparse_indices_mode"),
        "cmp_sparse_indices_mode": params.get("cmp_sparse_indices_mode"),
        "ori_topk_length": str(params.get("ori_topk_length")),
        "cmp_topk_length": str(params.get("cmp_topk_length")),
        "result": result,
        "fulfill_percent": fulfill_percent,
    }

    # 检查文件是否存在
    if result_path.exists():
        try:
            df = pd.read_excel(result_path)
            if set(df.columns) != set(row_data.keys()):
                print("警告：变量名与Excel列名不匹配！")
                print(f"Excel列名: {list(df.columns)}")
                print(f"变量名: {list(row_data.keys())}")
                print("请检查变量名或Excel文件")
                return False
            new_df = pd.DataFrame([row_data])
            df = pd.concat([df, new_df], ignore_index=True)
        except Exception:
            print(f"警告: {result_path} 文件损坏，将重建")
            df = pd.DataFrame([row_data])
    else:
        # 文件不存在，创建新的DataFrame
        df = pd.DataFrame([row_data])

    # 保存到Excel
    df.to_excel(result_path, index=False)
