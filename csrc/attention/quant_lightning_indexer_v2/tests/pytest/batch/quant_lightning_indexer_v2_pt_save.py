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

import os

import numpy as np
import pandas as pd
import torch
from quant_lightning_indexer_v2_golden import generate_qliv2_test_data

try:
    import torch_npu
except ImportError:
    torch_npu = None
import argparse

import pytest


def load_excel_test_cases(excel_file_path: str, sheetname: str):
    """
    从 Excel 文件加载测试用例。

    参数:
        excel_file_path (str): Excel 文件的路径。
        sheetname (str, optional): 工作表名称。若未提供，则默认 'Sheet1'。

    返回:
        list[tuple]: 测试用例元组列表，每个元组包含 20+ 个字段。
                       若失败或跳过，则返回空列表。
    """
    # 优先使用传入的 sheetname，否则尝试从环境变量获取
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
            "batch_size",
            "q_seq",
            "k_seq",
            "q_t_size",
            "k_t_size",
            "q_head_num",
            "k_head_num",
            "head_dim",
            "block_size",
            "block_num",
            "qk_dtype",
            "dequant_dtype",
            "actual_seq_dtype",
            "cu_seqlens_q",
            "cu_seqlens_k",
            "seqused_q",
            "seqused_k",
            "cmp_residual_k",
            "max_seqlen_q",
            "quant_mode",
            "layout_query",
            "layout_key",
            "sparse_count",
            "sparse_mode",
            "query_datarange",
            "key_datarange",
            "weights_datarange",
            "q_scale_datarange",
            "k_scale_datarange",
            "cmp_ratio",
            "return_value",
            "output_idx_offset",
        ]

        # 检查是否缺少必要列
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            pytest.skip(
                f"Missing required columns in Excel: {missing_cols}",
                allow_module_level=True,
            )

        # 构建测试用例列表
        test_cases = []
        for _, row in df.iterrows():
            test_cases.append(
                (
                    row["Testcase_Name"],
                    row["batch_size"],
                    row["q_seq"],
                    row["k_seq"],
                    row["q_t_size"],
                    row["k_t_size"],
                    row["q_head_num"],
                    row["k_head_num"],
                    row["head_dim"],
                    row["block_size"],
                    row["block_num"],
                    row["qk_dtype"],
                    row["weight_dtype"]
                    if "weight_dtype" in row and row["weight_dtype"] is not None
                    else row["dequant_dtype"],
                    row["dequant_dtype"],
                    row["actual_seq_dtype"],
                    row["cu_seqlens_q"],
                    row["cu_seqlens_k"],
                    row["seqused_q"],
                    row["seqused_k"],
                    row["cmp_residual_k"],
                    row["max_seqlen_q"],
                    row["quant_mode"],
                    row["layout_query"],
                    row["layout_key"],
                    row["sparse_count"],
                    row["sparse_mode"],
                    row["query_datarange"],
                    row["key_datarange"],
                    row["weights_datarange"],
                    row["q_scale_datarange"],
                    row["k_scale_datarange"],
                    row["cmp_ratio"],
                    row["return_value"],
                    row["output_idx_offset"],
                )
            )

        return test_cases

    except Exception as e:
        pytest.skip(f"Failed to read Excel file: {e}", allow_module_level=True)
        return None


def save_test_case(test_cases, file_path):
    print("正在保存pt文件...")
    # 创建输出目录
    os.makedirs(file_path, exist_ok=True)

    for idx, case in enumerate(test_cases):
        try:
            case_name = case[0]
            output_tensors = generate_qliv2_test_data(case[1:])
            # 生成文件名
            input_filename = f"{case_name}.pt"
            input_filepath = os.path.join(file_path, input_filename)

            # 保存数据
            torch.save(output_tensors, input_filepath)
            print(f"测试用例已保存到: {input_filepath}")

        except Exception as e:
            print(f"[失败] 生成 pt 文件失败: {case[0]} (索引: {idx})")
            print(f"错误详情: {e}")


def main():
    parser = argparse.ArgumentParser(description="qliv2_pt_save.py 接收路径参数")
    parser.add_argument("path1", type=str, help="第一个路径")
    parser.add_argument("path2", type=str, help="第二个路径")
    parser.add_argument("--sheet", "-S", type=str, default="Sheet1", help="Sheet 页名（默认: Sheet1）")
    args = parser.parse_args()
    path1 = args.path1
    path2 = args.path2
    testcase = load_excel_test_cases(path1, args.sheet)
    save_test_case(testcase, path2)


if __name__ == "__main__":
    main()
