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

import itertools
import os
from pathlib import Path

import pytest
import quant_lightning_indexer_v2_golden
import result_compare_method
import torch
import torch_npu
from batch import quant_lightning_indexer_v2_pt_loadprocess
from qliv2_test_utils import QliV2ResultWriter, ensure_comparison_passed
from test_quant_lightning_indexer_v2_paramset import ENABLED_PARAMSETS

SAVE_PT_DIR = os.environ.get("QLIV2_SINGLE_SAVE_PT_DIR", "").strip()
RESULT_PATH = os.environ.get("QLIV2_SINGLE_RESULT_PATH", "").strip()

param_names = [
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
    "weight_dtype",
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
    "run_mode",
]

param_combinations = []
for paramset_name, params in ENABLED_PARAMSETS:
    param_values = [
        params.get(name, params["dequant_dtype"] if name == "weight_dtype" else ["eager"]) for name in param_names
    ]
    combinations = list(itertools.product(*param_values))
    for combo_index, combo in enumerate(combinations, start=1):
        param_dict = dict(zip(param_names, combo))
        param_dict["case_name"] = paramset_name if len(combinations) == 1 else f"{paramset_name}_{combo_index:03d}"
        param_combinations.append(param_dict)


@pytest.mark.ci
@pytest.mark.parametrize("param_combinations", param_combinations)
def test_qliv2(param_combinations):  # Init params and tensors
    batch_size = param_combinations["batch_size"]
    q_seq = param_combinations["q_seq"]
    k_seq = param_combinations["k_seq"]
    q_t_size = param_combinations["q_t_size"]
    k_t_size = param_combinations["k_t_size"]
    q_head_num = param_combinations["q_head_num"]
    k_head_num = param_combinations["k_head_num"]
    head_dim = param_combinations["head_dim"]
    block_size = param_combinations["block_size"]
    block_num = param_combinations["block_num"]
    qk_dtype = param_combinations["qk_dtype"]
    weight_dtype = param_combinations["weight_dtype"]
    dequant_dtype = param_combinations["dequant_dtype"]
    actual_seq_dtype = param_combinations["actual_seq_dtype"]
    cu_seqlens_q = param_combinations["cu_seqlens_q"]
    cu_seqlens_k = param_combinations["cu_seqlens_k"]
    seqused_q = param_combinations["seqused_q"]
    seqused_k = param_combinations["seqused_k"]
    cmp_residual_k = param_combinations["cmp_residual_k"]
    max_seqlen_q = param_combinations["max_seqlen_q"]
    quant_mode = param_combinations["quant_mode"]
    layout_query = param_combinations["layout_query"]
    layout_key = param_combinations["layout_key"]
    sparse_count = param_combinations["sparse_count"]
    sparse_mode = param_combinations["sparse_mode"]
    query_datarange = param_combinations["query_datarange"]
    key_datarange = param_combinations["key_datarange"]
    weights_datarange = param_combinations["weights_datarange"]
    q_scale_datarange = param_combinations["q_scale_datarange"]
    k_scale_datarange = param_combinations["k_scale_datarange"]
    cmp_ratio = param_combinations["cmp_ratio"]
    return_value = param_combinations["return_value"]
    output_idx_offset = param_combinations["output_idx_offset"]
    run_mode = os.environ.get("QLIV2_RUN_MODE", param_combinations["run_mode"]).strip().lower()
    torch_npu.npu.set_device(0)
    test_data = (
        batch_size,
        q_seq,
        k_seq,
        q_t_size,
        k_t_size,
        q_head_num,
        k_head_num,
        head_dim,
        block_size,
        block_num,
        qk_dtype,
        weight_dtype,
        dequant_dtype,
        actual_seq_dtype,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        max_seqlen_q,
        quant_mode,
        layout_query,
        layout_key,
        sparse_count,
        sparse_mode,
        query_datarange,
        key_datarange,
        weights_datarange,
        q_scale_datarange,
        k_scale_datarange,
        cmp_ratio,
        return_value,
        output_idx_offset,
    )

    case_name = QliV2ResultWriter.case_name(
        test_data,
        explicit_name=param_combinations["case_name"],
    )
    if SAVE_PT_DIR:
        case_data = quant_lightning_indexer_v2_golden.generate_qliv2_test_data(test_data)
        case_path = Path(SAVE_PT_DIR) / f"{case_name}.pt"
        case_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(case_data, case_path)
        print(f"当前用例 PT 已保存: {case_path}")
        if run_mode == "eager":
            (
                cpu_result,
                npu_result,
                topk_value,
                cpu_topk_value,
                npu_topk_value,
                output_idx_offset,
                _,
            ) = quant_lightning_indexer_v2_pt_loadprocess.test_qliv2_process(case_path, device_id=0)
        elif run_mode == "graph":
            (
                cpu_result,
                npu_result,
                topk_value,
                cpu_topk_value,
                npu_topk_value,
                output_idx_offset,
                _,
            ) = quant_lightning_indexer_v2_pt_loadprocess.test_qliv2_process_graph(case_path, device_id=0)
        else:
            raise ValueError(f"unsupported run_mode: {run_mode}")
    elif run_mode == "eager":
        cpu_result, npu_result, topk_value, cpu_topk_value, npu_topk_value = (
            quant_lightning_indexer_v2_golden.qliv2_output_single(test_data)
        )
    elif run_mode == "graph":
        import quant_lightning_indexer_v2_acl_graph

        cpu_result, npu_result, topk_value, cpu_topk_value, npu_topk_value = (
            quant_lightning_indexer_v2_acl_graph.qliv2_output_acl_graph(test_data)
        )
    else:
        raise ValueError(f"unsupported run_mode: {run_mode}")
    # print("npu_result", npu_result)
    # print("cpu_result:", cpu_result)
    # Compare result accuracy
    result, fulfill_percent = result_compare_method.check_result(
        cpu_result,
        npu_result,
        topk_value,
        output_idx_offset,
        test_data,
        cpu_topk_value,
        npu_topk_value,
    )
    print("result", result)
    print("result", fulfill_percent)
    result_return_value = "N/A"
    fulfill_precent_return_value = 0
    if return_value:
        result_return_value, fulfill_precent_return_value = result_compare_method.check_result_return_value(
            cpu_topk_value,
            npu_topk_value,
            test_data,
            cpu_result,
            npu_result,
            topk_value,
            output_idx_offset,
        )
        print(f"result_return_value: {result_return_value}")
        print(f"result_return_value: {fulfill_precent_return_value}")

    if RESULT_PATH:
        row = QliV2ResultWriter.row(
            case_name,
            test_data,
            result,
            fulfill_percent,
            result_return_value,
            fulfill_precent_return_value,
        )
        QliV2ResultWriter.append(RESULT_PATH, row)
        print(f"当前用例结果已写入: {RESULT_PATH}")

    ensure_comparison_passed(
        case_name,
        result,
        fulfill_percent,
        result_return_value,
        fulfill_precent_return_value,
    )
