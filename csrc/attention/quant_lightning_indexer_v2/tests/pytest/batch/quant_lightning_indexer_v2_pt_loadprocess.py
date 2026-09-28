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


import torch
import torch_npu
from qliv2_parameter_normalization import normalize_qliv2_params

QUANT_MODE_MXFP4 = 5


def test_qliv2_process(filepath, device_id=0):
    # 加载测试数据
    test_data = torch.load(filepath, map_location="cpu")

    params = normalize_qliv2_params(test_data["params"])
    cpu_result = test_data["cpu_result"]
    topk_value = test_data["topk_value"]
    cpu_topk_value = test_data["cpu_topk_value"]
    print("执行用例：", filepath)
    torch_npu.npu.set_device(device_id)

    quant_mode = test_data["quant_mode"]
    if quant_mode == QUANT_MODE_MXFP4:
        query = test_data["query"].view(torch.uint8).npu()
        key = test_data["key"].view(torch.uint8).npu()
        if "blockFusion" in test_data and test_data["blockFusion"] is not None:
            blockFusion = test_data["blockFusion"].view(torch.uint8).npu()
    elif params[10] == "FLOAT8_E4M3FN" or params[10] == torch.float8_e4m3fn:
        query = test_data["query"].to(dtype=torch.float8_e4m3fn).npu()
        key = test_data["key"].to(dtype=torch.float8_e4m3fn).npu()
        if "blockFusion" in test_data and test_data["blockFusion"] is not None:
            blockFusion = test_data["blockFusion"]
            if blockFusion.dtype == torch.uint8:
                blockFusion = blockFusion.view(torch.float8_e4m3fn)
            else:
                blockFusion = blockFusion.to(dtype=torch.float8_e4m3fn)
            blockFusion = blockFusion.npu()
    else:
        query = test_data["query"].npu()
        key = test_data["key"].npu()
        if "blockFusion" in test_data and test_data["blockFusion"] is not None:
            blockFusion = test_data["blockFusion"].npu()

    max_seqlen_q = params[19]
    return_value = params[31]
    weights = test_data["weights"].npu()
    query_dequant_scale = test_data["query_dequant_scale"].npu()
    key_dequant_scale = test_data["key_dequant_scale"].npu()
    if "blockFusion" in test_data and test_data["blockFusion"] is not None:
        block_num = params[9]
        block_size = params[8]
        head_dim = params[7]
        k_head_num = params[6]
        k_head_num = int(k_head_num)
        head_dim = int(head_dim)
        block_size = int(block_size)
        block_num = int(block_num)
        dequant_dtype_str = params[12]
        if dequant_dtype_str == "FP16" or dequant_dtype_str == torch.float16:
            dequant_dtype = torch.float16
        elif dequant_dtype_str == "FP32" or dequant_dtype_str == torch.float32:
            dequant_dtype = torch.float32
        else:
            dequant_dtype = torch.float16
        key = blockFusion[:, : block_size * k_head_num * head_dim].view(block_num, block_size, k_head_num, head_dim)
        key_dequant_scale = (
            blockFusion[:, block_size * k_head_num * head_dim :]
            .view(dequant_dtype)
            .view(block_num, block_size, k_head_num)
        )
    if test_data["seqused_q"] is not None:
        seqused_q = test_data["seqused_q"].npu()
    else:
        seqused_q = None
    if test_data["seqused_k"] is not None:
        seqused_k = test_data["seqused_k"].npu()
    else:
        seqused_k = None
    if test_data["output_idx_offset"] is not None:
        output_idx_offset = test_data["output_idx_offset"].npu()
    else:
        output_idx_offset = None
    if test_data["cu_seqlens_query"] is not None:
        cu_seqlens_query = test_data["cu_seqlens_query"].npu()
    else:
        cu_seqlens_query = None
    if test_data["cu_seqlens_key"] is not None:
        cu_seqlens_key = test_data["cu_seqlens_key"].npu()
    else:
        cu_seqlens_key = None
    if test_data["block_table"] is not None:
        block_table = test_data["block_table"].npu()
    else:
        block_table = None
    layout_query = test_data["layout_query"]
    layout_key = test_data["layout_key"]
    sparse_count = test_data["sparse_count"]
    sparse_mode = test_data["sparse_mode"]
    cmp_ratio = test_data["cmp_ratio"]
    if test_data["cmp_residual_k_for_npu"] is not None:
        cmp_residual_k_for_npu = test_data["cmp_residual_k_for_npu"].npu()
    else:
        cmp_residual_k_for_npu = None

    max_seqlen_q_meta = test_data["max_seqlen_q_meta"]
    max_seqlen_k_meta = test_data["max_seqlen_k_meta"]
    metadata = torch.ops.cann_ops_transformer.quant_lightning_indexer_metadata(
        cu_seqlens_q=cu_seqlens_query,
        cu_seqlens_k=cu_seqlens_key,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k_for_npu,
        batch_size=params[0],
        max_seqlen_q=max_seqlen_q_meta,
        max_seqlen_k=max_seqlen_k_meta,
        num_heads_q=params[5],
        num_heads_k=params[6],
        head_dim=params[7],
        topk=sparse_count,
        quant_mode=quant_mode,
        mask_mode=sparse_mode,
        layout_q=layout_query,
        layout_k=layout_key,
        cmp_ratio=cmp_ratio,
    )
    metadata = metadata.npu()

    # 调用qli算子
    npu_result, npu_value = torch.ops.cann_ops_transformer.quant_lightning_indexer(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        cu_seqlens_q=cu_seqlens_query,
        cu_seqlens_k=cu_seqlens_key,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k_for_npu,
        output_idx_offset=output_idx_offset,
        max_seqlen_q=max_seqlen_q,
        block_table=block_table,
        metadata=metadata,
        quant_mode=quant_mode,
        layout_q=layout_query,
        layout_k=layout_key,
        topk=sparse_count,
        mask_mode=sparse_mode,
        cmp_ratio=cmp_ratio,
        return_value=return_value,
    )

    torch.npu.synchronize()
    npu_topk_value = npu_value
    if return_value:
        if npu_topk_value.shape != npu_result.shape:
            raise RuntimeError(
                "sparse_values and sparse_indices must have the same shape when return_value=1, "
                f"but got {tuple(npu_topk_value.shape)} and {tuple(npu_result.shape)}"
            )
        npu_topk_value, npu_sort_order = npu_topk_value.sort(dim=-1, descending=True)
        npu_result = torch.gather(npu_result, dim=-1, index=npu_sort_order)
    return (
        cpu_result,
        npu_result,
        topk_value,
        cpu_topk_value,
        npu_topk_value,
        output_idx_offset,
        params,
    )


def test_qliv2_process_graph(filepath, device_id=0):
    """
    graph 模式：从 .pt 文件加载 pre-computed tensor，走 torch.compile + torchair 后端执行算子，
    跳过 generate_qliv2_test_data 的随机数据重新生成和 CPU golden 重算。
    与 eager 模式共用相同的 .pt 数据，仅算子调用路径不同（compile vs eager）。
    """
    import quant_lightning_indexer_v2_acl_graph

    test_data = torch.load(filepath, map_location="cpu")
    params = normalize_qliv2_params(test_data["params"])
    output_idx_offset = test_data.get("output_idx_offset", None)

    torch_npu.npu.set_device(device_id)
    cpu_result, npu_result, topk_value, cpu_topk_value, npu_topk_value = (
        quant_lightning_indexer_v2_acl_graph.qliv2_output_acl_graph_from_pt(params, test_data)
    )

    return (
        cpu_result,
        npu_result,
        topk_value,
        cpu_topk_value,
        npu_topk_value,
        output_idx_offset,
        params,
    )
