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

import quant_lightning_indexer_v2_golden
import torch
import torch.nn as nn
import torchair
from torchair.configs.compiler_config import CompilerConfig

QUANT_MODE_MXFP4 = 5


class QLIV2Network(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        output_idx_offset,
        max_seqlen_q,
        block_table,
        metadata,
        quant_mode,
        layout_query,
        layout_key,
        sparse_count,
        sparse_mode,
        cmp_ratio,
        return_value,
    ):
        return torch.ops.cann_ops_transformer.quant_lightning_indexer(
            query,
            key,
            weights,
            query_dequant_scale,
            key_dequant_scale,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            cmp_residual_k=cmp_residual_k,
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


def _qliv2_prepare_tensors_and_metadata(params, tensor_dict):
    """
    统一处理 tensor 准备和 metadata 构造（共用逻辑）。
    兼容两个来源：generate_qliv2_test_data 返回值和 .pt 文件加载，二者都在 CPU。
    """
    qk_dtype = params[10]
    quant_mode = tensor_dict["quant_mode"]

    if quant_mode == QUANT_MODE_MXFP4:
        # TorchAir通过foreach_copy搬运图输入，当前不支持FP4 shell dtype；使用相同存储的
        # packed uint8视图，C++入口会根据quant_mode恢复ACL_FLOAT4_E2M1语义。
        query = tensor_dict["query"].view(torch.uint8).npu()
        key = tensor_dict["key"].view(torch.uint8).npu()
        if "blockFusion" in tensor_dict and tensor_dict["blockFusion"] is not None:
            blockFusion = tensor_dict["blockFusion"].view(torch.uint8).npu()
    elif qk_dtype == "FLOAT8_E4M3FN" or qk_dtype == torch.float8_e4m3fn:
        query = tensor_dict["query"].to(dtype=torch.float8_e4m3fn).npu()
        key = tensor_dict["key"].to(dtype=torch.float8_e4m3fn).npu()
        if "blockFusion" in tensor_dict and tensor_dict["blockFusion"] is not None:
            blockFusion = tensor_dict["blockFusion"]
            if blockFusion.dtype == torch.uint8:
                blockFusion = blockFusion.view(torch.float8_e4m3fn)
            else:
                blockFusion = blockFusion.to(dtype=torch.float8_e4m3fn)
            blockFusion = blockFusion.npu()
    else:
        query = tensor_dict["query"].npu()
        key = tensor_dict["key"].npu()
        if "blockFusion" in tensor_dict and tensor_dict["blockFusion"] is not None:
            blockFusion = tensor_dict["blockFusion"].npu()

    weights = tensor_dict["weights"].npu()
    query_dequant_scale = tensor_dict["query_dequant_scale"].npu()
    key_dequant_scale = tensor_dict["key_dequant_scale"].npu()

    if "blockFusion" in tensor_dict and tensor_dict["blockFusion"] is not None:
        block_num = int(params[9])
        block_size = int(params[8])
        head_dim = int(params[7])
        k_head_num = int(params[6])
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

    cu_seqlens_query = tensor_dict["cu_seqlens_query"].npu() if tensor_dict["cu_seqlens_query"] is not None else None
    cu_seqlens_key = tensor_dict["cu_seqlens_key"].npu() if tensor_dict["cu_seqlens_key"] is not None else None
    seqused_q = tensor_dict["seqused_q"].npu() if tensor_dict["seqused_q"] is not None else None
    seqused_k = tensor_dict["seqused_k"].npu() if tensor_dict["seqused_k"] is not None else None
    output_idx_offset = tensor_dict["output_idx_offset"].npu() if tensor_dict["output_idx_offset"] is not None else None
    block_table = tensor_dict["block_table"].npu() if tensor_dict["block_table"] is not None else None
    cmp_residual_k_for_npu = (
        tensor_dict["cmp_residual_k_for_npu"].npu() if tensor_dict.get("cmp_residual_k_for_npu") is not None else None
    )

    layout_query = tensor_dict["layout_query"]
    layout_key = tensor_dict["layout_key"]
    sparse_count = tensor_dict["sparse_count"]
    sparse_mode = tensor_dict["sparse_mode"]
    cmp_ratio = tensor_dict["cmp_ratio"]
    max_seqlen_q_meta = tensor_dict["max_seqlen_q_meta"]
    max_seqlen_k_meta = tensor_dict["max_seqlen_k_meta"]

    q_head_num = int(params[5])
    k_head_num = int(params[6])
    head_dim = int(params[7])
    batch_size = int(params[0])

    metadata = torch.ops.cann_ops_transformer.quant_lightning_indexer_metadata(
        cu_seqlens_q=cu_seqlens_query,
        cu_seqlens_k=cu_seqlens_key,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k_for_npu,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q_meta,
        max_seqlen_k=max_seqlen_k_meta,
        num_heads_q=q_head_num,
        num_heads_k=k_head_num,
        head_dim=head_dim,
        topk=sparse_count,
        quant_mode=quant_mode,
        mask_mode=sparse_mode,
        layout_q=layout_query,
        layout_k=layout_key,
        cmp_ratio=cmp_ratio,
    )
    metadata = metadata.npu()

    run_args = {
        "query": query,
        "key": key,
        "weights": weights,
        "query_dequant_scale": query_dequant_scale,
        "key_dequant_scale": key_dequant_scale,
        "cu_seqlens_q": cu_seqlens_query,
        "cu_seqlens_k": cu_seqlens_key,
        "seqused_q": seqused_q,
        "seqused_k": seqused_k,
        "cmp_residual_k": cmp_residual_k_for_npu,
        "output_idx_offset": output_idx_offset,
        "max_seqlen_q": params[19],
        "block_table": block_table,
        "metadata": metadata,
        "quant_mode": quant_mode,
        "layout_query": layout_query,
        "layout_key": layout_key,
        "sparse_count": sparse_count,
        "sparse_mode": sparse_mode,
        "cmp_ratio": cmp_ratio,
        "return_value": params[31],
    }
    return run_args


def _qliv2_run_compiled_graph(run_args):
    """
    通过 torch.compile + torchair 后端执行算子（共用逻辑）。
    """
    config = CompilerConfig()
    config.mode = "reduce-overhead"
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    torch._dynamo.reset()
    npu_mode = torch.compile(QLIV2Network().npu(), fullgraph=False, backend=npu_backend, dynamic=False)
    npu_result, npu_topk_value = npu_mode(**run_args)
    torch.npu.synchronize()
    if run_args["return_value"]:
        if npu_topk_value.shape != npu_result.shape:
            raise RuntimeError(
                "sparse_values and sparse_indices must have the same shape when return_value=1, "
                f"but got {tuple(npu_topk_value.shape)} and {tuple(npu_result.shape)}"
            )
        npu_topk_value, npu_sort_order = npu_topk_value.sort(dim=-1, descending=True)
        npu_result = torch.gather(npu_result, dim=-1, index=npu_sort_order)
    return npu_result, npu_topk_value


def qliv2_output_acl_graph(params):
    """
    graph 模式入口（single 用例使用）：即时生成随机 tensor + CPU golden，再走 torch.compile 执行。
    """
    print("acl_graph")
    tensor_dict = quant_lightning_indexer_v2_golden.generate_qliv2_test_data(params)
    cpu_result = tensor_dict["cpu_result"]
    topk_value = tensor_dict["topk_value"]
    cpu_topk_value = tensor_dict["cpu_topk_value"]

    run_args = _qliv2_prepare_tensors_and_metadata(params, tensor_dict)
    npu_result, npu_topk_value = _qliv2_run_compiled_graph(run_args)

    return cpu_result, npu_result, topk_value, cpu_topk_value, npu_topk_value


def qliv2_output_acl_graph_from_pt(params, tensor_dict):
    """
    graph 模式入口（batch 用例使用）：从 .pt 文件加载 pre-computed tensor，走 torch.compile 执行。
    跳过 generate_qliv2_test_data 的随机数据重新生成和 CPU golden 重算。
    """
    cpu_result = tensor_dict["cpu_result"]
    topk_value = tensor_dict["topk_value"]
    cpu_topk_value = tensor_dict["cpu_topk_value"]

    run_args = _qliv2_prepare_tensors_and_metadata(params, tensor_dict)
    npu_result, npu_topk_value = _qliv2_run_compiled_graph(run_args)

    return cpu_result, npu_result, topk_value, cpu_topk_value, npu_topk_value
