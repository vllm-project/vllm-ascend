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

import os
import torch
import torch_npu
import check_valid_param
import pytest
import random
import numpy as np
import math
from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env(include_vendor_lib=True)
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped] # noqa: E402,F401


class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

    def forward(
        self,
        q,
        ori_kv,
        cmp_kv,
        ori_sparse_indices,
        cmp_sparse_indices,
        ori_block_table,
        cmp_block_table,
        cu_seqlens_q,
        seqused_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        seqused_ori_kv,
        seqused_cmp_kv,
        cmp_residual_kv,
        ori_topk_length,
        cmp_topk_length,
        sinks,
        metadata,
        quant_mode,
        rope_head_dim,
        softmax_scale,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
        layout_q,
        layout_kv,
        topk_value_mode,
        return_softmax_lse,
    ):
        npu_result, npu_lse = (
            torch.ops._C_ascend.npu_mixed_quant_sparse_flash_mla(
                q=q,
                ori_kv=ori_kv,
                cmp_kv=cmp_kv,
                ori_sparse_indices=ori_sparse_indices,
                cmp_sparse_indices=cmp_sparse_indices,
                ori_block_table=ori_block_table,
                cmp_block_table=cmp_block_table,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_ori_kv=cu_seqlens_ori_kv,
                cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
                seqused_q=seqused_q,
                seqused_ori_kv=seqused_ori_kv,
                seqused_cmp_kv=seqused_cmp_kv,
                cmp_residual_kv=cmp_residual_kv,
                ori_topk_length=ori_topk_length,
                cmp_topk_length=cmp_topk_length,
                sinks=sinks,
                metadata=metadata,
                quant_mode=quant_mode,
                rope_head_dim=rope_head_dim,
                softmax_scale=softmax_scale,
                cmp_ratio=cmp_ratio,
                ori_mask_mode=ori_mask_mode,
                cmp_mask_mode=cmp_mask_mode,
                **({"ori_win_left": ori_win_left} if ori_win_left is not None else {}),
                **(
                    {"ori_win_right": ori_win_right}
                    if ori_win_right is not None
                    else {}
                ),
                layout_q=layout_q,
                layout_kv=layout_kv,
                topk_value_mode=topk_value_mode,
                return_softmax_lse=return_softmax_lse,
                key_dtype=None,
                value_dtype=None,
            )
        )
        return npu_result, npu_lse


def test_mqsmla_quant_process_graph(test_data, device_id=0):
    import torchair
    from torchair.configs.compiler_config import CompilerConfig

    params = test_data["params"]
    metadata_input = test_data["metadata_input"]
    op_input = test_data["op_input"]
    cpu_output = test_data["cpu_output"]
    cpu_lse = test_data.get("cpu_lse")
    torch_npu.npu.set_device(device_id)

    torch._dynamo.reset()
    npu_mode = Network().npu()
    config = CompilerConfig()
    config.mode = "reduce-overhead"
    config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
    config.experimental_config.aclgraph._aclnn_static_shape_kernel_build_dir = "./"
    config.experimental_config.frozen_parameter = True
    config.experimental_config.tiling_schedule_optimize = True
    config.experimental_config.topology_sorting_strategy = "StableRDFS"
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    npu_mode = torch.compile(
        npu_mode, fullgraph=True, backend=npu_backend, dynamic=False
    )

    print("test_data:", params)
    print("mixed_quant_sparse_flash_mla_metadata...")
    metadata = torch.ops._C_ascend.npu_mixed_quant_sparse_flash_mla_metadata(
        num_heads_q=metadata_input["num_heads_q"],
        num_heads_kv=metadata_input["num_heads_kv"],
        head_dim=metadata_input["head_dim"],
        cu_seqlens_q=metadata_input["cu_seqlens_q"].npu()
        if op_input["cu_seqlens_q"] is not None
        else torch.tensor([]).npu(),
        cu_seqlens_ori_kv=op_input["cu_seqlens_ori_kv"].npu()
        if op_input["cu_seqlens_ori_kv"] is not None
        else torch.tensor([]).npu(),
        cu_seqlens_cmp_kv=op_input["cu_seqlens_cmp_kv"].npu()
        if op_input["cu_seqlens_cmp_kv"] is not None
        else torch.tensor([]).npu(),
        seqused_q=op_input["seqused_q"].npu()
        if op_input["seqused_q"] is not None
        else torch.tensor([]).npu(),
        seqused_ori_kv=op_input["seqused_ori_kv"].npu()
        if op_input["seqused_ori_kv"] is not None
        else None,
        seqused_cmp_kv=op_input["seqused_cmp_kv"].npu()
        if op_input["seqused_cmp_kv"] is not None
        else None,
        cmp_residual_kv=op_input["cmp_residual_kv"].npu()
        if op_input["cmp_residual_kv"] is not None
        else None,
        ori_topk_length=op_input["ori_topk_length"].npu()
        if op_input.get("ori_topk_length") is not None
        else None,
        cmp_topk_length=op_input["cmp_topk_length"].npu()
        if op_input.get("cmp_topk_length") is not None
        else None,
        quant_mode=op_input["quant_mode"],
        batch_size=metadata_input["batch_size"],
        max_seqlen_q=metadata_input["max_seqlen_q"],
        max_seqlen_ori_kv=metadata_input["max_seqlen_ori_kv"],
        max_seqlen_cmp_kv=metadata_input["max_seqlen_cmp_kv"],
        ori_topk=metadata_input.get("ori_topk", 0),
        cmp_topk=metadata_input["topk"],
        rope_head_dim=op_input["rope_head_dim"],
        cmp_ratio=metadata_input["cmp_ratio"],
        ori_mask_mode=metadata_input["ori_mask_mode"],
        cmp_mask_mode=metadata_input["cmp_mask_mode"],
        **(
            {"ori_win_left": metadata_input["ori_win_left"]}
            if metadata_input.get("ori_win_left") is not None
            else {}
        ),
        **(
            {"ori_win_right": metadata_input["ori_win_right"]}
            if metadata_input.get("ori_win_right") is not None
            else {}
        ),
        layout_q=metadata_input["layout_q"],
        layout_kv=metadata_input["layout_kv"],
        has_ori_kv=metadata_input["has_ori_kv"],
        has_cmp_kv=metadata_input["has_cmp_kv"],
    )

    torch.npu.synchronize()
    metadata.npu()

    print("mixed_quant_sparse_flash_mla...")
    npu_result, npu_lse = npu_mode(
        q=op_input["q"].npu() if op_input["q"] is not None else None,
        ori_kv=op_input["ori_kv"].npu() if op_input["ori_kv"] is not None else None,
        cmp_kv=op_input["cmp_kv"].npu() if op_input["cmp_kv"] is not None else None,
        ori_sparse_indices=op_input["ori_sparse_indices"].npu()
        if op_input["ori_sparse_indices"] is not None
        else None,
        cmp_sparse_indices=op_input["cmp_sparse_indices"].npu()
        if op_input["cmp_sparse_indices"] is not None
        else None,
        ori_block_table=op_input["ori_block_table"].npu()
        if op_input["ori_block_table"] is not None
        else None,
        cmp_block_table=op_input["cmp_block_table"].npu()
        if op_input["cmp_block_table"] is not None
        else None,
        cu_seqlens_q=op_input["cu_seqlens_q"].npu()
        if op_input["cu_seqlens_q"] is not None
        else None,
        seqused_q=op_input["seqused_q"].npu()
        if op_input["seqused_q"] is not None
        else torch.tensor([]).npu(),
        cu_seqlens_ori_kv=op_input["cu_seqlens_ori_kv"].npu()
        if op_input["cu_seqlens_ori_kv"] is not None
        else None,
        cu_seqlens_cmp_kv=op_input["cu_seqlens_cmp_kv"].npu()
        if op_input["cu_seqlens_cmp_kv"] is not None
        else None,
        seqused_ori_kv=op_input["seqused_ori_kv"].npu()
        if op_input["seqused_ori_kv"] is not None
        else None,
        seqused_cmp_kv=op_input["seqused_cmp_kv"].npu()
        if op_input["seqused_cmp_kv"] is not None
        else None,
        cmp_residual_kv=op_input["cmp_residual_kv"].npu()
        if op_input["cmp_residual_kv"] is not None
        else None,
        ori_topk_length=op_input["ori_topk_length"].npu()
        if op_input.get("ori_topk_length") is not None
        else None,
        cmp_topk_length=op_input["cmp_topk_length"].npu()
        if op_input.get("cmp_topk_length") is not None
        else None,
        sinks=op_input["sinks"].npu() if op_input["sinks"] is not None else None,
        metadata=metadata,
        quant_mode=op_input["quant_mode"],
        rope_head_dim=op_input["rope_head_dim"],
        softmax_scale=op_input["softmax_scale"],
        cmp_ratio=op_input["cmp_ratio"],
        ori_mask_mode=op_input["ori_mask_mode"],
        cmp_mask_mode=op_input["cmp_mask_mode"],
        ori_win_left=op_input["ori_win_left"],
        ori_win_right=op_input["ori_win_right"],
        layout_q=op_input["layout_q"],
        layout_kv=op_input["layout_kv"],
        topk_value_mode=op_input.get("topk_value_mode", 1),
        return_softmax_lse=op_input.get("return_softmax_lse", False),
    )

    torch.npu.synchronize()
    return npu_result, cpu_output, npu_lse, cpu_lse


def test_mqsmla_quant_process_ci(test_data, device_id=0):
    params = test_data["params"]
    metadata_input = test_data["metadata_input"]
    op_input = test_data["op_input"]
    cpu_output = test_data["cpu_output"]
    cpu_lse = test_data.get("cpu_lse")
    torch_npu.npu.set_device(device_id)

    print("test_data:", params)
    print("mixed_quant_sparse_flash_mla_metadata...")
    metadata = torch.ops._C_ascend.npu_mixed_quant_sparse_flash_mla_metadata(
        num_heads_q=metadata_input["num_heads_q"],
        num_heads_kv=metadata_input["num_heads_kv"],
        head_dim=metadata_input["head_dim"],
        cu_seqlens_q=metadata_input["cu_seqlens_q"].npu()
        if op_input["cu_seqlens_q"] is not None
        else torch.tensor([]).npu(),
        cu_seqlens_ori_kv=op_input["cu_seqlens_ori_kv"].npu()
        if op_input["cu_seqlens_ori_kv"] is not None
        else torch.tensor([]).npu(),
        cu_seqlens_cmp_kv=op_input["cu_seqlens_cmp_kv"].npu()
        if op_input["cu_seqlens_cmp_kv"] is not None
        else torch.tensor([]).npu(),
        seqused_q=op_input["seqused_q"].npu()
        if op_input["seqused_q"] is not None
        else torch.tensor([]).npu(),
        seqused_ori_kv=op_input["seqused_ori_kv"].npu()
        if op_input["seqused_ori_kv"] is not None
        else None,
        seqused_cmp_kv=op_input["seqused_cmp_kv"].npu()
        if op_input["seqused_cmp_kv"] is not None
        else None,
        cmp_residual_kv=op_input["cmp_residual_kv"].npu()
        if op_input["cmp_residual_kv"] is not None
        else None,
        ori_topk_length=op_input["ori_topk_length"].npu()
        if op_input.get("ori_topk_length") is not None
        else None,
        cmp_topk_length=op_input["cmp_topk_length"].npu()
        if op_input.get("cmp_topk_length") is not None
        else None,
        quant_mode=op_input["quant_mode"],
        batch_size=metadata_input["batch_size"],
        max_seqlen_q=metadata_input["max_seqlen_q"],
        max_seqlen_ori_kv=metadata_input["max_seqlen_ori_kv"],
        max_seqlen_cmp_kv=metadata_input["max_seqlen_cmp_kv"],
        ori_topk=metadata_input.get("ori_topk", 0),
        cmp_topk=metadata_input["topk"],
        rope_head_dim=op_input["rope_head_dim"],
        cmp_ratio=metadata_input["cmp_ratio"],
        ori_mask_mode=metadata_input["ori_mask_mode"],
        cmp_mask_mode=metadata_input["cmp_mask_mode"],
        **(
            {"ori_win_left": metadata_input["ori_win_left"]}
            if metadata_input.get("ori_win_left") is not None
            else {}
        ),
        **(
            {"ori_win_right": metadata_input["ori_win_right"]}
            if metadata_input.get("ori_win_right") is not None
            else {}
        ),
        layout_q=metadata_input["layout_q"],
        layout_kv=metadata_input["layout_kv"],
        has_ori_kv=metadata_input["has_ori_kv"],
        has_cmp_kv=metadata_input["has_cmp_kv"],
    )
    torch.npu.synchronize()
    metadata.npu()
    print("mixed_quant_sparse_flash_mla...")
    npu_result, npu_lse = torch.ops._C_ascend.npu_mixed_quant_sparse_flash_mla(
        q=op_input["q"].npu() if op_input["q"] is not None else None,
        ori_kv=op_input["ori_kv"].npu() if op_input["ori_kv"] is not None else None,
        cmp_kv=op_input["cmp_kv"].npu() if op_input["cmp_kv"] is not None else None,
        ori_sparse_indices=op_input["ori_sparse_indices"].npu()
        if op_input.get("ori_sparse_indices") is not None
        else None,
        cmp_sparse_indices=op_input["cmp_sparse_indices"].npu()
        if op_input["cmp_sparse_indices"] is not None
        else None,
        ori_block_table=op_input["ori_block_table"].npu()
        if op_input["ori_block_table"] is not None
        else None,
        cmp_block_table=op_input["cmp_block_table"].npu()
        if op_input["cmp_block_table"] is not None
        else None,
        cu_seqlens_q=op_input["cu_seqlens_q"].npu()
        if op_input["cu_seqlens_q"] is not None
        else None,
        cu_seqlens_ori_kv=op_input["cu_seqlens_ori_kv"].npu()
        if op_input["cu_seqlens_ori_kv"] is not None
        else None,
        cu_seqlens_cmp_kv=op_input["cu_seqlens_cmp_kv"].npu()
        if op_input["cu_seqlens_cmp_kv"] is not None
        else None,
        seqused_q=op_input["seqused_q"].npu()
        if op_input["seqused_q"] is not None
        else torch.tensor([]).npu(),
        seqused_ori_kv=op_input["seqused_ori_kv"].npu()
        if op_input["seqused_ori_kv"] is not None
        else None,
        seqused_cmp_kv=op_input["seqused_cmp_kv"].npu()
        if op_input["seqused_cmp_kv"] is not None
        else None,
        cmp_residual_kv=op_input["cmp_residual_kv"].npu()
        if op_input["cmp_residual_kv"] is not None
        else None,
        ori_topk_length=op_input["ori_topk_length"].npu()
        if op_input.get("ori_topk_length") is not None
        else None,
        cmp_topk_length=op_input["cmp_topk_length"].npu()
        if op_input.get("cmp_topk_length") is not None
        else None,
        sinks=op_input["sinks"].npu() if op_input["sinks"] is not None else None,
        metadata=metadata,
        quant_mode=op_input["quant_mode"],
        rope_head_dim=op_input["rope_head_dim"],
        softmax_scale=op_input["softmax_scale"],
        cmp_ratio=op_input["cmp_ratio"],
        ori_mask_mode=op_input["ori_mask_mode"],
        cmp_mask_mode=op_input["cmp_mask_mode"],
        **(
            {"ori_win_left": op_input["ori_win_left"]}
            if op_input.get("ori_win_left") is not None
            else {}
        ),
        **(
            {"ori_win_right": op_input["ori_win_right"]}
            if op_input.get("ori_win_right") is not None
            else {}
        ),
        layout_q=op_input["layout_q"],
        layout_kv=op_input["layout_kv"],
        topk_value_mode=op_input.get("topk_value_mode", 1),
        return_softmax_lse=op_input.get("return_softmax_lse", False),
        key_dtype=None,
        value_dtype=None,
    )

    torch.npu.synchronize()

    return npu_result, cpu_output, cpu_lse, npu_lse
