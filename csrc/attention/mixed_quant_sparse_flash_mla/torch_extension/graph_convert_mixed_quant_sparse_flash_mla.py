# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# GE Converter for Graph Mode

try:
    import torch
    import torch_npu
    import torchair
    from torch.library import impl
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair.ge._ge_graph import Tensor, TensorSpec
    from torchair._ge_concrete_graph.fx2ge_converter import (
        declare_supported,
        register_fx_node_ge_converter,
    )
    from torchair._ge_concrete_graph.supported_declaration import Support
    from typing import Any, Dict, List, Tuple, Union, Callable, Optional
    from torchair._ge_concrete_graph.ge_ir_pb2 import (
        GraphDef,
        OpDef,
        TensorDescriptor,
        TensorDef,
    )
    from torchair.ge._ge_graph import get_default_ge_graph, next_unique_name
    from torchair.ge._ge_graph import auto_convert_to_tensor
    from torchair.ge._ge_graph import DataType, TensorType
    from torchair.ge._ge_graph import compat_as_bytes, compat_as_bytes_list
    from torchair.ge._ge_graph import trans_to_list_list_int, trans_to_list_list_float
    from torchair.ge._ge_graph import get_invalid_desc
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair.ge import attr

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.mixed_quant_sparse_flash_mla_metadata.default
    )
    def convert_mixed_quant_sparse_flash_mla_metadata(
        num_heads_q: int,
        num_heads_kv: int,
        head_dim: int,
        quant_mode: int,
        *,
        cu_seqlens_q: Optional[Tensor] = None,
        cu_seqlens_ori_kv: Optional[Tensor] = None,
        cu_seqlens_cmp_kv: Optional[Tensor] = None,
        seqused_q: Optional[Tensor] = None,
        seqused_ori_kv: Optional[Tensor] = None,
        seqused_cmp_kv: Optional[Tensor] = None,
        cmp_residual_kv: Optional[Tensor] = None,
        ori_topk_length: Optional[Tensor] = None,
        cmp_topk_length: Optional[Tensor] = None,
        batch_size: Optional[int] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_ori_kv: Optional[int] = None,
        max_seqlen_cmp_kv: Optional[int] = None,
        ori_topk: Optional[int] = None,
        cmp_topk: Optional[int] = None,
        rope_head_dim: Optional[int] = None,
        cmp_ratio: Optional[int] = None,
        ori_mask_mode: Optional[int] = None,
        cmp_mask_mode: Optional[int] = None,
        ori_win_left: Optional[int] = None,
        ori_win_right: Optional[int] = None,
        layout_q: Optional[str] = None,
        layout_kv: Optional[str] = None,
        has_ori_kv: Optional[bool] = None,
        has_cmp_kv: Optional[bool] = None,
        meta_outputs: TensorSpec = None,
    ):
        raise RuntimeError(
            "GE converter doesn't support op: 'mixed_quant_sparse_flash_mla_metadata'"
        )
