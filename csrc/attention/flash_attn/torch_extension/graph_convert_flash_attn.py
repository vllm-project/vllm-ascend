# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

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

    @auto_convert_to_tensor(
        [False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, True, True, True, True, True, True, True],
    )
    def FlashAttn(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        block_table: Optional[Tensor],
        cu_seqlens_q: Optional[Tensor],
        cu_seqlens_kv: Optional[Tensor],
        seqused_q: Optional[Tensor],
        seqused_kv: Optional[Tensor],
        sinks: Optional[Tensor],
        attn_mask: Optional[Tensor],
        metadata: Optional[Tensor],
        softmax_scale: float = 1.0,
        mask_mode: int = 0,
        win_left: int = -1,
        win_right: int = -1,
        max_seqlen_q: int = -1,
        max_seqlen_kv: int = -1,
        layout_q: str = "BSND",
        layout_kv: str = "BSND",
        layout_out: str = "BSND",
        return_softmax_lse: bool = False,
        deterministic: int = 0,
    ):
        result = q.new_empty(q.size())
        return result

    @register_fx_node_ge_converter(torch.ops.cann_ops_transformer.flash_attn.default)
    def convert_flash_attn(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        block_table: Tensor = None,
        cu_seqlens_q: Tensor = None,
        cu_seqlens_kv: Tensor = None,
        seqused_q: Tensor = None,
        seqused_kv: Tensor = None,
        sinks: Tensor = None,
        attn_mask: Tensor = None,
        metadata: Tensor = None,
        softmax_scale: float = 1.0,
        mask_mode: int = 0,
        win_left: int = -1,
        win_right: int = -1,
        max_seqlen_q: int = -1,
        max_seqlen_kv: int = -1,
        layout_q: str = "BSND",
        layout_kv: str = "BSND",
        layout_out: str = "BSND",
        return_softmax_lse: bool = False,
        deterministic: int = 0,
    ):
        raise AssertionError("GE not supported!")
