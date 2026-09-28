# ruff: noqa
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
    from collections.abc import Callable
    from typing import Any, Dict, List, Optional, Tuple, Union

    import torch
    import torch_npu
    import torchair
    from torch.library import impl
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair._ge_concrete_graph.compat_ir import IrDef, ge_op
    from torchair._ge_concrete_graph.fx2ge_converter import (
        declare_supported,
        register_fx_node_ge_converter,
    )
    from torchair._ge_concrete_graph.ge_ir_pb2 import (
        GraphDef,
        OpDef,
        TensorDef,
        TensorDescriptor,
    )
    from torchair._ge_concrete_graph.supported_declaration import Support
    from torchair.ge import attr
    from torchair.ge._ge_graph import (
        DataType,
        Tensor,
        TensorSpec,
        TensorType,
        auto_convert_to_tensor,
        compat_as_bytes,
        compat_as_bytes_list,
        get_default_ge_graph,
        get_invalid_desc,
        next_unique_name,
        trans_to_list_list_float,
        trans_to_list_list_int,
    )

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:

    @register_fx_node_ge_converter(torch.ops.cann_ops_transformer.quant_lightning_indexer_metadata.default)
    def convert_quant_lightning_indexer_metadata(
        num_heads_q: int,
        num_heads_kv: int,
        head_dim: int,
        topk: int,
        quant_mode: int,
        *,
        cu_seqlens_q: Tensor | None = None,
        cu_seqlens_k: Tensor | None = None,
        seqused_q: Tensor | None = None,
        seqused_k: Tensor | None = None,
        cmp_residual_k: Tensor | None = None,
        batch_size: int | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
        layout_q: str | None = None,
        layout_k: str | None = None,
        mask_mode: int | None = None,
        cmp_ratio: int | None = None,
        meta_outputs: TensorSpec = None,
    ):
        raise RuntimeError("GE converter doesn't support op: 'quant_lightning_indexer_metadata'")
