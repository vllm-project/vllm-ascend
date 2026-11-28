# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch_npu

LONG_KV_INT8_GROUP = 64


def pack_long_kv_int8(rows: torch.Tensor) -> torch.Tensor:
    """Pack each post-RoPE latent as 512 signed codes followed by eight FP32 scales."""
    row_count, width = rows.shape
    values, scales = torch_npu.npu_dynamic_quant(rows.reshape(-1, LONG_KV_INT8_GROUP), dst_type=torch.int8)
    scale_bytes = scales.reshape(row_count, width // LONG_KV_INT8_GROUP).view(torch.int8)
    return torch.cat((values.view_as(rows), scale_bytes), dim=-1)
