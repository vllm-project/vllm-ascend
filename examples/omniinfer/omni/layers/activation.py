# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Union, Dict, Any

import torch
import torch_npu
import torch.nn as nn

class SiluAndMul(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Union[dict[str, Any], torch.Tensor], quant_symbol: bool = False) -> Union[dict[str, Any], Any]:
        if quant_symbol and isinstance(x, Dict):
            kwargs = {
                        "weight_scale": x.get('out_scale').to(torch.float32),
                        "quant_scale": x.get('in_scale', None),  #.to(torch.float32), for smooth quant scale
                        "activation_scale": x.get('pertoken_scale', None),  # or activation_scale if error
                        "bias": None,
                        "quant_offset": None,
                        "group_index": None,
                        "activate_left": True,
                        "quant_mode": 1
                    }
            h, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
                x.get('x_int8'), **kwargs)
            return {"x_int8": h, "pertoken_scale": pertoken_scale}
        return torch_npu.npu_swiglu(x)

