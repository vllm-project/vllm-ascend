from typing import Optional, Tuple, Union

import torch
from vllm.model_executor.layers.layernorm import RMSNorm


def forward_oot(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    import torch_npu

    if residual is not None:
        x, _, residual = torch_npu.npu_add_rms_norm(x, residual, self.weight,
                                                    self.variance_epsilon)
        return x, residual

    x, residual = torch_npu.npu_rms_norm(x, self.weight, self.variance_epsilon)
    return x


RMSNorm.forward_oot = forward_oot
