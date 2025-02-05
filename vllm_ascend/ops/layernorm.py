from typing import Optional, Tuple, Union

import torch
from vllm.model_executor.layers.layernorm import RMSNorm

try:
    from mindie_turbo import RMSNormWithAntiOutlier
except:
    pass


def forward_oot(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if hasattr(self, "module"):
        return self.module.forward_anti_outlier(x, residual)
    
    import torch_npu

    if residual is not None:
        x, _, residual = torch_npu.npu_add_rms_norm(x, residual, self.weight,
                                                    self.variance_epsilon)
        return x, residual

    x, residual = torch_npu.npu_rms_norm(x, self.weight, self.variance_epsilon)
    return x


def enable_rmsnorm_with_antioutlier():
    def init(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
        has_weight: bool = True,
    ) -> None:
        super(RMSNorm, self).__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.variance_size_override = (None if var_hidden_size == hidden_size
                                       else var_hidden_size)
        self.has_weight = has_weight

        self.weight = torch.ones(hidden_size)
        if self.has_weight:
            self.weight = torch.nn.Parameter(self.weight)

        self.module = RMSNormWithAntiOutlier(self.hidden_size)
    
    RMSNorm.__init__ = init


RMSNorm.forward_oot = forward_oot
