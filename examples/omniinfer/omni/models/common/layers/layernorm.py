# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Custom normalization layers."""
import torch
import torch_npu
from typing import Optional, Union, Any
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNormGPU
from vllm.distributed import get_tp_group
from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank

from omni.models.common.config.model_config import model_extra_config


class RMSNorm(RMSNormGPU):
    def forward(
            self,
            x: torch.Tensor,
            residual: Optional[torch.Tensor] = None,
            quant_symbol: bool = False,
    ) -> Union[tuple[dict[str, Any], Any], Any]:
        if residual is not None:
            x, _, residual = torch_npu.npu_add_rms_norm(x, residual, self.weight, self.variance_epsilon)
            if quant_symbol:
                x_int8, pertoken_scale = torch_npu.npu_dynamic_quant(x)
                x = {"x_int8": x_int8, "pertoken_scale": pertoken_scale}
            return x, residual

        return torch_npu.npu_rms_norm(
            x,
            self.weight.data,
            self.variance_epsilon,
        )[0]

class RMSNormFlashComm(RMSNorm):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
        module_name: Optional[str] = "",
    ) -> None:
        super().__init__(hidden_size, eps, var_hidden_size)
        self.module_name = module_name
        self.tp_size = get_tensor_model_parallel_world_size() # get tp size for each module
        self.tp_rank = get_tensor_model_parallel_rank() # get tp rank for each module

    def forward(
            self,
            x: torch.Tensor,
            residual: Optional[torch.Tensor] = None,
            y_transform: str = "",
    ) -> Union[tuple[dict[str, Any], Any], Any]:
        if residual is not None:
            x, _, residual = torch_npu.npu_add_rms_norm(x, residual, self.weight, self.variance_epsilon)
            if y_transform == "AG":
                x = get_tp_group().all_gather(x, dim=0)
            return x, residual
        else:
            return torch_npu.npu_rms_norm(
                x,
                self.weight.data,
                self.variance_epsilon,
            )[0]

    def forward_with_residual(
            self,
            x: torch.Tensor,
            residual: Optional[torch.Tensor] = None,
            y_transform: str = "",
    ) -> Union[tuple[dict[str, Any], Any], Any]:
        if residual is not None:
            x, _, residual = torch_npu.npu_add_rms_norm(x, residual, self.weight, self.variance_epsilon)
            if y_transform == "AG":
                x = get_tp_group().all_gather(x, dim=0)
            return x, residual
        else:
            residual = x
            x = torch_npu.npu_rms_norm(
                x,
                self.weight.data,
                self.variance_epsilon,
            )[0]
            return x, residual