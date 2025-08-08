"""
Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
This file is a part of the vllm-ascend project.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Optional, Union

import torch
import torch_npu
import vllm
from torch.distributed import ProcessGroup
from torch.nn.parameter import Parameter
from vllm.distributed import (get_tensor_model_parallel_rank,
                              split_tensor_along_last_dim)
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import logger
from vllm.model_executor.layers.linear import RowParallelLinear

from vllm_ascend import envs

_HCOMM_INFO = None


class AscendRowParallelLinear(RowParallelLinear):
    """
    AscendRowParallelLinear is a custom implementation of RowParallelLinear
    that overrides the forward method to handle Ascend-specific operations.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the AscendRowParallelLinear layer.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        tp_group = get_tp_group().device_group
        hcomm_info = self.get_hcomm_info(tp_group)
        self.hcomm_info = hcomm_info
        super().__init__(*args, **kwargs)
        self.weight_t = self.weight.t()

    @staticmethod
    def get_hcomm_info(group: ProcessGroup) -> str:
        """Get the HCCL communication information for the given group.

        Args:
            group (ProcessGroup): The process group for which to get the HCCL communication info.

        Returns:
            str: The HCCL communication name for the given group.
        """
        global _HCOMM_INFO
        if _HCOMM_INFO is not None:
            return _HCOMM_INFO

        rank = torch.distributed.get_rank(group)
        if torch.__version__ > "2.0":
            global_rank = torch.distributed.get_global_rank(group, rank)
            _HCOMM_INFO = group._get_backend(
                torch.device("npu")).get_hccl_comm_name(global_rank)

        else:
            _HCOMM_INFO = group.get_hccl_comm_name(rank)
        return _HCOMM_INFO

    def forward(
        self, input_: torch.Tensor
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        """Forward pass for the AscendRowParallelLinear layer.

        Args:
            input_ (torch.Tensor): the input tensor to the layer.

        Returns:
            Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]: 
                The output tensor after applying the linear transformation,
                and optionally the bias if `return_bias` is True.
        """
        input_parallel = self.calc_input(input_)

        # Matrix multiply.
        assert self.quant_method is not None
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        output = self.calc_output(input_parallel)

        output_bias = self.bias if self.skip_bias_add else None

        if not self.return_bias:
            return output
        return output, output_bias

    def calc_input(self, input_: torch.Tensor) -> torch.Tensor:
        """Calculate the input tensor for parallel processing.

        Args:
            input_ (torch.Tensor): the input tensor to be processed.

        Returns:
            torch.Tensor: The input tensor split along the last dimension
            for tensor model parallelism, or the original input if not parallel.
        """
        if self.input_is_parallel:
            return input_
        tp_rank = get_tensor_model_parallel_rank()
        splitted_input = split_tensor_along_last_dim(
            input_, num_partitions=self.tp_size)
        return splitted_input[tp_rank].contiguous()

    def calc_output(self, input_parallel: torch.Tensor) -> torch.Tensor:
        """Calculate the output tensor of forward by considering
        fusing communication and computation.

        Args:
            input_parallel (_type_): the input tensor to be processed in parallel.

        Returns:
             torch.Tensor: the output tensor after applying the linear transformation
             and optionally handle communication between tensor model parallel ranks.
        """
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        if self.reduce_results and self.tp_size > 1:
            output = torch_npu.npu_mm_all_reduce_base(input_parallel,
                                                      self.weight_t,
                                                      self.hcomm_info,
                                                      bias=bias_)
        else:
            output = self.quant_method.apply(self, input_parallel, bias=bias_)
        return output


if envs.VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE:
    logger.info("AscendRowParallelLinear: Matmul all-reduce is enabled. ")
    vllm.model_executor.layers.linear.RowParallelLinear = AscendRowParallelLinear
