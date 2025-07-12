import itertools
from abc import abstractmethod
from typing import Any, Literal, Optional, Union

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter, UninitializedParameter

from vllm.distributed import (divide, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              split_tensor_along_last_dim,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce,
                              tensor_model_parallel_reduce_scatter)
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
# yapf: disable
from vllm.model_executor.parameter import (BasevLLMParameter,
                                           BlockQuantScaleParameter,
                                           PackedColumnParameter,
                                           PackedvLLMParameter,
                                           PerTensorScaleParameter,
                                           RowvLLMParameter)
# yapf: enable
from vllm.model_executor.utils import set_weight_attrs

from vllm.model_executor.layers.linear import RowParallelLinear
import torch.nn.functional as F
from vllm.forward_context import get_forward_context

logger = init_logger(__name__)


class MaybeScatterRowParallelLinear(RowParallelLinear):
    # Replace all_reduce with reduce_scatter in flashcomm_v1 situation
    
    def forward(
        self, input_, use_scatter, pad_size,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        if self.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[tp_rank].contiguous()

        # Matrix multiply.
        assert self.quant_method is not None
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = self.quant_method.apply(self,
                                                  input_parallel,
                                                  bias=bias_)
        if self.reduce_results and self.tp_size > 1:
            if use_scatter:
                if pad_size > 0:
                    output_parallel = F.pad(output_parallel, (0, 0, 0, pad_size))
                output = tensor_model_parallel_reduce_scatter(output_parallel, 0)
            else:
                output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None

        if not self.return_bias:
            return output
        return output, output_bias
    