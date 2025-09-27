# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This file extends the functionality of linear operations by encapsulating custom
communication groups and forward functions into classes (linear ops).

Current class inheritance structure:
CustomTensorParallelOp
├── CustomColumnParallelOp
│   ├── MLPColumnParallelOp
│   ├── DenseOptimMergedColumnParallelOp
│   └── DenseOptimQKVParallelOp
└── CustomRowParallelOp
    ├── MLPRowParallelOp
    ├── OProjRowParallelOp
    ├── MatmulAllreduceRowParallelOp
    └── DenseOptimRowParallelOp

How to extend a new linear op? Taking column parallel op as an example:
1. Inherit from CustomColumnParallelOp and create a new class MyColumnParallelOp
2. [Optional] The default communication group is the TP group. If a custom communication group is needed, override the comm_group method
3. Override the apply method according to requirements, which will replace the original linear.forward
4. Add selection logic for MyColumnParallelOp in the get_column_parallel_op method, typically based on prefix and configuration judgments
Row parallel op follows a similar approach - inherit from RowColumnParallelOp and register the new class in get_row_parallel_op.
"""

from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch_npu
from torch.distributed import ProcessGroup
from torch.nn.parameter import Parameter
from vllm.distributed import split_tensor_along_last_dim
from vllm.distributed.parallel_state import get_tp_group

from vllm_ascend.distributed.parallel_state import (get_mlp_tp_group,
                                                    get_otp_group)
from vllm_ascend.utils import (dense_optim_enable, enable_sp,
                               matmul_allreduce_enable, mlp_tp_enable,
                               oproj_tp_enable)


class CustomTensorParallelOp:

    def __init__(self, layer):
        self.layer = layer
        self.bias = None
        self.skip_bias_add = None
        self.return_bias = None
        self.quant_method = None

    # Custom communication group, while determining weight sharding
    @property
    def comm_group(self):
        return get_tp_group()

    @property
    def tp_rank(self):
        return self.comm_group.rank_in_group

    @property
    def tp_size(self):
        return self.comm_group.world_size

    # Update the attributes required by apply(), obtaining them from the layer.
    # Call this after the layer completes its initialization, specifically at the end of layer.init().
    def update_attrs(self):
        if hasattr(self.layer, "bias"):
            self.bias = self.layer.bias
        self.skip_bias_add = self.layer.skip_bias_add
        self.return_bias = self.layer.return_bias
        self.quant_method = self.layer.quant_method
        self.prefix = self.layer.prefix

    def apply_impl(self, input_):
        raise NotImplementedError

    # Replace layer.forward to customize the layer computation process.
    def apply(self, input_):
        output, output_bias = self.apply_impl(input_)
        if not self.return_bias:
            return output
        return output, output_bias


class CustomColumnParallelOp(CustomTensorParallelOp):

    def __init__(self, layer):
        super().__init__(layer)
        self.gather_output = None

    def update_attrs(self):
        super().update_attrs()
        self.gather_output = self.layer.gather_output


class CustomRowParallelOp(CustomTensorParallelOp):

    def __init__(self, layer):
        super().__init__(layer)
        self.reduce_results = None
        self.input_is_parallel = None
        self.input_size_per_partition = None

    def update_attrs(self):
        super().update_attrs()
        self.input_is_parallel = self.layer.input_is_parallel
        self.reduce_results = self.layer.reduce_results
        self.input_size_per_partition = self.layer.input_size_per_partition

    def apply(self, input_):
        output, output_bias = self.apply_impl(input_)
        if dense_optim_enable():
            torch.ops.vllm.maybe_prefetch_mlp_gate_up_proj(output, self.prefix)
        if not self.return_bias:
            return output
        return output, output_bias


class MLPColumnParallelOp(CustomColumnParallelOp):

    def __init__(self, layer):
        super().__init__(layer)

    @property
    def comm_group(self):
        return get_mlp_tp_group()

    def apply_impl(
        self,
        input_: torch.Tensor,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        bias = self.bias if not self.skip_bias_add else None
        # Matrix multiply.
        assert self.quant_method is not None
        input_parallel = self.comm_group.all_gather(input_, 0)
        output = self.quant_method.apply(self.layer, input_parallel, bias)

        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class SequenceMergedColumnParallelOp(CustomColumnParallelOp):

    def apply_impl(
        self, input_: torch.Tensor
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        """Linear layer with column parallelism.

        Implemented multiple optimization projects for dense models, such as FlashComm and
        communication-computation fusion.
        """

        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        assert self.quant_method is not None

        input_ = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(input_, True)
        output_parallel = self.quant_method.apply(self.layer, input_, bias)

        if self.gather_output:
            # All-gather across the partitions.
            output = self.comm_group.all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class SequenceQKVParallelOp(CustomColumnParallelOp):

    def __init__(self, layer, prefix):
        super().__init__(layer)
        self.prefix = prefix

    def apply_impl(
        self, input_: torch.Tensor
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        """Linear layer with column parallelism.

        Implemented multiple optimization projects for dense models, such as FlashComm and
        communication-computation fusion.
        """

        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        assert self.quant_method is not None

        layer_num = self.prefix.split('.')[2]

        input_ = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            input_, layer_num != '0')
        output_parallel = self.quant_method.apply(self.layer, input_, bias)

        if self.gather_output:
            # All-gather across the partitions.
            output = self.comm_group.all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class MLPRowParallelOp(CustomRowParallelOp):

    def __init__(self, layer):
        super().__init__(layer)

    @property
    def comm_group(self):
        return get_mlp_tp_group()

    def apply_impl(
        self, input_: torch.Tensor
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        if self.input_is_parallel:
            input_parallel = input_
        else:
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[self.tp_rank].contiguous()

        assert self.quant_method is not None
        bias_ = None if (self.tp_rank > 0
                         or self.skip_bias_add) else self.layer.bias
        output_parallel = self.quant_method.apply(self.layer,
                                                  input_parallel,
                                                  bias=bias_)
        output = self.comm_group.reduce_scatter(output_parallel, 0)

        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class OProjRowParallelOp(CustomRowParallelOp):

    def __init__(self, layer):
        super().__init__(layer)

    @property
    def comm_group(self):
        return get_otp_group()

    def apply_impl(
        self,
        input_: torch.Tensor,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:

        if self.input_is_parallel:
            input_parallel = input_
        else:
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[self.tp_rank].contiguous()

        # Prepare tensors for all-to-all communication
        local_batch_size = input_parallel.size(0)
        chunk_size = self.input_size_per_partition
        total_batch_size = local_batch_size * self.tp_size

        # Reshape tensor for efficient cross-device transfer:
        # [batch, dim] -> [tp_size, batch, chunk] -> flattened
        send_buf = (input_parallel.reshape(-1,
                                           self.tp_size, chunk_size).transpose(
                                               0, 1).contiguous().view(-1))

        # Create receive buffer
        recv_buf = torch.empty(total_batch_size * chunk_size,
                               dtype=input_parallel.dtype,
                               device=input_parallel.device)

        # Perform all-to-all communication
        dist.all_to_all_single(recv_buf,
                               send_buf,
                               group=self.comm_group.device_group)
        input_parallel = recv_buf.view(total_batch_size, chunk_size)

        # Only fuse bias add for rank 0 to avoid duplicate bias addition in TP>1
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        assert self.quant_method is not None
        output_parallel = self.quant_method.apply(self.layer,
                                                  input_parallel,
                                                  bias=bias_)

        # otp-specific: Combine partial results across devices
        output = self.comm_group.reduce_scatter(output_parallel, dim=0)
        output = output.view(input_.shape[0], self.layer.output_size)

        # Handle bias return based on configuration
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def update_attrs(self):
        super().update_attrs()
        self.input_is_parallel = self.layer.input_is_parallel
        self.input_size_per_partition = self.layer.input_size_per_partition


class MatmulAllreduceRowParallelOp(CustomRowParallelOp):
    _HCOMM_INFO = None

    def __init__(self, layer):
        super().__init__(layer)
        self.hcomm_info = self.get_hcomm_info(self.comm_group.device_group)

    def apply_impl(
        self, input_: torch.Tensor
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        if self.input_is_parallel:
            input_parallel = input_
        else:
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[self.tp_rank].contiguous()
        """Calculate the output tensor of forward by considering
        fusing communication and computation."""
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        if self.reduce_results and self.tp_size > 1:
            output = torch_npu.npu_mm_all_reduce_base(input_parallel,
                                                      self.weight_t,
                                                      self.hcomm_info,
                                                      bias=bias_)
        else:
            assert self.quant_method is not None
            output = self.quant_method.apply(self.layer,
                                             input_parallel,
                                             bias=bias_)

        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    @classmethod
    def get_hcomm_info(cls, group: ProcessGroup) -> str:
        """Get the HCCL communication information for the given group."""
        if cls._HCOMM_INFO is not None:
            return cls._HCOMM_INFO

        rank = torch.distributed.get_rank(group)
        if torch.__version__ > "2.0":
            global_rank = torch.distributed.get_global_rank(group, rank)
            cls._HCOMM_INFO = group._get_backend(
                torch.device("npu")).get_hccl_comm_name(global_rank)
        else:
            cls._HCOMM_INFO = group.get_hccl_comm_name(rank)
        return cls._HCOMM_INFO

    def update_attrs(self):
        super().update_attrs()
        self.weight_t = self.layer.weight.t()


class SequenceRowParallelOp(CustomRowParallelOp):

    def __init__(self, layer, prefix):
        super().__init__(layer)
        self.prefix = prefix

    def apply_impl(
        self, input_: torch.Tensor
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        """Linear layer with column parallelism.

        Implemented multiple optimization projects for dense models, such as FlashComm and
        communication-computation fusion.
        """

        if self.input_is_parallel:
            input_parallel = input_
        else:
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[self.tp_rank].contiguous()

        assert self.quant_method is not None
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias

        if self.tp_size == 1 or not self.reduce_results:
            output = self.quant_method.apply(self.layer,
                                             input_parallel,
                                             bias=bias_)
        else:
            output_parallel = self.quant_method.apply(self.layer,
                                                      input_parallel,
                                                      bias=bias_)
            output = torch.ops.vllm.maybe_pad_and_reduce(output_parallel)

        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def update_attrs(self):
        super().update_attrs()
        self.input_is_parallel = self.layer.input_is_parallel
        self.reduce_results = self.layer.reduce_results


def get_column_parallel_op(
    disable_tp, prefix, layer
) -> Tuple[Optional[Union[MLPColumnParallelOp, SequenceMergedColumnParallelOp,
                          SequenceQKVParallelOp]], int, int]:
    if disable_tp:
        return None, 0, 1

    custom_op: Optional[Union[
        MLPColumnParallelOp,
        SequenceMergedColumnParallelOp,
        SequenceQKVParallelOp,
    ]] = None
    if "gate_up_proj" in prefix and mlp_tp_enable():
        custom_op = MLPColumnParallelOp(layer)
    elif "gate_up_proj" in prefix and enable_sp():
        custom_op = SequenceMergedColumnParallelOp(layer)
    elif enable_sp():
        custom_op = SequenceQKVParallelOp(layer, prefix)

    if custom_op is not None:
        return custom_op, custom_op.tp_rank, custom_op.tp_size

    return None, get_tp_group().rank_in_group, get_tp_group().world_size


def get_row_parallel_op(
    disable_tp, prefix, layer
) -> Tuple[Optional[Union[MLPRowParallelOp, OProjRowParallelOp,
                          MatmulAllreduceRowParallelOp,
                          SequenceRowParallelOp]], int, int]:
    if disable_tp:
        return None, 0, 1

    custom_op: Optional[Union[MLPRowParallelOp, OProjRowParallelOp,
                              MatmulAllreduceRowParallelOp,
                              SequenceRowParallelOp]] = None
    if "down_proj" in prefix and mlp_tp_enable():
        custom_op = MLPRowParallelOp(layer)
    elif "o_proj" in prefix and oproj_tp_enable():
        custom_op = OProjRowParallelOp(layer)
    elif matmul_allreduce_enable():
        custom_op = MatmulAllreduceRowParallelOp(layer)
    elif enable_sp():
        custom_op = SequenceRowParallelOp(layer, prefix)

    if custom_op is not None:
        return custom_op, custom_op.tp_rank, custom_op.tp_size

    return None, get_tp_group().rank_in_group, get_tp_group().world_size
