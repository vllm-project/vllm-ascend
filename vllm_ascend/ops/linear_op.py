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
CustomLinearOp
├── CustomColumnParallelOp
│   ├── MLPColumnParallelOp
│   ├── SequenceColumnParallelOp
└── CustomRowParallelOp
│   ├── MLPRowParallelOp
│   ├── OProjRowParallelOp
|   ├── Flashcomm2OProjRowParallelOp
│   ├── MatmulAllreduceRowParallelOp
│   └── SequenceRowParallelOp
└── CustomReplicatedOp
How to extend a new linear op? Taking column parallel op as an example:
1. Inherit from CustomColumnParallelOp and create a new class MyColumnParallelOp
2. [Optional] The default communication group is the TP group. If a custom communication group is needed, override the comm_group method
3. Override the apply method according to requirements, which will replace the original linear.forward
4. Add selection logic for MyColumnParallelOp in the get_column_parallel_op method, typically based on prefix and configuration judgments
Row parallel op follows a similar approach - inherit from RowColumnParallelOp and register the new class in get_row_parallel_op.
"""

from typing import Optional, Union

import torch
import torch.distributed as dist
import torch_npu
from torch import nn
from torch.distributed import ProcessGroup
from torch.nn.parameter import Parameter
from vllm.distributed import split_tensor_along_last_dim
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.parallel_state import (get_flashcomm2_odp_group,
                                                    get_flashcomm2_otp_group,
                                                    get_mlp_tp_group,
                                                    get_otp_group)
from vllm_ascend.quantization.w8a8 import AscendW8A8LinearMethod
from vllm_ascend.utils import (dense_optim_enable, enable_sp,
                               flashcomm2_enable,
                               get_flashcomm2_reorgnized_batch_ids,
                               matmul_allreduce_enable, mlp_tp_enable,
                               oproj_tp_enable, shared_expert_dp_enabled)


class CustomLinearOp:

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


class CustomColumnParallelOp(CustomLinearOp):

    def __init__(self, layer):
        super().__init__(layer)
        self.gather_output = None

    def update_attrs(self):
        super().update_attrs()
        self.gather_output = self.layer.gather_output


class CustomRowParallelOp(CustomLinearOp):

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


class CustomReplicatedOp(CustomLinearOp):

    def apply_impl(self, input_):
        bias = self.bias if not self.skip_bias_add else None
        assert self.quant_method is not None

        output = self.quant_method.apply(self.layer, input_, bias)
        output_bias = self.bias if self.skip_bias_add else None

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


class Flashcomm2OProjRowParallelOp(CustomRowParallelOp):

    def __init__(self, layer):
        super().__init__(layer)
        self.odp_group = get_flashcomm2_odp_group()
        self.odp_size = self.odp_group.world_size
        self.reorgnized_batch_ids = get_flashcomm2_reorgnized_batch_ids(
            get_tp_group().world_size)
        self.group_indices = torch.tensor(self.reorgnized_batch_ids).npu()
        self._quant_comm_config = {}

    @property
    def comm_group(self):
        return get_flashcomm2_otp_group()

    @property
    def tp_rank(self):
        if get_ascend_config().flashcomm2_oproj_tensor_parallel_size == 1:
            return 0
        return self.comm_group.rank_in_group

    @property
    def tp_size(self):
        if get_ascend_config().flashcomm2_oproj_tensor_parallel_size == 1:
            return 1
        return self.comm_group.world_size

    def apply_impl(
        self,
        input_: torch.Tensor,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        """Linear layer for Flashcomm2.
            Input.ahspe = [batchsize*seqlength, headnum*headdim/TP]
            Output.shape = [(batchsize*seqlength+padsize)/TP, hiddensize]
        """
        # Handle input parallelism - split or use as-is
        if self.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = self.tp_rank
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[tp_rank].contiguous()

        # padding for all-to-all
        forward_context = get_forward_context()
        num_padding_tokens = forward_context.pad_size
        if num_padding_tokens > 0:
            input_parallel = nn.functional.pad(input_parallel,
                                               (0, 0, 0, num_padding_tokens))

        def otp_maybe_quant_comm(x):

            # Reorganize the tensor so that the batch id and rank id correspond to each other.
            chunk_num = len(self.reorgnized_batch_ids) * len(
                self.reorgnized_batch_ids[0])
            batch_size = x.size(0)

            assert batch_size % chunk_num == 0, f"Batch_size({batch_size}) must be divisible by chunk_num({chunk_num})"

            batch_size_per_chunk = batch_size // chunk_num
            # Indices of reorganized tensor
            chunked = x.view(chunk_num, batch_size_per_chunk, x.shape[1])
            reorganized_chunks = chunked[self.group_indices]
            send_buf = reorganized_chunks.flatten(1, 2)

            # all-to-all operation parameters
            all2all_tp_size = self.odp_size
            local_intermediate_size = x.size(1)
            chunk_size = x.size(0) // all2all_tp_size
            total_intermediate_size = local_intermediate_size * all2all_tp_size

            # Create receive buffer
            recv_buf = torch.empty(total_intermediate_size * chunk_size,
                                   dtype=x.dtype,
                                   device=x.device)

            # Perform all-to-all communication
            dist.all_to_all_single(recv_buf,
                                   send_buf,
                                   group=self.odp_group.device_group)

            return recv_buf.view(all2all_tp_size, chunk_size,
                                 -1).transpose(0, 1).reshape(chunk_size, -1)

        if not hasattr(self, '_quant_comm_config'):
            self._quant_comm_config = {}
        self._quant_comm_config['communication_fn'] = otp_maybe_quant_comm

        # Matrix multiply.
        assert self.quant_method is not None

        actual_quant_method = getattr(self.quant_method, 'quant_method',
                                      self.quant_method)

        if not isinstance(actual_quant_method, AscendW8A8LinearMethod):
            # Check if w8a8 quantization is enabled. If not, communicate immediately.
            otp_maybe_quant_comm(input_parallel)

        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias

        output_parallel = self.quant_method.apply(self.layer,
                                                  input_parallel,
                                                  bias=bias_)
        # output_parallel shape: [bs/(TP/flashcomm2_otp_size), hiddenstate]
        if self.tp_size > 1:
            # flashcomm2 with reduce-scatter
            output = self.comm_group.reduce_scatter(output_parallel, dim=0)
        else:
            output = output_parallel

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


class SequenceColumnParallelOp(CustomColumnParallelOp):

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


class SequenceRowParallelOp(CustomRowParallelOp):

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
            output = torch.ops.vllm.matmul_and_reduce(input_parallel,
                                                      self.prefix)

        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def matmul_and_reduce(self, input_parallel: torch.Tensor,
                          bias_: Optional[Parameter]) -> torch.Tensor:
        assert self.quant_method is not None
        output_parallel = self.quant_method.apply(self.layer,
                                                  input_parallel,
                                                  bias=bias_)
        from vllm_ascend.ops.register_custom_ops import \
            _maybe_pad_and_reduce_impl
        output = _maybe_pad_and_reduce_impl(output_parallel)
        return output

    def update_attrs(self):
        super().update_attrs()
        self.input_is_parallel = self.layer.input_is_parallel
        self.reduce_results = self.layer.reduce_results


def _get_column_parallel_op(
        prefix, layer
) -> Optional[Union[MLPColumnParallelOp, SequenceColumnParallelOp]]:
    if mlp_tp_enable() and "gate_up_proj" in prefix:
        return MLPColumnParallelOp(layer)
    if enable_sp() or flashcomm2_enable():
        if "shared_expert" in prefix:
            return None
        if "gate_up_proj" in prefix:
            return SequenceColumnParallelOp(layer)
        if "in_proj" in prefix:
            return SequenceColumnParallelOp(layer)
        if "qkv_proj" in prefix or "conv1d" in prefix:
            return SequenceColumnParallelOp(layer)

    return None


def _get_row_parallel_op(
    prefix, layer
) -> Optional[Union[MLPRowParallelOp, OProjRowParallelOp,
                    Flashcomm2OProjRowParallelOp, MatmulAllreduceRowParallelOp,
                    SequenceRowParallelOp]]:
    if "down_proj" in prefix and mlp_tp_enable():
        return MLPRowParallelOp(layer)
    if "o_proj" in prefix and oproj_tp_enable():
        return OProjRowParallelOp(layer)
    if matmul_allreduce_enable():
        return MatmulAllreduceRowParallelOp(layer)
    if flashcomm2_enable():
        if "shared_expert" in prefix:
            return None
        if "o_proj" in prefix or "out_proj" in prefix:
            return Flashcomm2OProjRowParallelOp(layer)
        if "down_proj" in prefix:
            return SequenceRowParallelOp(layer)
    if enable_sp():
        if "shared_expert" in prefix:
            return None
        if "o_proj" in prefix or "out_proj" in prefix or "down_proj" in prefix:
            return SequenceRowParallelOp(layer)

    return None


def get_parallel_op(disable_tp, prefix, layer, direct):
    if disable_tp or ("shared_experts" in prefix
                      and shared_expert_dp_enabled()):
        return None, 0, 1
    custom_op: Optional[Union[MLPColumnParallelOp, SequenceColumnParallelOp,
                              MLPRowParallelOp, OProjRowParallelOp,
                              Flashcomm2OProjRowParallelOp,
                              MatmulAllreduceRowParallelOp,
                              SequenceRowParallelOp]] = None
    if direct == "row":
        custom_op = _get_row_parallel_op(prefix, layer)

    if direct == "column":
        custom_op = _get_column_parallel_op(prefix, layer)

    if custom_op is not None:
        return custom_op, custom_op.tp_rank, custom_op.tp_size

    return None, get_tp_group().rank_in_group, get_tp_group().world_size


def get_replicated_op(disable_tp, prefix,
                      layer) -> Optional[Union[CustomReplicatedOp]]:
    if disable_tp:
        return None

    return CustomReplicatedOp(layer)
