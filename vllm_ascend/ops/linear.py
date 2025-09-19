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
import torch.distributed as dist
import torch.nn as nn
import torch_npu
from torch.distributed import ProcessGroup
from torch.nn.parameter import Parameter
from vllm.distributed import (divide, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              split_tensor_along_last_dim,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce,
                              tensor_model_parallel_reduce_scatter)
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import (  # noqa
    WEIGHT_LOADER_V2_SUPPORTED, ColumnParallelLinear, LinearBase,
    MergedColumnParallelLinear, QKVParallelLinear, QuantizeMethodBase,
    RowParallelLinear, UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization.base_config import \
    QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs

from vllm_ascend.distributed.parallel_state import (get_mlp_tp_group,
                                                    get_otp_group,
                                                    is_sp_enabled)
from vllm_ascend.utils import (dense_optim_enable, matmul_allreduce_enable,
                               mlp_tp_enable, oproj_tp_enable)

_HCOMM_INFO = None


class AscendColumnParallelLinear(ColumnParallelLinear):
    """Linear layer with column parallelism.

    Use the MLP tensor parallelism group in the MLP module,
    and the original TP group in other modules.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        output_sizes: Optional[list[int]] = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        self.comm_group = None
        self.enable_sp = is_sp_enabled()
        if prefix.find("gate_up_proj") != -1 and mlp_tp_enable(
        ) and not self.enable_sp:
            self.comm_group = get_mlp_tp_group()
        else:
            self.comm_group = get_tp_group()

        self.tp_size = self.comm_group.world_size
        self.tp_rank = self.comm_group.rank_in_group

        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]
        # If QKV or MergedColumn, use output size of each partition.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [
                divide(output_size, self.tp_size)
                for output_size in self.output_sizes
            ]
        AscendLinearBase.__init__(self,
                                  input_size,
                                  output_size,
                                  skip_bias_add,
                                  params_dtype,
                                  quant_config,
                                  prefix,
                                  return_bias=return_bias,
                                  disable_tp=disable_tp)

        self.gather_output = gather_output

        if output_sizes is None:
            output_sizes = [output_size]

        assert self.quant_method is not None
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2 if self.quant_method.__class__.__name__
                in WEIGHT_LOADER_V2_SUPPORTED else self.weight_loader))
        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size_per_partition,
                            dtype=params_dtype))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)


class AscendRowParallelLinear(RowParallelLinear):
    """Linear layer with row parallelism.
    Use the MLP tensor parallelism group in the MLP module,
    and the original TP group in other modules.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        self.enable_sp = is_sp_enabled()
        if prefix.find(
                "down_proj") != -1 and mlp_tp_enable() and not self.enable_sp:
            comm_group = get_mlp_tp_group()
            self.forward_type = "mlp_tp"
        elif prefix.find("o_proj") != -1 and oproj_tp_enable():
            comm_group = get_otp_group()
            self.forward_type = "oproj_tp"
        elif matmul_allreduce_enable():
            comm_group = get_tp_group()
            self.forward_type = "matmul_allreduce"
            self.hcomm_info = self.get_hcomm_info(comm_group.device_group)
        elif dense_optim_enable():
            comm_group = get_tp_group()
            self.forward_type = "dense_optim"
        elif self.enable_sp:
            comm_group = get_tp_group()
            self.forward_type = "mlp_sp"
        else:
            comm_group = get_tp_group()
            self.forward_type = "normal"
        self.comm_group = comm_group

        # TODO: check for disable_tp
        self.tp_size = self.comm_group.world_size
        self.tp_rank = self.comm_group.rank_in_group

        # Divide the weight matrix along the first dimension.
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size
        self.output_partition_sizes = [output_size]

        AscendLinearBase.__init__(self,
                                  input_size,
                                  output_size,
                                  skip_bias_add,
                                  params_dtype,
                                  quant_config,
                                  prefix,
                                  return_bias=return_bias,
                                  disable_tp=disable_tp)

        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        assert self.quant_method is not None
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2 if self.quant_method.__class__.__name__
                in WEIGHT_LOADER_V2_SUPPORTED else self.weight_loader))
        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError("When not reduce the results, adding bias to the "
                             "results can lead to incorrect results")

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size, dtype=params_dtype))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)

        if matmul_allreduce_enable():
            self.weight_t = self.weight.t()

    @staticmethod
    def get_hcomm_info(group: ProcessGroup) -> str:
        """Get the HCCL communication information for the given group."""
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
        self,
        input_,
        is_prefill: bool = True,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        # Choose different forward function according to the type of TP group
        if self.forward_type == "oproj_tp":
            return self._forward_oproj_tp(input_)
        elif self.forward_type == "mlp_tp":
            return self._forward_mlp_tp(input_)
        elif self.forward_type == "matmul_allreduce":
            return self._forward_matmul_allreduce(input_)
        elif self.forward_type == "dense_optim":
            return self._forward_dense_optim(input_)
        elif self.forward_type == "mlp_sp":
            return self._forward_mlp_sp(input_)
        else:
            return super().forward(input_)

    # enable sp tensor parallel
    def _forward_mlp_sp(self, input_: torch.Tensor) -> torch.Tensor:
        forward_context = get_forward_context()
        self.enable_sp = is_sp_enabled()
        attn_metadata = forward_context.attn_metadata
        is_prefill = attn_metadata.num_prefills if attn_metadata else False
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
        if self.reduce_results and self.enable_sp and is_prefill:
            sp_size = get_tensor_model_parallel_world_size()
            if not isinstance(input_, tuple):
                original_len = input_.shape[0]
            else:
                original_len = input_[0].shape[0]
            reminder = original_len % sp_size
            if reminder != 0:
                padding_len = sp_size - reminder
                output_parallel = nn.functional.pad(output_parallel,
                                                    (0, 0, 0, padding_len),
                                                    mode='constant',
                                                    value=0)
            output = tensor_model_parallel_reduce_scatter(
                output_parallel.movedim(0, -1)).movedim(-1, 0)
        elif self.reduce_results and self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None

        if not self.return_bias:
            return output
        return output, output_bias

    # enable custom MLP tensor parallel
    def _forward_mlp_tp(self, input_: torch.Tensor) -> torch.Tensor:

        if self.input_is_parallel:
            input_parallel = input_
        else:
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[self.tp_rank].contiguous()

        assert self.quant_method is not None
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = self.quant_method.apply(self,
                                                  input_parallel,
                                                  bias=bias_)
        output = self.comm_group.reduce_scatter(output_parallel, 0)

        output_bias = self.bias if self.skip_bias_add else None
        if not self.return_bias:
            return output
        return output, output_bias

    # enable custom Oproj tensor parallel
    def _forward_oproj_tp(
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
        output_parallel = self.quant_method.apply(self,
                                                  input_parallel,
                                                  bias=bias_)

        # otp-specific: Combine partial results across devices
        output = self.comm_group.reduce_scatter(output_parallel, dim=0)

        # Handle bias return based on configuration
        output_bias = self.bias if self.skip_bias_add else None
        if not self.return_bias:
            return output
        return output, output_bias

    def _forward_matmul_allreduce(
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
            output = self.quant_method.apply(self, input_parallel, bias=bias_)

        output_bias = self.bias if self.skip_bias_add else None
        if not self.return_bias:
            return output
        return output, output_bias

    def _forward_dense_optim(
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
            output = self.quant_method.apply(self, input_parallel, bias=bias_)
        else:
            output_parallel = self.quant_method.apply(self,
                                                      input_parallel,
                                                      bias=bias_)
            output = torch.ops.vllm.maybe_pad_and_reduce(output_parallel)
            torch.ops.vllm.maybe_prefetch_mlp_gate_up_proj(output, self.prefix)

        output_bias = self.bias if self.skip_bias_add else None

        if not self.return_bias:
            return output
        return output, output_bias


class AscendMergedColumnParallelLinear(MergedColumnParallelLinear):
    """Packed linear layers with column parallelism.

    Similar to ColumnParallelLinear, but the weight matrix is concatenated
    along the output dimension. When the weight matrix is loaded, the
    different partitions are sharded separately.

    Use the MLP tensor parallelism group in the MLP module,
    and the original TP group in other modules.
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        self.enable_sp = is_sp_enabled()
        if prefix.find("gate_up_proj") != -1 and mlp_tp_enable(
        ) and not self.enable_sp:
            comm_group = get_mlp_tp_group()
            self.forward_type = "mlp_tp"
        elif dense_optim_enable():
            comm_group = get_tp_group()
            self.forward_type = "dense_optim"
        elif self.enable_sp:
            comm_group = get_tp_group()
            self.forward_type = "mlp_sp"
        else:
            comm_group = get_tp_group()
            self.forward_type = "normal_tp"
        self.comm_group = comm_group
        # TODO: check for disable_tp
        self.tp_rank = comm_group.rank_in_group
        self.tp_size = comm_group.world_size

        self.output_sizes = output_sizes
        assert all(output_size % self.tp_size == 0
                   for output_size in output_sizes)
        AscendColumnParallelLinear.__init__(self,
                                            input_size=input_size,
                                            output_size=sum(output_sizes),
                                            bias=bias,
                                            gather_output=gather_output,
                                            skip_bias_add=skip_bias_add,
                                            params_dtype=params_dtype,
                                            quant_config=quant_config,
                                            prefix=prefix,
                                            return_bias=return_bias,
                                            disable_tp=disable_tp)

    def forward(
        self,
        input_,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        if self.forward_type == "mlp_tp":
            return self._forward_mlp_tp(input_)
        elif self.forward_type == "dense_optim":
            return self._forward_dense_optim(input_)
        elif self.forward_type == "mlp_sp":
            return self._forward_mlp_sp(input_)
        else:
            return super().forward(input_)

    def _forward_mlp_sp(
        self,
        input_: torch.Tensor,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        forward_context = get_forward_context()
        self.enable_sp = is_sp_enabled()
        attn_metadata = forward_context.attn_metadata
        is_prefill = attn_metadata.num_prefills if attn_metadata else False
        if self.enable_sp and is_prefill:
            input_ = get_tp_group().all_gather(input_, 0)
            input_ = input_[:attn_metadata.num_input_tokens]
        bias = self.bias if not self.skip_bias_add else None
        # self.global_batch_size = vllm_config.scheduler_config.max_num_seqs
        # Matrix multiply.
        assert self.quant_method is not None
        output_parallel = self.quant_method.apply(self, input_, bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None
        if not self.return_bias:
            return output
        return output, output_bias

    def _forward_mlp_tp(
        self,
        input_: torch.Tensor,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        bias = self.bias if not self.skip_bias_add else None
        # Matrix multiply.
        assert self.quant_method is not None
        input_parallel = get_mlp_tp_group().all_gather(input_, 0)
        output = self.quant_method.apply(self, input_parallel, bias)

        output_bias = self.bias if self.skip_bias_add else None
        if not self.return_bias:
            return output
        return output, output_bias

    def _forward_dense_optim(
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
        output_parallel = self.quant_method.apply(self, input_, bias)

        if self.gather_output:
            # All-gather across the partitions.
            output = self.comm_group.all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        if not self.return_bias:
            return output
        return output, output_bias


class AscendQKVParallelLinear(QKVParallelLinear):
    """Linear layers for the attention's QKV transformation.

    Linear layers for the linear transformation of the query, key, and value
    vectors in the attention layer. The weight matrix is concatenated along
    the output dimension. The layer is parallelized along the head dimension.
    When the number of key/value heads is smaller than the number of query
    heads (e.g., multi-query/grouped-query attention), the key/value head may
    be replicated while the query heads are partitioned.
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: Optional[int] = None,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        if dense_optim_enable():
            self.forward_type = "dense_optim"
        else:
            self.forward_type = "normal_tp"
        self.comm_group = get_tp_group()
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        # Divide the weight matrix along the last dimension.
        # TODO: check for disable_tp
        tp_size = self.comm_group.world_size
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size,
                                               self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        input_size = self.hidden_size
        output_size = (self.num_heads +
                       2 * self.num_kv_heads) * tp_size * self.head_size
        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,  # q_proj
            self.num_kv_heads * self.head_size * tp_size,  # k_proj
            self.num_kv_heads * self.head_size * tp_size,  # v_proj
        ]
        AscendColumnParallelLinear.__init__(self,
                                            input_size=input_size,
                                            output_size=output_size,
                                            bias=bias,
                                            gather_output=False,
                                            skip_bias_add=skip_bias_add,
                                            params_dtype=params_dtype,
                                            quant_config=quant_config,
                                            prefix=prefix,
                                            return_bias=return_bias,
                                            disable_tp=disable_tp)

    def forward(
        self,
        input_,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        if self.forward_type == "dense_optim":
            return self._forward_dense_optim(input_)
        else:
            return super().forward(input_)

    def _forward_dense_optim(
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
        output_parallel = self.quant_method.apply(self, input_, bias)

        if self.gather_output:
            # All-gather across the partitions.
            output = self.comm_group.all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        if not self.return_bias:
            return output
        return output, output_bias


class AscendLinearBase(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        nn.Module.__init__(self)

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.quant_config = quant_config
        self.prefix = prefix
        if quant_config is None:
            self.quant_method: Optional[
                QuantizeMethodBase] = UnquantizedLinearMethod()
        else:
            self.quant_method = quant_config.get_quant_method(self,
                                                              prefix=prefix)
        self.return_bias = return_bias
        self.disable_tp = disable_tp
