# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from abc import abstractmethod
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch_npu
from omni.adaptors.vllm.distributed.communication_op import \
    mla_tensor_model_parallel_reduce_scatter
from omni.adaptors.vllm.distributed.parallel_state import (GroupCoordinator,
                                                           get_mlp_tp_group,
                                                           get_o_proj_tp_group)
from omni.models.common.config.model_config import model_extra_config
from torch.nn.parameter import Parameter, UninitializedParameter
from vllm import logger
from vllm.distributed import (divide, get_tensor_model_parallel_rank,
                              get_tp_group, split_tensor_along_last_dim,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce,
                              tensor_model_parallel_reduce_scatter)
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               LinearBase, LinearMethodBase,
                                               ReplicatedLinear)
from vllm.model_executor.layers.linear import \
    RowParallelLinear as RowParallelLinearGPU
from vllm.model_executor.layers.linear import (UnquantizedLinearMethod,
                                               adjust_marlin_shard,
                                               adjust_scalar_to_fused_array)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform


class AscendUnquantizedLinearMethod(UnquantizedLinearMethod):

    def process_weights_after_loading(self, layer):
        if model_extra_config.operator_opt_config.unquant_bmm_nz:
            weight = layer.weight
            weight.data = torch_npu.npu_format_cast(weight.data, 29)
            layer.weight = Parameter(weight, requires_grad=False)
        return


class AscendMergedColumnParallelLinear(LinearBase):

    def __init__(self,
                 input_size: int,
                 output_sizes: List[int],
                 bias: bool = True,
                 gather_output: bool = False,
                 skip_bias_add: bool = False,
                 tp_size: int = 1,
                 tp_rank: int = 0,
                 params_dtype: Optional[torch.dtype] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        self.output_sizes = output_sizes
        self.tp_size = tp_size
        if not all(output_size % tp_size == 0 for output_size in output_sizes):
            raise RuntimeError("All output_sizes must be divisible by tp_size")
        self.tp_rank = tp_rank
        output_size = sum(output_sizes)
        super().__init__(input_size, output_size, skip_bias_add, params_dtype,
                         quant_config, prefix)

        self.gather_output = gather_output

        # Divide the weight matrix along the last dimension.
        if self.quant_method is None:
            raise RuntimeError("self.quant_method must not be None")
        self.output_size_per_partition = divide(self.output_size, tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]
        # If QKV or MergedColumn, use output size of each partition.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [
                divide(output_size, self.tp_size)
                for output_size in self.output_sizes
            ]
        output_sizes = [output_size]

        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=self.weight_loader)
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
        self.throw_dequant = True

    def forward(self, input_):
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        if self.quant_method is None:
            raise RuntimeError("self.quant_method must not be None")
        output_parallel = self.quant_method.apply(self, input_, bias)
        if self.gather_output:
            if not isinstance(output_parallel, torch.Tensor):
                raise RuntimeError(
                    "not support throw dequant when need gather output")
            # All-gather across the partitions.
            output = get_mlp_tp_group().all_gather(output_parallel, dim=-1)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def weight_loader(self,
                      param: Parameter,
                      loaded_weight: torch.Tensor,
                      loaded_shard_id: Optional[int] = None):

        # Special case for GGUF
        # initialize GGUF param after we know the quantize type
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.data[loaded_shard_id].copy_(loaded_weight)
            param.shard_weight_type[loaded_shard_id] = loaded_weight.item()
            return

        if is_gguf_weight:
            output_dim = getattr(param, "output_dim", None)
            shard_size = loaded_weight.size(output_dim) // self.tp_size
            start_idx = self.tp_rank * shard_size

            loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                 shard_size)

            param.shard_id.append(loaded_shard_id)
            param.shard_id_map[loaded_shard_id] = len(param.data_container)
            param.data_container.append(loaded_weight)
            if len(param.data_container) == 2:
                self.qweight = param.materialize_nested()
            return

        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        # Special case for AQLM codebooks.
        is_metadata = getattr(param, "is_metadata", False)
        # Special case for per-tensor scale to load scalar into fused array.
        needs_scalar_to_array = getattr(param, "needs_scalar_to_array", False)

        if loaded_shard_id is None:
            # Loaded weight is already fused on disk (qkv/mlp).
            if output_dim is None:
                if needs_scalar_to_array:
                    param_data, loaded_weight = adjust_scalar_to_fused_array(
                        param_data, loaded_weight, 0)

                if param_data.shape != loaded_weight.shape:
                    raise RuntimeError(
                        "param_data.shape != loaded_weight.shape")
                param_data.copy_(loaded_weight)
                return
            current_shard_offset = 0
            shard_offsets: List[Tuple[int, int, int]] = []
            for i, output_size in enumerate(self.output_sizes):
                shard_offsets.append((i, current_shard_offset, output_size))
                current_shard_offset += output_size
            packed_dim = getattr(param, "packed_dim", None)
            for shard_id, shard_offset, shard_size in shard_offsets:
                # Special case for Quantization.
                # If quantized, we need to adjust the offset and size to account
                # for the packing.
                if packed_dim == output_dim:
                    shard_size = shard_size // param.pack_factor
                    shard_offset = shard_offset // param.pack_factor
                    # Special case for Marlin.
                    shard_size, shard_offset = adjust_marlin_shard(
                        param, shard_size, shard_offset)

                loaded_weight_shard = loaded_weight.narrow(
                    output_dim, shard_offset, shard_size)
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return

        if loaded_shard_id >= len(self.output_sizes):
            raise RuntimeError(
                "loaded_shard_id must be less than the length of self.output_sizes"
            )
        if output_dim is not None:
            shard_offset = sum(
                self.output_sizes[:loaded_shard_id]) // self.tp_size
            shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
            # Special case for quantization.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            packed_dim = getattr(param, "packed_dim", None)
            if packed_dim == output_dim:
                shard_size = shard_size // param.pack_factor
                shard_offset = shard_offset // param.pack_factor
                # Special case for Marlin.
                shard_size, shard_offset = adjust_marlin_shard(
                    param, shard_size, shard_offset)

            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit",
                                            False)
            if use_bitsandbytes_4bit:
                shard_size = loaded_weight.shape[output_dim]
                shard_offset = loaded_weight.shape[output_dim] * \
                    loaded_shard_id

            param_data = param_data.narrow(output_dim, shard_offset,
                                           shard_size)
            start_idx = self.tp_rank * shard_size
            # bitsandbytes loads the weights of the specific portion
            # no need to narrow here
            if not use_bitsandbytes_4bit:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                     shard_size)
        # Special case for AQLM codebooks.
        elif is_metadata:
            # metadata indicates fixed size concatenated along dim 0
            shard_size = loaded_weight.shape[0]
            shard_offset = loaded_shard_id * shard_size
            param_data = param_data.narrow(0, shard_offset, shard_size)

        # Special case for per-tensor scales in fused case.
        elif needs_scalar_to_array:
            param_data, loaded_weight = adjust_scalar_to_fused_array(
                param_data, loaded_weight, loaded_shard_id)

        else:
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "MergedColumnParallelLinear, assume the weight is "
                    "the same for all partitions.")

        if param_data.shape != loaded_weight.shape:
            raise RuntimeError("param_data.shape != loaded_weight.shape")
        param_data.copy_(loaded_weight)


class AscendRowParallelLinear(LinearBase):

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 tp_size: int = 1,
                 tp_rank: int = 0,
                 bias: bool = True,
                 input_is_parallel: bool = True,
                 skip_bias_add: bool = False,
                 params_dtype: Optional[torch.dtype] = None,
                 reduce_results: bool = True,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__(input_size, output_size, skip_bias_add, params_dtype,
                         quant_config, prefix)

        if self.quant_method is None:
            raise RuntimeError("self.quant_method must not be None")

        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError("When not reduce the results, adding bias to the "
                             "results can lead to incorrect results")

        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        self.tp_size = tp_size
        self.tp_rank = tp_rank
        # Divide the weight matrix along the last dimension.
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size
        self.output_partition_sizes = [output_size]

        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=self.weight_loader)

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size, dtype=params_dtype))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        input_dim = getattr(param, "input_dim", None)
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)

        # Special case for GGUF
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.weight_type = loaded_weight.item()

        # Materialize GGUF UninitializedParameter
        if is_gguf_weight and isinstance(param, UninitializedParameter):
            weight_shape = list(loaded_weight.shape)
            if input_dim:
                weight_shape[
                    input_dim] = weight_shape[input_dim] // self.tp_size
            param.materialize(tuple(weight_shape), dtype=loaded_weight.dtype)

        param_data = param.data
        # bitsandbytes loads the weights of the specific portion
        # no need to narrow here
        if input_dim is not None and not use_bitsandbytes_4bit:
            shard_size = param_data.shape[input_dim]
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(input_dim, start_idx,
                                                 shard_size)

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        if param_data.shape != loaded_weight.shape:
            raise RuntimeError("param_data.shape != loaded_weight.shape")
        param_data.copy_(loaded_weight)

    def forward(self, input_):
        if self.input_is_parallel:
            input_parallel = input_
        else:
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[self.tp_rank].contiguous()

        # Matrix multiply.
        if self.quant_method is None:
            raise RuntimeError("self.quant_method is None")
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = self.quant_method.apply(self,
                                                  input_parallel,
                                                  bias=bias_)
        if self.reduce_results and self.tp_size > 1:
            output = get_mlp_tp_group().all_reduce(output_parallel)
        else:
            output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None

        return output, output_bias

    def extra_repr(self) -> str:
        s = f"input_features={self.input_size_per_partition}"
        s += f", output_features={self.output_size}"
        s += f", bias={self.bias is not None}"
        s += f", tp_size={self.tp_size}"
        s += f", reduce_results={self.reduce_results}"
        return s


class DP2TPRowParallelLinear(AscendRowParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(input_size, output_size, tp_size, tp_rank, bias,
                         input_is_parallel, skip_bias_add, params_dtype,
                         reduce_results, quant_config, prefix)

    def forward(
        self,
        input_,
        bsz,
        q_len,
        num_heads,
        v_head_dim,
    ):
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_ = input_.view(bsz * q_len, self.tp_size,
                                 self.input_size_per_partition).transpose(
                                     0, 1).contiguous()
            all_to_all_o_proj_shape = [bsz * q_len * num_heads * v_head_dim]
            input_ = input_.view(all_to_all_o_proj_shape)
            input_parallel = torch.empty(all_to_all_o_proj_shape,
                                         dtype=input_.dtype,
                                         device=current_platform.device_type)
            dist.all_to_all_single(input_parallel,
                                   input_,
                                   group=get_o_proj_tp_group().device_group)
            torch_npu.npu_prefetch(self.weight, input_, 25 * 1024 * 1024)
            input_parallel = input_parallel.view(bsz * q_len * self.tp_size,
                                                 self.input_size_per_partition)

        # Matrix multiply.
        assert self.quant_method is not None
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = self.quant_method.apply(self,
                                                  input_parallel,
                                                  bias=bias_)
        if self.reduce_results and self.tp_size > 1:
            output = get_o_proj_tp_group().reduce_scatter(output_parallel)
        else:
            output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None

        return output, output_bias


class Tp2DpAndTpRowParallelLinear(AscendRowParallelLinear):

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 tp_size: int = 1,
                 tp_rank: int = 0,
                 bias: bool = True,
                 input_is_parallel: bool = True,
                 skip_bias_add: bool = False,
                 params_dtype: Optional[torch.dtype] = None,
                 reduce_results: bool = True,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__(input_size, output_size, tp_size, tp_rank, bias,
                         input_is_parallel, skip_bias_add, params_dtype,
                         reduce_results, quant_config, prefix)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        input_dim = getattr(param, "input_dim", None)
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)

        # Special case for GGUF
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.weight_type = loaded_weight.item()

        # Materialize GGUF UninitializedParameter
        if is_gguf_weight and isinstance(param, UninitializedParameter):
            weight_shape = list(loaded_weight.shape)
            if input_dim:
                weight_shape[
                    input_dim] = weight_shape[input_dim] // self.tp_size
            param.materialize(tuple(weight_shape), dtype=loaded_weight.dtype)

        param_data = param.data
        # bitsandbytes loads the weights of the specific portion
        # adapter
        world_size = torch.distributed.get_world_size()
        rank_list = torch.arange(world_size).reshape(-1, self.tp_size).T
        dp_size = world_size // self.tp_size
        if input_dim is not None and not use_bitsandbytes_4bit:
            shard_size = param_data.shape[input_dim] // dp_size
            res = []
            for rank in rank_list[self.tp_rank]:
                start_idx = rank * shard_size
                tmp_weight = loaded_weight.narrow(input_dim, start_idx,
                                                  shard_size)
                res.append(tmp_weight)
            loaded_weight = torch.cat(res, dim=input_dim)

        # adapter end

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        if param_data.shape != loaded_weight.shape:
            raise RuntimeError("param_data.shape != loaded_weight.shape")
        param_data.copy_(loaded_weight)

    def forward(self, input_):
        if self.input_is_parallel:
            input_parallel = input_
        else:
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[self.tp_rank].contiguous()

        # Matrix multiply.
        if self.quant_method is None:
            raise RuntimeError("self.quant_method is None")
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = self.quant_method.apply(self,
                                                  input_parallel,
                                                  bias=bias_)
        if self.reduce_results and self.tp_size > 1:
            output = get_o_proj_tp_group().reduce_scatter(output_parallel)
        else:
            output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None

        return output, output_bias


class ColumnParallelLinearQuantGather(ColumnParallelLinear):

    def __init__(self, input_size, output_size, bias, quant_config, prefix):
        super().__init__(input_size=input_size,
                         output_size=output_size,
                         bias=bias,
                         quant_config=quant_config,
                         prefix=prefix)

    def forward(self, input_):
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        if self.quant_method is None:
            raise RuntimeError("self.quant_method is not None")
        output_parallel = self.quant_method.apply(self,
                                                  input_,
                                                  bias,
                                                  inner_gather=True)
        if self.gather_output:
            # All-gather across the partitions.
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class RowParallelLinear(RowParallelLinearGPU):

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 bias: bool = True,
                 input_is_parallel: bool = True,
                 skip_bias_add: bool = False,
                 params_dtype: Optional[torch.dtype] = None,
                 reduce_results: bool = True,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__(input_size, output_size, bias, input_is_parallel,
                         skip_bias_add, params_dtype, reduce_results,
                         quant_config, prefix)

    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[self.tp_rank].contiguous()

        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias

        output_parallel = self.quant_method.apply(self,
                                                  input_parallel,
                                                  bias=bias_)
        if self.reduce_results and self.tp_size > 1:
            output_ = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output_ = output_parallel
        # Adapt end.
        output_bias = self.bias if self.skip_bias_add else None

        return output_, output_bias


class RowParallelLinearWithReduceScatter(RowParallelLinear):

    def __init__(self, *args, **kwargs):
        super(RowParallelLinearWithReduceScatter,
              self).__init__(*args, **kwargs)
        if self.bias is not None:
            raise RuntimeError("self.bias is not None")

    def forward(self, input_, comm_group: Optional[GroupCoordinator] = None):
        if self.input_is_parallel:
            input_parallel = input_
        else:
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[self.tp_rank].contiguous()

        # Matrix multiply.
        if self.quant_method is None:
            raise RuntimeError("self.quant_method is None")
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = self.quant_method.apply(self,
                                                  input_parallel,
                                                  bias=bias_)
        if self.reduce_results and self.tp_size > 1:
            output = mla_tensor_model_parallel_reduce_scatter(
                output_parallel, comm_group=comm_group)
        else:
            output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None

        return output, output_bias


class MergedReplicatedLinear(ReplicatedLinear):

    def __init__(self,
                 input_size: int,
                 output_sizes: List[int],
                 bias: bool = True,
                 skip_bias_add: bool = False,
                 params_dtype: Optional[torch.dtype] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        self.output_sizes = output_sizes
        super().__init__(input_size=input_size,
                         output_size=sum(output_sizes),
                         bias=bias,
                         skip_bias_add=skip_bias_add,
                         params_dtype=params_dtype,
                         quant_config=quant_config,
                         prefix=prefix)

    def weight_loader(self,
                      param: Parameter,
                      loaded_weight: torch.Tensor,
                      loaded_shard_id: Optional[int] = None):

        # Special case for GGUF
        # initialize GGUF param after we know the quantize type
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.data[loaded_shard_id].copy_(loaded_weight)
            param.shard_weight_type[loaded_shard_id] = loaded_weight.item()
            return

        if is_gguf_weight:
            output_dim = getattr(param, "output_dim", None)
            shard_size = loaded_weight.size(output_dim)
            start_idx = 0

            loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                 shard_size)

            param.shard_id.append(loaded_shard_id)
            param.shard_id_map[loaded_shard_id] = len(param.data_container)
            param.data_container.append(loaded_weight)
            if len(param.data_container) == 2:
                self.qweight = param.materialize_nested()
            return

        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        # Special case for AQLM codebooks.
        is_metadata = getattr(param, "is_metadata", False)
        # Special case for per-tensor scale to load scalar into fused array.
        needs_scalar_to_array = getattr(param, "needs_scalar_to_array", False)

        if loaded_shard_id is None:
            # Loaded weight is already fused on disk (qkv/mlp).
            if output_dim is None:
                if needs_scalar_to_array:
                    param_data, loaded_weight = adjust_scalar_to_fused_array(
                        param_data, loaded_weight, 0)

                if param_data.shape != loaded_weight.shape:
                    raise RuntimeError(
                        "param_data.shape != loaded_weight.shape")
                param_data.copy_(loaded_weight)
                return
            current_shard_offset = 0
            shard_offsets: List[Tuple[int, int, int]] = []
            for i, output_size in enumerate(self.output_sizes):
                shard_offsets.append((i, current_shard_offset, output_size))
                current_shard_offset += output_size
            packed_dim = getattr(param, "packed_dim", None)
            for shard_id, shard_offset, shard_size in shard_offsets:
                # Special case for Quantization.
                # If quantized, we need to adjust the offset and size to account
                # for the packing.
                if packed_dim == output_dim:
                    shard_size = shard_size // param.pack_factor
                    shard_offset = shard_offset // param.pack_factor
                    # Special case for Marlin.
                    shard_size, shard_offset = adjust_marlin_shard(
                        param, shard_size, shard_offset)

                loaded_weight_shard = loaded_weight.narrow(
                    output_dim, shard_offset, shard_size)
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return

        if loaded_shard_id >= len(self.output_sizes):
            raise RuntimeError("loaded_shard_id >= len(self.output_sizes)")

        if output_dim is not None:
            shard_offset = sum(self.output_sizes[:loaded_shard_id])
            shard_size = self.output_sizes[loaded_shard_id]
            # Special case for quantization.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            packed_dim = getattr(param, "packed_dim", None)
            if packed_dim == output_dim:
                shard_size = shard_size // param.pack_factor
                shard_offset = shard_offset // param.pack_factor
                # Special case for Marlin.
                shard_size, shard_offset = adjust_marlin_shard(
                    param, shard_size, shard_offset)

            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit",
                                            False)
            if use_bitsandbytes_4bit:
                shard_size = loaded_weight.shape[output_dim]
                shard_offset = loaded_weight.shape[output_dim] * \
                    loaded_shard_id

            param_data = param_data.narrow(output_dim, shard_offset,
                                           shard_size)
            start_idx = 0
            # bitsandbytes loads the weights of the specific portion
            # no need to narrow here
            if not use_bitsandbytes_4bit:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                     shard_size)
        # Special case for AQLM codebooks.
        elif is_metadata:
            # metadata indicates fixed size concatenated along dim 0
            shard_size = loaded_weight.shape[0]
            shard_offset = loaded_shard_id * shard_size
            param_data = param_data.narrow(0, shard_offset, shard_size)

        # Special case for per-tensor scales in fused case.
        elif needs_scalar_to_array:
            param_data, loaded_weight = adjust_scalar_to_fused_array(
                param_data, loaded_weight, loaded_shard_id)

        else:
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "MergedColumnParallelLinear, assume the weight is "
                    "the same for all partitions.")

        if param_data.shape != loaded_weight.shape:
            raise RuntimeError("param_data.shape != loaded_weight.shape")
        param_data.copy_(loaded_weight)


class RowParallelLinearCross(LinearBase):

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 bias: bool = True,
                 tp_size=1,
                 tp_rank=0,
                 input_is_parallel: bool = True,
                 skip_bias_add: bool = False,
                 params_dtype: Optional[torch.dtype] = None,
                 reduce_results: bool = True,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__(input_size, output_size, skip_bias_add, params_dtype,
                         quant_config, prefix)

        self.quant_config = quant_config
        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        # Divide the weight matrix along the last dimension
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.input_size_per_partition = divide(input_size, self.tp_size)
        assert self.quant_method is not None

        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=[self.output_size],
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=self.weight_loader)
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

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        input_dim = getattr(param, "input_dim", None)
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)

        # Special case for GGUF
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.weight_type = loaded_weight.item()

        # Materialize GGUF UninitializedParameter
        if is_gguf_weight and isinstance(param, UninitializedParameter):
            weight_shape = list(loaded_weight.shape)
            if input_dim:
                weight_shape[
                    input_dim] = weight_shape[input_dim] // self.tp_size
            param.materialize(tuple(weight_shape), dtype=loaded_weight.dtype)

        param_data = param.data
        # bitsandbytes loads the weights of the specific portion
        # no need to narrow here
        if input_dim is not None and not use_bitsandbytes_4bit:
            shard_size = param_data.shape[input_dim]
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(input_dim, start_idx,
                                                 shard_size)

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8)
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def forward(self, input_):
        # todo: check mc2
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
        output = self.quant_method.apply(self, input_parallel, bias=bias_)

        output_bias = self.bias if self.skip_bias_add else None

        return output, output_bias


class FlashCommLinearMethodBase(LinearMethodBase):
    """Base class for different (maybe quantized) linear methods."""

    @abstractmethod
    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None,
              module_name: Optional[str] = "",
              x_transform: Optional[str] = None) -> torch.Tensor:
        """Apply the weights in layer to the input tensor.
        Expects create_weights to have been called before on the layer."""
        raise NotImplementedError


class UnquantizedFlashCommLinearMethod(FlashCommLinearMethodBase):
    """Linear method without quantization."""

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        weight = Parameter(torch.empty(sum(output_partition_sizes),
                                       input_size_per_partition,
                                       dtype=params_dtype),
                           requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        weight_data = layer.weight.data.t().contiguous()
        layer.weight.data = weight_data
        set_weight_attrs(layer.weight, {"is_weight_transposed": True})
        # weight_data = torch_npu.npu_format_cast(layer.weight.data.t().contiguous(), 29)
        # layer.weight = torch.nn.Parameter(weight_data, requires_grad=False)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None,
              module_name: Optional[str] = "",
              x_transform: Optional[str] = None,
              is_prefill: Optional[bool] = True) -> torch.Tensor:

        if x_transform == "AG":
            x = get_tp_group().all_gather(x, dim=0)
        elif x_transform == "A2A":
            x = get_tp_group().all_to_all(x)

        if bias is not None:
            # return F.linear(x, layer.weight, bias)
            return torch.addmm(bias, x, layer.weight)
        else:
            return torch.matmul(x, layer.weight)


class FlashCommLinearBase(torch.nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.skip_bias_add = skip_bias_add
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        if quant_config is None or prefix in quant_config.ignore:
            self.quant_method: Optional[
                QuantizeMethodBase] = UnquantizedFlashCommLinearMethod()
        else:
            self.quant_method = quant_config.get_quant_method(self,
                                                              prefix=prefix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


"""
Similar to vllm's original RowParallelLinear except for:
1. layerwise TP configurations. call get_tensor_model_parallel_world_size/rank.
2. flexible communication applied to x or y during forward (controlled by parameters of forward function).
"""


class RowParallelFlashCommLinear(FlashCommLinearBase):

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 tp_size: int = 1,
                 tp_rank: int = 0,
                 bias: bool = True,
                 skip_bias_add: bool = False,
                 params_dtype: Optional[torch.dtype] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__(input_size, output_size, tp_size, tp_rank,
                         skip_bias_add, params_dtype, quant_config, prefix)
        # Divide the weight matrix along the first dimension.
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.prefix = prefix

        assert self.quant_method is not None
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=[self.output_size],
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=self.weight_loader)

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size, dtype=params_dtype))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        # veRL special case: transpose the weight back to original shape
        is_weight_transposed = getattr(param, "is_weight_transposed", False)
        if is_weight_transposed:
            param.data = param.data.t().contiguous()
        input_dim = getattr(param, "input_dim", None)
        param_data = param.data

        if input_dim is not None:
            shard_size = param_data.shape[input_dim]
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(input_dim, start_idx,
                                                 shard_size)

        loaded_weight = torch.squeeze(loaded_weight)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)
        # veRL special case: transpose the weight to use torch npu operator
        if is_weight_transposed:
            param.data = param.data.t().contiguous()

    def forward(self, input_, reduce_type="AR", x_transform=None):
        input_parallel = input_

        # Matrix multiply.
        assert self.quant_method is not None
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = self.quant_method.apply(self,
                                                  input_parallel,
                                                  bias=bias_,
                                                  module_name=self.prefix,
                                                  x_transform=x_transform)
        if self.tp_size > 1:
            if reduce_type == "AR":
                output = tensor_model_parallel_all_reduce(output_parallel)
            elif reduce_type == "RS":
                output = tensor_model_parallel_reduce_scatter(output_parallel)
            else:
                output = output_parallel
        else:
            output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None

        return output, output_bias


class ColumnParallelFlashCommLinear(FlashCommLinearBase):

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 tp_size: int = 1,
                 tp_rank: int = 0,
                 bias: bool = True,
                 skip_bias_add: bool = False,
                 params_dtype: Optional[torch.dtype] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 output_sizes: Optional[List[int]] = None,
                 prefix: str = ""):
        super().__init__(input_size, output_size, tp_size, tp_rank,
                         skip_bias_add, params_dtype, quant_config, prefix)

        # Divide the weight matrix along the last dimension.
        assert self.quant_method is not None
        self.output_size_per_partition = divide(self.output_size, self.tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]
        # If QKV or MergedColumn, use output size of each partition.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [
                divide(output_size, self.tp_size)
                for output_size in self.output_sizes
            ]

        if output_sizes is None:
            output_sizes = [output_size]

        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=self.weight_loader)
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
        self.prefix = prefix

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        # veRL special case: transpose the weight back to original shape
        is_weight_transposed = getattr(param, "is_weight_transposed", False)
        if is_weight_transposed:
            param.data = param.data.t().contiguous()
        output_dim = getattr(param, "output_dim", None)

        param_data = param.data
        if output_dim is not None:
            shard_size = param_data.shape[output_dim]
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                 shard_size)

        loaded_weight = torch.squeeze(loaded_weight)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)
        # veRL special case: transpose the weight to use torch npu operator
        if is_weight_transposed:
            param.data = param.data.t().contiguous()

    def forward(self, input_, x_transform=None, is_prefill=True):
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        assert self.quant_method is not None
        output = self.quant_method.apply(self,
                                         input_,
                                         bias,
                                         module_name=self.prefix,
                                         x_transform=x_transform,
                                         is_prefill=is_prefill)
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def extra_repr(self) -> str:
        s = f"in_features={self.input_size}"
        s += f", output_features={self.output_size_per_partition}"
        s += f", bias={self.bias is not None}"
        s += f", tp_size={self.tp_size}"
        return s


class QKVParallelFlashCommLinear(ColumnParallelFlashCommLinear):

    def __init__(self,
                 hidden_size: int,
                 head_size: int,
                 total_num_heads: int,
                 total_num_kv_heads: Optional[int] = None,
                 tp_size: int = 1,
                 tp_rank: int = 0,
                 bias: bool = True,
                 skip_bias_add: bool = False,
                 params_dtype: Optional[torch.dtype] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        # Divide the weight matrix along the last dimension.
        self.prefix = prefix
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

        super().__init__(input_size=input_size,
                         output_size=output_size,
                         tp_size=tp_size,
                         tp_rank=tp_rank,
                         bias=bias,
                         skip_bias_add=skip_bias_add,
                         params_dtype=params_dtype,
                         quant_config=quant_config,
                         prefix=prefix)

    def weight_loader(self,
                      param: Parameter,
                      loaded_weight: torch.Tensor,
                      loaded_shard_id: Optional[str] = None):
        # veRL special case: transpose the weight back to original shape
        is_weight_transposed = getattr(param, "is_weight_transposed", False)
        if is_weight_transposed:
            param.data = param.data.t().contiguous()
        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        assert loaded_shard_id in ["q", "k", "v"]

        # If output dim is defined, use the default loading process.
        assert output_dim is not None
        if loaded_shard_id == "q":
            shard_offset = 0
            shard_size = self.num_heads * self.head_size
        elif loaded_shard_id == "k":
            shard_offset = self.num_heads * self.head_size
            shard_size = self.num_kv_heads * self.head_size
        elif loaded_shard_id == "v":
            shard_offset = (self.num_heads +
                            self.num_kv_heads) * self.head_size
            shard_size = self.num_kv_heads * self.head_size

        param_data = param_data.narrow(output_dim, shard_offset, shard_size)
        if loaded_shard_id == "q":
            shard_id = self.tp_rank
        else:
            shard_id = self.tp_rank // self.num_kv_head_replicas
        start_idx = shard_id * shard_size

        loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        loaded_weight = torch.squeeze(loaded_weight)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)
        # veRL special case: transpose the weight to use torch npu operator
        if is_weight_transposed:
            param.data = param.data.t().contiguous()


class MergedColumnParallelFlashCommLinear(ColumnParallelFlashCommLinear):

    def __init__(self,
                 input_size: int,
                 output_sizes: List[int],
                 tp_size: int = 1,
                 tp_rank: int = 0,
                 bias: bool = True,
                 skip_bias_add: bool = False,
                 params_dtype: Optional[torch.dtype] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        self.output_sizes = output_sizes
        self.prefix = prefix
        assert all(output_size % tp_size == 0 for output_size in output_sizes)
        super().__init__(input_size=input_size,
                         output_size=sum(output_sizes),
                         tp_size=tp_size,
                         tp_rank=tp_rank,
                         bias=bias,
                         skip_bias_add=skip_bias_add,
                         params_dtype=params_dtype,
                         quant_config=quant_config,
                         prefix=prefix)

    def weight_loader(self,
                      param: Parameter,
                      loaded_weight: torch.Tensor,
                      loaded_shard_id: Optional[int] = None):
        # veRL special case: transpose the weight back to original shape
        is_weight_transposed = getattr(param, "is_weight_transposed", False)
        if is_weight_transposed:
            param.data = param.data.t().contiguous()
        param_data = param.data
        output_dim = getattr(param, "output_dim", None)

        assert loaded_shard_id is not None
        assert loaded_shard_id < len(self.output_sizes)
        assert output_dim is not None
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size

        param_data = param_data.narrow(output_dim, shard_offset, shard_size)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        loaded_weight = torch.squeeze(loaded_weight)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)
        # veRL special case: transpose the weight to use torch npu operator
        if is_weight_transposed:
            param.data = param.data.t().contiguous()
