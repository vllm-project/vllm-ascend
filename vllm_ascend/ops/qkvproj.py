from typing import Optional, Union
import numpy as np
import torch.distributed as dist

import torch
from torch.nn.parameter import Parameter

from vllm.distributed import divide
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
# yapf: disable
from vllm.model_executor.parameter import (BasevLLMParameter,
                                           BlockQuantScaleParameter,
                                           PackedColumnParameter,
                                           PackedvLLMParameter,
                                           PerTensorScaleParameter,
                                           RowvLLMParameter)
from vllm.distributed.parallel_state import get_dp_group
# yapf: enable
from vllm.model_executor.utils import set_weight_attrs

from vllm_ascend.distributed.parallel_state import get_qkvtp_group
from vllm.model_executor.layers.linear import (WEIGHT_LOADER_V2_SUPPORTED, 
                                               LinearBase,
                                               adjust_scalar_to_fused_array,
                                               adjust_marlin_shard,
                                               adjust_bitsandbytes_4bit_shard)
from vllm.forward_context import get_forward_context

logger = init_logger(__name__)

class CustomColumnParallelLinear(LinearBase):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Args:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias.
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        output_sizes: list of output sizes packed into one output, like for QKV
                       the list would be size 3.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj) 
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
    ):
        # Divide the weight matrix along the last dimension.
        self.tp_size = get_qkvtp_group().world_size
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]
        # If QKV or MergedColumn, use output size of each partition.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [
                divide(output_size, self.tp_size)
                for output_size in self.output_sizes
            ]

        super().__init__(input_size,
                         output_size,
                         skip_bias_add,
                         params_dtype,
                         quant_config,
                         prefix,
                         return_bias=return_bias)

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

class qkvproj_QKVParallelLinear(CustomColumnParallelLinear):
    """Linear layers for the attention's QKV transformation.

    Linear layers for the linear transformation of the query, key, and value
    vectors in the attention layer. The weight matrix is concatenated along
    the output dimension. The layer is parallelized along the head dimension.
    When the number of key/value heads is smaller than the number of query
    heads (e.g., multi-query/grouped-query attention), the key/value head may
    be replicated while the query heads are partitioned.

    Args:
        hidden_size: input hidden state size of the transformer.
        head_size: size of each attention head.
        total_num_heads: total number of attention query heads.
        total_num_kv_heads: total number of attention key/value heads. If
                            None, assume total_num_kv_heads = total_num_heads.
        bias: If true, add bias.
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
        return_bias: If true, return bias together with outputs in forward pass.
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
    ):
        self.tp_rank = get_qkvtp_group().rank_in_group
        self.dp_rank = get_dp_group().rank_in_group
        self.tp_size = get_qkvtp_group().world_size
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        # Divide the weight matrix along the last dimension.

        self.num_heads = divide(self.total_num_heads, self.tp_size)
        if self.tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(self.tp_size,
                                               self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, self.tp_size)
            self.num_kv_head_replicas = 1
        input_size = self.hidden_size
        output_size = (self.num_heads +
                       2 * self.num_kv_heads) * self.tp_size * self.head_size
        self.output_sizes = [
            self.num_heads * self.head_size * self.tp_size,  # q_proj
            self.num_kv_heads * self.head_size * self.tp_size,  # k_proj
            self.num_kv_heads * self.head_size * self.tp_size,  # v_proj 
        ]

        super().__init__(input_size=input_size,
                         output_size=output_size,
                         bias=bias,
                         gather_output=False,
                         skip_bias_add=skip_bias_add,
                         params_dtype=params_dtype,
                         quant_config=quant_config,
                         prefix=prefix,
                         return_bias=return_bias)

    def _get_shard_offset_mapping(self, loaded_shard_id: str):
        shard_offset_mapping = {
            "q": 0,
            "k": self.num_heads * self.head_size,
            "v": (self.num_heads + self.num_kv_heads) * self.head_size,
            "total": (self.num_heads + 2 * self.num_kv_heads) * self.head_size
        }
        return shard_offset_mapping.get(loaded_shard_id)

    def _get_shard_size_mapping(self, loaded_shard_id: str):
        shard_size_mapping = {
            "q": self.num_heads * self.head_size,
            "k": self.num_kv_heads * self.head_size,
            "v": self.num_kv_heads * self.head_size,
        }
        return shard_size_mapping.get(loaded_shard_id)

    def _load_fused_module_from_checkpoint(self, param: BasevLLMParameter,
                                           loaded_weight: torch.Tensor):
        """
        Handle special case for models where QKV layers are already 
        fused on disk. In this case, we have no shard id. This function
        determmines the shard id by splitting these layers and then calls
        the weight loader using the shard id.

        An example of a model with these fused layers:
        https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
        """
        shard_offsets = [
            # (shard_id, shard_offset, shard_size)
            ("q", 0, self.total_num_heads * self.head_size),
            ("k", self.total_num_heads * self.head_size,
             self.total_num_kv_heads * self.head_size),
            ("v",
             (self.total_num_heads + self.total_num_kv_heads) * self.head_size,
             self.total_num_kv_heads * self.head_size),
        ]

        for shard_id, shard_offset, shard_size in shard_offsets:
            # Special case for Quantization.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            if isinstance(param, (PackedColumnParameter, PackedvLLMParameter
                                  )) and param.packed_dim == param.output_dim:
                shard_size, shard_offset = \
                    param.adjust_shard_indexes_for_packing(
                    shard_size=shard_size, shard_offset=shard_offset)

            loaded_weight_shard = loaded_weight.narrow(param.output_dim,
                                                       shard_offset,
                                                       shard_size)
            self.weight_loader_v2(param, loaded_weight_shard, shard_id)

    def weight_loader_v2(self,
                         param: BasevLLMParameter,
                         loaded_weight: torch.Tensor,
                         loaded_shard_id: Optional[str] = None):
        if loaded_shard_id is None:  # special case for certain models
            if isinstance(param, PerTensorScaleParameter):
                param.load_qkv_weight(loaded_weight=loaded_weight, shard_id=0)
                return
            elif type(param) in (RowvLLMParameter, BasevLLMParameter):
                param.load_qkv_weight(loaded_weight=loaded_weight)
                return
            # TODO: @dsikka - move to parameter.py
            self._load_fused_module_from_checkpoint(param, loaded_weight)
            return

        assert loaded_shard_id in ["q", "k", "v"]

        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)

        # Note(simon): This is needed for Qwen3's fp8 quantization.
        if isinstance(param, BlockQuantScaleParameter):
            assert self.quant_method is not None
            assert hasattr(self.quant_method, "quant_config")
            weight_block_size = self.quant_method.quant_config.weight_block_size
            block_n, _ = weight_block_size[0], weight_block_size[1]
            shard_offset = (shard_offset + block_n - 1) // block_n
            shard_size = (shard_size + block_n - 1) // block_n

        param.load_qkv_weight(loaded_weight=loaded_weight,
                              num_heads=self.num_kv_head_replicas,
                              shard_id=loaded_shard_id,
                              shard_offset=shard_offset,
                              shard_size=shard_size)

    def weight_loader(self,
                      param: Parameter,
                      loaded_weight: torch.Tensor,
                      loaded_shard_id: Optional[str] = None):

        # Special case for GGUF
        # initialize GGUF param after we know the quantize type
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            idx_map = {"q": 0, "k": 1, "v": 2}
            if loaded_shard_id is not None:
                param.data[idx_map[loaded_shard_id]].copy_(loaded_weight)
                param.shard_weight_type[loaded_shard_id] = loaded_weight.item()
            else:
                param.shard_weight_type = {
                    k: loaded_weight.item()
                    for k in idx_map
                }
            return

        if is_gguf_weight:
            tp_size = self.tp_size
            tp_rank = self.tp_rank

            output_dim = getattr(param, "output_dim", None)
            shard_size = loaded_weight.size(output_dim) // tp_size
            start_idx = tp_rank * shard_size

            if loaded_shard_id is not None:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                     shard_size)
                param.shard_id.append(loaded_shard_id)
                param.shard_id_map[loaded_shard_id] = len(param.data_container)
                param.data_container.append(loaded_weight)
                return

        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        # Special case for AQLM codebooks.
        is_metadata = getattr(param, "is_metadata", False)

        # Special case for per-tensor scales in fused case.
        needs_scalar_to_array = getattr(param, "needs_scalar_to_array", False)

        if loaded_shard_id is None:
            # Loaded weight is already fused on disk (qkv).
            # (e.g., Phi-3's qkv_proj).
            if output_dim is None:
                if needs_scalar_to_array:
                    param_data, loaded_weight = adjust_scalar_to_fused_array(
                        param_data, loaded_weight, 0)

                assert param_data.shape == loaded_weight.shape
                param_data.copy_(loaded_weight)
                return
            shard_offsets = [
                # (shard_id, shard_offset, shard_size)
                ("q", 0, self.total_num_heads * self.head_size),
                ("k", self.total_num_heads * self.head_size,
                 self.total_num_kv_heads * self.head_size),
                ("v", (self.total_num_heads + self.total_num_kv_heads) *
                 self.head_size, self.total_num_kv_heads * self.head_size),
            ]
            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit",
                                            False)

            packed_dim = getattr(param, "packed_dim", None)
            for shard_id, shard_offset, shard_size in shard_offsets:
                # Special case for Quantized Weights.
                # If quantized, we need to adjust the offset and size to account
                # for the packing.
                if packed_dim == output_dim:
                    shard_size = shard_size // param.pack_factor
                    shard_offset = shard_offset // param.pack_factor

                    # Special case for Marlin.
                    shard_size, shard_offset = adjust_marlin_shard(
                        param, shard_size, shard_offset)

                if use_bitsandbytes_4bit:
                    orig_qkv_offsets = {
                        "q": (0, self.total_num_heads * self.head_size),
                        "k": (self.total_num_heads * self.head_size,
                              self.total_num_kv_heads * self.head_size),
                        "v":
                        ((self.total_num_heads + self.total_num_kv_heads) *
                         self.head_size,
                         self.total_num_kv_heads * self.head_size),
                        "total":
                        ((self.total_num_heads + 2 * self.total_num_kv_heads) *
                         self.head_size, 0)
                    }

                    shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
                        param, orig_qkv_offsets, shard_id)

                loaded_weight_shard = loaded_weight.narrow(
                    output_dim, shard_offset, shard_size)
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return

        tp_rank = self.tp_rank
        assert loaded_shard_id in ["q", "k", "v"]

        # If output dim is defined, use the default loading process.
        if output_dim is not None:
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
            # Special case for Quantized Weights.
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
            is_sharded_weight = getattr(param, "is_sharded_weight", False)
            # bitsandbytes loads the weights of the specific portion
            # no need to narrow
            is_sharded_weight = is_sharded_weight or use_bitsandbytes_4bit

            if use_bitsandbytes_4bit:
                orig_qkv_offsets = {
                    "q": (0, self.num_heads * self.head_size),
                    "k": (self.num_heads * self.head_size,
                          self.num_kv_heads * self.head_size),
                    "v":
                    ((self.num_heads + self.num_kv_heads) * self.head_size,
                     self.num_kv_heads * self.head_size),
                    "total":
                    ((self.num_heads + 2 * self.num_kv_heads) * self.head_size,
                     0)
                }
                shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
                    param, orig_qkv_offsets, loaded_shard_id)

            param_data = param_data.narrow(output_dim, shard_offset,
                                           shard_size)
            if loaded_shard_id == "q":
                shard_id = tp_rank
            else:
                shard_id = tp_rank // self.num_kv_head_replicas
            start_idx = shard_id * shard_size

            if not is_sharded_weight:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                     shard_size)

        # Special case for for AQLM codebooks.
        elif is_metadata:
            # metadata indicates fixed size concatenated along dim 0
            shard_size = loaded_weight.shape[0]
            shard_index = ["q", "k", "v"].index(loaded_shard_id)
            param_data = param_data.narrow(0, shard_index * shard_size,
                                           shard_size)
        # Special case for per-tensor scales in fused case.
        elif needs_scalar_to_array:
            param_data, loaded_weight = adjust_scalar_to_fused_array(
                param_data, loaded_weight, loaded_shard_id)
        else:
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "QKVParallelLinear, assume the weight is the same "
                    "for all partitions.")

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)
    
    def forward(
        self, input_
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        bias = self.bias if not self.skip_bias_add else None
        # 判断是否是prefill阶段，prefill阶段走单算子模式，会走alltoallv
        forward_context = get_forward_context()
        with_prefill = forward_context.with_prefill
        local_batch_size = input_.size(0)

        if not with_prefill:
            gather_output = get_qkvtp_group().all_gather(input_, dim=0)
        else:
            cu_tokens_across_dp_cpu = forward_context.dp_metadata.cu_tokens_across_dp_cpu
            prefix_array = cu_tokens_across_dp_cpu.numpy()
            global_batch_size = np.concatenate(
                ([prefix_array[0]], np.diff(prefix_array)))
            qkv_tp_group_id = self.dp_rank // self.tp_size
            qkv_tp_group_batchsize = global_batch_size[qkv_tp_group_id*self.tp_size: (qkv_tp_group_id+1)*self.tp_size]

            torch_list = [torch.zeros(size, input_.size(1), dtype=input_.dtype, device=input_.device) 
                        for size in qkv_tp_group_batchsize]
            dist.all_gather(torch_list, input_, group=get_qkvtp_group().device_group)
            gather_output = torch.concat(torch_list, dim=0)

        # Matrix multiply.
        assert self.quant_method is not None
        output_parallel = self.quant_method.apply(self, gather_output, bias)

        chunk_size = output_parallel.size(1)
        send_buf = (
            output_parallel
            .contiguous()
            .view(-1))
        
        # Create receive buffer
        recv_buf = torch.empty(
            local_batch_size * chunk_size * self.tp_size,
            dtype=output_parallel.dtype,
            device=output_parallel.device)

        if self.gather_output:
            # TODO gather output
            return get_qkvtp_group().all_gather(output_parallel, dim=1)
        elif not with_prefill:
            dist.all_to_all_single(recv_buf, send_buf, group=get_qkvtp_group().device_group)
        else:
            # Create split array
            send_splits = [size * chunk_size for size in qkv_tp_group_batchsize]
            recv_splits = [local_batch_size * chunk_size] * self.tp_size

            # Perform all-to-all communication
            dist.all_to_all_single(
                recv_buf, 
                send_buf,
                recv_splits,
                send_splits,
                group=get_qkvtp_group().device_group)
        
        output = recv_buf.view(self.tp_size, local_batch_size, chunk_size)
        sizes = [self.num_heads * self.head_size, self.num_kv_heads * self.head_size, self.num_kv_heads * self.head_size]
    
        output_permuted = output.permute(1, 0, 2).contiguous()  # [tp_size, local_batch_size, -1]

        # 再 split（在最后一维拆分）
        split_parts = torch.split(output_permuted, split_size_or_sections=sizes, dim=-1)

        # 此时每个 part 形状为 [tp_size, local_batch_size, size_per_tp]
        # 直接 reshape 即可：[tp_size, B, size_per_tp] -> [B, tp_size * size_per_tp]
        outs = [
            part.reshape(local_batch_size, -1) for part in split_parts
        ]
        output = outs

        output_bias = self.bias if self.skip_bias_add else None
        if not self.return_bias:
            return output
        return output, output_bias
