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
To customize linear communication groups or forward of classes in this file,
extend new linear operations in linear_op.py.
The classes in this file should not be modified, including AscendQKVParallelLinear,
AscendMergedColumnParallelLinear, AscendMergedColumnParallelLinear,
AscendRowParallelLinear and AscendColumnParallelLinear.
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from vllm.config import get_current_vllm_config
from vllm.distributed import divide, get_tensor_model_parallel_rank
from vllm.model_executor.layers.linear import (  # noqa
    WEIGHT_LOADER_V2_SUPPORTED,
    ColumnParallelLinear,
    LinearBase,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    QuantizeMethodBase,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.device.hardware_profile import HardwareCapability, WeightLayoutPolicy, get_current_hardware_profile
from vllm_ascend.ops.linear_op import get_parallel_op, get_replicated_op
from vllm_ascend.utils import maybe_trans_nz
from vllm_ascend.weight_switch import WeightSwitchGatherSpec, WeightSwitchMixin


def unquantized_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.nn.functional.linear(x, weight, bias)


def unquantized_gemm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    op_name="unquantized_gemm",
    op_func=unquantized_gemm,
    fake_impl=unquantized_gemm_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)


def _should_keep_nd_for_compatibility_weight(weight: torch.Tensor) -> bool:
    return (
        get_current_hardware_profile().weight_layout_policy is WeightLayoutPolicy.FORCE_NZ
        and weight.ndim >= 2
        and (weight.shape[-1] == 1 or weight.shape[-2] == 1)
    )


def _should_reshape_wo_a_to_3d(prefix: str, dtype: torch.dtype) -> bool:
    """Whether a DSV4 wo_a weight must be reshaped to
    [n_local_groups, hidden_size, o_lora_rank] for npu_transpose_batchmatmul.

    For bf16 wo_a (unquantized), the reshape must happen here regardless of
    whether the model has a global quant config: partially-quantized checkpoints
    (e.g. ModelSlim) keep unquantized FLOAT wo_a alongside a non-None quant
    config, and no quant method's process_weights_after_loading will run for
    those layers. Quantized (fp8/int8) wo_a is unaffected and still handled by
    the quantization path.
    """
    supports_dynamic_mx_quant_fusion = get_current_hardware_profile().supports(
        HardwareCapability.DYNAMIC_MX_QUANT_FUSION
    )
    reshape_bf16_wo_a = "wo_a" in prefix and supports_dynamic_mx_quant_fusion and dtype == torch.bfloat16
    return "wo_a" in prefix and (not supports_dynamic_mx_quant_fusion or reshape_bf16_wo_a)


class AscendUnquantizedLinearMethod(WeightSwitchMixin, UnquantizedLinearMethod):
    """Linear method without quantization"""

    supports_weight_preprocessing = True

    weight_switch_gather_specs = (WeightSwitchGatherSpec("weight", gather_dim=1),)
    weight_switch_output_gather_specs = (WeightSwitchGatherSpec("weight"),)
    supports_weight_switch = True

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "is_weights_processed", False) is True:
            return
        prepare = getattr(layer, "prepare_weights_for_processing", None)
        if prepare is not None:
            prepare()
        super().process_weights_after_loading(layer)
        keep_nd_weight = _should_keep_nd_for_compatibility_weight(layer.weight.data)
        skip_weight_nz_conversion = getattr(layer, "skip_weight_nz_conversion", False)
        # must use fp32 to avoid accuracy degradation in dsv4.
        if getattr(layer, "precast_fp32_weight", False):
            weight_fp32 = layer.weight.data.to(torch.float32)
            new_fp32 = weight_fp32 if keep_nd_weight or skip_weight_nz_conversion else maybe_trans_nz(weight_fp32)
            # keep the captured graph's weight reference to the updated weight
            # during RL weight updates.
            replace_parameter(layer, "weight_fp32", new_fp32, prefer_copy=True)
        if "conv1d" not in layer.prefix and not skip_weight_nz_conversion:
            # 310P torch_npu rejects FRACTAL_NZ matmul when the weight-side
            # matrix has n=1 or k=1. Keep scalar gates such as Qwen MoE's
            # shared_expert_gate in ND format, leaving non-310P policy intact.
            if not keep_nd_weight:
                layer.weight.data = maybe_trans_nz(layer.weight.data)

        # DSV4 wo_a is consumed by npu_transpose_batchmatmul in the 3D layout
        # [n_local_groups, hidden_size, o_lora_rank]. Reshape it here so it
        # applies to load-format=dummy too, where weight_loader never runs.
        if _should_reshape_wo_a_to_3d(layer.prefix, layer.weight.data.dtype) and layer.weight.data.ndim == 2:
            layer.weight.data = (
                layer.weight.data.view(layer.n_local_groups, layer.o_lora_rank, -1).transpose(2, 1).contiguous()
            )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.ops.vllm.unquantized_gemm(x, layer.weight, bias)


# TODO(realliujiaxu): Remove this class after linear of vllm supports custom comm group
class AscendLinearBase(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
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
            self.quant_method: QuantizeMethodBase | None = AscendUnquantizedLinearMethod()
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)
        self.return_bias = return_bias
        self.disable_tp = disable_tp


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
        total_num_kv_heads: int | None = None,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
        v_head_size: int | None = None,
    ):
        self.v_head_size = v_head_size if v_head_size is not None else head_size
        self.custom_op, _, tp_size = get_parallel_op(disable_tp, prefix, self, "column")
        # TODO(realliujiaxu): Replace the initialization code below with super().__init__ after
        # linear of vllm supports custom comm group
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        # Divide the weight matrix along the last dimension.
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        input_size = self.hidden_size
        output_size = (self.num_heads + 2 * self.num_kv_heads) * tp_size * self.head_size
        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,  # q_proj
            self.num_kv_heads * self.head_size * tp_size,  # k_proj
            self.num_kv_heads * self.head_size * tp_size,  # v_proj
        ]
        AscendColumnParallelLinear.__init__(
            self,
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            gather_output=False,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        if self.custom_op is not None:
            return self.custom_op.apply(input_)

        return super().forward(input_)


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
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        self.custom_op, self.tp_rank, self.tp_size = get_parallel_op(disable_tp, prefix, self, "column")
        # TODO(realliujiaxu): Replace the initialization code below with super().__init__ after
        # linear of vllm supports custom comm group
        self.output_sizes = output_sizes
        assert all(output_size % self.tp_size == 0 for output_size in output_sizes)
        AscendColumnParallelLinear.__init__(
            self,
            input_size=input_size,
            output_size=sum(output_sizes),
            bias=bias,
            gather_output=gather_output,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        if self.custom_op is not None:
            return self.custom_op.apply(input_)

        return super().forward(input_)


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
        params_dtype: torch.dtype | None = None,
        out_dtype: torch.dtype | None = None,
        reduce_results: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        self.custom_op, self.tp_rank, self.tp_size = get_parallel_op(disable_tp, prefix, self, "row")
        # TODO(realliujiaxu): Replace the initialization code below with super().__init__ after
        # linear of vllm supports custom comm group
        # Divide the weight matrix along the first dimension.
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size
        self.output_partition_sizes = [output_size]
        self.out_dtype = out_dtype

        AscendLinearBase.__init__(
            self,
            input_size,
            output_size,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

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
                self.weight_loader_v2
                if self.quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED
                else self.weight_loader
            ),
        )
        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError("When not reduce the results, adding bias to the results can lead to incorrect results")

        if bias:
            self.bias = Parameter(torch.empty(self.output_size, dtype=params_dtype))
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)

        if self.custom_op is not None:
            self.custom_op.update_attrs()

    def forward(
        self,
        input_,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        if self.custom_op is not None:
            return self.custom_op.apply(input_)

        return super().forward(input_)


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
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        output_sizes: list[int] | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        #
        self.custom_op, self.tp_rank, self.tp_size = get_parallel_op(disable_tp, prefix, self, "column")
        # TODO(realliujiaxu): Replace the initialization code below with super().__init__ after
        # linear of vllm supports custom comm group
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]
        # If QKV or MergedColumn, use output size of each partition.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [divide(output_size, self.tp_size) for output_size in self.output_sizes]

        AscendLinearBase.__init__(
            self,
            input_size,
            output_size,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

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
                self.weight_loader_v2
                if self.quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED
                else self.weight_loader
            ),
        )
        if bias:
            self.bias = Parameter(torch.empty(self.output_size_per_partition, dtype=params_dtype))
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)

        if self.custom_op is not None:
            self.custom_op.update_attrs()
        self.prefix = prefix
        if "wo_a" in prefix:
            hf_config = get_current_vllm_config().model_config.hf_text_config
            self.n_local_groups = getattr(hf_config, "o_groups", 0) // self.tp_size
            self.o_lora_rank = getattr(hf_config, "o_lora_rank", 0)

    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        if self.custom_op is not None:
            return self.custom_op.apply(input_)

        return super().forward(input_)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        if _should_reshape_wo_a_to_3d(self.prefix, loaded_weight.dtype):
            if self.weight.ndim == 2:
                # Keep the raw 2D layout here. The 2D -> 3D reshape happens in
                # process_weights_after_loading so it also runs for
                # load-format=dummy, where weight_loader is never called.
                super().weight_loader(param, loaded_weight)
            else:
                # In RL update flows, wo_a can be loaded again after being
                # transformed into [n_local_groups, hidden_size, o_lora_rank].
                shard_size = self.n_local_groups * self.o_lora_rank
                start_idx = self.tp_rank * shard_size
                if loaded_weight.shape[0] != shard_size:
                    loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
                loaded_weight = (
                    loaded_weight.view(
                        self.n_local_groups,
                        self.o_lora_rank,
                        -1,
                    )
                    .transpose(2, 1)
                    .contiguous()
                )

                if loaded_weight.shape != self.weight.shape:
                    raise ValueError(
                        f"Unexpected wo_a weight shape {tuple(loaded_weight.shape)}, "
                        f"expected {tuple(self.weight.shape)}"
                    )
                self.weight.data.copy_(loaded_weight)
        else:
            super().weight_loader(param, loaded_weight)


class _DCPDerivedColumnParallelLinear(AscendColumnParallelLinear):
    """Local projection prepared by its group parent, once per weight load."""

    _processed_weights: dict[str, torch.Tensor] | None = None

    @property
    def is_weights_processed(self) -> bool:
        return self._processed_weights is not None

    def finish_weight_processing(self, previous: "_DCPDerivedColumnParallelLinear | None") -> None:
        tensors = dict(self.named_parameters(recurse=False)) | dict(self.named_buffers(recurse=False))
        # Quantization may also create unregistered tensors such as weight_scale_fp32.
        tensors.update((name, value) for name, value in vars(self).items() if isinstance(value, torch.Tensor))
        if previous is not None:
            previous_weights = previous._processed_weights
            assert previous_weights is not None and previous_weights.keys() == tensors.keys()
            for name, value in tensors.items():
                target = previous_weights[name]
                if (target.shape, target.dtype, target.device) != (value.shape, value.dtype, value.device):
                    raise ValueError(f"DCP local Q parameter {name} changed layout during weight reload")
            # Retain graph addresses even if the layerwise loader removed the old parameters.
            for name, value in tensors.items():
                target = previous_weights[name]
                target.data.copy_(value)
                setattr(self, name, target)
            tensors = previous_weights
        self._processed_weights = tensors


class AscendDCPGroupColumnParallelLinear(AscendColumnParallelLinear):
    """Load group Q weights; derive local prefill weights before quantization processing."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        bias: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        parallel = get_current_vllm_config().parallel_config
        self.group_size = parallel.decode_context_parallel_size
        tp_size = parallel.tensor_parallel_size
        if output_size % tp_size:
            raise ValueError("DCP Q output size must be divisible by TP size")
        self.qrep_active = self.group_size > 1  # Upstream MLA interface.
        self.rank_in_group = get_tensor_model_parallel_rank() % self.group_size
        super().__init__(input_size, output_size, bias=bias, quant_config=quant_config, prefix=prefix)
        if not getattr(self.quant_method, "supports_weight_preprocessing", False):
            raise ValueError(f"DCP Q replication cannot prepare local weights with {type(self.quant_method).__name__}")
        self.local_proj: _DCPDerivedColumnParallelLinear | None = None
        self.update_param_tp_status()

    def update_param_tp_status(self) -> None:
        # The child uses local TP; do not overwrite it with the group TP.
        for param in self.parameters(recurse=False):
            if isinstance(param, BasevLLMParameter):
                param.tp_rank = self.tp_rank
                param.tp_size = self.tp_size

    def prepare_weights_for_processing(self) -> None:
        """Prepare local weights before processing the group weights."""
        group_params = dict(self.named_parameters(recurse=False))
        device = next(iter(group_params.values())).device
        with torch.device(device):
            local_proj = _DCPDerivedColumnParallelLinear(
                self.input_size,
                self.output_size,
                bias=self.bias is not None,
                skip_bias_add=self.skip_bias_add,
                params_dtype=self.params_dtype,
                quant_config=self.quant_config,
                prefix=self.prefix,
                return_bias=self.return_bias,
            )
        assert local_proj.tp_size == self.tp_size * self.group_size
        assert local_proj.tp_rank == self.tp_rank * self.group_size + self.rank_in_group
        block_size = getattr(self, "weight_block_size", None)
        if block_size is not None and local_proj.output_size_per_partition % block_size[0]:
            raise ValueError("DCP Q replication requires local Q heads to align with weight quantization blocks")

        for name, local_param in local_proj.named_parameters(recurse=False):
            group_param = group_params[name]
            output_dim = getattr(group_param, "output_dim", None)
            local_data = group_param.detach()
            if output_dim is not None:
                if (
                    getattr(group_param, "packed_dim", None) == output_dim
                    and local_proj.output_size_per_partition % group_param.packed_factor
                ):
                    raise ValueError(f"DCP Q parameter {name} has a local output boundary inside a packed element")
                width = local_param.shape[output_dim]
                if group_param.shape[output_dim] != width * self.group_size:
                    raise ValueError(f"DCP Q parameter {name} cannot be split into equal local output shards")
                local_data = local_data.narrow(output_dim, self.rank_in_group * width, width)
            if local_data.shape != local_param.shape or local_data.dtype != local_param.dtype:
                raise ValueError(f"DCP Q parameter {name} does not match the local parameter shape or dtype")
            local_param.data.copy_(local_data)
        local_proj.quant_method.process_weights_after_loading(local_proj)
        local_proj.update_param_tp_status()
        local_proj.finish_weight_processing(self.local_proj)
        self.local_proj = local_proj

    def forward_local(self, x):
        """Project only this TP rank's heads using the standard Ascend linear."""
        if self.local_proj is None:
            raise RuntimeError("DCP local Q projection must be prepared during weight processing")
        return self.local_proj(x)


class AscendReplicatedLinear(ReplicatedLinear):
    """Ascend Replicated linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        bias: If true, add bias.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
        return_bias: If true, return bias together with outputs in forward pass.
        disable_tp: Take no effect for replicated linear layers.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        self.custom_op, self.tp_rank, self.tp_size = get_replicated_op(disable_tp, prefix, self)
        # If MergedReplicatedLinear, use output size of each partition.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = self.output_sizes
        else:
            self.output_partition_sizes = [output_size]

        AscendLinearBase.__init__(
            self,
            input_size,
            output_size,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

        # All the linear layer supports quant method.
        assert self.quant_method is not None
        self.quant_method.create_weights(
            self,
            self.input_size,
            [self.output_size],
            self.input_size,
            self.output_size,
            self.params_dtype,
            weight_loader=self.weight_loader,
        )

        if bias:
            self.bias = Parameter(torch.empty(self.output_size, dtype=self.params_dtype))
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)

        if self.custom_op is not None:
            self.custom_op.update_attrs()

    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        if self.custom_op is not None:
            return self.custom_op.apply(input_)

        return super().forward(input_)
