# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA head partitions shared by projection loaders and recurrent caches."""

from vllm.distributed import divide, get_pcp_group, get_tp_group
from vllm.model_executor.layers.linear import ColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.models.kimi_k3.nvidia.kda import _KimiGDNMergedColumnParallelLinear


def get_kda_parallel_size(parallel_config) -> int:
    return parallel_config.tensor_parallel_size * parallel_config.prefill_context_parallel_size


def get_kda_parallel_rank() -> int:
    # Mooncake worker IDs use the same TP-major, PCP-minor order.
    return get_tp_group().rank_in_group * get_pcp_group().world_size + get_pcp_group().rank_in_group


def get_kda_state_shape(vllm_config):
    config = vllm_config.model_config.hf_text_config.linear_attn_config
    size = get_kda_parallel_size(vllm_config.parallel_config)
    num_spec = vllm_config.speculative_config.num_speculative_tokens if vllm_config.speculative_config else 0
    shapes = MambaStateShapeCalculator.kda_state_shape(
        size,
        config["num_heads"],
        config["head_dim"],
        conv_kernel_size=config["short_conv_kernel_size"],
        num_spec=num_spec,
    )
    if vllm_config.cache_config.use_kda_recoverssm:
        return MambaStateShapeCalculator.append_kda_recoverssm_record(
            shapes,
            config["num_heads"],
            config["head_dim"],
            tp_world_size=size,
            spec_query_len=1 + num_spec,
        )
    return shapes


class KDAColumnParallelLinear(ColumnParallelLinear):
    def __init__(self, input_size: int, output_size: int, **kwargs) -> None:
        self.pcp_group = get_pcp_group()
        self.tp_group = get_tp_group()
        super().__init__(
            input_size,
            output_size,
            tp_rank=get_kda_parallel_rank(),
            tp_size=self.tp_group.world_size * self.pcp_group.world_size,
            **kwargs,
        )

    def forward(self, input_):
        bias = self.bias if not self.skip_bias_add else None
        output = self.quant_method.apply(self, input_, bias)
        if self.gather_output:
            if self.pcp_group.world_size > 1:
                output = self.pcp_group.all_gather(output, dim=-1)
            if self.tp_group.world_size > 1:
                output = self.tp_group.all_gather(output, dim=-1)
        if not self.return_bias:
            return output
        return output, self.bias if self.skip_bias_add else None


class KDAMergedColumnParallelLinear(MergedColumnParallelLinear, KDAColumnParallelLinear):
    def __init__(self, input_size: int, output_sizes: list[int], **kwargs) -> None:
        self.output_sizes = output_sizes.copy()
        # Keep the merged checkpoint loaders, but allocate with explicit ranks.
        KDAColumnParallelLinear.__init__(self, input_size, sum(self.output_sizes), **kwargs)


class KDAInputParallelLinear(_KimiGDNMergedColumnParallelLinear, KDAColumnParallelLinear):
    def __init__(self, input_size: int, output_sizes: list[int], tp_size: int, **kwargs) -> None:
        self.replicated_shard_id = 4
        self.output_sizes = output_sizes.copy()
        self.output_sizes[self.replicated_shard_id] *= tp_size
        KDAColumnParallelLinear.__init__(self, input_size, sum(self.output_sizes), **kwargs)


class KDARowParallelLinear(RowParallelLinear):
    def __init__(self, input_size: int, output_size: int, **kwargs) -> None:
        self.pcp_group = get_pcp_group()
        self.tp_group = get_tp_group()
        size = self.tp_group.world_size * self.pcp_group.world_size
        # The release RowParallelLinear has no explicit rank/size arguments.
        super().__init__(divide(input_size, size), output_size, disable_tp=True, **kwargs)
        self.input_size = input_size
        self.disable_tp = False
        self.tp_size = size
        self.tp_rank = get_kda_parallel_rank()
        self.update_param_tp_status()

    def forward(self, input_):
        if not self.input_is_parallel:
            input_ = input_.chunk(self.tp_size, dim=-1)[self.tp_rank].contiguous()
        bias = None if self.tp_rank > 0 or self.skip_bias_add else self.bias
        output = self.quant_method.apply(self, input_, bias)
        # Sequence parallelism defers only TP reduction to the decoder.
        if self.pcp_group.world_size > 1:
            output = self.pcp_group.all_reduce(output)
        if self.reduce_results:
            if self.tp_group.world_size > 1:
                output = self.tp_group.all_reduce(output)
        if not self.return_bias:
            return output
        return output, self.bias if self.skip_bias_add else None
