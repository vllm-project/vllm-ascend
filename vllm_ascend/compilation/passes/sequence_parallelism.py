import torch
import torch._inductor.pattern_matcher as pm
from torch._inductor.pattern_matcher import PatternMatcherPass
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group, tensor_model_parallel_all_reduce
from vllm.logger import logger

from vllm_ascend.compilation.passes.noop_elimination import NoOpEliminationPass
from vllm_ascend.utils import is_moe_model

SP_MIN_TOKEN_NUM_DEFAULT = 1000


def get_sp_min_token_num(config: VllmConfig) -> int:
    if is_moe_model(config):
        return 1

    return SP_MIN_TOKEN_NUM_DEFAULT


class _SequenceParallelPatternHelper:
    """Helper for sequence parallelism patterns.

    Provides TP communication helper methods: _all_reduce, _reduce_scatter,
    _all_gather, and tensor creation utilities.
    """

    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str,
    ):
        self.eps = epsilon
        self.dtype = dtype
        self.device = device
        self.tp_group = get_tp_group()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tp_group().rank_in_group

    def _all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return tensor_model_parallel_all_reduce(x)

    def _reduce_scatter(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.reduce_scatter(x, dim=0, world_size=self.tp_size, group_name=self.tp_group.unique_name)

    def _all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.all_gather(x, dim=0, world_size=self.tp_size, group_name=self.tp_group.unique_name)

    def empty(self, *args, **kws):
        return torch.empty(*args, dtype=self.dtype, device="npu", **kws)


class MiddleAllReduceRMSNormPattern(_SequenceParallelPatternHelper):
    """Replaces all_reduce + AddRMSNormBias with reduce_scatter + AddRMSNormBias
    + all_gather for middle-layer sequence parallelism."""

    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(eps, vllm_config.model_config.dtype, torch.npu.current_device())

    def empty(self, *args, **kws):
        return torch.empty(*args, dtype=self.dtype, device="npu", **kws)

    def get_inputs(self):
        """
        Generate example inputs.
        """
        input = self.empty(8, 16)
        weight = self.empty(16)
        residual = self.empty(8, 16)
        return [input, weight, residual]

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x = self._all_reduce(input)
            result, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(x, residual, weight, None, self.eps)

            return result, residual

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            reduce_scatter = self._reduce_scatter(input)
            residual = torch.ops.vllm.maybe_chunk_residual(reduce_scatter, residual)
            result, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(
                reduce_scatter, residual, weight, None, self.eps
            )
            all_gather = self._all_gather(result)
            return all_gather, residual

        pm.register_replacement(pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass)


class LastAllReduceRMSNormPattern(_SequenceParallelPatternHelper):
    """Same as MiddleAllReduceRMSNormPattern but for the last layer
    (no residual backprop)."""

    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(eps, vllm_config.model_config.dtype, torch.npu.current_device())

    def get_inputs(self):
        input = self.empty(8, 16)
        weight = self.empty(16)
        residual = self.empty(8, 16)
        return [input, weight, residual]

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
        ) -> torch.Tensor:
            x = self._all_reduce(input)
            result, _, _ = torch.ops._C_ascend.npu_add_rms_norm_bias(x, residual, weight, None, self.eps)

            return result

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
        ) -> torch.Tensor:
            reduce_scatter = self._reduce_scatter(input)
            residual = torch.ops.vllm.maybe_chunk_residual(reduce_scatter, residual)
            result, _, _ = torch.ops._C_ascend.npu_add_rms_norm_bias(reduce_scatter, residual, weight, None, self.eps)
            all_gather = self._all_gather(result)
            return all_gather

        pm.register_replacement(pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass)


class Qwen3VLMiddleAllReduceRMSNormPattern(_SequenceParallelPatternHelper):
    """For Qwen3-VL middle layers with hidden_states + deepstack_input_embeds add.

    Replaces all_reduce + add + AddRMSNormBias with reduce_scatter +
    chunk(deepstack_input_embeds) + add + AddRMSNormBias + all_gather.
    """

    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(eps, vllm_config.model_config.dtype, torch.npu.current_device())

    def get_inputs(self):
        input = self.empty(8, 16)
        weight = self.empty(16)
        residual = self.empty(8, 16)
        deepstack_input_embeds = self.empty(8, 16)
        return [input, weight, residual, deepstack_input_embeds]

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
            deepstack_input_embeds: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x = self._all_reduce(input)
            add_ = x + deepstack_input_embeds
            result, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(add_, residual, weight, None, self.eps)

            return result, residual

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
            deepstack_input_embeds: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            reduce_scatter = self._reduce_scatter(input)
            chunk = deepstack_input_embeds.chunk(self.tp_size)[self.tp_rank]
            add_ = reduce_scatter + chunk
            residual = torch.ops.vllm.maybe_chunk_residual(reduce_scatter, residual)
            result, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(add_, residual, weight, None, self.eps)
            all_gather = self._all_gather(result)
            return all_gather, residual

        pm.register_replacement(pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass)


class SequenceParallelismPass(VllmInductorPass):
    """Sequence parallelism compilation pass.

    Registers and applies the above patterns. Runs noop cleanup first, then
    uses token range to determine whether to enable SP.
    """

    def __init__(self, config: VllmConfig):
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(pass_name="npu_sequence_parallelism_pass")
        self.noop_cleanup = NoOpEliminationPass(config)

        for epsilon in [1e-5, 1e-6]:
            MiddleAllReduceRMSNormPattern(config, epsilon).register(self.patterns)

            LastAllReduceRMSNormPattern(config, epsilon).register(self.patterns)

            Qwen3VLMiddleAllReduceRMSNormPattern(config, epsilon).register(self.patterns)

        self.min_tokens = get_sp_min_token_num(config)

    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        self.noop_cleanup(graph)  # Eliminate redundant view-like operations
        logger.debug(f"after noop_cleanup {graph.graph}")
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)
        logger.debug(f"after apply replacement {graph.graph}")

        from torch._inductor.pattern_matcher import PatternPrettyPrinter

        pattern_idx = 0
        for pattern_entry in self.patterns.patterns.values():
            for p in pattern_entry:
                p_str = PatternPrettyPrinter.run(p.pattern)
                logger.debug("Pattern %d: %s", pattern_idx, p_str)
                pattern_idx += 1

        self.end_and_log()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        """
        Check if the pass is applicable for the current configuration.
        """
        applicable = compile_range.start >= self.min_tokens
        logger.debug(f"SequenceParallelismPass {compile_range=} {applicable=}")
        return applicable
