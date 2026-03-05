import torch
import torch._inductor.pattern_matcher as pm
from torch._inductor.pattern_matcher import PatternMatcherPass
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group, tensor_model_parallel_all_reduce
from vllm.logger import logger

from vllm_ascend.compilation.passes.noop_elimination import AscendNoOpEliminationPass
from vllm_ascend.utils import is_moe_model

SP_THRESHOLD = 1000


def get_sp_threshold(config: VllmConfig):
    logger.info(f"is_moe_model {is_moe_model(config)}")
    if is_moe_model(config):
        return 1

    additional_config = config.additional_config if config.additional_config is not None else {}
    return additional_config.get("sp_threshold", SP_THRESHOLD)


class _SequenceParallelPatternHelper:
    """Helper for sequence parallelism patterns."""

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


class AscendMiddleAllReduceRMSNormPattern(_SequenceParallelPatternHelper):
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


class AscendLastAllReduceRMSNormPattern(_SequenceParallelPatternHelper):
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


class AscendQwen3VLMiddleAllReduceRMSNormPattern(_SequenceParallelPatternHelper):
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


class AllgatherRMSNormPattern(_SequenceParallelPatternHelper
                                                 ):

    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(eps, vllm_config.model_config.dtype,
                         torch.npu.current_device())

    def get_inputs(self):
        input = self.empty(5, 16)
        weight = self.empty(16)
        residual = self.empty(8, 16)
        # num_tokens = 8
        return [input, weight, residual]
    
    def get_scalar_inputs(self):
        return {"num_tokens": 8}

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
            num_tokens
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_gather = self._all_gather(input)
            x_sliced = all_gather[:num_tokens]
            result, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(
                x_sliced, residual, weight, None, self.eps)

            return result, residual

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
            num_tokens
        ) -> tuple[torch.Tensor, torch.Tensor]:
            residual = torch.ops.vllm.maybe_chunk_residual(input, residual)
            result, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(
                input, residual, weight, None, self.eps)
            all_gather = self._all_gather(result)
            return all_gather, residual

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass, scalar_workaround=self.get_scalar_inputs())

class LastLayerAllgatherRMSNormPattern(_SequenceParallelPatternHelper
                                                 ):

    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(eps, vllm_config.model_config.dtype,
                         torch.npu.current_device())

    def get_inputs(self):
        input = self.empty(5, 16)
        weight = self.empty(16)
        residual = self.empty(8, 16)
        return [input, weight, residual]

    def get_scalar_inputs(self):
        return {"num_tokens": 8}

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
            num_tokens
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_gather = self._all_gather(input)
            x_sliced = all_gather[:num_tokens]
            result, _, _ = torch.ops._C_ascend.npu_add_rms_norm_bias(
                x_sliced, residual, weight, None, self.eps)

            return result

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
            num_tokens
        ) -> tuple[torch.Tensor, torch.Tensor]:
            residual = torch.ops.vllm.maybe_chunk_residual(input, residual)
            result, _, _ = torch.ops._C_ascend.npu_add_rms_norm_bias(
                input, residual, weight, None, self.eps)
            all_gather = self._all_gather(result)
            return all_gather

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass, scalar_workaround=self.get_scalar_inputs())


class AllgatherAddRMSNormPattern(_SequenceParallelPatternHelper
                                                 ):

    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(eps, vllm_config.model_config.dtype,
                         torch.npu.current_device())

    def get_inputs(self):
        input = self.empty(5, 16)
        weight = self.empty(16)
        residual = self.empty(8, 16)
        deepstack_input_embeds = self.empty(8, 16)
        return [input, weight, residual, deepstack_input_embeds]

    def get_scalar_inputs(self):
        return {"num_tokens": 8}

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
            deepstack_input_embeds: torch.Tensor,
            num_tokens
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_gather = self._all_gather(input)
            x_sliced = all_gather[:num_tokens]
            add_ = x_sliced + deepstack_input_embeds
            result, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(
                add_, residual, weight, None, self.eps)

            return result, residual

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
            deepstack_input_embeds: torch.Tensor,
            num_tokens
        ) -> tuple[torch.Tensor, torch.Tensor]:
            chunk = deepstack_input_embeds.chunk(self.tp_size)[self.tp_rank]
            add_ = input + chunk
            residual = torch.ops.vllm.maybe_chunk_residual(input, residual)
            result, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(
                add_, residual, weight, None, self.eps)
            all_gather = self._all_gather(result)
            return all_gather, residual

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass, scalar_workaround=self.get_scalar_inputs())     




class AscendSequenceParallelismPass(VllmInductorPass):
    def __init__(self, config: VllmConfig):
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(pass_name="npu_sequence_parallelism_pass")
        self.noop_cleanup = AscendNoOpEliminationPass(config)

        for epsilon in [1e-5, 1e-6]:
            AscendMiddleAllReduceRMSNormPattern(config, epsilon).register(self.patterns)

            AscendLastAllReduceRMSNormPattern(config, epsilon).register(self.patterns)

            AscendQwen3VLMiddleAllReduceRMSNormPattern(config, epsilon).register(self.patterns)
            
            AllgatherRMSNormPattern(config, epsilon).register(self.patterns)
            LastLayerAllgatherRMSNormPattern(config, epsilon).register(self.patterns)
            AllgatherAddRMSNormPattern(config, epsilon).register(self.patterns)

        self.min_tokens = get_sp_threshold(config)

    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        self.noop_cleanup(graph)
        logger.info(f"after noop_cleanup {graph.graph}")
        self.matched_count = self.patterns.apply(graph)
        logger.info("Replaced %s patterns", self.matched_count)
        logger.info(f"after apply replacement {graph.graph}")
        
        from torch._inductor.pattern_matcher import PatternPrettyPrinter
        pattern_idx = 0
        for pattern_entry in self.patterns.patterns.values():
            for p in pattern_entry:
                p_str = PatternPrettyPrinter.run(p.pattern)
                logger.info("Pattern %d: %s", pattern_idx, p_str)
                pattern_idx += 1
        
        self.end_and_log()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        """
        Check if the pass is applicable for the current configuration.
        """
        applicable = compile_range.start >= self.min_tokens
        logger.info(f"SequenceParallelismPass {compile_range=} {applicable=}")
        return applicable
