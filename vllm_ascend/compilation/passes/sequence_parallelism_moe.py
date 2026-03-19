import torch
import torch._inductor.pattern_matcher as pm
from torch._inductor.pattern_matcher import PatternMatcherPass
from vllm.compilation.passes.vllm_inductor_pass import PatternPrettyPrinter, VllmInductorPass
from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.logger import logger

from vllm_ascend.compilation.passes.sequence_parallelism import (
    _SequenceParallelPatternHelper,
    get_sp_min_token_num,
)


class MiddleLayerAllgatherAddRMSNormPattern(_SequenceParallelPatternHelper):
    """Replaces all_gather + slice + AddRMSNormBias with AddRMSNormBias +
    all_gather to avoid middle-layer shape mismatch."""

    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(eps, vllm_config.model_config.dtype, torch.npu.current_device())

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
            input: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor, num_tokens
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_gather = self._all_gather(input)
            x_sliced = all_gather[:num_tokens]
            result, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(x_sliced, residual, weight, None, self.eps)

            return result, residual

        def replacement(
            input: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor, num_tokens
        ) -> tuple[torch.Tensor, torch.Tensor]:
            residual = torch.ops.vllm.maybe_chunk_residual(input, residual)
            result, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(input, residual, weight, None, self.eps)
            all_gather = self._all_gather(result)
            return all_gather, residual

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass, scalar_workaround=self.get_scalar_inputs()
        )


class LastLayerAllgatherRMSNormPattern(_SequenceParallelPatternHelper):
    """Same as MiddleLayerAllgatherAddRMSNormPattern but for the last layer (no residual)
    all_gather + RMSNorm fusion."""

    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(eps, vllm_config.model_config.dtype, torch.npu.current_device())

    def get_inputs(self):
        input = self.empty(5, 16)
        weight = self.empty(16)
        residual = self.empty(8, 16)
        return [input, weight, residual]

    def get_scalar_inputs(self):
        return {"num_tokens": 8}

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(
            input: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor, num_tokens
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_gather = self._all_gather(input)
            x_sliced = all_gather[:num_tokens]
            result, _, _ = torch.ops._C_ascend.npu_add_rms_norm_bias(x_sliced, residual, weight, None, self.eps)

            return result

        def replacement(
            input: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor, num_tokens
        ) -> tuple[torch.Tensor, torch.Tensor]:
            residual = torch.ops.vllm.maybe_chunk_residual(input, residual)
            result, _, _ = torch.ops._C_ascend.npu_add_rms_norm_bias(input, residual, weight, None, self.eps)
            all_gather = self._all_gather(result)
            return all_gather

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass, scalar_workaround=self.get_scalar_inputs()
        )


class Qwen3VLMiddleLayerAllgatherAddRMSNormPattern(_SequenceParallelPatternHelper):
    """Replaces all_gather + slice + add + AddRMSNormBias with add(chunk) +
    AddRMSNormBias + all_gather for Qwen3-VL-style all_gather path."""

    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(eps, vllm_config.model_config.dtype, torch.npu.current_device())

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
            num_tokens,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_gather = self._all_gather(input)
            x_sliced = all_gather[:num_tokens]
            add_ = x_sliced + deepstack_input_embeds
            result, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(add_, residual, weight, None, self.eps)

            return result, residual

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
            deepstack_input_embeds: torch.Tensor,
            num_tokens,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            chunk = deepstack_input_embeds.chunk(self.tp_size)[self.tp_rank]
            add_ = input + chunk
            residual = torch.ops.vllm.maybe_chunk_residual(input, residual)
            result, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(add_, residual, weight, None, self.eps)
            all_gather = self._all_gather(result)
            return all_gather, residual

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass, scalar_workaround=self.get_scalar_inputs()
        )


class AllGatherChunkNoOpPattern(_SequenceParallelPatternHelper):
    """Folds all_gather + sequence_parallel_chunk_impl into identity (no-op)."""

    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(eps, vllm_config.model_config.dtype, torch.npu.current_device())

    def get_inputs(self):
        return [self.empty(8, 16)]

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(input: torch.Tensor) -> torch.Tensor:
            gathered = self._all_gather(input)
            return torch.ops.vllm.sequence_parallel_chunk_impl(gathered)

        def replacement(input: torch.Tensor) -> torch.Tensor:
            return input

        pm.register_replacement(pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass)


class SequenceParallelismMoePass(VllmInductorPass):
    """Sequence parallelism AllGather epilogue pass.

    Applies AllGather-based patterns: MiddleLayerAllgatherAddRMSNormPattern,
    LastLayerAllgatherRMSNormPattern, Qwen3VLMiddleLayerAllgatherAddRMSNormPattern,
    and AllGatherChunkNoOpPattern (all_gather + sequence_parallel_chunk_impl -> identity).
    """

    def __init__(self, config: VllmConfig):
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(pass_name="npu_sequence_parallelism_allgather_ep_pass")

        for epsilon in [1e-5, 1e-6]:
            MiddleLayerAllgatherAddRMSNormPattern(config, epsilon).register(self.patterns)
            LastLayerAllgatherRMSNormPattern(config, epsilon).register(self.patterns)
            Qwen3VLMiddleLayerAllgatherAddRMSNormPattern(config, epsilon).register(self.patterns)

        AllGatherChunkNoOpPattern(config).register(self.patterns)

        self.min_tokens = get_sp_min_token_num(config)

    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        logger.debug(f"before apply replacement {graph}")
        self.matched_count = self.patterns.apply(graph)
        logger.debug(f"after apply replacement {graph}")
        logger.debug("SequenceParallelismMoePass replaced %s patterns", self.matched_count)
        pattern_idx = 0
        for pattern_entry in self.patterns.patterns.values():
            for p in pattern_entry:
                p_str = PatternPrettyPrinter.run(p.pattern)
                logger.debug("Pattern %d: %s", pattern_idx, p_str)
                pattern_idx += 1
        self.end_and_log()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        applicable = compile_range.start >= self.min_tokens
        logger.debug(f"SequenceParallelismMoePass {compile_range=} {applicable=}")
        return applicable
