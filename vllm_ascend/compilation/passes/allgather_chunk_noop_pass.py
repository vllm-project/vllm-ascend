import torch
import torch._inductor.pattern_matcher as pm
from torch._inductor.pattern_matcher import PatternMatcherPass
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group
from vllm.logger import logger


class AllGatherChunkNoOpCleanupPass(VllmInductorPass):
    """Fold all_gather + sequence_parallel_chunk_impl into identity."""

    def __init__(self, config: VllmConfig):
        super().__init__(config)
        self.tp_group = get_tp_group()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.patterns: PatternMatcherPass = PatternMatcherPass(pass_name="npu_allgather_chunk_noop_cleanup_pass")
        self._register_patterns()

    def _all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.all_gather(x, dim=0, world_size=self.tp_size, group_name=self.tp_group.unique_name)

    def _empty(self, *args, **kwargs):
        return torch.empty(*args, dtype=self.model_dtype, device=self.device, **kwargs)

    def _register_patterns(self) -> None:
        def pattern(input: torch.Tensor) -> torch.Tensor:
            gathered = self._all_gather(input)
            return torch.ops.vllm.sequence_parallel_chunk_impl(gathered)

        def replacement(input: torch.Tensor) -> torch.Tensor:
            return input

        pm.register_replacement(pattern, replacement, [self._empty(8, 16)], pm.fwd_only, self.patterns)

    def __call__(self, graph: torch.fx.Graph) -> None:
        self.begin()
        matched_count = self.patterns.apply(graph)
        logger.debug("AllGatherChunkNoOpCleanupPass replaced %s patterns", matched_count)
        self.end_and_log()
