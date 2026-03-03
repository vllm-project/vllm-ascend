from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
import torch._inductor.pattern_matcher as pm
import torchair
from torch._inductor.pattern_matcher import PatternMatcherPass
from vllm.config import VllmConfig

from vllm_ascend.compilation.passes.utils.npugraph_ex_utils_check import extra_stream_scope_check

# Global set to track registered patterns and prevent duplicates
_registered_patterns: set[str] = set()


class BasePattern(ABC):
    def __init__(self, vllm_config: VllmConfig, pattern_id: str | None = None):
        self.vllm_config = vllm_config
        self.dtype = vllm_config.model_config.dtype
        self.pattern_id = pattern_id or self.__class__.__name__

    @abstractmethod
    def get_inputs(self) -> list[torch.Tensor]:
        pass

    @abstractmethod
    def get_pattern(self) -> Callable:
        pass

    @abstractmethod
    def get_replacement(self) -> Callable:
        pass

    def get_extra_stream_scope_check(self):
        return extra_stream_scope_check

    def register(self, pm_pass: PatternMatcherPass) -> None:
        # Skip registration if this pattern has already been registered globally
        if self.pattern_id in _registered_patterns:
            return

        pattern_fn = self.get_pattern()
        replacement_fn = self.get_replacement()
        example_inputs = self.get_inputs()

        pm.register_replacement(pattern_fn, replacement_fn, example_inputs, pm.fwd_only, pm_pass)

        torchair.register_replacement(
            search_fn=pattern_fn,
            replace_fn=replacement_fn,
            example_inputs=example_inputs,
            extra_check=self.get_extra_stream_scope_check(),
        )

        # Mark this pattern as registered
        _registered_patterns.add(self.pattern_id)
