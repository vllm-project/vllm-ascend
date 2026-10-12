from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch
import torch._inductor.pattern_matcher as pm
from torch._inductor.pattern_matcher import PatternMatcherPass
from vllm.config import VllmConfig

try:
    import npugraph_ex as nge
except ImportError:
    import torchair as nge

from vllm_ascend.compilation.passes.utils.npugraph_ex_utils_check import extra_stream_scope_check

# Global set to track registered patterns and prevent duplicates
_registered_patterns: set[str] = set()


class BasePattern(ABC):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        self.vllm_config = vllm_config
        self.dtype = vllm_config.model_config.dtype
        self.eps = eps

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

    def get_extra_check(self):
        return lambda match: True

    def get_scalar_workaround(self) -> dict[str, float | int] | None:
        return None

    def get_nge_inputs(self) -> list[Any]:
        return self.get_inputs()

    def pattern_key(self) -> str:
        return f"{self.__class__.__name__}_{self.eps}"

    def register(self, pm_pass: PatternMatcherPass) -> None:
        # Create a unique identifier for this pattern
        pattern_id = self.pattern_key()

        pattern_fn = self.get_pattern()
        replacement_fn = self.get_replacement()
        example_inputs = self.get_inputs()

        # PatternMatcherPass instances are local to a pass manager, so always
        # register the pattern on the supplied pass. Only the backend registry
        # is global and needs duplicate protection.
        pm.register_replacement(
            pattern_fn,
            replacement_fn,
            example_inputs,
            pm.fwd_only,
            pm_pass,
            extra_check=self.get_extra_check(),
            scalar_workaround=self.get_scalar_workaround(),
        )

        if pattern_id in _registered_patterns:
            return

        extra_check = self.get_extra_check()
        stream_check = self.get_extra_stream_scope_check()

        nge.register_replacement(
            search_fn=pattern_fn,
            replace_fn=replacement_fn,
            example_inputs=self.get_nge_inputs(),
            extra_check=lambda match: stream_check(match) and extra_check(match),
        )

        # Mark this pattern as registered
        _registered_patterns.add(pattern_id)
