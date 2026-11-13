from abc import ABC, abstractmethod
from collections.abc import Callable

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
        """Optional per-match predicate for constraints the pattern cannot express.

        Return ``None`` to accept every structural match. Use this for attribute
        constraints (dtype, static shapes, collective group) rather than encoding
        them as extra pattern variants.
        """
        return None

    def get_scalar_workaround(self) -> dict[str, float | int] | None:
        """Optional ``scalar_workaround`` for patterns taking non-tensor arguments."""
        return None

    def pattern_key(self) -> str:
        return f"{self.__class__.__name__}_{self.eps}"

    def register(self, pm_pass: PatternMatcherPass) -> None:
        # Create a unique identifier for this pattern
        pattern_id = self.pattern_key()

        # Skip registration if this pattern has already been registered globally
        if pattern_id in _registered_patterns:
            return

        pattern_fn = self.get_pattern()
        replacement_fn = self.get_replacement()
        example_inputs = self.get_inputs()
        extra_check = self.get_extra_check()
        scalar_workaround = self.get_scalar_workaround()

        # Only forward the optional kwargs when set, so the defaults of the
        # underlying registries keep applying.
        optional_kwargs: dict = {}
        if scalar_workaround:
            optional_kwargs["scalar_workaround"] = scalar_workaround

        pm.register_replacement(
            pattern_fn,
            replacement_fn,
            example_inputs,
            pm.fwd_only,
            pm_pass,
            **({"extra_check": extra_check} if extra_check is not None else {}),
            **optional_kwargs,
        )

        stream_check = self.get_extra_stream_scope_check()
        if extra_check is None:
            nge_check = stream_check
        else:

            def nge_check(match, _stream=stream_check, _extra=extra_check):
                return _stream(match) and _extra(match)

        nge.register_replacement(
            search_fn=pattern_fn,
            replace_fn=replacement_fn,
            example_inputs=example_inputs,
            extra_check=nge_check,
            **optional_kwargs,
        )

        # Mark this pattern as registered
        _registered_patterns.add(pattern_id)
