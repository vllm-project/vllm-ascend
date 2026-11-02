import weakref
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

# Patterns already handed to the process-global npugraph_ex backend registry.
_registered_patterns: set[str] = set()
# Pattern keys already registered into a given matcher. A module-level set cannot
# be used here because every compiled graph builds its own PatternMatcherPass; a
# weak-keyed map keeps the bookkeeping off the third-party matcher object and
# does not extend its lifetime.
_registered_by_matcher: "weakref.WeakKeyDictionary[PatternMatcherPass, set[str]]" = weakref.WeakKeyDictionary()


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

    def pattern_key(self) -> str:
        return f"{self.__class__.__name__}_{self.eps}"

    def register(self, pm_pass: PatternMatcherPass) -> None:
        # NOTE: every GraphFusionPassManager instance owns its own PatternMatcherPass,
        # and a new instance is built for each compiled graph (prefill range, decode
        # range, speculator, ...). Registering into ``pm_pass`` must therefore happen
        # unconditionally: an earlier global de-duplication skipped it for every graph
        # after the first one, which left their matchers empty and silently disabled
        # every FX fusion pass. Only the npugraph_ex backend registry is global, so
        # only that call is de-duplicated.
        pattern_id = self.pattern_key()

        registered = _registered_by_matcher.get(pm_pass)
        if registered is None:
            registered = set()
            _registered_by_matcher[pm_pass] = registered

        pattern_fn = self.get_pattern()
        replacement_fn = self.get_replacement()
        example_inputs = self.get_inputs()

        # The two registries are independent, so they are guarded and recorded
        # separately: a pattern that fails to register must stay retryable, and a
        # failure in one registry must not suppress the other. Record the key only
        # after the registration actually succeeded.
        if pattern_id not in registered:
            pm.register_replacement(pattern_fn, replacement_fn, example_inputs, pm.fwd_only, pm_pass)
            registered.add(pattern_id)

        if pattern_id not in _registered_patterns:
            nge.register_replacement(
                search_fn=pattern_fn,
                replace_fn=replacement_fn,
                example_inputs=example_inputs,
                extra_check=self.get_extra_stream_scope_check(),
            )
            # Mark this pattern as registered in the global backend registry
            _registered_patterns.add(pattern_id)
