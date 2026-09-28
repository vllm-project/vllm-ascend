#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
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
#
import functools
import hashlib
import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
import torch._inductor.pattern_matcher as pm
from torch._inductor.pattern_matcher import Match, PatternMatcherPass
from vllm.config import VllmConfig
from vllm.logger import logger

try:
    import npugraph_ex as nge
except ImportError:
    import torchair as nge

from vllm_ascend.compilation.passes.utils.npugraph_ex_utils_check import extra_stream_scope_check

# Global set to track registered patterns and prevent duplicates
_registered_patterns: set[str] = set()


def _example_inputs_signature(example_inputs: list) -> str:
    parts = []
    for inp in example_inputs:
        if isinstance(inp, torch.Tensor):
            parts.append(f"{tuple(inp.shape)}:{inp.dtype}")
        else:
            parts.append(type(inp).__name__)
    return ",".join(parts)


def _matched_main_input_has_expected_width(match: Match, main_argname: str, expected_width: int) -> bool | None:
    """Shape guard against cross-model pattern application.

    Returns True/False when the matched main input's last-dim width is
    verifiable from the match, and None when it is not (in which case the
    caller rejects the match to stay on the safe side).
    """
    if not hasattr(match, "kwargs"):
        return None
    main_node = match.kwargs.get(main_argname)
    if main_node is None or not hasattr(main_node, "meta"):
        return None
    val = main_node.meta.get("val")
    if not isinstance(val, torch.Tensor) or val.dim() < 2:
        return None
    try:
        return bool(val.shape[-1] == expected_width)
    except Exception:
        # Symbolic widths (SymInt) compare to a SymBool that may not be
        # statically resolvable (e.g. GuardOnDataDependentSymNode); bool()
        # would raise inside the caller's conditional and crash the
        # compilation. Treat as unverifiable and reject the match
        # (reject-on-unknown), mirroring _wrap_search_fn_with_width_guard.
        return None


def _wrap_search_fn_with_width_guard(
    search_fn: Callable,
    main_argname: str,
    expected_width: int,
    pattern_name: str,
) -> Callable:
    """Guard a pattern search function against cross-model application.

    torch's pattern_matcher verifies a structurally matched pattern by
    re-tracing search_fn with the real shapes from the matched graph (the
    check_fn wrapper created by register_replacement). Shape mismatches
    surface during that re-trace as tensor-op errors; e.g. split_with_sizes
    raises "Split sizes add up to X but got the tensor's size of Y" as a
    ValueError, which check_fn does NOT catch (it only catches RuntimeError),
    so the error escapes and kills the whole compilation. This is exactly how
    a target-sized fusion pattern crashes the spec-decode draft model's graph
    in FULL (npugraph_ex) mode, where the pattern registry is process-global.

    Raising RuntimeError ourselves when the main input's width provably
    differs turns the crash into torch's designed graceful rejection path
    (check_fn -> log_trace_failure -> return False): the pattern is skipped
    for that graph and compilation proceeds. The guard only fires when the
    width is a concrete int that definitely mismatches; symbolic or unknown
    widths fall through to the original re-trace verification.
    """
    params = list(inspect.signature(search_fn).parameters)
    if main_argname not in params:
        return search_fn
    main_idx = params.index(main_argname)

    @functools.wraps(search_fn)
    def guarded(*args, **kwargs):
        if main_idx < len(args):
            tensor = args[main_idx]
            shape = getattr(tensor, "shape", None)
            if shape is not None and len(shape) > 0:
                try:
                    width_matches = bool(shape[-1] == expected_width)
                except Exception:
                    width_matches = None
                if width_matches is False:
                    raise RuntimeError(
                        f"{pattern_name}: main input '{main_argname}' has width {shape[-1]}, "
                        f"but this pattern instance was registered for width {expected_width} "
                        "(another model configuration); declining to apply"
                    )
        return search_fn(*args, **kwargs)

    return guarded


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

    def _get_main_input_info(self, example_inputs: list[torch.Tensor]) -> tuple[str | None, int | None]:
        """Find (argname, last-dim width) of the widest >=2-D example input.

        The widest input is the main activation of the pattern (e.g. the fused
        qkv tensor for QKNormRope patterns). Its width is the shape identity
        used by the registration guard: the target model and the spec-decode
        draft model differ exactly in this width.
        """
        argnames = list(inspect.signature(self.get_pattern()).parameters.keys())
        main_argname: str | None = None
        main_width: int | None = None
        for argname, inp in zip(argnames, example_inputs):
            if isinstance(inp, torch.Tensor) and inp.dim() >= 2:
                if main_width is None or inp.shape[-1] > main_width:
                    main_argname = argname
                    main_width = inp.shape[-1]
        return main_argname, main_width

    def get_extra_check(self, main_argname: str | None = None, expected_width: int | None = None) -> Callable:
        stream_scope_check = self.get_extra_stream_scope_check()
        if main_argname is None or expected_width is None:
            return stream_scope_check

        pattern_name = self.__class__.__name__

        def check_with_shape_guard(match: Match) -> bool:
            if not stream_scope_check(match):
                return False
            width_matches = _matched_main_input_has_expected_width(match, main_argname, expected_width)
            if width_matches is None:
                # Cannot verify the input width from this match (unexpected
                # Match flavor). Reject rather than risk applying a pattern
                # baked with another model's shapes: a wrong fusion here
                # crashes compilation, while skipping it only loses fusion.
                logger.debug(
                    "Rejecting %s pattern: main input width not verifiable from match",
                    pattern_name,
                )
                return False
            if not width_matches:
                logger.debug(
                    "Rejecting %s pattern: main input width != %d (pattern registered for another model)",
                    pattern_name,
                    expected_width,
                )
                return False
            return True

        return check_with_shape_guard

    def register(self, pm_pass: PatternMatcherPass) -> None:
        pattern_fn = self.get_pattern()
        replacement_fn = self.get_replacement()
        example_inputs = self.get_inputs()

        # The pattern id must include the example-input shapes. The target
        # model and the spec-decode draft model instantiate the same pattern
        # classes with different head configurations, and a class+eps-only id
        # silently drops the second model's registration. Since the
        # npugraph_ex/torchair replacement registry is process-global, the
        # first-registered (target-sized) pattern then gets applied to the
        # draft graph and crashes with e.g. "Split sizes add up to 6144 but
        # got the tensor's size of 4096".
        signature = _example_inputs_signature(example_inputs)
        pattern_id = f"{self.__class__.__name__}_{self.eps}_{signature}"

        # Skip registration if this pattern has already been registered globally
        if pattern_id in _registered_patterns:
            return

        main_argname, main_width = self._get_main_input_info(example_inputs)

        # Wrap the search function with a width guard BEFORE registration so
        # that re-trace verification of a cross-model match raises RuntimeError
        # (caught by pattern_matcher's check_fn) instead of letting tensor ops
        # fail with e.g. ValueError, which escapes and crashes compilation.
        if main_argname is not None and main_width is not None:
            pattern_fn = _wrap_search_fn_with_width_guard(pattern_fn, main_argname, main_width, self.__class__.__name__)

        # Unique closure names per shape keep the target and draft variants
        # distinct in registries that key patterns by function name.
        shape_tag = hashlib.md5(signature.encode(), usedforsecurity=False).hexdigest()[:8]
        pattern_fn.__name__ = f"{pattern_fn.__name__}_{shape_tag}"
        replacement_fn.__name__ = f"{replacement_fn.__name__}_{shape_tag}"

        pm.register_replacement(pattern_fn, replacement_fn, example_inputs, pm.fwd_only, pm_pass)

        try:
            nge.register_replacement(
                search_fn=pattern_fn,
                replace_fn=replacement_fn,
                example_inputs=example_inputs,
                extra_check=self.get_extra_check(main_argname, main_width),
            )
        except RuntimeError as e:
            if "Duplicate pattern" in str(e):
                logger.warning(
                    "Pattern %s (eps=%s, inputs=%s) was rejected by the npugraph_ex/torchair "
                    "registry as a duplicate; matching graphs will run unfused.",
                    self.__class__.__name__,
                    self.eps,
                    signature,
                )
            else:
                raise

        # Mark this pattern as registered
        _registered_patterns.add(pattern_id)
