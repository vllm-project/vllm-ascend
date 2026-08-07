# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Breakable ACL graph capture/replay.

This is an alternative to :class:`ACLGraphWrapper` that replaces vLLM's
torch.compile-based FX graph splitting with runtime stream-capture
breaks.

The idea (inspired by sgl-project/sglang#19102): instead of pre-splitting
the model into many pieces at attention boundaries, a
single capture context drives the whole forward and intercepts
attention / kv-cache custom ops at the dispatcher to end the current
stream capture, run the op eagerly, and resume capture.

The captured artifact is a list of zero-arg callables -- the bound
``ACLGraph.replay`` for graph segments, or the user fn for eager
segments -- replayed in order at inference time.

Eager segments must operate on the same static buffers used during
capture so subsequent graph segments read the same memory addresses.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import get_forward_context

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.compilation.acl_graph import (
    get_draft_graph_params,
    get_draft_graph_prefill_params,
    get_graph_params,
    weak_ref_workspaces,
)


class BreakableACLGraphWrapper(BreakableCUDAGraphWrapper):
    def __init__(
        self,
        runnable: Callable[..., Any],
        vllm_config: VllmConfig,
        use_eagle: bool = False,
        enable_enpu: bool = False,
    ) -> None:
        super().__init__(
            runnable=runnable,
            vllm_config=vllm_config,
        )

        self.use_eagle = use_eagle
        self.enable_enpu = enable_enpu

    def _capture(
        self,
        entry: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        forward_context = get_forward_context()
        is_full_capture = forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
        if is_full_capture:
            # Ascend FULL graph attention creates task groups and records the
            # mutable graph parameters only while this flag is set.
            forward_context.capturing = True

        output = super()._capture(entry, args, kwargs)

        if is_full_capture:
            # Keep the same workspace lifetime contract as ACLGraphWrapper.
            weak_ref_workspaces(get_graph_params())
            weak_ref_workspaces(get_draft_graph_params())
            weak_ref_workspaces(get_draft_graph_prefill_params())

        return output

    def _replay(
        self,
        entry: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        forward_context = get_forward_context()
        if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL:
            # Match ACLGraphWrapper's ordering between async attention
            # parameter updates and the previous/current FULL graph replay.
            is_draft_eagle = _EXTRA_CTX.is_draft_model and self.use_eagle
            if not self.enable_enpu and not is_draft_eagle:
                torch.npu.current_stream().synchronize()
        super()._replay(entry, args, kwargs)
        return entry.output
