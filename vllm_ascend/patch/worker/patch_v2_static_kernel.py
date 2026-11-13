# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import sys
from collections.abc import Callable
from functools import wraps
from typing import Concatenate, ParamSpec

from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import logger
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor, CreateForwardFn, CudaGraphManager

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.worker.static_kernel import prepare_static_kernels
from vllm_ascend.worker.v2.aclgraph_utils import AscendGraphManagerMixin

P = ParamSpec("P")


def _wrap_capture(
    capture: Callable[Concatenate[CudaGraphManager, CreateForwardFn, P], None],
) -> Callable[Concatenate[CudaGraphManager, CreateForwardFn, P], None]:
    @wraps(capture)
    def capture_with_static_kernels(
        self: CudaGraphManager, create_forward_fn: CreateForwardFn, *args: P.args, **kwargs: P.kwargs
    ) -> None:
        if not isinstance(self, AscendGraphManagerMixin):
            return capture(self, create_forward_fn, *args, **kwargs)
        config = get_ascend_config().ascend_compilation_config
        manager_name = type(self).__name__
        if not config.enable_static_kernel or config.enable_npugraph_ex or not self.static_kernel_warmup_enabled:
            logger.debug(
                "Skipping static kernel warmup for %s: static_kernel=%s, npugraph_ex=%s, warmup_enabled=%s",
                manager_name,
                config.enable_static_kernel,
                config.enable_npugraph_ex,
                self.static_kernel_warmup_enabled,
            )
            return capture(self, create_forward_fn, *args, **kwargs)

        def create_prepared_forward(desc: BatchExecutionDescriptor, warmup: bool) -> Callable[[CUDAGraphMode], None]:
            forward_fn = create_forward_fn(desc, warmup=warmup)
            if not warmup:
                return forward_fn

            def forward(mode: CUDAGraphMode) -> None:
                if mode == CUDAGraphMode.NONE:
                    logger.debug("Static kernel warmup started: manager=%s, descriptor=%s", manager_name, desc)
                    prepare_static_kernels(forward_fn, config)
                    logger.debug(
                        "Static kernel warmup finished: manager=%s, descriptor=%s",
                        manager_name,
                        desc,
                    )
                else:
                    forward_fn(mode)

            return forward

        return capture(self, create_prepared_forward, *args, **kwargs)

    return capture_with_static_kernels


def _wrap_end_segment(
    end_segment: Callable[[BreakableCUDAGraphCapture], None],
) -> Callable[[BreakableCUDAGraphCapture], None]:
    @wraps(end_segment)
    def end_segment_with_super_kernel(self: BreakableCUDAGraphCapture) -> None:
        graph = self._current_graph if self._capturing else None
        end_segment(self)
        # TODO: Use sys.exception() once our minimum Python version matches upstream (3.11+).
        if graph is not None and sys.exc_info()[1] is None:
            if get_ascend_config().ascend_compilation_config.enable_super_kernel:
                graph.super_kernel_optimize(optimize_options={"dcci_after_kernel_end": [".*"]})
                logger.info_once("Super kernel optimization is enabled for Breakable ACL graph segments.")

    return end_segment_with_super_kernel


CudaGraphManager.capture = _wrap_capture(CudaGraphManager.capture)
BreakableCUDAGraphCapture._end_segment = _wrap_end_segment(BreakableCUDAGraphCapture._end_segment)
