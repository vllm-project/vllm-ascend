# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
from typing import Optional
from enum import Enum

import torch

from vllm import forward_context
from vllm.forward_context import ForwardContext
from vllm.v1.worker.ubatching import UBatchContext
from vllm_ascend import envs
from vllm_ascend.utils import dbo_current_stream, dbo_set_stream

_THREAD_ID_TO_CONTEXT: dict = {}
_CURRENT_CONTEXTS: list[Optional['AscendUBatchContext']] = [None, None]


class UBatchEventKey(Enum):
    ATTN_PRE = 0
    ATTN_POST = 1
    MOE_DISPATCH = 2
    MOE_COMBINE = 3
    DEFAULT = 4


class AscendUBatchContext(UBatchContext):
    """
    Context manager for micro-batching synchronization using threading events.
    """

    def __init__(self,
                 id: int,
                 comm_stream: torch.npu.Stream,
                 compute_stream: torch.npu.Stream,
                 forward_context: ForwardContext,
                 ready_barrier: threading.Barrier,
                 cpu_wait_event: threading.Event,
                 cpu_signal_event: threading.Event,
                 gpu_comm_done_event: dict[UBatchEventKey, torch.npu.Event],
                 gpu_compute_done_event: dict[UBatchEventKey, torch.npu.Event],
                 schedule: str = "default"):
        self.id = id
        self.comm_stream = comm_stream
        self.compute_stream = compute_stream
        self.forward_context = forward_context
        self.ready_barrier = ready_barrier
        self.cpu_wait_event = cpu_wait_event
        self.cpu_signal_event = cpu_signal_event
        self.current_stream = compute_stream
        self.gpu_comm_done_event = gpu_comm_done_event
        self.gpu_compute_done_event = gpu_compute_done_event
        self.schedule = schedule
        self.recv_hook = None
        # set aic/aiv core num according to the env param
        self.comm_cube_core = envs.VLLM_ASCEND_DBO_COMM_AIC_NUM
        self.comm_vector_core = envs.VLLM_ASCEND_DBO_COMM_AIV_NUM
        # get the total core num
        props = torch.npu.get_device_limit(torch.npu.current_device())
        self.comp_cube_core = props[
            "cube_core_num"] - self.comm_cube_core if self.comm_cube_core != -1 else -1
        self.comp_vector_core = props[
            "vector_core_num"] - self.comm_vector_core if self.comm_vector_core != -1 else -1

    def __enter__(self):
        global _CURRENT_CONTEXTS, _THREAD_ID_TO_CONTEXT
        _THREAD_ID_TO_CONTEXT[threading.get_ident()] = self.id
        _CURRENT_CONTEXTS[self.id] = self
        self.ready_barrier.wait()

        self.cpu_wait_event.wait()
        self.cpu_wait_event.clear()
        self._restore_context()
        forward_context._forward_context.dbo_enabled = True
        # Assume we want to start on the compute stream
        self.update_stream(self.compute_stream)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _CURRENT_CONTEXTS, _THREAD_ID_TO_CONTEXT
        _CURRENT_CONTEXTS[self.id] = None
        del _THREAD_ID_TO_CONTEXT[threading.get_ident()]
        self.maybe_run_recv_hook()
        self.cpu_signal_event.set()
        self.cpu_wait_event.clear()
        return False

    def _restore_context(self):
        forward_context._forward_context = self.forward_context

    def update_stream(self, stream):
        self.current_stream = stream
        if dbo_current_stream() != self.current_stream:
            dbo_set_stream(self.current_stream)

    def _signal_comm_done(self, event=UBatchEventKey.DEFAULT):
        self.gpu_comm_done_event[event].record(self.comm_stream)

    def _signal_compute_done(self, event=UBatchEventKey.DEFAULT):
        self.gpu_compute_done_event[event].record(self.compute_stream)

    def _wait_compute_done(self, event=UBatchEventKey.DEFAULT):
        self.comm_stream.wait_event(self.gpu_compute_done_event[event])

    def _wait_comm_done(self, event=UBatchEventKey.DEFAULT):
        self.compute_stream.wait_event(self.gpu_comm_done_event[event])

    def _cpu_yield(self):
        # It is critical for correctness that only one thread is running
        # at a time. These asserts just make sure that this is the only
        # thread running before waking the other one up and going to sleep
        assert forward_context._forward_context == self.forward_context
        assert dbo_current_stream() == self.current_stream
        assert not self.cpu_wait_event.is_set()

        self.cpu_signal_event.set()
        self.cpu_wait_event.wait()
        self.cpu_wait_event.clear()
        self._restore_context()

    def switch_to_comm(self):
        self.update_stream(self.comm_stream)

    def switch_to_compute(self):
        self.update_stream(self.compute_stream)

    def switch_to_comm_sync(self, event=UBatchEventKey.DEFAULT):
        self._signal_compute_done(event)
        self.update_stream(self.comm_stream)
        self._wait_compute_done(event)

    def switch_to_compute_sync(self, event=UBatchEventKey.DEFAULT):
        self._signal_comm_done(event)
        self.update_stream(self.compute_stream)
        self._wait_comm_done(event)

    def record_current_stream(self, event=UBatchEventKey.DEFAULT):
        # set core limit for communication block
        if self.comm_cube_core != -1 or self.comm_vector_core != -1:
            torch.npu.set_stream_limit(dbo_current_stream(),
                                       cube_num=self.comm_cube_core,
                                       vector_num=self.comm_vector_core)
        self._signal_compute_done(event)

    def wait_current_stream_and_yield(self,
                                      event=UBatchEventKey.DEFAULT,
                                      wait: bool = True):
        if wait:
            self._wait_compute_done(event)
        self._cpu_yield()
        # set core limit for comp part
        if self.comm_cube_core != -1 or self.comm_vector_core != -1:
            torch.npu.set_stream_limit(dbo_current_stream(),
                                       cube_num=self.comp_cube_core,
                                       vector_num=self.comp_vector_core)

    def maybe_run_recv_hook(self):
        if self.recv_hook is not None:
            self.recv_hook()
            self.recv_hook = None

    def yield_(self):
        self.current_stream = dbo_current_stream()
        self._cpu_yield()

    # switch func for two stream overlap
    # 1. yield from stream 1 thread1 to stream2 thread2
    def yield_and_switch_from_compute_to_comm(self,
                                              event=UBatchEventKey.DEFAULT):
        assert dbo_current_stream() == self.compute_stream
        self._signal_compute_done(event)
        self._cpu_yield()
        assert self.current_stream == self.compute_stream
        self.update_stream(self.comm_stream)
        self._wait_compute_done(event)

    # 1. yield from stream 2 thread2 to stream1 thread1
    def yield_and_switch_from_comm_to_compute(self,
                                              event=UBatchEventKey.DEFAULT):
        assert dbo_current_stream() == self.comm_stream
        self._signal_comm_done(event)
        self._cpu_yield()
        assert self.current_stream == self.comm_stream
        self.update_stream(self.compute_stream)
        self._wait_comm_done(event)


def dbo_enabled() -> bool:
    return len(_THREAD_ID_TO_CONTEXT) > 0


def dbo_current_ubatch_id() -> int:
    if len(_THREAD_ID_TO_CONTEXT) == 0:
        return 0
    return _THREAD_ID_TO_CONTEXT[threading.get_ident()]


def _register_ubatch_function(func):

    def wrapper(*args, **kwargs):
        if len(_THREAD_ID_TO_CONTEXT) > 0:
            ctx_idx = _THREAD_ID_TO_CONTEXT[threading.get_ident()]
            ctx = _CURRENT_CONTEXTS[ctx_idx]
            func(ctx, *args, **kwargs)

    return wrapper


dbo_maybe_run_recv_hook = _register_ubatch_function(
    AscendUBatchContext.maybe_run_recv_hook)
dbo_yield = _register_ubatch_function(AscendUBatchContext.yield_)
dbo_yield_and_switch_from_compute_to_comm = _register_ubatch_function(
    AscendUBatchContext.yield_and_switch_from_compute_to_comm)
dbo_yield_and_switch_from_comm_to_compute = _register_ubatch_function(
    AscendUBatchContext.yield_and_switch_from_comm_to_compute)
dbo_switch_to_comm = _register_ubatch_function(
    AscendUBatchContext.switch_to_comm)
dbo_switch_to_compute = _register_ubatch_function(
    AscendUBatchContext.switch_to_compute)
dbo_switch_to_comm_sync = _register_ubatch_function(
    AscendUBatchContext.switch_to_comm_sync)
dbo_switch_to_compute_sync = _register_ubatch_function(
    AscendUBatchContext.switch_to_compute_sync)

dbo_record_current_stream = _register_ubatch_function(
    AscendUBatchContext.record_current_stream)
dbo_wait_current_stream_and_yield = _register_ubatch_function(
    AscendUBatchContext.wait_current_stream_and_yield)


def dbo_register_recv_hook(recv_hook):
    if len(_THREAD_ID_TO_CONTEXT) > 0:
        ctx_idx = _THREAD_ID_TO_CONTEXT[threading.get_ident()]
        next_ctx = _CURRENT_CONTEXTS[(ctx_idx + 1) % 2]
        next_ctx.recv_hook = recv_hook


def dbo_get_previous_event(func, *args, **kwargs):
    if len(_THREAD_ID_TO_CONTEXT) > 0:
        ctx_idx = _THREAD_ID_TO_CONTEXT[threading.get_ident()]
        ctx = _CURRENT_CONTEXTS[ctx_idx]
        # execute callable on the ubatch compute stream to record/wait events there
        with torch.npu.stream(ctx.compute_stream):
            return func(*args, **kwargs)


def make_ubatch_contexts(
    num_micro_batches: int,
    compute_stream: torch.npu.Stream,
    comm_stream: torch.npu.Stream,
    forward_contexts: list[ForwardContext],
    ready_barrier: threading.Barrier,
    schedule: str = "default",
) -> list[AscendUBatchContext]:
    assert num_micro_batches == 2, "only been tested with 2 micro-batches"
    """
    Create a context manager for micro-batching synchronization.
    """
    key_list = [
        UBatchEventKey.ATTN_PRE, UBatchEventKey.ATTN_POST,
        UBatchEventKey.MOE_DISPATCH, UBatchEventKey.MOE_COMBINE,
        UBatchEventKey.DEFAULT
    ]
    cpu_events = [threading.Event() for _ in range(num_micro_batches)]
    gpu_comm_done_events = [{
        key: torch.npu.Event()
        for key in key_list
    } for _ in range(num_micro_batches)]
    gpu_compute_done_events = [{
        key: torch.npu.Event()
        for key in key_list
    } for _ in range(num_micro_batches)]

    assert len(forward_contexts) == 2

    ctxs = []
    current_microbatch_stream = compute_stream
    other_microbatch_stream = comm_stream
    for i in range(num_micro_batches):
        if i == 0:
            current_microbatch_stream = compute_stream
            other_microbatch_stream = comm_stream
        else:
            current_microbatch_stream = comm_stream
            other_microbatch_stream = compute_stream
        ctx = AscendUBatchContext(
            id=i,
            compute_stream=current_microbatch_stream,
            comm_stream=other_microbatch_stream,
            forward_context=forward_contexts[i],
            ready_barrier=ready_barrier,
            cpu_wait_event=cpu_events[i],
            cpu_signal_event=cpu_events[(i + 1) % num_micro_batches],
            gpu_comm_done_event=gpu_comm_done_events[i],
            gpu_compute_done_event=gpu_compute_done_events[i],
            schedule=schedule)
        ctxs.append(ctx)

    return ctxs
