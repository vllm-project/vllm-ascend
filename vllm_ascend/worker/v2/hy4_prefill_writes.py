# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import cache

import torch
from vllm.v1.worker.gpu.buffer_utils import FusedStagedWriter, StagedWriteTensor
from vllm.v1.worker.gpu.sample.penalties import use_penalty

_pending_writes: ContextVar[list[StagedWriteTensor] | None] = ContextVar("hy4_prefill_writes", default=None)


@cache
def _install_write_hooks():
    apply_write = StagedWriteTensor.apply_write
    fused_apply = FusedStagedWriter.apply

    def defer_write(tensor):
        pending = _pending_writes.get()
        if pending is not None and tensor.dtype == torch.int32 and tensor._staged_write_indices:
            pending.append(tensor)
        else:
            apply_write(tensor)

    def defer_group(writer, tensors, output_ptrs, output_strides):
        pending = _pending_writes.get()
        if pending is not None and all(tensor.dtype == torch.int32 for tensor in tensors):
            pending.extend(tensor for tensor in tensors if tensor._staged_write_indices)
        else:
            fused_apply(writer, tensors, output_ptrs, output_strides)

    StagedWriteTensor.apply_write = defer_write
    FusedStagedWriter.apply = defer_group


class PrefillWrites:
    def __init__(self, device):
        self.device = device
        self.layouts = {}
        _install_write_hooks()

    def for_step(self, runner, output):
        sampler = runner.sampler
        if (
            output.scheduled_new_reqs
            and type(runner.model_state).__name__ == "FlashModelState"
            and sampler is not None
            and len(sampler.logits_processors) == 3
            and all(
                req.sampling_params is not None and not use_penalty(req.sampling_params) and not req.mm_features
                for req in output.scheduled_new_reqs
            )
        ):
            return self._collect()
        return nullcontext()

    @contextmanager
    def _collect(self):
        scope = _pending_writes.set([])
        try:
            yield
            self.flush()
        finally:
            _pending_writes.reset(scope)

    def flush(self):
        pending = _pending_writes.get()
        if not pending:
            return
        tensors = tuple(dict.fromkeys(pending))
        pending.clear()
        scope = _pending_writes.set(None)
        try:
            self._apply(tensors)
        finally:
            _pending_writes.reset(scope)

    def _apply(self, tensors: Sequence[StagedWriteTensor]):
        if len(tensors) <= 1 or any(len(t._staged_write_indices) != len(set(t._staged_write_indices)) for t in tensors):
            for tensor in tensors:
                tensor.apply_write()
            return

        layout = self.layouts.get(tensors)
        if layout is None:
            writer = FusedStagedWriter(self.device, sum(t.num_rows for t in tensors))
            ptrs = torch.tensor([t.gpu.data_ptr() for t in tensors], dtype=torch.uint64, device=self.device)
            strides = torch.tensor([t.gpu.stride(0) for t in tensors], dtype=torch.int64, device=self.device)
            layout = self.layouts[tensors] = (writer, ptrs, strides)
        writer, ptrs, strides = layout
        writer.apply(tensors, ptrs, strides)
