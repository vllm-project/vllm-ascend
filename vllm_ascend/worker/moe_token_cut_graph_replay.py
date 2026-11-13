# SPDX-License-Identifier: Apache-2.0
"""Native operator graph replay using v0.27's task-update/event pattern.

The kernel's integer layer_idx is a communication generation, not a Tensor
control. Refresh its captured task before replay, as v0.27 Attention refreshes
captured operator parameters. Actual CANN/HCCL support requires NPU validation.
"""

import torch

from vllm_ascend.ops.fused_moe.moe_token_cut_native import (
    MOE_TOKEN_CUT_SCHEMA,
    REBATCH_AND_UPDATE_SCHEMA,
    invoke_native_op,
)
from vllm_ascend.worker.moe_token_cut_native_rebatch import INT32_MAX


class MoETokenCutGraphReplay:
    """Serial diagnostic graph, not a model-runner graph wrapper.

    All EP ranks must call capture/replay in the same order, even with no live
    rows. The caller must ensure every rank completes the previous invocation
    before another can overwrite its HCCL mailbox (the smoke uses a barrier).
    Each instance owns its buffers, task handle, graph and external event.
    """

    def __init__(self, buffers, *, trace=None):
        self.buffers = buffers
        self.trace = trace
        self.graph = None
        self.handle = None
        self.event = None
        self.update_stream = None
        self.generation = -1
        self.failed = False
        self._addresses = buffers.tensor_addresses()

    def capture(self):
        if self.graph is not None or self.failed:
            raise RuntimeError("token-cut graph capture is one-shot")
        self.update_stream = torch.npu.Stream(device=self.buffers.device)
        self.event = torch.npu.ExternalEvent()
        self.graph = torch.npu.NPUGraph()
        try:
            with torch.npu.graph(self.graph):
                stream = torch.npu.current_stream()
                self.event.wait(stream)
                self.event.reset(stream)
                torch.npu.graph_task_group_begin(stream)
                invoke_native_op(MOE_TOKEN_CUT_SCHEMA, self.buffers.cut_arguments)
                self.handle = torch.npu.graph_task_group_end(stream)
                invoke_native_op(REBATCH_AND_UPDATE_SCHEMA, self.buffers.rebatch_arguments)
            # Also release the event for the capture iteration before its
            # caller synchronizes/clears dummy state, as the runner does for
            # full-graph Attention task groups on the capture iteration.
            self._update_generation(0)
        except Exception:
            self.failed = True
            raise

    def _update_generation(self, generation):
        if not 0 <= generation < INT32_MAX:
            raise RuntimeError("token-cut graph communication generation exhausted")
        if self.trace is not None:
            self.trace("generation_update_before", generation)
        arguments = dict(self.buffers.cut_arguments, layer_idx=generation)
        with torch.npu.stream(self.update_stream):
            torch.npu.graph_task_update_begin(self.update_stream, self.handle)
            try:
                invoke_native_op(MOE_TOKEN_CUT_SCHEMA, arguments)
            finally:
                torch.npu.graph_task_update_end(self.update_stream)
            self.event.record(self.update_stream)
        self.generation = generation
        if self.trace is not None:
            self.trace("generation_update_submitted", generation)

    def replay(self):
        if self.graph is None or self.failed:
            raise RuntimeError("token-cut graph is not ready for replay")
        if self.buffers.tensor_addresses() != self._addresses:
            raise RuntimeError("token-cut graph buffers changed after capture")
        try:
            # Same conservative ordering as v0.27's full-graph Attention path:
            # don't update a task while its previous execution is in flight.
            torch.npu.current_stream().synchronize()
            self._update_generation(self.generation + 1)
            self.graph.replay()
        except Exception:
            self.failed = True
            raise
