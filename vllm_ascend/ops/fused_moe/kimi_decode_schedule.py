# SPDX-License-Identifier: Apache-2.0
"""Kimi Decode dependencies layered on the 0.30 shared-expert executor."""

import torch
import torch_npu

from vllm_ascend.compilation.kimi_moe_superkernel import moe_superkernel_scope
from vllm_ascend.utils import npu_stream_switch, shared_experts_calculation_stream

ROUTER_CORE_LIMIT = (32, 1)
BF16_SHARED_CORE_LIMIT = (32, 1)
MXFP_SHARED_CORE_LIMIT = (16, 32)


class KimiDecodeSchedule:
    """One invocation's shared tensors/events; never retained by a communicator.

    Dispatch -> shared SiTU -> routed GMM1 -> shared Down.
    MC2 Decode submits shared ReduceScatter after shared Down, overlapping
    routed activation/quantization/GMM2. Other paths keep the 0.30 schedule.
    """

    def __init__(self, shared, prepared_input, router_ready, *, enable_superkernel=False, comm_stream=None):
        self.shared = shared
        self.enable_superkernel = enable_superkernel
        self.comm_stream = comm_stream if comm_stream is not None else shared_experts_calculation_stream()
        self.prepared = shared.prepare_for_router_overlap(prepared_input, router_ready)
        assert self.prepared is not None
        self.submitted = None
        self.down_input = None
        self.output = None
        self.output_done = None

    def before_dispatch(self, dispatch_ready):
        """Release Gate/Up at the same main-stream boundary as Dispatch."""
        limit = (
            MXFP_SHARED_CORE_LIMIT
            if hasattr(self.shared.layer.gate_up_proj, "weight_scale")
            else BF16_SHARED_CORE_LIMIT
        )
        # Select the producer stream before applying its core limit.
        with (
            npu_stream_switch(shared_experts_calculation_stream()),
            torch_npu.npu.npugraph_ex.scope.limit_core_num(*limit),
        ):
            self.submitted = self.shared.enqueue_gate_up_after_router(self.prepared, dispatch_ready)

    def after_dispatch(self):
        if self.submitted is None:
            raise RuntimeError("Shared Gate/Up must be submitted at the Dispatch boundary")
        main = torch.npu.current_stream()
        dispatch_done = main.record_event()
        with npu_stream_switch(shared_experts_calculation_stream()):
            torch.npu.current_stream().wait_event(dispatch_done)
            # PR51 releases GMM1 after SiTU alone. Keep Down's input
            # quantization in its registered linear, after GMM1 completes.
            self.down_input = self.shared.apply_activation(self.submitted.gate_up)
            activation_done = torch.npu.current_stream().record_event()
        # Both BF16 and MXFP shared paths release GMM1 only after SiTU.
        main.wait_event(activation_done)

    def after_gmm1(self):
        if self.down_input is None:
            raise RuntimeError("Shared SiTU must be submitted before routed GMM1")
        gmm1_done = torch.npu.current_stream().record_event()
        with npu_stream_switch(shared_experts_calculation_stream()):
            torch.npu.current_stream().wait_event(gmm1_done)
            self.output = self.shared.part2(self.submitted.hidden_states, self.down_input)
            down_done = torch.npu.current_stream().record_event()
            self.output.record_stream(self.comm_stream)
        # Dedicated communication stream waits only for its producer, never
        # for routed Combine. The final join consumes this invocation's event.
        with moe_superkernel_scope(None, self.enable_superkernel), npu_stream_switch(self.comm_stream):
            torch.npu.current_stream().wait_event(down_done)
            self.output = self.shared._pad_and_reduce_scatter(self.output)
            self.output_done = torch.npu.current_stream().record_event()

    def finalize(self):
        if self.output is None or self.output_done is None:
            raise RuntimeError("Routed MLP did not publish its GMM1 completion")
        self.output.record_stream(torch.npu.current_stream())
        return self.output
