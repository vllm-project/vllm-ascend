# SPDX-License-Identifier: Apache-2.0
"""Kimi Decode dependencies layered on the 0.30 shared-expert executor."""

import torch
import torch_npu

from vllm_ascend.utils import npu_stream_switch, shared_experts_calculation_stream

ROUTER_CORE_LIMIT = (32, 1)
BF16_SHARED_CORE_LIMIT = (32, 1)
MXFP_SHARED_CORE_LIMIT = (16, 32)


class KimiDecodeSchedule:
    """One invocation's shared tensors/events; never retained by a communicator.

    Dispatch -> shared SiTU -> routed GMM1 -> shared Down. Keep the 0.30
    HCCL ordering: shared ReduceScatter follows routed finalize, even though
    shared Down's Cube work overlaps routed activation/quantization/GMM2.
    """

    def __init__(self, shared, prepared_input, router_ready):
        self.shared = shared
        prepared = shared.prepare_for_router_overlap(prepared_input, router_ready)
        assert prepared is not None
        limit = MXFP_SHARED_CORE_LIMIT if hasattr(shared.layer.gate_up_proj, "weight_scale") else BF16_SHARED_CORE_LIMIT
        with torch_npu.npu.npugraph_ex.scope.limit_core_num(*limit):
            self.submitted = shared.enqueue_gate_up_after_router(prepared, router_ready)
        self.down_input = None
        self.output = None

    def after_dispatch(self):
        main = torch.npu.current_stream()
        dispatch_done = main.record_event()
        with npu_stream_switch(shared_experts_calculation_stream()):
            torch.npu.current_stream().wait_event(dispatch_done)
            activated = self.shared.apply_activation(self.submitted.gate_up)
            self.down_input = self.shared._prepare_linear_input_for_overlap(self.shared.layer.down_proj, activated)
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

    def finalize(self):
        if self.output is None:
            raise RuntimeError("Routed MLP did not publish its GMM1 completion")
        # Preserve 0.30's cross-process-group collective ordering.
        routed_done = torch.npu.current_stream().record_event()
        with npu_stream_switch(shared_experts_calculation_stream()):
            torch.npu.current_stream().wait_event(routed_done)
            self.output = self.shared._pad_and_reduce_scatter(self.output)
        self.output.record_stream(torch.npu.current_stream())
        return self.output
