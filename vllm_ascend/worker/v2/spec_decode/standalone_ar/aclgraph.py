# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import SpeculatorCudaGraphManager

from vllm_ascend.worker.v2.aclgraph_utils import model_capture_wrapper
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata_wrapper
from vllm_ascend.worker.v2.spec_decode.target_dependent_ar.aclgraph import AutoRegressiveAclGraphManager
from vllm_ascend.worker.v2.utils import communicator_switch


class StandaloneARAclGraphManager(AutoRegressiveAclGraphManager):
    """Update NPU attention parameters around upstream's single-step graphs."""

    def capture(self, forward_fn, *args, **kwargs) -> None:
        # Standalone AR replays one decode step at a time. Keep the upstream
        # capture factory, including its expanded slot-mapping buffer, instead
        # of the target-dependent adapter's merged multi-step capture.
        with (
            communicator_switch(),
            model_capture_wrapper(self.speculator, self.is_draft_model_prefill),
            build_attn_metadata_wrapper(),
        ):
            SpeculatorCudaGraphManager.capture(self, forward_fn, *args, **kwargs)
