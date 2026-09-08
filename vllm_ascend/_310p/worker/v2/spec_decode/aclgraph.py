# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# mypy: ignore-errors

"""310P draft ACLGraph manager: refresh capture seq_lens (no FIA graph_task)."""

from __future__ import annotations

import torch
from vllm.logger import logger
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)

from vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph import (
    AutoRegressiveAclGraphManager,
)


class AutoRegressiveAclGraphManager310(AutoRegressiveAclGraphManager):
    """310P draft FULL replay: in-place seq_lens refresh, then pure replay.

    310P attention is captured as direct NPU ops without FIA ``graph_task``
    handles, so mainline ``update_full_graph_params`` is a no-op here. Mirror
    target ``prepare_attn`` / ``_refresh_capture_seq_lens`` instead.
    """

    def run_fullgraph(self, desc: BatchExecutionDescriptor) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        num_tokens = desc.num_tokens
        if self.is_draft_model_prefill:
            logger.info_once(
                "AutoRegressiveAclGraphManager310: draft prefill run_fullgraph with num_tokens=%s",
                num_tokens,
            )
        else:
            logger.info_once(
                "AutoRegressiveAclGraphManager310: draft run_fullgraph with num_tokens=%s",
                num_tokens,
            )

        # Ensure H2D into capture-stable buffers is visible before replay.
        torch.npu.current_stream().synchronize()
        ms = self.speculator.model_state
        runtime_seq_lens = self.speculator.target_input_buffers.seq_lens
        refresh = getattr(ms, "_refresh_capture_seq_lens", None)
        if callable(refresh):
            refresh(runtime_seq_lens)
        return CudaGraphManager.run_fullgraph(self, desc)
