# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Workaround for FIA parameter updates during DFlash and DSpark graph replay.

Delete this module once the FIA operator is retired.
"""

from collections.abc import Mapping
from typing import Any

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor


class FIADraftGraphMetadataMixin:
    # Cache builder metadata for FIA parameter updates during UpdatableGraph replay.
    draft_graph_replay_metadata: dict[str, Any] | None = None

    def _build_uniform_attn_metadata(
        self,
        batch_desc: BatchExecutionDescriptor,
        num_reqs: int,
        num_query_per_req: int,
        seq_lens_cpu_upper_bound: torch.Tensor,
        step: int,
        causal: bool | Mapping[int, bool] = True,
        dcp_local_seq_lens: torch.Tensor | None = None,
    ) -> dict[str, Any] | None:
        metadata = super()._build_uniform_attn_metadata(
            batch_desc=batch_desc,
            num_reqs=num_reqs,
            num_query_per_req=num_query_per_req,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            step=step,
            causal=causal,
            dcp_local_seq_lens=dcp_local_seq_lens,
        )
        if batch_desc.cg_mode == CUDAGraphMode.FULL and metadata is not None:
            self.draft_graph_replay_metadata = metadata
        return metadata

    def prepare_draft_graph_replay_metadata(self, desc: BatchExecutionDescriptor) -> dict[str, Any]:
        metadata = self.draft_graph_replay_metadata
        assert metadata is not None, "FIA graph replay requires attention metadata from the draft builder."
        assert desc.num_reqs is not None
        self._update_draft_attn_metadata(metadata, desc.num_reqs)
        return metadata

    def _update_draft_attn_metadata(self, attn_metadata: dict[str, Any], num_reqs_padded: int) -> None:
        """Adjust FIA metadata for the padded request count used by the graph."""
        raise NotImplementedError
