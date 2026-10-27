# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode.standalone_ar.speculator import StandaloneARSpeculator

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.v2.attn_utils import (
    build_attn_metadata_factory,
    build_attn_metadata_wrapper,
)


class AscendStandaloneARSpeculator(StandaloneARSpeculator):
    def _build_attn_metadata(
        self,
        num_reqs: int,
        batch_desc: BatchExecutionDescriptor,
        query_start_loc_np: np.ndarray,
        seq_lens_cpu_upper_bound: torch.Tensor,
        step: int,
        causal: bool | Mapping[int, bool] = True,
        dcp_local_seq_lens: torch.Tensor | None = None,
        slot_mappings: torch.Tensor | None = None,
    ) -> dict[str, Any] | None:
        draft_seq_lens_cpu_upper_bound = seq_lens_cpu_upper_bound
        if step == 0:
            # TODO: Remove this workaround once upstream includes the correction
            # token in the standalone AR prefill sequence-length bound.
            # Draft prefill appends the target's correction token to the
            # expanded query, so the upstream builder must receive target
            # length + 1. Keep the input batch's bounds unchanged.
            draft_seq_lens_cpu_upper_bound = (seq_lens_cpu_upper_bound[:num_reqs] + 1).clamp(max=self.max_model_len)

        num_tokens = int(query_start_loc_np[-1])
        positions = self.expanded_positions if step == 0 else self.input_buffers.positions
        attn_state = AscendAttentionState.DecodeOnly

        with (
            build_attn_metadata_wrapper(),
            build_attn_metadata_factory(
                positions,
                num_tokens,
                self.draft_is_prefilling[:num_reqs],
                attn_state=attn_state,
                parallel_config=self.vllm_config.parallel_config,
            ),
        ):
            return super()._build_attn_metadata(
                num_reqs,
                batch_desc,
                query_start_loc_np,
                draft_seq_lens_cpu_upper_bound,
                step,
                causal,
                dcp_local_seq_lens,
                slot_mappings,
            )
