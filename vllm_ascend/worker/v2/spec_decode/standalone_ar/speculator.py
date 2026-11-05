# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode.standalone_ar.speculator import StandaloneARSpeculator

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.attn_utils import (
    build_attn_metadata_factory,
    build_attn_metadata_wrapper,
)

# vLLM #60989 added draft graph capture and moved correction-token accounting
# to the caller. The release and verified main pins still use the old API.
_UPSTREAM_STANDALONE_AR_GRAPHS = hasattr(StandaloneARSpeculator, "_prefill_capture_slot_mappings")


class AscendStandaloneARSpeculator(StandaloneARSpeculator):
    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # Keep the draft eager until standalone graphs support Ascend's host
        # FIA metadata refresh. The target can still use its own ACL graphs.
        super().init_cudagraph_manager(CUDAGraphMode.NONE)

    def capture(self) -> None:
        # The upstream capture method enters a graph context even in NONE mode.
        return

    if not vllm_version_is("0.32.0"):

        def supports_batch_sharded_draft_logits(self) -> bool:
            # NPU draft-logits kernels index the cache by request state.
            return False

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
        if step == 0 and not _UPSTREAM_STANDALONE_AR_GRAPHS:
            # Older upstream callers pass the target's bound unchanged.
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
