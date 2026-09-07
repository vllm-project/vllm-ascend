# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Speculative decoding support for Model Runner V2 PP."""

import numpy as np
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu

_DRAFT_UPDATE_PATCHED = "_vllm_ascend_spec_pp_draft_update_patched"


def install_spec_pp_draft_update(pp_handler) -> None:
    """Apply upstream PP draft receives without NPU boolean indexing."""
    if getattr(pp_handler, _DRAFT_UPDATE_PATCHED, False):
        return

    original_get_prev_sampled_outputs = pp_handler.get_prev_sampled_outputs

    def get_prev_sampled_outputs(draft_tokens_to_update=None):
        slot = pp_handler.queue[0] if pp_handler.queue else None
        # vLLM #50514 receives drafts separately. Let the parent wait for the
        # receive and filter sampled outputs, then update drafts below using
        # CPU row indices instead of its device boolean-indexing path.
        outputs = original_get_prev_sampled_outputs()
        if outputs is None or draft_tokens_to_update is None:
            return outputs
        assert slot is not None
        draft_tokens = slot.draft_tokens
        if draft_tokens is None:
            return outputs

        # Preserve valid rows on CPU; NPU bool indexing lowers to NonzeroV2.
        freed = pp_handler.req_idx_gen_np[slot.idx_mapping_np] != slot.gen_at_receive_np
        exclude_mask = freed | ~slot.need_sampled_mask
        if exclude_mask.any():
            valid_rows = np.flatnonzero(~exclude_mask)
            update_indices = np.stack(
                (valid_rows, slot.idx_mapping_np[valid_rows]),
            )
            draft_rows, draft_req_indices = async_copy_to_gpu(
                update_indices,
                device=pp_handler.device,
            ).unbind(dim=0)
            draft_tokens_to_update.index_copy_(
                0,
                draft_req_indices,
                draft_tokens.index_select(0, draft_rows),
            )
        else:
            draft_tokens_to_update.index_copy_(
                0,
                outputs["idx_mapping"],
                draft_tokens,
            )
        return outputs

    pp_handler.get_prev_sampled_outputs = get_prev_sampled_outputs
    setattr(pp_handler, _DRAFT_UPDATE_PATCHED, True)
