# mypy: ignore-errors
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""310P workaround for the aicpu IndexPut crash (error 507018).

On Ascend 310P, bool-mask `index_put_` dispatches to the aicpu IndexPut
kernel which aborts the device context (error 507018), killing EngineCore
on any multimodal request. See vllm-ascend issue #12086.

This patch replaces the bool-mask assignment in vLLM's
`_merge_multimodal_embeddings` with an equivalent
`torch.nonzero` + `index_copy_` sequence, which does not touch the broken
kernel. Applied on 310P only.
"""

import torch
import vllm.model_executor.models.utils as model_utils
from vllm.model_executor.models.utils import (
    _embedding_count_expression,
    _flatten_embeddings,
)


def _merge_multimodal_embeddings_310p(
    inputs_embeds: torch.Tensor,
    multimodal_embeddings,
    is_multimodal: torch.Tensor,
) -> torch.Tensor:
    """Drop-in replacement for vLLM's `_merge_multimodal_embeddings`.

    Semantically identical to the upstream implementation; only the write
    mechanism differs (integer-index `index_copy_` instead of bool-mask
    `index_put_`).
    """
    if len(multimodal_embeddings) == 0:
        return inputs_embeds

    mm_embeds_flat = _flatten_embeddings(multimodal_embeddings)
    input_dtype = inputs_embeds.dtype

    try:
        # NOTE: `is_multimodal` may reside on CPU (e.g. the Qwen3-VL
        # deepstack path), so the index must be moved to the embedding
        # device explicitly.
        mm_idx = torch.nonzero(is_multimodal, as_tuple=True)[0].to(device=inputs_embeds.device, non_blocking=True)
        inputs_embeds.index_copy_(
            0,
            mm_idx,
            mm_embeds_flat.to(device=inputs_embeds.device, dtype=input_dtype),
        )
    except RuntimeError as e:
        num_actual_tokens = len(mm_embeds_flat)
        num_expected_tokens = is_multimodal.sum().item()

        if num_actual_tokens != num_expected_tokens:
            expr = _embedding_count_expression(multimodal_embeddings)
            raise ValueError(
                f"Attempted to assign {expr} = {num_actual_tokens} "
                f"multimodal tokens to {num_expected_tokens} placeholders"
            ) from e

        raise ValueError("Error during index put operation") from e

    return inputs_embeds


# Model modules import the symbol with
# `from .utils import _merge_multimodal_embeddings`; vllm-ascend platform
# patches are imported before any model module loads, so rebinding the
# module attribute here is sufficient.
model_utils._merge_multimodal_embeddings = _merge_multimodal_embeddings_310p
