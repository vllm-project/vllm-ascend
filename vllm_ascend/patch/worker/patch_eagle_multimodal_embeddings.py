# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Apply vLLM #57633 to drafters that still borrow the multimodal method.

Remove this patch when all supported vLLM refs include the explicit embedding
implementations from #57633. Fixed upstream implementations are left intact.
"""

import torch
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    _require_is_multimodal,
)
from vllm.model_executor.models.llama4_eagle import EagleLlama4ForCausalLM
from vllm.model_executor.models.mistral_large_3_eagle import EagleMistralLarge3ForCausalLM
from vllm.model_executor.models.utils import _merge_multimodal_embeddings


def _embed_input_ids(
    self,
    input_ids: torch.Tensor,
    multimodal_embeddings: MultiModalEmbeddings | None = None,
    *,
    is_multimodal: torch.Tensor | None = None,
) -> torch.Tensor:
    inputs_embeds = self.model.embed_input_ids(input_ids)
    if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
        return inputs_embeds
    return _merge_multimodal_embeddings(
        inputs_embeds=inputs_embeds,
        multimodal_embeddings=multimodal_embeddings,
        is_multimodal=_require_is_multimodal(is_multimodal),
    )


def _patch_draft_embedding(model_cls) -> None:
    if model_cls.embed_input_ids is SupportsMultiModal.embed_input_ids:
        model_cls.embed_input_ids = _embed_input_ids


_patch_draft_embedding(EagleLlama4ForCausalLM)
_patch_draft_embedding(EagleMistralLarge3ForCausalLM)
