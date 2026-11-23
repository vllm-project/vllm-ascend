# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""No-op MLA prefill backend for Ascend.

Ascend's AscendSFAImpl/AscendMLAImpl handles the full forward pass (including
prefill) via ``impl.forward()``, so ``prefill_backend.run_prefill_*`` is never
called. This backend exists only so that ``MLAAttention.__init__`` completes
without requiring a CUDA flash-attn prefill backend; it is returned by
``NPUPlatform.get_mla_prefill_backend_cls``.
"""

from typing import TYPE_CHECKING

import torch
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )


class AscendMLAPrefillBackend(MLAPrefillBackend):
    @staticmethod
    def get_name() -> str:
        return "ASCEND"

    @classmethod
    def is_available(cls) -> bool:
        return True

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Ascend MLA prefill is handled by AscendSFAImpl/AscendMLAImpl")

    def run_prefill_context_chunk(
        self,
        chunk: "MLACommonPrefillMetadata.ContextChunk",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Ascend MLA prefill is handled by AscendSFAImpl/AscendMLAImpl")
