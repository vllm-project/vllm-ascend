# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MegaTree construction for the DFlash2 candidate lattice."""

from typing import TYPE_CHECKING

import torch

from vllm_ascend.worker.v2.spec_decode.tree.layout import TreeLayout

if TYPE_CHECKING:
    from vllm_ascend.models.qwen3_dflash2 import DFlash2Qwen3ForCausalLM


class DFlash2MegaTreeBuilder:
    """Project the DFlash2 hidden states and construct a tree on device."""

    def __init__(self, model: "DFlash2Qwen3ForCausalLM") -> None:
        selector = model.model.candidate_selector
        self.predecessor_codebook = selector.predecessor_codebook
        self.successor_codebook = selector.successor_codebook
        self.hidden_projection = selector.hidden_projection
        for book in (self.predecessor_codebook, self.successor_codebook):
            if (
                book.device.type != "npu"
                or book.ndim != 2
                or book.shape[-1] != 256
                or book.dtype != torch.bfloat16
                or not book.is_contiguous()
            ):
                raise ValueError("MegaTree requires contiguous rank-256 BF16 codebooks on Ascend")

    def build(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        draft_hidden: torch.Tensor,
        root_token_ids: torch.Tensor,
        out: TreeLayout,
    ) -> TreeLayout:
        from vllm_ascend.ops.triton.spec_decode.tree.mega_tree import build_dflash2_tree

        # Ascend's unquantized GEMM expects a matrix during graph capture.
        projected = self.hidden_projection(draft_hidden.flatten(0, 1)).view(*draft_hidden.shape[:-1], -1)
        return build_dflash2_tree(
            candidate_ids.contiguous(),
            unary_logits.contiguous(),
            projected.contiguous(),
            root_token_ids.contiguous(),
            self.predecessor_codebook,
            self.successor_codebook,
            out,
        )
