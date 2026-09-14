# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 MLA-only additions to the shared Ascend DSpark runtime."""

from contextlib import contextmanager

import torch
import vllm.v1.worker.gpu.spec_decode.dflash.cudagraph as dflash_cudagraph

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator


class AscendMLADSparkSpeculator(AscendDSparkSpeculator):
    """Reuse shared loading, input preparation, graph dispatch and replay.

    Unlike the GQA draft, K3 MLA consumes post-layer raw prefix sums and
    stores FIA query lengths in ``metadata.decode``.
    """

    @contextmanager
    def draft_capture_context(self):
        """Supply the MLA RoPE positions omitted by upstream graph capture."""
        original = dflash_cudagraph.build_attn_metadata

        def build_mla_metadata(*args, **kwargs):
            kwargs["positions"] = self.input_buffers.positions[: kwargs["num_tokens"]]
            kwargs["is_prefilling"] = torch.zeros(kwargs["num_reqs"], dtype=torch.bool)
            kwargs["attn_state"] = AscendAttentionState.SpecDecoding
            return build_attn_metadata(*args, **kwargs)

        try:
            dflash_cudagraph.build_attn_metadata = build_mla_metadata
            yield
        finally:
            dflash_cudagraph.build_attn_metadata = original

    def _get_draft_is_prefilling(self, num_reqs_padded: int) -> torch.Tensor:
        # The query graph contains only the speculative query block, even
        # when its context was produced by a target prefill. Padded rows must
        # not inherit prefill flags from a previous batch.
        return torch.zeros(num_reqs_padded, dtype=torch.bool)

    def _update_draft_attn_metadata(self, attn_metadata, num_reqs_padded):
        query_lengths = [(i + 1) * self.num_query_per_req for i in range(num_reqs_padded)]
        if not attn_metadata:
            raise RuntimeError("K3 MLA draft graph requires attention metadata.")
        for name, metadata in attn_metadata.items():
            decode = getattr(metadata, "decode", None)
            if decode is None or not hasattr(decode, "actual_seq_lengths_q"):
                raise TypeError(f"K3 MLA draft layer {name} has no decode query-length metadata.")
            # Match capture: a parallel draft block is TND speculative decode.
            metadata.attn_state = AscendAttentionState.SpecDecoding
            decode.actual_seq_lengths_q = query_lengths
        return attn_metadata
