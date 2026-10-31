# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-specific MLA adapter, imported only for the vLLM main lane."""

from vllm.models.kimi_k3.amd.mla import KimiK3MultiHeadLatentAttentionWrapper

from vllm_ascend.ops.mla import AscendMultiHeadLatentAttention


class AscendKimiK3MultiHeadLatentAttention(AscendMultiHeadLatentAttention, KimiK3MultiHeadLatentAttentionWrapper):
    """Keep Ascend MLA dispatch after vLLM #52494 introduced an AMD subclass.

    OOT lookup uses the concrete class name. Inherit the upstream subclass too
    so Python runs the Ascend initializer on the replacement instance.
    """
