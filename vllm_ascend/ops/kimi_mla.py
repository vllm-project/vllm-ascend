# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-specific MLA adapter, imported only for the vLLM main lane."""

from vllm.models.kimi_k3.amd.mla import KimiK3MultiHeadLatentAttentionWrapper

from vllm_ascend.ops.mla import AscendMultiHeadLatentAttention
from vllm_ascend.utils import vllm_version_is


class AscendKimiK3MultiHeadLatentAttention(AscendMultiHeadLatentAttention, KimiK3MultiHeadLatentAttentionWrapper):
    """Keep Ascend MLA dispatch after vLLM #52494 introduced an AMD subclass.

    OOT lookup uses the concrete class name. Inherit the upstream subclass too
    so Python runs the Ascend initializer on the replacement instance.
    """

    if not vllm_version_is("0.32.0"):
        # vLLM #60110 added this flag and reads it in the upstream
        # process_weights_after_loading hook. The AMD initializer never runs on
        # Ascend, so expose the fold as disabled.
        _fold_qb_uk = False
