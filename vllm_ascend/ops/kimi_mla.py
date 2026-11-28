# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-specific MLA adapter, imported only for the vLLM main lane."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from vllm.models.kimi_k3.amd.mla import KimiK3MultiHeadLatentAttentionWrapper

from vllm_ascend.ops.mla import AscendMultiHeadLatentAttention

# Upstream Kimi constructs its wrapper without forwarding attention options.
_draft_window: ContextVar[tuple[int | None, bool]] = ContextVar("kimi_mla_draft_window", default=(None, False))


@contextmanager
def kimi_mla_draft_window(window: int | None, *, uniform_causal: bool) -> Iterator[None]:
    token = _draft_window.set((window, uniform_causal))
    try:
        yield
    finally:
        _draft_window.reset(token)


class AscendKimiK3MultiHeadLatentAttention(AscendMultiHeadLatentAttention, KimiK3MultiHeadLatentAttentionWrapper):
    """Keep Ascend MLA dispatch after vLLM #52494 introduced an AMD subclass.

    OOT lookup uses the concrete class name. Inherit the upstream subclass too
    so Python runs the Ascend initializer on the replacement instance.
    """

    # vLLM #60110 added this flag and reads it in the upstream
    # process_weights_after_loading hook. The AMD initializer never runs on
    # Ascend, so expose the fold as disabled.
    _fold_qb_uk = False

    def __init__(self, *args, **kwargs) -> None:
        window, uniform_causal = _draft_window.get()
        super().__init__(*args, sliding_window=window, dflash_draft_window=uniform_causal, **kwargs)
