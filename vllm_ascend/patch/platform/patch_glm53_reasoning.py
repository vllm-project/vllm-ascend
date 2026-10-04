# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Backport GLM-5.3 always-on reasoning from vllm-project/vllm#56994.

Remove once all supported vLLM versions include the upstream fix.
"""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING

from vllm.logger import logger
from vllm.parser import glm47_moe

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool


def _glm53_always_thinks(tokenizer: TokenizerLike) -> bool:
    # Keep the positive template signature from upstream PR #56994. Older
    # GLM templates with an enable_thinking switch must retain their behavior.
    template = getattr(tokenizer, "chat_template", None)
    return (
        isinstance(template, str)
        and "[gMASK]<sop>" in template
        and "Reasoning Effort:" in template
        and "enable_thinking" not in template
        and "<tool_call>" in template
        and "<arg_key>" in template
        and "<arg_value>" in template
    )


def _patch_glm53_reasoning() -> None:
    # Upstream PR #56994 adds this helper together with the constructor fix.
    if hasattr(glm47_moe, "_glm53_always_thinks"):
        return

    original_init = glm47_moe.Glm47MoeParser.__init__
    if getattr(original_init, "_vllm_ascend_glm53_reasoning", False):
        return

    @wraps(original_init)
    def patched_init(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        thinking = chat_kwargs.get("thinking")
        enable_thinking = chat_kwargs.get("enable_thinking")
        if (thinking is False or enable_thinking is False) and _glm53_always_thinks(tokenizer):
            logger.warning_once(
                "Ignoring enable_thinking/thinking: the GLM-5.3 chat template "
                "has no thinking switch, so reasoning is always on and "
                "disabling extraction would leak it into the content."
            )
            # Normalize only the parser's copy, before it selects its initial
            # state and reasoning terminals. Never mutate caller-owned kwargs.
            kwargs["chat_template_kwargs"] = dict(chat_kwargs, thinking=None, enable_thinking=None)
        original_init(self, tokenizer, tools, **kwargs)

    patched_init._vllm_ascend_glm53_reasoning = True  # type: ignore[attr-defined]
    glm47_moe.Glm47MoeParser.__init__ = patched_init


_patch_glm53_reasoning()
