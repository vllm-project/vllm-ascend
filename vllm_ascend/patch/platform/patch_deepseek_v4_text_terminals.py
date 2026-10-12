# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Keep DeepSeek V4 text-form DSML terminals out of streamed content.

`StreamingParserEngine._process_lex_tokens` demotes a terminal to content once the
stream has carried any token id, including a terminal that was lexed from plain
text. DeepSeek V4 emits its DSML markers as text on the MTP path, so the whole tool
block reaches the client as content instead of a tool call. The same markers also
arrive as dedicated token ids on other paths, so the outcome must not depend on the
encoding that arrives first.

Remove this patch once the supported vLLM keeps text-form terminals authoritative
for the DeepSeek V4 parser config.
"""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any

from vllm.logger import logger

if TYPE_CHECKING:
    from vllm.parser.engine.events import SemanticEvent
    from vllm.parser.engine.incremental_lexer import LexToken

_PATCHED_ATTR = "_ascend_text_form_dsv4_terminals_patched"

# DeepSeek V4 declares these DSML tool markers both as lexical text and as token
# ids. The engine can only demote the text form while the markers are token-id
# terminals, so a config that no longer declares them makes this patch obsolete.
_TOKEN_ID_MARKERS = frozenset({"TOOL_START", "TOOL_END"})

try:
    from vllm.parser.deepseek_v4 import deepseek_v4_config
    from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine
except ImportError:  # pragma: no cover - a vLLM revision without the parser engine
    logger.warning_once(
        "The DeepSeek V4 text-form terminal patch is skipped: the supported vLLM "
        "does not provide the parser engine this patch targets."
    )
    deepseek_v4_config = None  # type: ignore[assignment]
    StreamingParserEngine = None  # type: ignore[assignment]


def _install() -> None:
    """Make text-form terminals authoritative for the DeepSeek V4 parser config."""
    if StreamingParserEngine is None or deepseek_v4_config is None:
        return

    if getattr(StreamingParserEngine, _PATCHED_ATTR, False):
        return

    config = deepseek_v4_config()
    missing = _TOKEN_ID_MARKERS - set(config.token_id_terminals)
    if missing:
        logger.warning_once(
            "The DeepSeek V4 text-form terminal patch is obsolete: the supported vLLM no "
            f"longer declares {sorted(missing)} as token-id terminals. Remove this patch."
        )
        return

    original_process_lex_tokens = StreamingParserEngine._process_lex_tokens

    @wraps(original_process_lex_tokens)
    def _patched_process_lex_tokens(self: Any, tokens: list[LexToken]) -> list[SemanticEvent]:
        # The engine rebuilds its strict set from `_token_id_terminal_names` on every
        # call, so emptying that set for the duration of the call keeps text-form
        # terminals authoritative without mirroring the upstream loop body. Restore it
        # in a `finally` block so one call never leaks the change into the next one.
        if not self._ever_had_token_ids or getattr(self.config, "name", None) != config.name:
            return original_process_lex_tokens(self, tokens)

        saved = self._token_id_terminal_names
        self._token_id_terminal_names = frozenset()
        try:
            return original_process_lex_tokens(self, tokens)
        finally:
            self._token_id_terminal_names = saved

    StreamingParserEngine._process_lex_tokens = _patched_process_lex_tokens
    setattr(StreamingParserEngine, _PATCHED_ATTR, True)


_install()
