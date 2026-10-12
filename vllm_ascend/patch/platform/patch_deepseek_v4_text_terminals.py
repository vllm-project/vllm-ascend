#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# DeepSeek V4 text-form DSML terminal patch.
#

from __future__ import annotations

from typing import Any

from vllm_ascend.utils import vllm_version_is

_PATCHED_ATTR = "_ascend_text_form_dsv4_terminals_patched"

# This module is imported unconditionally by `vllm_ascend.patch.platform`, so every
# import and definition that depends on the parser engine of the supported vLLM
# version stays behind the version gate; on any other version the module loads and
# does nothing.
if vllm_version_is("0.27.1"):
    from functools import wraps

    from vllm.parser.deepseek_v4 import deepseek_v4_config
    from vllm.parser.engine.events import SemanticEvent
    from vllm.parser.engine.incremental_lexer import LexToken
    from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine

    _DSV4_CONFIG_NAME = deepseek_v4_config().name
    _TOKEN_ID_MARKERS = frozenset({"TOOL_START", "TOOL_END"})

    _original_process_lex_tokens = StreamingParserEngine._process_lex_tokens

    def _keeps_text_terminals(engine: Any) -> bool:
        """Report whether a parser config must keep text-form terminals active."""
        return getattr(engine.config, "name", None) == _DSV4_CONFIG_NAME

    @wraps(_original_process_lex_tokens)
    def _patched_process_lex_tokens(
        self: Any,
        tokens: list[LexToken],
    ) -> list[SemanticEvent]:
        # Upstream demotes a terminal to content once the stream has carried any token
        # id, even when that terminal was lexed from plain text. DeepSeek V4 emits its
        # DSML markers as text on the MTP path, so the demotion emits the whole tool
        # block as content. DeepSeek V4 also emits the same markers as dedicated token
        # ids on other paths, so the demotion must not depend on the encoding that
        # happened to arrive first.
        #
        # The engine rebuilds its strict set from `_token_id_terminal_names` on every
        # call, so emptying that set for the duration of the call keeps text-form
        # terminals authoritative without mirroring the upstream loop body. Only the
        # DeepSeek V4 parser config is affected; every other config keeps the upstream
        # behavior.
        if not self._ever_had_token_ids or not _keeps_text_terminals(self):
            return _original_process_lex_tokens(self, tokens)

        saved = self._token_id_terminal_names
        self._token_id_terminal_names = frozenset()
        try:
            return _original_process_lex_tokens(self, tokens)
        finally:
            self._token_id_terminal_names = saved

    def _install() -> None:
        if getattr(StreamingParserEngine, _PATCHED_ATTR, False):
            return

        missing = _TOKEN_ID_MARKERS - set(deepseek_v4_config().token_id_terminals)
        if missing:
            raise RuntimeError(
                "The DeepSeek V4 text-form terminal patch is obsolete: the supported vLLM "
                f"no longer declares {sorted(missing)} as token-id terminals. Remove this patch."
            )

        StreamingParserEngine._process_lex_tokens = _patched_process_lex_tokens
        setattr(StreamingParserEngine, _PATCHED_ATTR, True)

    _install()
