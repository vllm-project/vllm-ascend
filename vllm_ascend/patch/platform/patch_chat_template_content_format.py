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
# Backport the Qwen3.5 chat-template content-format fix from vLLM #42660.
#
# vLLM v0.18.0 only recognizes loops that read message content through
# ``message['content']``. Qwen3.5 passes content to a macro whose loop uses
# ``for item in content``. Auto detection therefore returns ``string`` and
# reorders multimodal items in tool responses, which prevents the template
# from rebuilding historical assistant reasoning in later agent turns.

from __future__ import annotations

import jinja2
from vllm.renderers import hf as hf_renderer

from vllm_ascend.utils import vllm_version_is

_original_iter_nodes_assign_content_item = hf_renderer._iter_nodes_assign_content_item
_PATCHED_ATTR = "_vllm_ascend_chat_template_content_format_patched"


def _iter_nodes_assign_content_item(root: jinja2.nodes.Node):
    seen_loops: set[int] = set()

    for loop_ast, loop_target in _original_iter_nodes_assign_content_item(root):
        seen_loops.add(id(loop_ast))
        yield loop_ast, loop_target

    # Match the fallback added by upstream
    # https://github.com/vllm-project/vllm/pull/42660.
    for loop_ast in root.find_all(jinja2.nodes.For):
        if id(loop_ast) in seen_loops:
            continue

        loop_iter = loop_ast.iter
        loop_target = loop_ast.target
        if (
            isinstance(loop_iter, jinja2.nodes.Name)
            and loop_iter.name == "content"
            and isinstance(loop_target, jinja2.nodes.Name)
        ):
            yield loop_ast, loop_target.name


def _install_patch() -> None:
    if getattr(hf_renderer, _PATCHED_ATTR, False):
        return

    hf_renderer._iter_nodes_assign_content_item = _iter_nodes_assign_content_item
    hf_renderer._detect_content_format.cache_clear()
    setattr(hf_renderer, _PATCHED_ATTR, True)


if vllm_version_is("0.18.0"):
    _install_patch()
