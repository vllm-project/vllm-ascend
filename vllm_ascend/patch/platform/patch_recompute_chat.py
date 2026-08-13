# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from vllm.entrypoints.serve.render.serving import OpenAIServingRender

from vllm_ascend.recompute_proxy import replace_rendered_chat_inputs

_original_render_chat = OpenAIServingRender.render_chat


async def _render_chat_with_recompute_tokens(self, request, *, skip_mm_cache=False):
    result = await _original_render_chat(self, request, skip_mm_cache=skip_mm_cache)
    return replace_rendered_chat_inputs(request, result)


OpenAIServingRender.render_chat = _render_chat_with_recompute_tokens
