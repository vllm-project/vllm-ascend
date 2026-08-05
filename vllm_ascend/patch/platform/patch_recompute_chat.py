# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from functools import wraps

from vllm.renderers.online_renderer import OnlineRenderer

from vllm_ascend.patch.recompute_proxy import replace_rendered_chat_inputs

_ORIGINAL_RENDER_CHAT_ATTR = "_ascend_original_recompute_render_chat"

if not hasattr(OnlineRenderer, _ORIGINAL_RENDER_CHAT_ATTR):
    setattr(
        OnlineRenderer,
        _ORIGINAL_RENDER_CHAT_ATTR,
        OnlineRenderer.render_chat,
    )


@wraps(getattr(OnlineRenderer, _ORIGINAL_RENDER_CHAT_ATTR))
async def _render_chat_with_recompute_tokens(
    self: OnlineRenderer,
    request,
    *,
    skip_mm_cache: bool = False,
):
    original = getattr(type(self), _ORIGINAL_RENDER_CHAT_ATTR)
    result = await original(self, request, skip_mm_cache=skip_mm_cache)
    return replace_rendered_chat_inputs(request, result)


OnlineRenderer.render_chat = _render_chat_with_recompute_tokens
