# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse vLLM media loading while encoding the original V4.1 messages."""

import copy

from vllm.entrypoints.chat_utils import parse_chat_messages, parse_chat_messages_async
from vllm.renderers.deepseek_v4 import DeepseekV4Renderer
from vllm.renderers.inputs.preprocess import parse_dec_only_prompt


def _media_blocks(blocks):
    """Expose images nested in reference tool_result/content_blocks to vLLM."""
    result = []
    for block in blocks:
        if block.get("type") == "tool_result":
            content = block.get("content", "")
            result.extend(_media_blocks(content) if isinstance(content, list) else [{"type": "text", "text": content}])
        elif block.get("type") == "image":
            source = block.get("source") or {}
            url = block.get("url") or source.get("url")
            if source.get("type") == "base64":
                url = f"data:{source['media_type']};base64,{source['data']}"
            url = url or block.get("data")
            result.append({"type": "image_url", "image_url": {"url": url}})
        elif block.get("type") == "image_url" and isinstance(block.get("image_url"), str):
            result.append({**block, "image_url": {"url": block["image_url"]}})
        else:
            result.append(block)
    return result


def _media_messages(messages):
    messages = copy.deepcopy(messages)
    for message in messages:
        content = message.pop("content_blocks", message.get("content"))
        if isinstance(content, list):
            message["content"] = _media_blocks(content)
        if "reasoning_content" in message and "reasoning" not in message:
            message["reasoning"] = message["reasoning_content"]
    return messages


class DeepseekV41Renderer(DeepseekV4Renderer):
    def _template_kwargs(self, params):
        return {**params.get_apply_chat_template_kwargs(), "response_format": params.response_format}

    @staticmethod
    def _prompt(raw, media, uuids):
        prompt = parse_dec_only_prompt(raw)
        if media is not None:
            prompt["multi_modal_data"] = media
        if uuids is not None:
            prompt["multi_modal_uuids"] = uuids
        return prompt

    def render_messages(self, messages, params):
        conversation, media, uuids = parse_chat_messages(
            _media_messages(messages),
            self.model_config,
            content_format="string",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )
        raw = self._apply_chat_template(messages=messages, **self._template_kwargs(params))
        return conversation, self._prompt(raw, media, uuids)

    async def render_messages_async(self, messages, params):
        conversation, media, uuids = await parse_chat_messages_async(
            _media_messages(messages),
            self.model_config,
            content_format="string",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )
        raw = await self._apply_chat_template_async(messages=messages, **self._template_kwargs(params))
        return conversation, self._prompt(raw, media, uuids)
