# SPDX-License-Identifier: Apache-2.0

from vllm.tokenizers import deepseek_v4_encoding

import vllm_ascend.patch.platform.patch_deepseek_v4_trailing_system  # noqa: F401


def test_trailing_system_appends_assistant_transition():
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "question"},
        {"role": "system", "content": "final rules"},
    ]

    chat_prompt = deepseek_v4_encoding.encode_messages(messages, thinking_mode="chat")
    thinking_prompt = deepseek_v4_encoding.encode_messages(
        messages,
        thinking_mode="thinking",
    )

    prefix = "<｜begin▁of▁sentence｜>rules<｜User｜>questionfinal rules"
    assert chat_prompt == prefix + "<｜Assistant｜></think>"
    assert thinking_prompt == prefix + "<｜Assistant｜><think>"


def test_system_to_assistant_appends_transition():
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "assistant", "reasoning": "reason", "content": "answer"},
    ]

    chat_prompt = deepseek_v4_encoding.encode_messages(messages, thinking_mode="chat")
    thinking_prompt = deepseek_v4_encoding.encode_messages(
        messages,
        thinking_mode="thinking",
    )

    assert chat_prompt == ("<｜begin▁of▁sentence｜>rules<｜Assistant｜></think>answer<｜end▁of▁sentence｜>")
    assert thinking_prompt == (
        "<｜begin▁of▁sentence｜>rules<｜Assistant｜><think>reason</think>answer<｜end▁of▁sentence｜>"
    )


def test_mid_conversation_system_does_not_append_transition():
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "system", "content": "new rules"},
        {"role": "user", "content": "second"},
    ]

    prompt = deepseek_v4_encoding.encode_messages(messages, thinking_mode="chat")

    assert "new rules<｜User｜>second" in prompt
    assert "new rules<｜Assistant｜>" not in prompt
