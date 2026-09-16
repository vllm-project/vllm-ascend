# SPDX-License-Identifier: Apache-2.0

from vllm.renderers import hf

from vllm_ascend.patch.platform import patch_chat_template_content_format  # noqa: F401
from vllm_ascend.patch.platform.patch_chat_template_content_format import _install_patch

_QWEN35_MACRO_CONTENT_TEMPLATE = """
{%- macro render_content(content) %}
    {%- if content is string %}
        {{- content }}
    {%- elif content is iterable and content is not mapping %}
        {%- for item in content %}
            {%- if 'image' in item or item.type == 'image' %}
                {{- '<|vision_start|><|image_pad|><|vision_end|>' }}
            {%- elif 'text' in item %}
                {{- item.text }}
            {%- endif %}
        {%- endfor %}
    {%- endif %}
{%- endmacro %}
{%- for message in messages %}
    {{- render_content(message.content) }}
{%- endfor %}
"""

_STRING_CONTENT_TEMPLATE = """
{%- macro render_content(items) %}
    {%- for item in items %}
        {{- item }}
    {%- endfor %}
{%- endmacro %}
{%- for message in messages %}
    {{- render_content(message.content) }}
{%- endfor %}
"""


def _resolve_content_format(chat_template: str) -> str:
    _install_patch()
    hf._detect_content_format.cache_clear()
    return hf._detect_content_format(chat_template, default="string")


def test_qwen35_macro_content_loop_is_detected_as_openai():
    assert _resolve_content_format(_QWEN35_MACRO_CONTENT_TEMPLATE) == "openai"


def test_unrelated_macro_loop_stays_string():
    assert _resolve_content_format(_STRING_CONTENT_TEMPLATE) == "string"


def test_patch_installation_is_idempotent():
    patched_detector = hf._iter_nodes_assign_content_item

    _install_patch()

    assert hf._iter_nodes_assign_content_item is patched_detector
