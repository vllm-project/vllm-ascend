#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Unit tests for :mod:`vllm_ascend.patch.platform.patch_glm_thinking_mode`.

Tests focus on the monkey-patched ``ChatCompletionRequest.build_chat_params``
method, specifically the mapping:

* ``thinking: {type: "disabled"}`` -> ``enable_thinking: False`` injected into
  ``chat_template_kwargs``.
* ``thinking: {type: "enabled"}`` -> ``enable_thinking: True`` injected.
* No ``thinking`` field -> original ``chat_template_kwargs`` returned unchanged.
* ``enable_thinking`` already set in ``chat_template_kwargs`` -> not overridden.
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field, replace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Lightweight stubs for vllm types that we cannot import in a plain unit-test
# context without a full vllm installation.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeChatParams:
    """Minimal stand-in for ``vllm.renderers.ChatParams``."""

    chat_template: str | None = None
    chat_template_content_format: str = "auto"
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    extra_field: str | None = None

    def model_copy(self, update: dict[str, Any] | None = None) -> "_FakeChatParams":
        """Minimal ``BaseModel.model_copy`` compatible helper for tests."""
        return replace(self, **(update or {}))


def _merge_kwargs(
    defaults: dict[str, Any] | None,
    overrides: dict[str, Any] | None,
    *,
    unset_values: tuple[object, ...] = (None, "auto"),
) -> dict[str, Any]:
    """Replica of ``vllm.renderers.merge_kwargs`` for offline tests."""
    if defaults is None:
        defaults = {}
    if overrides is None:
        overrides = {}
    return defaults | {k: v for k, v in overrides.items() if v not in unset_values}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(
    thinking: Any | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
    chat_template: str | None = None,
    model_extra: dict[str, Any] | None = None,
    pydantic_extra: dict[str, Any] | None = None,
) -> MagicMock:
    """Create a ``MagicMock`` that mimics ``ChatCompletionRequest``."""
    request = MagicMock()
    default_extra = {} if thinking is None else {"thinking": thinking}
    request.model_extra = default_extra if model_extra is None else model_extra
    request.__pydantic_extra__ = pydantic_extra
    request.chat_template = chat_template
    request.chat_template_kwargs = chat_template_kwargs
    return request


def _make_original_result(
    chat_template_kwargs: dict[str, Any] | None = None,
    chat_template: str | None = None,
    extra_field: str | None = None,
) -> _FakeChatParams:
    return _FakeChatParams(
        chat_template=chat_template,
        chat_template_content_format="auto",
        chat_template_kwargs=chat_template_kwargs or {},
        extra_field=extra_field,
    )


# ---------------------------------------------------------------------------
# Patch the heavy-weight vllm module imports *before* importing our patch.
# ---------------------------------------------------------------------------

_fake_chat_completion_protocol = MagicMock()

# ``ChatCompletionRequest`` will be replaced after we import the patch module.
_fake_chat_completion_cls = MagicMock()
_fake_chat_completion_protocol.ChatCompletionRequest = _fake_chat_completion_cls

_fake_renderers = MagicMock()
_fake_renderers.ChatParams = _FakeChatParams
_fake_renderers.merge_kwargs = _merge_kwargs


@pytest.fixture(autouse=True, scope="module")
def _patch_vllm_imports():
    """Inject lightweight stubs so the patch module can be imported."""
    with patch.dict(
        sys.modules,
        {
            "vllm": MagicMock(),
            "vllm.renderers": _fake_renderers,
            "vllm.entrypoints": MagicMock(),
            "vllm.entrypoints.openai": MagicMock(),
            "vllm.entrypoints.openai.chat_completion": MagicMock(),
            "vllm.entrypoints.openai.chat_completion.protocol": _fake_chat_completion_protocol,
        },
    ):
        module_name = "vllm_ascend.patch.platform.patch_glm_thinking_mode"
        if module_name in sys.modules:
            del sys.modules[module_name]
        module = importlib.import_module(module_name)
        yield module


# ---------------------------------------------------------------------------
# Utility: extract the patched function from the module
# ---------------------------------------------------------------------------


def _get_patched_fn():
    """Return the current patched function from the module."""
    module_name = "vllm_ascend.patch.platform.patch_glm_thinking_mode"
    mod = sys.modules[module_name]
    return mod._patched_build_chat_params


def _call_patched(
    thinking: Any | None = None,
    existing_kwargs: dict[str, Any] | None = None,
    chat_template: str | None = None,
    model_extra: dict[str, Any] | None = None,
    pydantic_extra: dict[str, Any] | None = None,
    extra_field: str | None = None,
):
    """Exercise the patched function end-to-end with mocked collaborators."""
    original_result = _make_original_result(
        chat_template_kwargs=existing_kwargs,
        chat_template=chat_template,
        extra_field=extra_field,
    )

    patched_fn = _get_patched_fn()
    module = sys.modules["vllm_ascend.patch.platform.patch_glm_thinking_mode"]
    original_original = module._original_build_chat_params
    module._original_build_chat_params = MagicMock(return_value=original_result)

    try:
        request = _make_request(
            thinking=thinking,
            chat_template_kwargs=existing_kwargs,
            chat_template=chat_template,
            model_extra=model_extra,
            pydantic_extra=pydantic_extra,
        )
        return patched_fn(request, None, "auto")
    finally:
        module._original_build_chat_params = original_original


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGlmThinkingModeDisabled:
    """``thinking: {type: "disabled"}`` -> ``enable_thinking: False``."""

    def test_enable_thinking_false_injected(self):
        result = _call_patched(thinking={"type": "disabled"})
        assert isinstance(result, _FakeChatParams)
        assert result.chat_template_kwargs.get("enable_thinking") is False

    def test_existing_kwargs_preserved(self):
        result = _call_patched(
            thinking={"type": "disabled"},
            existing_kwargs={"foo": "bar"},
        )
        assert result.chat_template_kwargs.get("foo") == "bar"
        assert result.chat_template_kwargs.get("enable_thinking") is False

    def test_enable_thinking_not_overridden_when_already_set(self):
        """A user-supplied ``enable_thinking`` in ``chat_template_kwargs`` wins."""
        result = _call_patched(
            thinking={"type": "disabled"},
            existing_kwargs={"enable_thinking": True},
        )
        assert result.chat_template_kwargs.get("enable_thinking") is True

    def test_falls_back_to_pydantic_extra(self):
        result = _call_patched(
            thinking=None,
            model_extra=None,
            pydantic_extra={"thinking": {"type": "disabled"}},
        )
        assert result.chat_template_kwargs.get("enable_thinking") is False

    def test_preserves_other_chat_params_fields(self):
        result = _call_patched(
            thinking={"type": "disabled"},
            extra_field="keep-me",
        )
        assert result.extra_field == "keep-me"


class TestGlmThinkingModeEnabled:
    """``thinking: {type: "enabled"}`` -> ``enable_thinking: True``."""

    def test_enable_thinking_true_injected(self):
        result = _call_patched(thinking={"type": "enabled"})
        assert result.chat_template_kwargs.get("enable_thinking") is True

    def test_enable_thinking_not_overridden_when_already_false(self):
        """``chat_template_kwargs`` takes precedence."""
        result = _call_patched(
            thinking={"type": "enabled"},
            existing_kwargs={"enable_thinking": False},
        )
        assert result.chat_template_kwargs.get("enable_thinking") is False


class TestGlmThinkingModeAbsent:
    """When no ``thinking`` field is present, result must be unchanged."""

    def test_no_thinking_field(self):
        original = _make_original_result(chat_template_kwargs={"a": 1})
        module = sys.modules["vllm_ascend.patch.platform.patch_glm_thinking_mode"]
        original_original = module._original_build_chat_params
        module._original_build_chat_params = MagicMock(return_value=original)
        patched_fn = _get_patched_fn()
        try:
            request = _make_request(thinking=None)
            result = patched_fn(request, None, "auto")
        finally:
            module._original_build_chat_params = original_original

        assert result is original

    def test_thinking_is_not_dict(self):
        """Non-dict ``thinking`` values are ignored."""
        original = _make_original_result(chat_template_kwargs={"a": 1})
        module = sys.modules["vllm_ascend.patch.platform.patch_glm_thinking_mode"]
        original_original = module._original_build_chat_params
        module._original_build_chat_params = MagicMock(return_value=original)
        patched_fn = _get_patched_fn()
        try:
            request = _make_request(thinking="disabled")
            result = patched_fn(request, None, "auto")
        finally:
            module._original_build_chat_params = original_original

        assert result is original


class TestGlmThinkingModeUnknownType:
    """Unknown ``thinking.type`` values leave the result unchanged."""

    def test_unknown_type_returns_original(self):
        original = _make_original_result()
        module = sys.modules["vllm_ascend.patch.platform.patch_glm_thinking_mode"]
        original_original = module._original_build_chat_params
        module._original_build_chat_params = MagicMock(return_value=original)
        patched_fn = _get_patched_fn()
        try:
            request = _make_request(thinking={"type": "auto"})
            result = patched_fn(request, None, "auto")
        finally:
            module._original_build_chat_params = original_original

        assert result is original
