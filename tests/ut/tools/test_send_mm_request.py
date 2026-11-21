# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools import send_mm_request


@pytest.mark.parametrize(
    "url",
    ["https://example.com/image.jpg", "http://example.com/image.png", "data:image/png;base64,aW1hZ2U="],
)
def test_ready_image_url_needs_no_loading(monkeypatch, url):
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url, "detail": "low"}}]}]
    mm = {"messages": messages}
    original = deepcopy(mm)
    load_image = Mock(side_effect=AssertionError("ready image URLs must not load files or datasets"))
    monkeypatch.setattr(send_mm_request, "_load_image_data", load_image)

    assert not send_mm_request._messages_need_preload(mm)
    assert send_mm_request._build_messages(mm, []) == messages
    assert mm == original
    load_image.assert_not_called()


@pytest.mark.parametrize("content", ["Describe the image", ""])
def test_string_message_content_is_preserved(content):
    messages = [{"role": "system", "content": content}]
    mm = {"messages": messages}

    assert not send_mm_request._messages_need_preload(mm)
    assert send_mm_request._build_messages(mm, []) == messages


@pytest.mark.parametrize("url", ["", "{IMAGE_0}", "/tmp/image.png", "file:///tmp/image.jpg"])
def test_images_requiring_preload_are_still_detected(url):
    mm = {"messages": [{"content": [{"type": "image_url", "image_url": {"url": url}}]}]}
    assert send_mm_request._messages_need_preload(mm)


def test_placeholder_and_empty_urls_use_preloaded_images():
    mm = {
        "messages": [
            {
                "content": [
                    {"type": "image_url", "image_url": {"url": "{IMAGE_1}"}},
                    {"type": "image_url", "image_url": {"url": ""}},
                    {"type": "image_url", "image_url": {"url": ""}},
                ]
            }
        ]
    }
    original = deepcopy(mm)
    messages = send_mm_request._build_messages(mm, ["first", "second"])

    assert messages[0]["role"] == "user"
    assert [part["image_url"]["url"] for part in messages[0]["content"]] == [
        "data:image/jpeg;base64,second",
        "data:image/jpeg;base64,first",
        "data:image/jpeg;base64,second",
    ]
    assert mm == original


def test_local_image_still_uses_loader(monkeypatch):
    load_image = Mock(return_value="local-image")
    monkeypatch.setattr(send_mm_request, "_load_image_data", load_image)
    mm = {"messages": [{"content": [{"type": "image_url", "image_url": {"url": "file:///tmp/image.jpg"}}]}]}

    messages = send_mm_request._build_messages(mm, [])

    load_image.assert_called_once_with("file:///tmp/image.jpg")
    assert messages[0]["content"][0]["image_url"]["url"] == "data:image/jpeg;base64,local-image"


def test_send_inline_image_with_string_system_message(monkeypatch):
    messages = [
        {"role": "system", "content": "Describe images briefly."},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U="}}]},
    ]
    config = SimpleNamespace(model="test-model", mm_request={"messages": messages}, expected_response={})
    response_json = {"choices": [{"message": {"content": "An image."}}]}
    post = Mock(return_value=SimpleNamespace(status_code=200, json=lambda: response_json))
    load_image = Mock(side_effect=AssertionError("inline requests must not load files or datasets"))
    monkeypatch.setattr(send_mm_request.requests, "post", post)
    monkeypatch.setattr(send_mm_request, "_load_image_data", load_image)
    server = SimpleNamespace(url_for=lambda *parts: "http://localhost/" + "/".join(parts))

    assert send_mm_request.send_image_request(config, server) == response_json

    assert post.call_args.kwargs["json"]["messages"] == messages
    load_image.assert_not_called()


@pytest.mark.parametrize("content", [None, "", [{"type": "text", "text": "hello"}]])
def test_message_metadata_and_null_content_are_preserved(content):
    messages = [
        {
            "role": "assistant",
            "content": content,
            "name": "helper",
            "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
        },
        {"role": "tool", "content": "result", "tool_call_id": "call-1"},
    ]
    mm = {"messages": messages}
    original = deepcopy(mm)

    assert not send_mm_request._messages_need_preload(mm)
    assert send_mm_request._build_messages(mm, []) == messages
    assert mm == original


@pytest.mark.parametrize("mm", [{}, {"messages": None}, {"messages": []}])
def test_absent_or_empty_messages_keep_default_prompt(mm):
    assert not send_mm_request._messages_need_preload(mm)
    assert send_mm_request._build_messages(mm, []) == [
        {"role": "user", "content": [{"type": "text", "text": send_mm_request.DEFAULT_TEXT_PROMPT}]}
    ]


@pytest.mark.parametrize(
    ("messages", "error"),
    [
        ({}, "messages must be a list or null"),
        ("hello", "messages must be a list or null"),
        ([None], "messages[0] must be an object"),
        ([{"content": 0}], "messages[0].content must be a string, list, or null"),
        ([{"content": False}], "messages[0].content must be a string, list, or null"),
        ([{"content": {}}], "messages[0].content must be a string, list, or null"),
        ([{"content": [None]}], "messages[0].content[0] must be an object"),
        ([{"content": ["text"]}], "messages[0].content[0] must be an object"),
        ([{"content": [{"type": "text", "text": "valid"}, 1]}], "messages[0].content[1] must be an object"),
    ],
)
def test_invalid_message_containers_fail_explicitly(messages, error):
    mm = {"messages": messages}
    original = deepcopy(mm)

    for build in (lambda: send_mm_request._messages_need_preload(mm), lambda: send_mm_request._build_messages(mm, [])):
        with pytest.raises(ValueError) as failure:
            build()
        assert str(failure.value) == error
    assert mm == original


def test_invalid_later_message_fails_before_image_loading_or_http(monkeypatch):
    messages = [
        {"content": [{"type": "image_url", "image_url": {"url": "/tmp/image.png"}}]},
        {"content": [None]},
    ]
    config = SimpleNamespace(model="test-model", mm_request={"messages": messages}, expected_response={})
    original = deepcopy(config.mm_request)
    load_image = Mock(side_effect=AssertionError("invalid messages must not load images"))
    post = Mock(side_effect=AssertionError("invalid messages must not be submitted"))
    monkeypatch.setattr(send_mm_request, "_load_image_data", load_image)
    monkeypatch.setattr(send_mm_request.requests, "post", post)

    with pytest.raises(ValueError) as failure:
        send_mm_request.send_image_request(config, SimpleNamespace())

    assert str(failure.value) == "messages[1].content[0] must be an object"
    assert config.mm_request == original
    load_image.assert_not_called()
    post.assert_not_called()


def test_send_null_assistant_content_preserves_tool_call_metadata(monkeypatch):
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1", "type": "function"}]},
        {"role": "tool", "content": "done", "tool_call_id": "call-1"},
    ]
    config = SimpleNamespace(model="test-model", mm_request={"messages": messages}, expected_response={})
    response_json = {"choices": [{"message": {"content": "Done."}}]}
    post = Mock(return_value=SimpleNamespace(status_code=200, json=lambda: response_json))
    load_image = Mock(side_effect=AssertionError("null content must not load images"))
    monkeypatch.setattr(send_mm_request.requests, "post", post)
    monkeypatch.setattr(send_mm_request, "_load_image_data", load_image)
    server = SimpleNamespace(url_for=lambda *parts: "http://localhost/" + "/".join(parts))

    assert send_mm_request.send_image_request(config, server) == response_json
    assert post.call_args.kwargs["json"]["messages"] == messages
    load_image.assert_not_called()
