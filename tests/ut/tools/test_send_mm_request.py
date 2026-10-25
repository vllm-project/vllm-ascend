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
