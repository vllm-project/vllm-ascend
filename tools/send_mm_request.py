from __future__ import annotations

import base64
import os
from typing import TYPE_CHECKING, Any

import huggingface_hub
import regex as re
import requests
from modelscope import snapshot_download  # type: ignore

if TYPE_CHECKING:
    from tests.e2e.nightly.single_node.models.scripts.single_node_config import SingleNodeConfig

DEFAULT_TEXT_PROMPT = "What is the content of this image?"

DEFAULT_IMAGE_REQUEST_ARGS: dict[str, Any] = {
    "eos_token_id": [1, 106],
    "pad_token_id": 0,
    "top_k": 64,
    "top_p": 0.95,
    "max_tokens": 8192,
    "stream": False,
}

_HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}

_FILE_URL_RE = re.compile(r"^file://", re.IGNORECASE)


def _resolve_fs_path(path: str) -> str:
    return _FILE_URL_RE.sub("", path, count=1)


def _load_image_data(image_path: str | None = None) -> str:
    if not image_path:
        image_path = os.environ.get("MM_IMAGE_PATH")
    if not image_path:
        mm_dir = snapshot_download(
            "vllm-ascend/mm_request",
            repo_type="dataset",
            local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
        )
        image_path = os.path.join(mm_dir, "test_mm2.jpg")
    with open(_resolve_fs_path(image_path), "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def _load_images(paths: list[str | None], default_path: str | None = None) -> list[str]:
    if not default_path:
        default_path = os.environ.get("MM_IMAGE_PATH")
    return [_load_image_data(p or default_path) for p in paths]


def _messages_need_preload(mm: dict[str, Any]) -> bool:
    """Return True only when at least one image_url needs base64 preloading."""
    for msg in mm.get("messages") or []:
        for part in msg.get("content", []):
            if part.get("type") != "image_url":
                continue
            url = ((part.get("image_url") or {}).get("url") or "").strip()
            if not url:
                return True
            if re.search(r"\{IMAGE_\d+\}", url):
                return True
            if not url.startswith(("http://", "https://")):
                return True
    return bool(mm.get("images"))


def _build_messages(mm: dict[str, Any], image_data_list: list[str]) -> list[dict[str, Any]]:
    raw = mm.get("messages")

    if not raw:
        parts: list[dict[str, Any]] = [{"type": "text", "text": mm.get("text_prompt", DEFAULT_TEXT_PROMPT)}]
        for img in image_data_list:
            parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
        return [{"role": "user", "content": parts}]
    image_index = 0
    messages: list[dict[str, Any]] = []
    for msg in raw:
        content_parts: list[dict[str, Any]] = []
        for part in msg.get("content", []):
            p = dict(part)
            if p.get("type") == "image_url":
                img = dict(p.get("image_url") or {})
                url = (img.get("url") or "").strip()
                if url:
                    if re.search(r"\{IMAGE_\d+\}", url):
                        for i, data in enumerate(image_data_list):
                            url = url.replace(f"{{IMAGE_{i}}}", f"data:image/jpeg;base64,{data}")
                    elif url.startswith(("http://", "https://")):
                        pass
                    else:
                        url = f"data:image/jpeg;base64,{_load_image_data(url)}"
                elif image_index < len(image_data_list):
                    url = f"data:image/jpeg;base64,{image_data_list[image_index]}"
                    image_index += 1
                img["url"] = url
                p["image_url"] = img
            content_parts.append(p)
        messages.append({"role": msg.get("role", "user"), "content": content_parts})
    return messages


def _assert_response(status_code: int, response_json: dict[str, Any], expected: dict[str, Any]) -> None:
    expected_status = expected.get("status_code")
    if expected_status is not None:
        assert status_code == expected_status, f"expected status_code {expected_status}, got {status_code}"

    choices = response_json.get("choices", [])
    assert choices, "no choices in response"
    content = choices[0].get("message", {}).get("content", "")

    if expected.get("assert_not_empty", True):
        assert content, "empty response"

    for keyword in expected.get("content_contains", []):
        assert keyword in content, f"response missing expected keyword: '{keyword}'"

    max_length = expected.get("max_length")
    if max_length is not None:
        assert len(content) <= max_length, f"response too long: {len(content)} > {max_length}"

    min_length = expected.get("min_length")
    if min_length is not None:
        assert len(content) >= min_length, f"response too short: {len(content)} < {min_length}"


def send_image_request(config: SingleNodeConfig, server) -> dict[str, Any]:
    mm = config.mm_request or {}

    default_path = mm.get("image_path")
    image_data_list: list[str] = []

    if _messages_need_preload(mm):
        images = mm.get("images", [])
        if images:
            image_data_list = _load_images(images, default_path)
        elif default_path:
            image_data_list = [_load_image_data(default_path)]
        else:
            image_data_list = [_load_image_data()]

    messages = _build_messages(mm, image_data_list)
    api_args = {**DEFAULT_IMAGE_REQUEST_ARGS, **(mm.get("api_args") or {})}

    data: dict[str, Any] = {"model": config.model, "messages": messages}
    data.update(api_args)

    url = server.url_for("v1", "chat", "completions")
    response = requests.post(url, headers=_HEADERS, json=data)
    print("Status Code:", response.status_code)
    response_json = response.json()
    print("Response:", response_json)

    _assert_response(response.status_code, response_json, config.expected_response or {})
    return response_json
