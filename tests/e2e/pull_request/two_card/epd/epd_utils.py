# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0

"""Reusable helpers for the two-card EPD end-to-end tests.

The helpers only compose the reusable RemoteEPDServer / DisaggEpdProxy process
helpers from tests.e2e.conftest: every test starts real vllm serve and proxy
processes, and no product code is mocked or patched here.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import requests
from vllm.assets.image import ImageAsset
from vllm.multimodal.utils import encode_image_url
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import DisaggEpdProxy, RemoteEPDServer, RemoteOpenAIServer

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_MODEL_LEN = 10000
DEFAULT_MAX_TOKENS = 16
REQUEST_TIMEOUT_SECONDS = 600
EC_WAIT_SECONDS = 180.0
EC_FILE_GLOB = "*.safetensors"


def chat_completions_url(port: int) -> str:
    """Return the OpenAI chat-completions URL of a locally started instance."""
    return f"http://127.0.0.1:{port}/v1/chat/completions"


def health_url(port: int) -> str:
    """Return the /health URL of a locally started instance."""
    return f"http://127.0.0.1:{port}/health"


def unique_ec_path(name: str) -> str:
    """Return a unique EC shared-storage path so parallel tests never collide."""
    return f"/dev/shm/epd_n6a_{name}_{uuid.uuid4().hex[:8]}"


def clean_shared_storage(shared_storage_path: str) -> None:
    """Remove stale EC cache files left behind by an interrupted previous run."""
    shutil.rmtree(shared_storage_path, ignore_errors=True)


def snapshot_ec_files(shared_storage_path: str) -> set[Path]:
    """Return the EC cache files currently stored under shared_storage_path."""
    root = Path(shared_storage_path)
    if not root.is_dir():
        return set()
    return set(root.glob(f"**/{EC_FILE_GLOB}"))


def count_ec_files(shared_storage_path: str) -> int:
    """Return the number of EC cache files currently stored on disk."""
    return len(snapshot_ec_files(shared_storage_path))


def assert_ec_files(
    shared_storage_path: str,
    before: set[Path],
    after: set[Path],
    expected_new: int | None = None,
) -> set[Path]:
    """Assert that encoding produced new EC cache files and return them."""
    new_files = set(after) - set(before)
    assert new_files, (
        f"no new EC cache files were written under {shared_storage_path}: "
        f"before={sorted(before)}, after={sorted(after)}"
    )
    if expected_new is not None:
        assert len(new_files) == expected_new, (
            f"expected {expected_new} new EC cache files under {shared_storage_path}, "
            f"got {len(new_files)}: {sorted(new_files)}"
        )
    return new_files


def wait_until(
    predicate: Callable[[], bool],
    timeout: float,
    interval: float = 0.2,
    description: str = "condition",
) -> None:
    """Poll predicate until it is true or fail after timeout seconds."""
    deadline = time.monotonic() + timeout
    while True:
        if predicate():
            return
        if time.monotonic() >= deadline:
            pytest.fail(f"timed out after {timeout}s waiting for {description}")
        time.sleep(interval)


def wait_for_ec_files(
    shared_storage_path: str,
    before: set[Path],
    expected_new: int,
    timeout: float = EC_WAIT_SECONDS,
) -> set[Path]:
    """Wait until expected_new EC cache files appeared, then return them."""
    wait_until(
        lambda: len(snapshot_ec_files(shared_storage_path) - set(before)) >= expected_new,
        timeout=timeout,
        description=f"{expected_new} new EC cache files under {shared_storage_path}",
    )
    return snapshot_ec_files(shared_storage_path)


def text_payload(text: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> dict[str, Any]:
    """Build a deterministic text-only OpenAI chat-completions payload."""
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": [{"type": "text", "text": text}]}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 42,
    }


def image_payload(
    image_urls_list: list[str],
    text: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    """Build a deterministic image+text OpenAI chat-completions payload."""
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": image_url}} for image_url in image_urls_list
    ]
    content.append({"type": "text", "text": text})
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 42,
    }


def image_urls() -> list[str]:
    """Return four distinct image data URLs derived from repo-local assets.

    Only stop_sign and cherry_blossom are used by the existing EPD tests, so
    the four variants are produced with deterministic PIL transforms instead
    of relying on additional ImageAsset names. Rotated variants have different
    pixels, hence different mm_hash values.
    """
    stop_sign = ImageAsset("stop_sign").pil_image.convert("RGB").resize((448, 448))
    cherry_blossom = ImageAsset("cherry_blossom").pil_image.convert("RGB").resize((448, 448))
    return [
        encode_image_url(stop_sign),
        encode_image_url(stop_sign.rotate(90)),
        encode_image_url(cherry_blossom),
        encode_image_url(cherry_blossom.rotate(180)),
    ]


def baseline_server_args(port: int, mm_limit: int = 2) -> list[str]:
    """Return the CLI args of the non-EPD single-instance baseline server."""
    return [
        "--port",
        str(port),
        "--gpu-memory-utilization",
        "0.9",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-num-batched-tokens",
        str(MAX_MODEL_LEN),
        "--max-num-seqs",
        "128",
        "--limit-mm-per-prompt",
        json.dumps({"image": mm_limit, "video": 0}),
    ]


def _ec_transfer_config(ec_role: str, shared_storage_path: str) -> str:
    return json.dumps(
        {
            "ec_connector": "ECExampleConnector",
            "ec_role": ec_role,
            "ec_connector_extra_config": {"shared_storage_path": shared_storage_path},
        }
    )


def encoder_server_args(
    model: str,
    port: int,
    shared_storage_path: str,
    mm_limit: int = 2,
) -> list[str]:
    """Return the CLI args of the E (mm_encoder_only, EC producer) instance."""
    return [
        "--port",
        str(port),
        "--model",
        model,
        "--gpu-memory-utilization",
        "0.01",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-num-batched-tokens",
        str(MAX_MODEL_LEN),
        "--max-num-seqs",
        "1",
        "--limit-mm-per-prompt",
        json.dumps({"image": mm_limit, "video": 0}),
        "--ec-transfer-config",
        _ec_transfer_config("ec_producer", shared_storage_path),
    ]


def pd_server_args(
    model: str,
    port: int,
    shared_storage_path: str,
    mm_limit: int = 2,
) -> list[str]:
    """Return the CLI args of the PD (generative, EC consumer) instance."""
    return [
        "--port",
        str(port),
        "--model",
        model,
        "--gpu-memory-utilization",
        "0.95",
        "--enforce-eager",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-num-batched-tokens",
        str(MAX_MODEL_LEN),
        "--max-num-seqs",
        "128",
        "--limit-mm-per-prompt",
        json.dumps({"image": mm_limit, "video": 0}),
        "--ec-transfer-config",
        _ec_transfer_config("ec_consumer", shared_storage_path),
    ]


def reference_proxy_args(encode_port: int, decode_port: int, proxy_port: int) -> list[str]:
    """Return the CLI args of the reference EPD proxy for the E + PD topology."""
    return [
        "--host",
        "127.0.0.1",
        "--port",
        str(proxy_port),
        "--encode-servers-urls",
        f"http://127.0.0.1:{encode_port}",
        "--decode-servers-urls",
        f"http://127.0.0.1:{decode_port}",
        "--prefill-servers-urls",
        "disable",
    ]


def post_chat_request(
    url: str,
    payload: dict[str, Any],
    request_id: str | None = None,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> requests.Response:
    """POST one chat-completions payload without raising on HTTP errors."""
    headers = {"x-request-id": request_id} if request_id else None
    return requests.post(url, json=payload, headers=headers, timeout=timeout)


def post_chat(
    url: str,
    payload: dict[str, Any],
    request_id: str | None = None,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> str:
    """POST one chat-completions payload and return the generated text."""
    response = post_chat_request(url, payload, request_id=request_id, timeout=timeout)
    body = response.json()
    choices = body.get("choices") or []
    assert choices, f"no choices in response from {url}: status={response.status_code} body={body!r}"
    content = (choices[0].get("message") or {}).get("content") or ""
    assert content, f"empty completion from {url}: status={response.status_code} body={body!r}"
    return content


def assert_outputs_equal(actual: str, expected: str, context: str = "") -> None:
    """Assert the EPD output matches the non-EPD baseline verbatim."""
    label = f"{context}: " if context else ""
    assert actual == expected, (
        f"{label}EPD output differs from the non-EPD baseline.\n"
        f"baseline ({len(expected)} chars): {expected!r}\n"
        f"epd ({len(actual)} chars): {actual!r}"
    )


def collect_baseline(
    payloads: list[dict[str, Any]],
    model: str = MODEL,
    env_dict: dict[str, str] | None = None,
) -> list[str]:
    """Serve the model on a single instance and collect the non-EPD outputs."""
    port = get_open_port()
    env = {"ASCEND_RT_VISIBLE_DEVICES": "0"}
    if env_dict is not None:
        env.update(env_dict)
    with RemoteOpenAIServer(
        model,
        baseline_server_args(port),
        server_port=port,
        auto_port=False,
        env_dict=env,
    ):
        url = chat_completions_url(port)
        return [post_chat(url, payload) for payload in payloads]


@dataclass(frozen=True)
class EpdCluster:
    """Handles and ports of a running E + PD deployment behind the proxy."""

    servers: RemoteEPDServer
    proxy: DisaggEpdProxy
    encode_port: int
    decode_port: int
    proxy_port: int
    shared_storage_path: str

    @property
    def proxy_chat_url(self) -> str:
        return chat_completions_url(self.proxy_port)

    @property
    def encoder_chat_url(self) -> str:
        return chat_completions_url(self.encode_port)

    @property
    def pd_chat_url(self) -> str:
        return chat_completions_url(self.decode_port)


@contextmanager
def epd_pair(
    shared_storage_path: str,
    mm_limit: int = 2,
    model: str = MODEL,
    env_dict: dict[str, str] | None = None,
) -> Iterator[EpdCluster]:
    """Start E + PD instances and the reference proxy; always clean them up."""
    clean_shared_storage(shared_storage_path)
    encode_port = get_open_port()
    decode_port = get_open_port()
    proxy_port = get_open_port()
    server_args = [
        encoder_server_args(model, encode_port, shared_storage_path, mm_limit=mm_limit),
        pd_server_args(model, decode_port, shared_storage_path, mm_limit=mm_limit),
    ]
    with (
        RemoteEPDServer(vllm_serve_args=server_args, env_dict=env_dict) as servers,
        DisaggEpdProxy(
            proxy_args=reference_proxy_args(encode_port, decode_port, proxy_port),
            env_dict=env_dict,
        ) as proxy,
    ):
        yield EpdCluster(
            servers=servers,
            proxy=proxy,
            encode_port=encode_port,
            decode_port=decode_port,
            proxy_port=proxy_port,
            shared_storage_path=shared_storage_path,
        )
