# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""Two-card service E2E for Ascend Mooncake encoder-output transfer."""

from __future__ import annotations

import hashlib
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
import requests
from vllm.assets.image import ImageAsset
from vllm.multimodal.utils import encode_image_url
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import (
    RemoteEPDServer,
    RemoteOpenAIServer,
    wait_until_npu_memory_free,
)

MODEL = "Qwen/Qwen3.5-9B"
MAX_MODEL_LEN = 4096
MAX_NUM_SEQS = 2
NORMAL_STAGING_BYTES = 128 * 1024 * 1024
FORCED_FALLBACK_STAGING_BYTES = 1
CONSUMER_BUFFER_BYTES = 256 * 1024 * 1024
BOUNCE_ARENA_BYTES = 2 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 600


def _messages() -> dict[str, list[dict[str, Any]]]:
    stop_sign = ImageAsset("stop_sign").pil_image.resize((448, 448))
    cherry_blossom = ImageAsset("cherry_blossom").pil_image.resize((448, 448))
    stop_sign_url = encode_image_url(stop_sign)
    cherry_blossom_url = encode_image_url(cherry_blossom)
    return {
        "single image": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": stop_sign_url}},
                    {
                        "type": "text",
                        "text": "Describe the image in one short sentence.",
                    },
                ],
            }
        ],
        "two images": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": stop_sign_url}},
                    {
                        "type": "image_url",
                        "image_url": {"url": cherry_blossom_url},
                    },
                    {
                        "type": "text",
                        "text": "Describe both images briefly, in image order.",
                    },
                ],
            }
        ],
    }


def _request_body(messages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 64,
        "temperature": 0.0,
        "seed": 42,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _post_chat(url: str, body: dict[str, Any], request_id: str) -> str:
    response = requests.post(
        url,
        json=body,
        headers={"x-request-id": request_id},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    assert content, f"request {request_id} returned an empty completion"
    return content


def _run_requests(server: Any) -> dict[str, str]:
    cases = _messages()

    def complete(item: tuple[str, list[dict[str, Any]]]) -> tuple[str, str]:
        name, messages = item
        return name, _post_chat(
            server.url_for("v1", "chat", "completions"),
            _request_body(messages),
            uuid.uuid4().hex,
        )

    with ThreadPoolExecutor(max_workers=len(cases)) as executor:
        return dict(executor.map(complete, cases.items()))


def _content_uuid(item: dict[str, Any]) -> str:
    url = (item.get("image_url") or {}).get("url") or ""
    payload = url or json.dumps(item, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _flatten_metadata(metadata: dict[str, Any]) -> dict[str, list[Any]]:
    # This is the metadata-only image_embeds representation consumed by vLLM.
    # Keep it aligned with the upstream disaggregated-encoder proxy protocol.
    return {
        key: [value for item in values for value in (item if isinstance(item, list) else [item])]
        for key, values in metadata.items()
    }


def _prepare_decode_body(
    body: dict[str, Any],
    encode_url: str,
    consumer_zmq: str,
    request_id: str,
) -> dict[str, Any]:
    """Run the upstream E+PD control-plane exchange without a proxy process."""
    images = [
        item for message in body["messages"] for item in message.get("content", []) if item.get("type") == "image_url"
    ]

    def encode(
        item_with_index: tuple[int, dict[str, Any]],
    ) -> tuple[int, dict[str, Any], str, Any, dict[str, str]]:
        index, item = item_with_index
        item_uuid = _content_uuid(item)
        transfer_id = uuid.uuid4().hex
        producer_body = {
            "model": body["model"],
            "messages": [
                {
                    "role": "user",
                    "content": [{**item, "uuid": item_uuid}],
                }
            ],
            "stream": False,
            "ec_transfer_params": {
                "consumer_zmq": consumer_zmq,
                "ec_items": [{"transfer_id": transfer_id}],
            },
        }
        response = requests.post(
            encode_url,
            json=producer_body,
            headers={"x-request-id": f"{request_id}:{index}"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        params = response.json().get("ec_transfer_params") or {}
        ec_mm_hash = item_uuid
        reported = params.get(ec_mm_hash)
        if reported is None and len(params) == 1:
            ((ec_mm_hash, reported),) = params.items()
        assert reported, f"encoder returned no transfer handle for image {index}"
        metadata = reported.get("metadata") or {}
        assert metadata, f"encoder returned no metadata for image {index}"
        item_meta = {
            "uuid": item_uuid,
            "metadata": _flatten_metadata(metadata),
        }
        return (
            index,
            item_meta,
            ec_mm_hash,
            reported,
            {"mm_hash": ec_mm_hash, "transfer_id": transfer_id},
        )

    with ThreadPoolExecutor(max_workers=len(images)) as executor:
        encoded = list(executor.map(encode, enumerate(images)))

    item_meta = {index: meta for index, meta, _, _, _ in encoded}
    ec_handles = {ec_mm_hash: reported for _, _, ec_mm_hash, reported, _ in encoded}
    transfer_items = [transfer_item for _, _, _, _, transfer_item in encoded]

    image_index = 0
    rewritten_messages = []
    for message in body["messages"]:
        content = []
        for item in message.get("content", []):
            if item.get("type") != "image_url":
                content.append(item)
                continue
            meta = item_meta[image_index]
            content.append(
                {
                    "type": "image_embeds",
                    "image_embeds": meta["metadata"],
                    "uuid": meta["uuid"],
                }
            )
            image_index += 1
        rewritten_messages.append({**message, "content": content})

    return {
        **body,
        "messages": rewritten_messages,
        "ec_transfer_params": {
            "ec_items": transfer_items,
            **ec_handles,
        },
    }


def _run_epd_requests(
    encode_port: int,
    pd_port: int,
    reservation_port: int,
) -> dict[str, str]:
    cases = _messages()
    encode_url = f"http://127.0.0.1:{encode_port}/v1/chat/completions"
    decode_url = f"http://127.0.0.1:{pd_port}/v1/chat/completions"
    consumer_zmq = f"tcp://127.0.0.1:{reservation_port}"

    def complete(item: tuple[str, list[dict[str, Any]]]) -> tuple[str, str]:
        name, messages = item
        request_id = uuid.uuid4().hex
        body = _prepare_decode_body(_request_body(messages), encode_url, consumer_zmq, request_id)
        return name, _post_chat(decode_url, body, request_id)

    with ThreadPoolExecutor(max_workers=len(cases)) as executor:
        return dict(executor.map(complete, cases.items()))


def _common_server_args() -> list[str]:
    return [
        "--model",
        MODEL,
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-num-batched-tokens",
        str(MAX_MODEL_LEN),
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--limit-mm-per-prompt",
        json.dumps({"image": 2, "video": 0}),
        "--mm-tensor-ipc",
        "torch_shm",
    ]


def _run_epd(
    baseline_outputs: dict[str, str],
    producer_staging_bytes: int,
    repeat: int,
) -> None:
    encode_port = get_open_port()
    pd_port = get_open_port()
    reservation_port = get_open_port()

    producer_config = {
        "ec_connector": "ECMooncakeConnector",
        "ec_role": "ec_producer",
        "ec_buffer_size": producer_staging_bytes,
        "ec_buffer_device": "npu",
        "ec_connector_extra_config": {
            "mooncake_protocol": "ascend",
            "ascend_mooncake_bounce_arena_size": BOUNCE_ARENA_BYTES,
        },
    }
    consumer_config = {
        "ec_connector": "ECMooncakeConnector",
        "ec_role": "ec_consumer",
        "ec_ip": "127.0.0.1",
        "ec_port": reservation_port,
        "ec_buffer_size": CONSUMER_BUFFER_BYTES,
        "ec_buffer_device": "npu",
        "ec_connector_extra_config": {"mooncake_protocol": "ascend"},
    }
    common_args = _common_server_args()
    server_args = [
        [
            "--port",
            str(encode_port),
            *common_args,
            "--gpu-memory-utilization",
            "0.1",
            "--enable-request-id-headers",
            "--ec-transfer-config",
            json.dumps(producer_config),
        ],
        [
            "--port",
            str(pd_port),
            *common_args,
            "--gpu-memory-utilization",
            "0.7",
            "--enable-mm-embeds",
            "--enable-request-id-headers",
            "--ec-transfer-config",
            json.dumps(consumer_config),
        ],
    ]
    env_dict = {
        "EC_MOONCAKE_RESERVATION_PORT": str(reservation_port),
        "MOONCAKE_EC_PROTOCOL": "ascend",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
    }

    with RemoteEPDServer(vllm_serve_args=server_args, env_dict=env_dict):
        for _ in range(repeat):
            assert _run_epd_requests(encode_port, pd_port, reservation_port) == baseline_outputs


@pytest.fixture(scope="module")
def baseline_outputs() -> dict[str, str]:
    """Run the single-server reference once for both EPD paths."""
    baseline_port = get_open_port()
    baseline_args = [
        "--port",
        str(baseline_port),
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.7",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--no-enable-prefix-caching",
        "--limit-mm-per-prompt",
        json.dumps({"image": 2, "video": 0}),
    ]
    with RemoteOpenAIServer(
        MODEL,
        baseline_args,
        server_port=baseline_port,
        auto_port=False,
        env_dict={
            "ASCEND_RT_VISIBLE_DEVICES": "0",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
        },
    ) as baseline_server:
        return _run_requests(baseline_server)


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="multimodal",
    feature="",
    parallel="",
    deploy="epd",
    hardware="A3",
    quantization="BF16",
    graph_mode="eager",
)
@wait_until_npu_memory_free()
def test_mooncake_staging_matches_single_server(
    baseline_outputs: dict[str, str],
) -> None:
    """Validate the production-shaped staging-first transfer path."""
    _run_epd(baseline_outputs, NORMAL_STAGING_BYTES, repeat=1)


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="multimodal",
    feature="",
    parallel="",
    deploy="epd",
    hardware="A3",
    quantization="BF16",
    graph_mode="eager",
)
@wait_until_npu_memory_free()
def test_mooncake_forced_fallback_matches_single_server(
    baseline_outputs: dict[str, str],
) -> None:
    """Validate direct/bounce fallback and its resource reuse."""
    # One byte is deliberate fault injection used only by this case. It makes
    # stage() reject every non-empty encoder output and proves the insurance
    # path without depending on a particular model's encoder-output size.
    _run_epd(baseline_outputs, FORCED_FALLBACK_STAGING_BYTES, repeat=2)
