# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""Four-card regression for cross-Encoder Mooncake Store transfers."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import socket
import subprocess
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pybase64 as base64
import pytest
import requests
import torch
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
CONSUMER_BUFFER_BYTES = 256 * 1024 * 1024
BOUNCE_ARENA_BYTES = 2 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 600
STORE_LOG_TIMEOUT_SECONDS = 30
STORE_MASTER_START_TIMEOUT_SECONDS = 30
STORE_MASTER_STOP_TIMEOUT_SECONDS = 10
STORE_PUT_LOG = "Stored encoder output in Mooncake Store"
STORE_HIT_LOG = "encoder output(s) from Mooncake Store"
STORE_GLOBAL_SEGMENT_SIZE = "1GB"
STORE_LOCAL_BUFFER_SIZE = "512MB"


class _CapturingEPDServer(RemoteEPDServer):
    """Keep child logs so the Store test can prove that a GET was used."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.output_lines: list[str] = []
        super().__init__(*args, **kwargs)

    def _read_output(self, pipe: Any, prefix: str) -> None:
        with pipe:
            for line in iter(pipe.readline, ""):
                if line:
                    rendered = f"{prefix}: {line}"
                    self.output_lines.append(rendered)
                    print(rendered, end="")

    def wait_for_output(self, marker: str, process_prefix: str | None = None) -> bool:
        deadline = time.monotonic() + STORE_LOG_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if any(
                marker in line
                and (process_prefix is None or process_prefix in line)
                for line in self.output_lines
            ):
                return True
            time.sleep(0.1)
        return False


def _wait_for_mooncake_master(
    process: subprocess.Popen[Any],
    port: int,
    log_path: Path,
) -> None:
    deadline = time.monotonic() + STORE_MASTER_START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(
                "mooncake_master exited during startup:\n"
                + log_path.read_text(errors="replace")
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.1)
    pytest.fail(
        "timed out waiting for mooncake_master:\n"
        + log_path.read_text(errors="replace")
    )


@pytest.fixture(scope="module")
def mooncake_store_config(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    master = shutil.which("mooncake_master")
    if master is None:
        pytest.fail("mooncake_master is required for the Store E2E")

    work_dir = tmp_path_factory.mktemp("ec-mooncake-store")
    config_path = work_dir / "mooncake_store.json"
    log_path = work_dir / "mooncake_master.log"
    port = get_open_port()
    config_path.write_text(
        json.dumps(
            {
                "mode": "embedded",
                "metadata_server": "P2PHANDSHAKE",
                "master_server_address": f"127.0.0.1:{port}",
                "global_segment_size": STORE_GLOBAL_SEGMENT_SIZE,
                "local_buffer_size": STORE_LOCAL_BUFFER_SIZE,
                "protocol": "ascend",
                "device_name": "",
                "enable_offload": False,
            }
        )
    )

    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            [master, "--port", str(port)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_for_mooncake_master(process, port, log_path)
            yield str(config_path)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=STORE_MASTER_STOP_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


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
                        "text": ("Identify the traffic sign. Reply with exactly one lowercase word: stop or yield."),
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
                        "text": (
                            "For the first image choose stop or yield; for the "
                            "second choose flowers or building. Reply with exactly "
                            "two lowercase words separated by a comma."
                        ),
                    },
                ],
            }
        ],
    }


def _request_body(messages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 16,
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


def _encode_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    """Encode placeholder metadata using the image_embeds wire contract."""
    encoded = {}
    for key, values in metadata.items():
        buffer = io.BytesIO()
        tensor = torch.tensor(values, dtype=torch.long)
        # Qwen's placeholder parser expects one flat (t, h, w) grid per item.
        torch.save(tensor.reshape(-1)[:3], buffer)
        encoded[key] = base64.b64encode(buffer.getvalue()).decode()
    return encoded


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
    ) -> tuple[int, dict[str, Any], str, dict[str, str]]:
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
        assert len(params) == 1, f"encoder returned unexpected EC items for image {index}: {params!r}"
        ec_mm_hash, reported = next(iter(params.items()))
        assert ec_mm_hash, f"encoder returned an empty mm_hash for image {index}"
        assert isinstance(reported, dict), f"encoder returned a non-dict EC item for image {index}: {reported!r}"
        reported_transfer_id = reported.get("transfer_id")
        assert reported_transfer_id in (None, transfer_id), (
            f"encoder changed transfer_id for image {index}: expected {transfer_id!r}, got {reported_transfer_id!r}"
        )
        metadata = reported.get("metadata") or {}
        assert isinstance(metadata, dict), f"encoder returned non-dict metadata for image {index}: {metadata!r}"
        assert metadata, f"encoder returned no metadata for image {index}"
        item_meta = {
            "uuid": item_uuid,
            "metadata": _encode_metadata(metadata),
        }
        return (
            index,
            item_meta,
            ec_mm_hash,
            {"mm_hash": ec_mm_hash, "transfer_id": transfer_id},
        )

    with ThreadPoolExecutor(max_workers=len(images)) as executor:
        encoded = list(executor.map(encode, enumerate(images)))

    item_meta = {index: meta for index, meta, _, _ in encoded}
    transfer_items = [transfer_item for _, _, _, transfer_item in encoded]

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
        "--mm-processor-device",
        "cpu",
    ]


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


def _producer_config(cache_prefix: str) -> dict[str, Any]:
    return {
        "ec_connector": "ECMooncakeConnector",
        "ec_role": "ec_producer",
        "ec_buffer_size": NORMAL_STAGING_BYTES,
        "ec_buffer_device": "npu",
        "ec_connector_extra_config": {
            "mooncake_protocol": "ascend",
            "ascend_mooncake_bounce_arena_size": BOUNCE_ARENA_BYTES,
            "cross_encoder_cache": True,
            "embedding_cache_prefix": cache_prefix,
        },
    }


def _consumer_config(reservation_port: int) -> dict[str, Any]:
    return {
        "ec_connector": "ECMooncakeConnector",
        "ec_role": "ec_consumer",
        "ec_ip": "127.0.0.1",
        "ec_port": reservation_port,
        "ec_buffer_size": CONSUMER_BUFFER_BYTES,
        "ec_buffer_device": "npu",
        "ec_connector_extra_config": {"mooncake_protocol": "ascend"},
    }


def _producer_args(port: int, cache_prefix: str) -> list[str]:
    return [
        "--port",
        str(port),
        *_common_server_args(),
        "--gpu-memory-utilization",
        "0.1",
        "--enable-request-id-headers",
        "--ec-transfer-config",
        json.dumps(_producer_config(cache_prefix)),
    ]


def _consumer_args(port: int, reservation_port: int) -> list[str]:
    return [
        "--port",
        str(port),
        *_common_server_args(),
        "--gpu-memory-utilization",
        "0.7",
        "--enable-mm-embeds",
        "--enable-request-id-headers",
        "--ec-transfer-config",
        json.dumps(_consumer_config(reservation_port)),
    ]


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
def test_cross_encoder_store_put_then_remote_hit(
    baseline_outputs: dict[str, str],
    mooncake_store_config: str,
) -> None:
    """E0 publishes outputs that E1 reuses through the shared Store."""
    encoder0_port = get_open_port()
    encoder1_port = get_open_port()
    pd0_port = get_open_port()
    pd1_port = get_open_port()
    pd0_reservation_port = get_open_port()
    pd1_reservation_port = get_open_port()
    cache_prefix = f"cross-encoder-e2e-{uuid.uuid4().hex}"

    server_args = [
        _producer_args(encoder0_port, cache_prefix),
        _producer_args(encoder1_port, cache_prefix),
        _consumer_args(pd0_port, pd0_reservation_port),
        _consumer_args(pd1_port, pd1_reservation_port),
    ]
    env_dict = {
        "ASCEND_ENABLE_USE_FABRIC_MEM": "1",
        "MOONCAKE_CONFIG_PATH": mooncake_store_config,
        "MOONCAKE_EC_PROTOCOL": "ascend",
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
    }

    with _CapturingEPDServer(
        vllm_serve_args=server_args,
        env_dict=env_dict,
    ) as server:
        first_outputs = _run_epd_requests(
            encoder0_port,
            pd0_port,
            pd0_reservation_port,
        )
        assert first_outputs == baseline_outputs
        if not server.wait_for_output(STORE_PUT_LOG, "[VLLM_0]"):
            pytest.fail("E0 never completed its Mooncake Store PUT")

        # E1 has not encoded these images. Clearing it makes that invariant
        # explicit and prevents a future eager/local population change from
        # silently weakening this regression.
        response = requests.post(
            f"http://127.0.0.1:{encoder1_port}/reset_encoder_cache",
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()

        second_outputs = _run_epd_requests(
            encoder1_port,
            pd1_port,
            pd1_reservation_port,
        )
        assert second_outputs == baseline_outputs
        if not server.wait_for_output(STORE_HIT_LOG, "[VLLM_1]"):
            pytest.fail("E1 never loaded E0's encoder output from Mooncake Store")


