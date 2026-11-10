# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0

"""Two-card E2E: the EPD encoder-only (E) node serves independently."""

from __future__ import annotations

from pathlib import Path

import pytest
import requests
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteEPDServer, wait_until_npu_memory_free
from tests.e2e.pull_request.two_card.epd.epd_utils import (
    MODEL,
    REQUEST_TIMEOUT_SECONDS,
    assert_ec_files,
    chat_completions_url,
    clean_shared_storage,
    encoder_server_args,
    health_url,
    image_payload,
    image_urls,
    snapshot_ec_files,
    text_payload,
    unique_ec_path,
    wait_for_ec_files,
)


def _assert_health(port: int) -> None:
    response = requests.get(health_url(port), timeout=30)
    assert response.status_code == 200, f"encoder /health returned {response.status_code}: {response.text}"


def _post(port: int, payload: dict) -> requests.Response:
    return requests.post(chat_completions_url(port), json=payload, timeout=REQUEST_TIMEOUT_SECONDS)


@pytest.mark.e2e_model("Qwen/Qwen2.5-VL-7B-Instruct")
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
def test_encoder_only_node_serves_independently() -> None:
    shared_storage_path = unique_ec_path("encoder_only")
    clean_shared_storage(shared_storage_path)
    encode_port = get_open_port()
    with RemoteEPDServer(
        vllm_serve_args=[encoder_server_args(MODEL, encode_port, shared_storage_path)],
    ) as server:
        _assert_health(encode_port)

        text_response = _post(encode_port, text_payload("Say hi.", max_tokens=1))
        assert text_response.status_code == 200, (
            f"text-only encoder request returned {text_response.status_code}: {text_response.text}"
        )

        before = snapshot_ec_files(shared_storage_path)
        image_response = _post(
            encode_port,
            image_payload([image_urls()[0]], "Describe this image.", max_tokens=1),
        )
        assert image_response.status_code == 200, (
            f"image encoder request returned {image_response.status_code}: {image_response.text}"
        )
        after = wait_for_ec_files(shared_storage_path, before, expected_new=1)
        assert_ec_files(shared_storage_path, before, after, expected_new=1)

    assert server.exit_codes == [0], f"encoder exited with {server.exit_codes}, expected a graceful 0"
    assert not Path(shared_storage_path).exists(), (
        f"EC shared storage was not cleaned up after shutdown: {shared_storage_path}"
    )


@pytest.mark.e2e_model("Qwen/Qwen2.5-VL-7B-Instruct")
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
def test_encoder_only_text_only_request_creates_no_ec() -> None:
    shared_storage_path = unique_ec_path("encoder_only_text")
    clean_shared_storage(shared_storage_path)
    encode_port = get_open_port()
    with RemoteEPDServer(
        vllm_serve_args=[encoder_server_args(MODEL, encode_port, shared_storage_path)],
    ):
        _assert_health(encode_port)
        before = snapshot_ec_files(shared_storage_path)
        for index in range(2):
            response = _post(encode_port, text_payload(f"Hello {index}.", max_tokens=1))
            assert response.status_code == 200, (
                f"text-only encoder request returned {response.status_code}: {response.text}"
            )
        after = snapshot_ec_files(shared_storage_path)
        assert after == before, f"text-only requests created EC cache files: {sorted(after - before)}"
