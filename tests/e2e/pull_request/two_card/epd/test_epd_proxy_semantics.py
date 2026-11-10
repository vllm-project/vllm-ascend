# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0

"""Two-card E2E: reference EPD proxy semantics against a non-EPD baseline."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.e2e.conftest import wait_until_npu_memory_free
from tests.e2e.pull_request.two_card.epd.epd_utils import (
    assert_ec_files,
    assert_outputs_equal,
    collect_baseline,
    epd_pair,
    image_payload,
    image_urls,
    post_chat,
    post_chat_request,
    snapshot_ec_files,
    text_payload,
    unique_ec_path,
    wait_for_ec_files,
)


def _assert_4xx(response, description: str) -> None:
    assert 400 <= response.status_code < 500, (
        f"{description} returned {response.status_code}, expected a deterministic 4xx: {response.text}"
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
def test_text_only_matches_baseline() -> None:
    payload = text_payload("What is the capital of France? Answer in one word.", max_tokens=8)
    baseline = collect_baseline([payload])[0]
    with epd_pair(unique_ec_path("proxy_text")) as cluster:
        actual = post_chat(cluster.proxy_chat_url, payload)
    assert_outputs_equal(actual, baseline, context="text-only")


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
def test_empty_mm_input_returns_4xx() -> None:
    urls = image_urls()
    with epd_pair(unique_ec_path("empty_mm")) as cluster:
        empty_image_response = post_chat_request(
            cluster.proxy_chat_url,
            image_payload([""], "Describe this image.", max_tokens=8),
        )
        _assert_4xx(empty_image_response, "empty image_url request")

        empty_messages = text_payload("placeholder", max_tokens=8)
        empty_messages["messages"] = []
        empty_messages_response = post_chat_request(cluster.proxy_chat_url, empty_messages)
        _assert_4xx(empty_messages_response, "empty messages request")

        recovered = post_chat(
            cluster.proxy_chat_url,
            image_payload([urls[0]], "Describe this image.", max_tokens=8),
        )
        assert recovered


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
def test_overlong_images_rejected_then_recovers() -> None:
    urls = image_urls()
    with epd_pair(unique_ec_path("overlong"), mm_limit=1) as cluster:
        overlong_response = post_chat_request(
            cluster.proxy_chat_url,
            image_payload(urls[:2], "Compare these images.", max_tokens=8),
        )
        _assert_4xx(overlong_response, "two-image request with --limit-mm-per-prompt image=1")

        recovered = post_chat(
            cluster.proxy_chat_url,
            image_payload([urls[0]], "Describe this image.", max_tokens=8),
        )
        assert recovered


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
def test_concurrent_distinct_images_match_baseline() -> None:
    urls = image_urls()
    payloads = [image_payload([url], "Describe this image briefly.", max_tokens=16) for url in urls]
    baseline = collect_baseline(payloads)

    with epd_pair(unique_ec_path("concurrent")) as cluster:
        before = snapshot_ec_files(cluster.shared_storage_path)
        request_order = [0, 1, 2, 3, 0, 1, 2, 3]
        with ThreadPoolExecutor(max_workers=len(request_order)) as executor:
            actual = list(
                executor.map(
                    lambda image_index: post_chat(cluster.proxy_chat_url, payloads[image_index]),
                    request_order,
                )
            )

        for request_index, (image_index, content) in enumerate(zip(request_order, actual)):
            assert_outputs_equal(
                content,
                baseline[image_index],
                context=f"concurrent request {request_index} (image {image_index})",
            )

        after = wait_for_ec_files(
            cluster.shared_storage_path,
            before,
            expected_new=len(urls),
        )
        assert_ec_files(
            cluster.shared_storage_path,
            before,
            after,
            expected_new=len(urls),
        )
