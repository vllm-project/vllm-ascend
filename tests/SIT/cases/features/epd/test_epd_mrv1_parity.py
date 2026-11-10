# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0

"""Manual-only MRV1 vs MRV2 EPD output parity (Q5: not part of PR CI).

This file lives under tests/SIT/cases/features/epd/, not under
tests/e2e/pull_request/. test_selector.TEST_ROOTS and
select_tests._ALL_TESTS_ROOTS both start at tests/e2e/pull_request, and
coverage.py only scans tests/e2e/pull_request, so this file is excluded at the
directory level and never selected by PR jobs. Run it manually:

    VLLM_USE_MODELSCOPE=true pytest -sv \
        tests/SIT/cases/features/epd/test_epd_mrv1_parity.py
"""

from __future__ import annotations

import pytest

from tests.e2e.conftest import wait_until_npu_memory_free
from tests.e2e.pull_request.two_card.epd.epd_utils import (
    assert_outputs_equal,
    epd_pair,
    image_payload,
    image_urls,
    post_chat,
    unique_ec_path,
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
def test_mrv1_v2_outputs_match() -> None:
    payload = image_payload([image_urls()[0]], "Describe this image briefly.", max_tokens=16)

    with epd_pair(unique_ec_path("mrv1"), env_dict={"VLLM_USE_V2_MODEL_RUNNER": "0"}) as cluster:
        v1_output = post_chat(cluster.proxy_chat_url, payload)
    with epd_pair(unique_ec_path("mrv2"), env_dict={"VLLM_USE_V2_MODEL_RUNNER": "1"}) as cluster:
        v2_output = post_chat(cluster.proxy_chat_url, payload)

    assert_outputs_equal(v2_output, v1_output, context="MRV1 vs MRV2 EPD")
