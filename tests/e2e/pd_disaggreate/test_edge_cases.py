# SPDX-License-Identifier: Apache-2.0
# This code is from: https://github.com/vllm-project/vllm/blob/main/tests/v1/kv_connector/nixl_integration/test_edge_cases.py
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import os

import openai

PREFILL_PORT = os.getenv("PREFILL_PORT", None)
DECODE_PORT = os.getenv("DECODE_PORT", None)
PROXY_PORT = os.getenv("PROXY_PORT", None)

if PREFILL_PORT is None or DECODE_PORT is None or PROXY_PORT is None:
    raise ValueError(
        "Please set the PREFILL_PORT, DECODE_PORT, and PROXY_PORT.")

LONG_PROMPT = "Red Hat is the best company in the world to work for because it works on open source software, which means that all the contributions are delivered to the community. As a result, when working on projects like vLLM we are able to meet many amazing people from various organizations like AMD, Google, NVIDIA, "  # noqa: E501
PROMPT = "Red Hat is the best company in the world to work for because it works on open source software, which means that all the contributions are delivered to the community. As a result,"  # noqa: E501
SHORT_PROMPT = "Red Hat is "


def test_edge_cases():
    # Set the OpenAI API key and base URL
    decode_client = openai.OpenAI(
        api_key="MY_KEY",
        base_url=f"http://localhost:{DECODE_PORT}/v1",
    )
    prefill_client = openai.OpenAI(
        api_key="MY_KEY",
        base_url=f"http://localhost:{PREFILL_PORT}/v1",
    )
    proxy_client = openai.OpenAI(
        api_key="MY_KEY",
        base_url=f"http://localhost:{PROXY_PORT}/v1",
    )

    # Get the list of models
    models = decode_client.models.list()
    MODEL = models.data[0].id

    # (1) Check that we can handle a very short prompt,
    # less than the length of the block size.
    completion = proxy_client.completions.create(model=MODEL,
                                                 prompt=SHORT_PROMPT,
                                                 temperature=0)
    proxy_response = completion.choices[0].text
    print(f"SMALL PROMPT: {proxy_response=}")
    assert proxy_response is not None

    # (2) Check that we can handle a full prefix cache
    completion = proxy_client.completions.create(model=MODEL,
                                                 prompt=PROMPT,
                                                 temperature=0)
    proxy_response = completion.choices[0].text
    print(f"FULL CACHE HIT: {proxy_response=}")
    assert proxy_response is not None

    # (3) Check that we can handle a partial prefix cache
    completion = proxy_client.completions.create(model=MODEL,
                                                 prompt=LONG_PROMPT,
                                                 temperature=0)
    proxy_response = completion.choices[0].text
    print(f"PARTIAL CACHE HIT: {proxy_response=}")
    assert proxy_response is not None