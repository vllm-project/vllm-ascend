# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

from concurrent.futures import ThreadPoolExecutor
from itertools import cycle, islice
from typing import Any

import pytest
import requests

from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.weekly.single_node.features.structured_output.api import (
    create_plain_chat_completion,
    create_structured_chat_completion,
    raw_chat_completion_payload,
)
from tests.e2e.weekly.single_node.features.structured_output.assertions import (
    assert_plain_output,
    assert_structured_output,
)
from tests.e2e.weekly.single_node.features.structured_output.cases import (
    INVALID_STRUCTURED_OUTPUTS,
    SERVED_MODEL_NAME,
    STRUCTURED_OUTPUT_CASES,
    StructuredOutputCase,
)

CONCURRENT_REQUEST_COUNT = 32
CONCURRENT_WORKERS = 8


@pytest.mark.parametrize("case", STRUCTURED_OUTPUT_CASES, ids=lambda case: case.case_id)
def test_openai_structured_output(openai_client: Any, case: StructuredOutputCase) -> None:
    text = create_structured_chat_completion(openai_client, SERVED_MODEL_NAME, case)

    assert_structured_output(text, case)


@pytest.mark.parametrize("case", STRUCTURED_OUTPUT_CASES, ids=lambda case: case.case_id)
def test_openai_streaming_structured_output(openai_client: Any, case: StructuredOutputCase) -> None:
    text = create_structured_chat_completion(openai_client, SERVED_MODEL_NAME, case, stream=True)

    assert_structured_output(text, case)


def test_openai_mixed_concurrent_structured_outputs(openai_client: Any) -> None:
    request_cases = list(islice(cycle((*STRUCTURED_OUTPUT_CASES, None)), CONCURRENT_REQUEST_COUNT))

    def run_request(case: StructuredOutputCase | None) -> tuple[StructuredOutputCase | None, str]:
        if case is None:
            return None, create_plain_chat_completion(openai_client, SERVED_MODEL_NAME)
        return case, create_structured_chat_completion(openai_client, SERVED_MODEL_NAME, case)

    with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
        results = list(executor.map(run_request, request_cases))

    assert len(results) == CONCURRENT_REQUEST_COUNT
    for case, text in results:
        if case is None:
            assert_plain_output(text)
        else:
            assert_structured_output(text, case)


@pytest.mark.parametrize(
    "case_id,structured_outputs",
    INVALID_STRUCTURED_OUTPUTS,
    ids=[case_id for case_id, _ in INVALID_STRUCTURED_OUTPUTS],
)
def test_invalid_request_does_not_break_following_requests(
    openai_server: RemoteOpenAIServer,
    openai_client: Any,
    case_id: str,
    structured_outputs: dict[str, Any],
) -> None:
    response = requests.post(
        openai_server.url_for("v1", "chat", "completions"),
        headers={"Authorization": f"Bearer {openai_server.DUMMY_API_KEY}"},
        json=raw_chat_completion_payload(SERVED_MODEL_NAME, structured_outputs),
        timeout=120,
    )

    valid_case = STRUCTURED_OUTPUT_CASES[0]
    text = create_structured_chat_completion(openai_client, SERVED_MODEL_NAME, valid_case)
    assert_structured_output(text, valid_case)

    assert response.status_code in {
        400,
        422,
    }, f"{case_id} unexpectedly returned {response.status_code}: {response.text}"
