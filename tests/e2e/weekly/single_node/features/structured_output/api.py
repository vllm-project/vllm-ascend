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

from typing import Any

from tests.e2e.weekly.single_node.features.structured_output.cases import StructuredOutputCase


def structured_output_request_kwargs(case: StructuredOutputCase) -> dict[str, Any]:
    if case.constraint_type == "json":
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": case.case_id.replace("-", "_"),
                    "schema": case.constraint,
                },
            }
        }

    return {
        "extra_body": {
            "structured_outputs": case.structured_outputs_kwargs(),
        }
    }


def create_structured_chat_completion(
    client: Any,
    model: str,
    case: StructuredOutputCase,
    *,
    stream: bool = False,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": case.prompt}],
        temperature=0.0,
        max_tokens=case.max_tokens,
        stream=stream,
        **structured_output_request_kwargs(case),
    )

    if not stream:
        content = response.choices[0].message.content
        assert content is not None
        return content

    chunks: list[str] = []
    try:
        for chunk in response:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                chunks.append(content)
    finally:
        response.close()
    return "".join(chunks)


def create_plain_chat_completion(client: Any, model: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with a short greeting."}],
        temperature=0.0,
        max_tokens=32,
    )
    content = response.choices[0].message.content
    assert content is not None
    return content


def raw_chat_completion_payload(model: str, structured_outputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Generate constrained output."}],
        "temperature": 0.0,
        "max_tokens": 32,
        "structured_outputs": structured_outputs,
    }
