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

import json

import jsonschema
import regex

from tests.e2e.weekly.single_node.features.structured_output.cases import StructuredOutputCase


def assert_structured_output(text: str, case: StructuredOutputCase) -> None:
    assert text, f"{case.case_id} returned empty output"

    if case.constraint_type == "json":
        instance = json.loads(text)
        jsonschema.validate(instance=instance, schema=case.constraint)
        return

    if case.constraint_type == "regex":
        assert regex.fullmatch(case.constraint, text), (
            f"{case.case_id} output {text!r} does not match {case.constraint!r}"
        )
        return

    if case.constraint_type == "choice":
        assert text in case.constraint, f"{case.case_id} output {text!r} is not in {case.constraint!r}"
        return

    if case.constraint_type == "grammar":
        assert text in case.allowed_outputs, f"{case.case_id} output {text!r} is not one of {case.allowed_outputs!r}"
        return

    raise AssertionError(f"Unsupported constraint type: {case.constraint_type}")


def assert_plain_output(text: str) -> None:
    assert text and text.strip(), "plain request returned empty output"
