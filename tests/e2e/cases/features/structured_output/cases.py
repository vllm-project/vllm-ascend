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

from dataclasses import dataclass
from typing import Any, Literal

ConstraintType = Literal["json", "regex", "choice", "grammar"]

MODEL_NAME = "vllm-ascend/Qwen3-32B-W4A4"
SERVED_MODEL_NAME = "qwen3"

EMPLOYEE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "age": {"type": "integer", "minimum": 18, "maximum": 80},
        "department": {"type": "string", "enum": ["engineering", "sales", "finance"]},
        "skills": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 16},
            "minItems": 2,
            "maxItems": 4,
        },
    },
    "required": ["name", "age", "department", "skills"],
    "additionalProperties": False,
}

IPV4_REGEX = (
    r"(25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)"
    r"(\.(25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)){3}"
)

COLOR_CHOICES = ["red", "green", "blue"]
YES_NO_GRAMMAR = 'root ::= "yes" | "no"'


@dataclass(frozen=True)
class StructuredOutputCase:
    case_id: str
    prompt: str
    constraint_type: ConstraintType
    constraint: Any
    max_tokens: int
    allowed_outputs: tuple[str, ...] = ()

    def structured_outputs_kwargs(self) -> dict[str, Any]:
        return {self.constraint_type: self.constraint}


STRUCTURED_OUTPUT_CASES = (
    StructuredOutputCase(
        case_id="json-employee",
        prompt="Generate one employee record that follows the required JSON schema.",
        constraint_type="json",
        constraint=EMPLOYEE_SCHEMA,
        max_tokens=192,
    ),
    StructuredOutputCase(
        case_id="regex-ipv4",
        prompt="Return exactly one valid IPv4 address and nothing else.",
        constraint_type="regex",
        constraint=IPV4_REGEX,
        max_tokens=32,
    ),
    StructuredOutputCase(
        case_id="choice-color",
        prompt="Choose exactly one color from red, green, or blue.",
        constraint_type="choice",
        constraint=COLOR_CHOICES,
        max_tokens=8,
    ),
    StructuredOutputCase(
        case_id="grammar-yes-no",
        prompt="Answer the question 'Is the sky visible?' with exactly yes or no.",
        constraint_type="grammar",
        constraint=YES_NO_GRAMMAR,
        max_tokens=8,
        allowed_outputs=("yes", "no"),
    ),
)

INVALID_STRUCTURED_OUTPUTS = (
    (
        "invalid-json-schema",
        {"json": {"type": "not-a-json-schema-type"}},
    ),
    (
        "conflicting-constraints",
        {"json": EMPLOYEE_SCHEMA, "regex": IPV4_REGEX},
    ),
)
