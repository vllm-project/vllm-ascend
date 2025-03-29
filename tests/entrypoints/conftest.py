#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/blob/main/tests/conftest.py
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
#
import pytest


@pytest.fixture
def sample_regex():
    return (r"((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)\.){3}"
            r"(25[0-5]|(2[0-4]|1\d|[1-9]|)\d)")


@pytest.fixture
def sample_json_schema():
    return {
        "type": "object",
        "properties": {
            "name": {
                "type": "string"
            },
            "age": {
                "type": "integer"
            },
            "skills": {
                "type": "array",
                "items": {
                    "type": "string",
                    "maxLength": 10
                },
                "minItems": 3
            },
            "work_history": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "company": {
                            "type": "string"
                        },
                        "duration": {
                            "type": "number"
                        },
                        "position": {
                            "type": "string"
                        }
                    },
                    "required": ["company", "position"]
                }
            }
        },
        "required": ["name", "age", "skills", "work_history"]
    }
