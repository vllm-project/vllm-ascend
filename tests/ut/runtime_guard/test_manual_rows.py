#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

# mypy: ignore-errors
"""UT: iter_local_request_rows resolution order / fallbacks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_ascend.observability.runtime_guard.state import iter_local_request_rows


@pytest.mark.parametrize(
    ("runner", "scheduler_output", "expected"),
    [
        (
            SimpleNamespace(input_batch=SimpleNamespace(req_ids=["a", ""])),
            None,
            [("a", 0)],
        ),
        (
            SimpleNamespace(
                input_batch=None,
                execute_model_state=SimpleNamespace(
                    input_batch=SimpleNamespace(req_ids=["b1", "b2"]),
                ),
            ),
            None,
            [("b1", 0), ("b2", 1)],
        ),
        (
            SimpleNamespace(
                input_batch=None,
                execute_model_state=None,
                requests={"r1": object(), "r2": object()},
            ),
            None,
            [("r1", -1), ("r2", -1)],
        ),
        (
            SimpleNamespace(
                input_batch=None,
                execute_model_state=None,
                requests={},
                req_states=SimpleNamespace(req_id_to_index={"z": 5, "a": 1}),
            ),
            None,
            [("a", 1), ("z", 5)],
        ),
        (
            SimpleNamespace(
                input_batch=None,
                execute_model_state=None,
                requests={},
                req_states=SimpleNamespace(req_id_to_index={}),
            ),
            SimpleNamespace(num_scheduled_tokens={"s1": 10, "s2": 0, "": 5}),
            [("s1", -1)],
        ),
    ],
)
def test_iter_local_request_rows_sources(runner, scheduler_output, expected):
    assert iter_local_request_rows(runner, scheduler_output) == expected
