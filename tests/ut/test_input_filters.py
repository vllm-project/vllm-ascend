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

from __future__ import annotations

from vllm_ascend.dfx.detector.base import AnomalyDetector
from vllm_ascend.dfx.input_filters import (
    InputFilterManager,
    InputTokenIdPrefixFilter,
    PromptContainsTokenIdsFilter,
    PromptLengthFilter,
    build_input_filter_chain,
    matches_input_token_id_prefixes,
)


def test_matches_input_token_id_prefixes_or_semantics():
    assert matches_input_token_id_prefixes([1, 2, 3, 4], []) is True
    assert matches_input_token_id_prefixes([1, 2, 3, 4], [[1, 2]]) is True
    assert matches_input_token_id_prefixes([1, 2, 3, 4], [[9], [1, 2]]) is True
    assert matches_input_token_id_prefixes([1, 2, 3, 4], [[1, 9]]) is False
    assert matches_input_token_id_prefixes([1], [[1, 2]]) is False


def test_input_filter_manager_singleton_and_allow():
    InputFilterManager.reset_for_tests()
    a = InputFilterManager.get()
    b = InputFilterManager.get()
    assert a is b
    assert a.allow("r", prompt_token_ids=[1, 2, 3]) is True

    a.apply_configs(
        [
            {
                "type": "prompt_length",
                "mode": "include",
                "op": "gte",
                "value": 3,
            },
            {
                "type": "prompt_contains_token_ids",
                "mode": "exclude",
                "token_ids": [9],
                "match": "any",
            },
        ]
    )
    # Distinct req_ids: allow is cached per request (prompt is stable in prod).
    assert a.allow("r_ok", prompt_token_ids=[1, 2, 3, 4], log=False) is True
    assert a.allow("r_ok", prompt_token_ids=[1, 2], log=False) is True  # cached
    assert a.allow("r_short", prompt_token_ids=[1, 2], log=False) is False
    assert a.allow("r_excl", prompt_token_ids=[1, 2, 9, 4], log=False) is False
    assert a.allow("r_miss", prompt_token_ids=None, log=False) is False
    InputFilterManager.reset_for_tests()


def test_input_filter_length_before_prefix_and_allow_cache():
    # Config order is contains → prefix → length; eval order must be length first.
    chain = build_input_filter_chain(
        [
            {
                "type": "prompt_contains_token_ids",
                "mode": "include",
                "token_ids": [7],
                "match": "any",
            },
            {
                "type": "input_token_id_prefix",
                "mode": "include",
                "prefixes": [[1, 2]],
            },
            {
                "type": "prompt_length",
                "mode": "include",
                "op": "gte",
                "value": 4,
            },
        ]
    )
    assert [type(f) for f in chain._includes] == [
        PromptLengthFilter,
        InputTokenIdPrefixFilter,
        PromptContainsTokenIdsFilter,
    ]

    InputFilterManager.reset_for_tests()
    mgr = InputFilterManager.get()
    mgr.apply_configs(
        [
            {
                "type": "prompt_length",
                "mode": "include",
                "op": "gte",
                "value": 2,
            }
        ]
    )
    assert mgr.allow("req-a", prompt_token_ids=[1, 2], log=False) is True
    assert mgr._allow_cache.get("req-a") is True
    # Same configs on every refresh_config step must keep the allow cache.
    assert (
        mgr.apply_configs(
            [
                {
                    "type": "prompt_length",
                    "mode": "include",
                    "op": "gte",
                    "value": 2,
                }
            ]
        )
        is False
    )
    assert mgr._allow_cache.get("req-a") is True
    mgr.clear_req("req-a")
    assert "req-a" not in mgr._allow_cache
    assert mgr.allow("req-a", prompt_token_ids=[0], log=False) is False
    assert mgr.apply_configs([]) is True  # rebuild clears cache
    assert mgr._allow_cache == {}
    InputFilterManager.reset_for_tests()


def test_detector_skips_when_filter_rejects():
    InputFilterManager.reset_for_tests()
    InputFilterManager.get().apply_configs(
        [
            {
                "type": "prompt_length",
                "mode": "include",
                "op": "gte",
                "value": 100,
            }
        ]
    )
    det = AnomalyDetector()
    assert det._passes_input_filter("r1", 0, prompt_token_ids=[1, 2, 3], log=False) is False
    InputFilterManager.reset_for_tests()
