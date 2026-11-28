# SPDX-License-Identifier: Apache-2.0
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
#
# This file is a part of the vllm-ascend project.
#
"""Regression tests for the DP-wide ``skip_drafts`` agreement (#23592).

Upstream vLLM agrees on ``skip_drafts`` across the DP group with the same
all-reduce that carries the padding metadata. vLLM Ascend skips that
all-reduce on the MC2 paths, so without an explicit fallback every rank
keeps its own value and two ranks can disagree on whether the drafter runs.
The mismatched drafter passes leave unmatched EP/MC2 collectives behind and
deadlock the engine.
"""

from types import SimpleNamespace

from vllm.v1.core.sched.output import SchedulerOutput

from vllm_ascend.worker.v2.dp_sync import align_skip_drafts_across_dp


def _scheduler_output(skip_drafts: bool) -> SchedulerOutput:
    output = SchedulerOutput.make_empty()
    output.skip_drafts = skip_drafts
    return output


def _vllm_config(data_parallel_size: int) -> SimpleNamespace:
    return SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=data_parallel_size))


def test_skip_drafts_kept_when_dp_metadata_sync_is_not_skipped():
    output = _scheduler_output(True)

    aligned = align_skip_drafts_across_dp(output, _vllm_config(2), skip_dp_metadata_sync=False)

    assert aligned is output
    assert aligned.skip_drafts is True


def test_skip_drafts_kept_without_a_dp_group():
    output = _scheduler_output(True)

    aligned = align_skip_drafts_across_dp(output, _vllm_config(1), skip_dp_metadata_sync=True)

    assert aligned is output
    assert aligned.skip_drafts is True


def test_skip_drafts_kept_when_the_dp_group_size_is_unknown():
    output = _scheduler_output(True)

    aligned = align_skip_drafts_across_dp(output, None, skip_dp_metadata_sync=True)

    assert aligned is output
    assert aligned.skip_drafts is True


def test_skip_drafts_cleared_when_dp_metadata_sync_is_skipped():
    output = _scheduler_output(True)

    aligned = align_skip_drafts_across_dp(output, _vllm_config(2), skip_dp_metadata_sync=True)

    assert aligned is not output
    assert aligned.skip_drafts is False
    # The scheduler output of the caller must not be mutated in place.
    assert output.skip_drafts is True
    # Everything else is carried over by the shallow copy.
    assert aligned.num_scheduled_tokens is output.num_scheduled_tokens
    assert aligned.finished_req_ids is output.finished_req_ids


def test_skip_drafts_already_false_is_not_copied():
    output = _scheduler_output(False)

    aligned = align_skip_drafts_across_dp(output, _vllm_config(2), skip_dp_metadata_sync=True)

    assert aligned is output
    assert aligned.skip_drafts is False
