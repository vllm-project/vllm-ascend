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
"""CPU tests for the PP + speculation accepted-token derivation.

Non-final PP stages never sample, so the number of draft tokens accepted in the
previous verify step has to be recovered from ``num_computed_tokens``. These
tests execute the real ``NPUModelRunner._update_states`` with a CPU parent
instead of importing NPU workers.
"""

from __future__ import annotations
import __future__

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

RUNNER_SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v1/model_runner.py"


class _PPGroup(SimpleNamespace):
    world_size = 2
    is_last_rank = False


def _load_runner_cls():
    tree = ast.parse(RUNNER_SOURCE.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    cls.bases = [ast.Name(id="Parent", ctx=ast.Load())]
    cls.body = [node for node in cls.body if getattr(node, "name", None) in {"_update_states", "_is_pd_prefill_worker"}]

    class Parent:
        tracked_refs: int
        parent_calls: int
        input_batch: Any
        parent_delta: dict
        parent_drop: str | None

        def _update_replay_requests(self, scheduler_output):
            return None

        def _apply_pp_sampled_tokens_from_scheduler_output(self, scheduler_output):
            return None

        def _track_tmp_encoder_cache_refs(self, scheduler_output):
            self.tracked_refs += 1

        def _update_states(self, scheduler_output):
            self.parent_calls += 1
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                index = self.input_batch.req_id_to_index.get(req_id)
                if index is None:
                    continue
                self.input_batch.num_computed_tokens_cpu[index] += self.parent_delta.get(req_id, 0)
            if self.parent_drop is not None:
                self.input_batch.req_id_to_index.pop(self.parent_drop, None)
            return "sampling_metadata"

    pp_group = _PPGroup()
    namespace = {
        "Parent": Parent,
        "get_pp_group": lambda: pp_group,
        "Callable": object,
    }
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    exec(
        compile(module, str(RUNNER_SOURCE), "exec", flags=__future__.annotations.compiler_flag),
        namespace,
    )
    return namespace["NPUModelRunner"], pp_group


NPUModelRunner, PP_GROUP = _load_runner_cls()


def _make_runner(
    *,
    spec=True,
    num_spec_tokens=3,
    async_scheduling=False,
    world_size=2,
    last_rank=False,
    producer=False,
    consumer=False,
):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.parent_calls = 0
    runner.tracked_refs = 0
    runner.parent_delta = {"r1": 4}
    runner.parent_drop = None
    runner.speculative_config = SimpleNamespace(method="mtp") if spec else None
    runner.num_spec_tokens = num_spec_tokens
    runner.use_async_scheduling = async_scheduling
    runner.is_kv_producer = producer
    runner.is_kv_consumer = consumer
    runner.requests = {
        "r1": SimpleNamespace(prev_num_draft_len=3, num_computed_tokens=5, output_token_ids=[]),
        "r2": SimpleNamespace(prev_num_draft_len=0, num_computed_tokens=5, output_token_ids=[]),
    }
    runner.input_batch = SimpleNamespace(
        req_id_to_index={"r1": 0, "r2": 1},
        num_computed_tokens_cpu=[5, 5],
        num_accepted_tokens_cpu=[1, 1],
    )
    PP_GROUP.world_size = world_size
    PP_GROUP.is_last_rank = last_rank
    return runner


def _make_output(*, total=6, req_ids=("r1", "r2"), kv_copies=()):
    return SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=list(req_ids),
            num_computed_tokens=[0] * len(req_ids),
            num_output_tokens=[16] * len(req_ids),
            new_token_ids=[[1]] * len(req_ids),
        ),
        kv_cache_block_copies=list(kv_copies),
        total_num_scheduled_tokens=total,
    )


def test_non_last_stage_derives_accepted_count_from_computed_tokens():
    runner = _make_runner()
    result = runner._update_states(_make_output())

    assert result == "sampling_metadata"
    assert runner.parent_calls == 1
    assert runner.tracked_refs == 1
    # r1 accepted all 4 tokens verified since the snapshot (5 -> 9).
    assert runner.input_batch.num_accepted_tokens_cpu == [4, 1]


def test_requests_without_draft_tokens_keep_parent_value():
    runner = _make_runner()
    runner.parent_delta = {"r1": 4, "r2": 2}
    runner._update_states(_make_output())

    assert runner.input_batch.num_accepted_tokens_cpu == [4, 1]


def test_last_stage_leaves_count_to_parent():
    runner = _make_runner(last_rank=True)
    runner._update_states(_make_output())

    assert runner.input_batch.num_accepted_tokens_cpu == [1, 1]


def test_single_stage_leaves_count_to_parent():
    runner = _make_runner(world_size=1)
    runner._update_states(_make_output())

    assert runner.input_batch.num_accepted_tokens_cpu == [1, 1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spec": False},
        {"num_spec_tokens": 0},
        {"async_scheduling": True},
        {"producer": True},
    ],
)
def test_gated_off_paths_are_noops(kwargs):
    runner = _make_runner(**kwargs)
    runner._update_states(_make_output())

    assert runner.input_batch.num_accepted_tokens_cpu == [1, 1]


def test_no_scheduled_tokens_is_noop():
    runner = _make_runner()
    runner._update_states(_make_output(total=0))

    assert runner.input_batch.num_accepted_tokens_cpu == [1, 1]


def test_zero_delta_is_not_written_back():
    runner = _make_runner()
    runner.parent_delta = {}
    runner._update_states(_make_output())

    assert runner.input_batch.num_accepted_tokens_cpu == [1, 1]


def test_request_dropped_by_parent_is_skipped():
    runner = _make_runner()
    runner.parent_drop = "r1"
    runner._update_states(_make_output())

    assert runner.input_batch.num_accepted_tokens_cpu == [1, 1]


@pytest.mark.parametrize(
    "world_size,last_rank,expect_write",
    [
        (2, False, True),  # first stage
        (3, False, True),  # middle stage
        (3, True, False),  # last stage
    ],
)
def test_stage_roles_only_write_on_non_last_stage(world_size, last_rank, expect_write):
    runner = _make_runner(world_size=world_size, last_rank=last_rank)
    runner._update_states(_make_output())

    assert runner.input_batch.num_accepted_tokens_cpu == ([4, 1] if expect_write else [1, 1])
