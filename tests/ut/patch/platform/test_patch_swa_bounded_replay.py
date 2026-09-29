# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""SWA bounded replay: the V1 resume path.

Upstream reports a rewind on ``NewRequestData`` only, because the V2 model
runner folds resumed requests into the new-request list. The V1 runner resumes
a preempted request through ``CachedRequestData``, and a resumed request is
exactly the one that can rewind a second time, so the field has to be declared
there and filled from the resumed requests.
"""

import inspect
from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.output import CachedRequestData
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_ascend.patch.platform.patch_swa_bounded_replay import (
    _EXPECTED_PARAMETERS,
    _patch_cached_request_data_replay_start,
    _patch_make_cached_request_data,
    _upstream_carries_replay_start,
)

pytestmark = pytest.mark.skipif(
    not _upstream_carries_replay_start(),
    reason="The replay start arrives with vLLM #56227; on a pin that predates it the patch is "
    "inert by design, so there is no field to carry and nothing to fill.",
)

REPLAY_START = 64


def test_the_probe_finds_upstreams_rewind():
    """The probe is the module's own switch, and it is what the skip above
    rests on: it has to key on something that arrives with the feature."""
    assert _upstream_carries_replay_start()


def _scheduler() -> Scheduler:
    """A Scheduler instance with only what the payload builder reads.

    Deliberately minimal: if a pin bump makes the builder read more state, this
    has to fail here rather than silently build a payload that never happens.
    """
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.use_pp = True
    scheduler.use_v2_model_runner = False
    scheduler.scheduler_config = SimpleNamespace(async_scheduling=False)
    scheduler.prev_step_scheduled_req_ids = set()
    return scheduler


def _request(request_id: str, *, replay_start: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        all_token_ids=[11, 12, 13],
        num_computed_tokens=2,
        num_output_tokens=1,
        num_output_placeholders=0,
        replay_start=replay_start,
    )


def _blocks():
    return SimpleNamespace(get_block_ids=lambda allow_none: ([0],))


def _build(resumed_reqs):
    return Scheduler._make_cached_request_data(
        _scheduler(),
        running_reqs=[],
        resumed_reqs=resumed_reqs,
        num_scheduled_tokens={req.request_id: 1 for req in resumed_reqs},
        spec_decode_tokens={},
        req_to_new_blocks={req.request_id: _blocks() for req in resumed_reqs},
    )


# --- the field has to be declared to cross the process boundary --------------


def test_replay_start_is_a_declared_field():
    """msgpack encodes declared fields only, so an instance attribute set after
    construction would never reach the worker."""
    assert "replay_start" in CachedRequestData.__dataclass_fields__
    assert "replay_start" in inspect.signature(CachedRequestData.__init__).parameters


def test_declaring_the_field_is_idempotent():
    fields_before = dict(CachedRequestData.__dataclass_fields__)

    _patch_cached_request_data_replay_start()

    assert CachedRequestData.__dataclass_fields__ == fields_before


# --- a resumed request that rewinds has to say so ----------------------------


def test_resumed_request_replay_start_reaches_the_payload():
    cached_reqs = _build([_request("req-0", replay_start=REPLAY_START)])

    assert cached_reqs.resumed_req_ids == ["req-0"]
    assert cached_reqs.replay_start == {"req-0": REPLAY_START}


def test_resumed_request_that_does_not_rewind_contributes_nothing():
    """A preempted request can come back without a fresh hit; naming it would
    make the worker pad slots that were never replayed."""
    cached_reqs = _build([_request("req-0"), _request("req-1", replay_start=REPLAY_START)])

    assert cached_reqs.replay_start == {"req-1": REPLAY_START}


def test_payload_without_resumed_requests_carries_an_empty_map():
    """Absent has to mean "no replay" without depending on the payload's
    length, which is why this is a dict and not a list parallel to req_ids."""
    cached_reqs = _build([])

    assert cached_reqs.replay_start == {}


# --- a pin bump must fail loudly, not drop the field silently ----------------


def test_patch_is_idempotent():
    patched = Scheduler._make_cached_request_data

    _patch_make_cached_request_data()

    assert Scheduler._make_cached_request_data is patched


def test_signature_drift_raises_instead_of_silently_stopping(monkeypatch):
    def _renamed(self, running_reqs, resumed, num_scheduled_tokens, spec_decode_tokens, req_to_new_blocks):
        raise AssertionError("the patch must not have been applied")

    monkeypatch.setattr(Scheduler, "_make_cached_request_data", _renamed)

    with pytest.raises(RuntimeError, match="unexpected Scheduler._make_cached_request_data signature"):
        _patch_make_cached_request_data()


def test_expected_parameters_match_the_patched_method():
    """Guards the guard: the tuple above has to describe the method actually
    installed, or the check would pass on a signature it no longer recognizes."""
    assert tuple(inspect.signature(Scheduler._make_cached_request_data).parameters) == _EXPECTED_PARAMETERS
