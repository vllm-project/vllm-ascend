# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""A draft model must never advertise a seed.

Its topology is not final when the loader returns: the proposer then shares
embed_tokens / lm_head / topk buffers with the target, and a DSpark draft
rotates fc in place. Anything registered before that point can be rebound or
rewritten underneath a reading peer.
"""

from dataclasses import replace
from unittest.mock import Mock

import pytest

from tests.ut.model_loader.rfork.session_test_utils import dummy_model, make_session


def _session_for(runtime, *, is_draft_model):
    """A registered, leased session with the requested model kind."""
    runtime.identity = replace(runtime.identity, is_draft_model=is_draft_model)
    return make_session(runtime)


@pytest.mark.parametrize("processed_layout", [False, True])
def test_draft_session_refuses_to_start_a_seed_service(runtime, processed_layout):
    session = _session_for(runtime, is_draft_model=True)

    result = session.start_seed_service(dummy_model(), processed_layout)

    assert result is runtime.types.RForkSeedServiceStartResult.FAILED
    # Refused before touching the listener or the planner, so nothing to unwind.
    assert session.seed_server is None
    session.planner.report_seed_once.assert_not_called()


def test_draft_refusal_leaves_the_session_usable_for_inference(runtime):
    """The transferred draft still serves inference, so nothing may be torn down."""
    session = _session_for(runtime, is_draft_model=True)
    state_before = session.state

    session.start_seed_service(dummy_model(), True)

    assert session.state is state_before
    assert not session.lease_release_stop_event.is_set()


def test_main_model_session_still_advertises(runtime, monkeypatch):
    """The gate keys on the draft flag only; main models are unaffected."""
    session = _session_for(runtime, is_draft_model=False)
    monkeypatch.setattr(session, "_start_seed_service", Mock(return_value=True))

    result = session.start_seed_service(dummy_model(), True)

    assert result is runtime.types.RForkSeedServiceStartResult.STARTED
