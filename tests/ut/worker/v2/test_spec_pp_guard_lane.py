# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest

from vllm_ascend.worker.v2 import pp_utils
from vllm_ascend.worker.v2.pp_utils import (
    SpecPPSupport,
    bypass_upstream_spec_pp_guard,
    is_legacy_spec_pp_lane,
)

RELEASE_LANES = ("0.28.0", "0.29.0")


def _force_lane(monkeypatch, lane: str) -> None:
    """Pin the vLLM lane the module under test sees."""
    monkeypatch.setattr(pp_utils, "vllm_version_is", lambda version: version == lane)


@pytest.mark.parametrize("lane", RELEASE_LANES)
def test_release_lanes_need_the_ascend_path(monkeypatch, lane):
    _force_lane(monkeypatch, lane)
    assert is_legacy_spec_pp_lane()


def test_main_needs_no_ascend_path(monkeypatch):
    _force_lane(monkeypatch, "0.0.0-dev")
    assert not is_legacy_spec_pp_lane()


@pytest.mark.parametrize("lane", RELEASE_LANES)
def test_pp_guard_bypasses_on_release_lanes(monkeypatch, lane):
    _force_lane(monkeypatch, lane)
    config = SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=2))
    support = SpecPPSupport(bypass_upstream_pp_guard=True)

    with bypass_upstream_spec_pp_guard(config, support) as bypassed:
        assert bypassed
        assert config.parallel_config.pipeline_parallel_size == 1

    assert config.parallel_config.pipeline_parallel_size == 2


def test_pp_guard_is_inert_on_upstream_main(monkeypatch):
    """Upstream builds the PP state itself, so Ascend must not mask PP there."""
    _force_lane(monkeypatch, "0.0.0-dev")
    config = SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=2))
    support = SpecPPSupport(bypass_upstream_pp_guard=True)

    with bypass_upstream_spec_pp_guard(config, support) as bypassed:
        assert not bypassed
        assert config.parallel_config.pipeline_parallel_size == 2
