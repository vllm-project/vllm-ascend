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
"""SWA bounded replay: where this backend has to turn the switch off.

Upstream implements the rewind inside ``Scheduler.schedule()``. Four Ascend
scheduling modes replace that method with a pinned copy of an older upstream
body, so they neither rewind nor realize that the chunk they scheduled is a
replay. Activating one of them therefore has to turn the switch off -- and only
the switch, since the model wiring is what decides whether the sliding-window
group declares a replay window at all.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_ascend.platform import (
    _downgrade_swa_bounded_replay,
    _swa_bounded_replay_unsupported_mode,
)

# The modes whose ``schedule()`` is a full copy of an older upstream body.
UNSUPPORTED_MODES = [
    ("recompute_scheduler_enable", {"recompute_scheduler_enable": True}),
    ("dyntra_lb_config", {"dyntra_lb_config": SimpleNamespace(enabled=True)}),
    ("profiling_chunk_config", {"profiling_chunk_config": SimpleNamespace(enabled=True)}),
    ("enable_balance_scheduling", {"enable_balance_scheduling": True}),
]

V41 = ("DeepseekV41ForCausalLM",)


def _ascend_config(**overrides) -> SimpleNamespace:
    scheduler_config = SimpleNamespace(
        recompute_scheduler_enable=False,
        dyntra_lb_config=SimpleNamespace(enabled=False),
        profiling_chunk_config=SimpleNamespace(enabled=False),
        # Not a fork: reaches upstream through super().schedule().
        batch_job_sched_config=SimpleNamespace(enabled=True),
        # Not a fork either: does not override schedule() at all.
        short_request_first_config=SimpleNamespace(enabled=True),
        enable_balance_scheduling=False,
    )
    for name, value in overrides.items():
        setattr(scheduler_config, name, value)
    return SimpleNamespace(scheduler_config=scheduler_config, enable_dsa_cp=False)


def _vllm_config(*, swa_bounded_replay: bool = True, architectures=V41) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(swa_bounded_replay=swa_bounded_replay),
        model_config=SimpleNamespace(architectures=list(architectures)),
    )


# --- which modes are unsupported ---------------------------------------------


@pytest.mark.parametrize(("mode", "overrides"), UNSUPPORTED_MODES)
def test_unsupported_modes_are_named(mode, overrides):
    assert _swa_bounded_replay_unsupported_mode(_ascend_config(**overrides)) == mode


def test_modes_that_reach_upstream_are_left_alone():
    """BatchJobAware calls super().schedule(); ShortRequestFirst only installs a
    request queue. Both inherit the rewind and must not be downgraded."""
    assert _swa_bounded_replay_unsupported_mode(_ascend_config()) is None


# --- the downgrade itself ----------------------------------------------------


@pytest.mark.parametrize(("mode", "overrides"), UNSUPPORTED_MODES)
def test_unsupported_mode_turns_the_switch_off(mode, overrides):
    vllm_config = _vllm_config()

    _downgrade_swa_bounded_replay(vllm_config, _ascend_config(**overrides))

    assert not vllm_config.cache_config.swa_bounded_replay


def test_switch_already_off_is_not_touched():
    vllm_config = _vllm_config(swa_bounded_replay=False)

    _downgrade_swa_bounded_replay(vllm_config, _ascend_config(profiling_chunk_config=SimpleNamespace(enabled=True)))

    assert not vllm_config.cache_config.swa_bounded_replay


def test_switch_is_kept_on_when_nothing_conflicts():
    vllm_config = _vllm_config()

    _downgrade_swa_bounded_replay(vllm_config, _ascend_config())

    assert vllm_config.cache_config.swa_bounded_replay


def test_missing_cache_config_is_not_an_error():
    _downgrade_swa_bounded_replay(SimpleNamespace(cache_config=None), _ascend_config())


def test_release_lane_without_the_switch_is_not_an_error():
    """The release lane predates the switch; reading it must not raise, and the
    absence already means "off"."""
    cache_config = SimpleNamespace()

    _downgrade_swa_bounded_replay(
        SimpleNamespace(cache_config=cache_config, model_config=SimpleNamespace(architectures=list(V41))),
        _ascend_config(profiling_chunk_config=SimpleNamespace(enabled=True)),
    )

    assert not hasattr(cache_config, "swa_bounded_replay")


# --- the warning says which kind of conflict it was --------------------------


def _warning_args(overrides, **config) -> tuple:
    """The positional arguments the warning was called with.

    ``vllm_ascend`` logs with %s and passes its values as arguments rather than
    formatting eagerly, so the mode arrives here separately from the template.
    """
    with patch("vllm_ascend.platform.logger.warning_once") as warning:
        _downgrade_swa_bounded_replay(_vllm_config(**config), _ascend_config(**overrides))
    args, _ = warning.call_args
    return args


def test_the_warning_names_the_mode_that_conflicts():
    """A user who reads only the mode has to know which knob to turn off."""
    template, mode = _warning_args({"enable_balance_scheduling": True})

    assert mode == "enable_balance_scheduling"
    assert "%s schedules with a pinned copy of the scheduler" in template


def test_a_model_that_cannot_replay_is_not_warned_about():
    """The switch defaults to on for every model, but only a model the feature
    was written for reads it. Telling a V4 user that replay is disabled names a
    feature they never had -- and the switch is still cleared, so nothing can
    reach the rewound path through some other caller."""
    vllm_config = _vllm_config(architectures=("DeepseekV4ForCausalLM",))

    with patch("vllm_ascend.platform.logger.warning_once") as warning:
        _downgrade_swa_bounded_replay(vllm_config, _ascend_config(enable_balance_scheduling=True))

    warning.assert_not_called()
    assert not vllm_config.cache_config.swa_bounded_replay
