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

import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.ut.dfx_test_utils import make_dfx_config
from vllm_ascend.dfx.dumper import Dumper


def _make_dumper() -> Dumper:
    return Dumper.__new__(Dumper)


def test_finalize_dump_data_uses_debugger_specific_step_signature():
    dumper = _make_dumper()
    dumper._debugger = MagicMock()
    dumper._debugger_started = True
    dumper._msprobe_dump_active = False
    dumper._dump_needs_forward = False
    dumper._dump_forward_seen = False
    dumper.disable_msprobe_dump_if_needed = MagicMock()

    dumper.finalize_dump_data()

    dumper._debugger.stop.assert_called_once_with()
    dumper._debugger.step.assert_called_once_with()
    dumper.disable_msprobe_dump_if_needed.assert_called_once_with()


def test_finalize_dump_data_does_not_consume_dummy_forward():
    dumper = _make_dumper()
    dumper._debugger = MagicMock()
    dumper._debugger_started = True
    dumper._msprobe_dump_active = True
    dumper._dump_needs_forward = True
    dumper._dump_forward_seen = True
    dumper.disable_msprobe_dump_if_needed = MagicMock()

    dumper.finalize_dump_data(dump=False)

    dumper._debugger.step.assert_called_once_with(dump=False)
    assert not dumper._dump_forward_seen
    dumper.disable_msprobe_dump_if_needed.assert_not_called()


def test_handle_anomaly_alert_calls_on_alert_armed():
    from vllm_ascend.dfx.detector.alert import AnomalyAlert

    dumper = _make_dumper()
    dumper.enable_msprobe_dump_if_needed = MagicMock(return_value=True)
    detector = MagicMock()
    alert = AnomalyAlert(
        anomaly_type="token_logprob",
        req_id="req-1",
        req_idx=0,
        is_ill=True,
        ill_type=1,
        detail={"hits": 1},
        skip_related_check=True,
    )
    assert dumper.handle_anomaly_alert(alert, detector=detector) is True
    dumper.enable_msprobe_dump_if_needed.assert_called_once()
    detector.on_alert_armed.assert_called_once_with(alert)


def test_start_dump_data_marks_forward_when_active():
    dumper = _make_dumper()
    dumper._debugger = MagicMock()
    dumper._debugger_started = False
    dumper._msprobe_dump_active = True
    dumper._dump_needs_forward = True
    dumper._dump_forward_seen = False
    dumper.runner = SimpleNamespace(model=MagicMock())
    dumper.dump_rank_tag = MagicMock(return_value="tp0")
    dumper._dump_state_tag = MagicMock(return_value="active")

    dumper.start_dump_data()
    assert dumper._dump_forward_seen is True
    dumper._debugger.start.assert_called_once_with(dumper.runner.model)

    dumper._msprobe_dump_active = False
    dumper._dump_needs_forward = False
    dumper.start_dump_data()
    assert dumper._debugger.start.call_count == 1


def test_start_dump_data_skips_debugger_when_inactive():
    dumper = _make_dumper()
    dumper._debugger = MagicMock()
    dumper._debugger_started = False
    dumper._uses_aclgraph_dumper = False
    dumper._aclgraph_hooks_installed = False
    dumper._msprobe_dump_active = False
    dumper.runner = SimpleNamespace(model=MagicMock())

    dumper.start_dump_data()
    dumper._debugger.start.assert_not_called()
    assert dumper._debugger_started is False


def test_start_dump_data_installs_aclgraph_hooks_when_inactive():
    """ACLGraph must patch before capture; step() stays gated by dump window."""
    dumper = _make_dumper()
    dumper._debugger = MagicMock()
    dumper._debugger._running = False
    dumper._debugger_started = False
    dumper._uses_aclgraph_dumper = True
    dumper._aclgraph_hooks_installed = False
    dumper._msprobe_dump_active = False
    dumper._dump_needs_forward = False
    dumper.runner = SimpleNamespace(model=MagicMock(), tp_rank=0, dp_rank=0)
    dumper.dump_rank_tag = MagicMock(return_value="tp0")

    with patch.object(Dumper, "_clear_aclgraph_stats") as clear_stats:
        dumper.start_dump_data()

    dumper._debugger.start.assert_called_once_with(dumper.runner.model)
    assert dumper._aclgraph_hooks_installed is True
    assert dumper._debugger_started is False
    clear_stats.assert_called_once_with()


def test_start_dump_data_enables_aclgraph_collection_when_active():
    dumper = _make_dumper()
    dumper._debugger = MagicMock()
    dumper._debugger_started = False
    dumper._uses_aclgraph_dumper = True
    dumper._aclgraph_hooks_installed = True
    dumper._msprobe_dump_active = True
    dumper._dump_needs_forward = True
    dumper._dump_forward_seen = False
    dumper.runner = SimpleNamespace(model=MagicMock())
    dumper.dump_rank_tag = MagicMock(return_value="tp0")
    dumper._dump_state_tag = MagicMock(return_value="active")

    with patch.object(Dumper, "_clear_aclgraph_stats") as clear_stats:
        dumper.start_dump_data()

    dumper._debugger.start.assert_not_called()
    assert dumper._debugger_started is True
    assert dumper._dump_forward_seen is True
    clear_stats.assert_called_once_with()


def test_finalize_aclgraph_keeps_hooks_without_stop():
    dumper = _make_dumper()
    dumper._debugger = MagicMock()
    dumper._debugger._running = True
    dumper._debugger.dump_path = "/tmp/msprobe_out"
    dumper._debugger_started = True
    dumper._uses_aclgraph_dumper = True
    dumper._msprobe_dump_active = True
    dumper._dump_needs_forward = True
    dumper._dump_forward_seen = True
    dumper._use_pending_dump_sync = MagicMock(return_value=False)
    dumper.disable_msprobe_dump_if_needed = MagicMock()
    dumper.dump_rank_tag = MagicMock(return_value="tp0")
    dumper._dump_state_tag = MagicMock(return_value="active")

    dumper.finalize_dump_data()

    dumper._debugger.stop.assert_not_called()
    dumper._debugger.step.assert_called_once_with()
    assert dumper._debugger._running is True
    assert dumper._debugger_started is False
    dumper.disable_msprobe_dump_if_needed.assert_called_once_with()


def test_sync_dump_pending_or_does_not_touch_config():
    dumper = _make_dumper()
    dumper.dfx_config = MagicMock()
    dumper._pending_dump = False
    dumper._anomaly_dump_feature_enabled = MagicMock(return_value=False)
    dumper._use_pending_dump_sync = MagicMock(return_value=False)
    dumper.dump_rank_tag = MagicMock(return_value="tp0")
    dumper.apply_dfx_config = MagicMock()

    assert dumper.sync_dump_pending_or() is False
    dumper.dfx_config.sync_dfx_config.assert_not_called()
    dumper.apply_dfx_config.assert_not_called()


def test_async_pending_does_not_consume_quota_before_activation():
    req_id = "req-1"
    dumper = _make_dumper()
    dumper._debugger = MagicMock()
    dumper.dfx_config = MagicMock()
    dumper.dfx_config.dump_enabled.return_value = True
    dumper.dfx_config.dump_max_times.return_value = 2
    dumper._pending_dump = False
    dumper._pending_dump_req_id = None
    dumper._pending_dump_skip_quota = False
    dumper._msprobe_dump_active = False
    dumper._msprobe_dumped_req_ids = set()
    dumper._msprobe_dump_total_count = 0
    dumper._dump_max_times = 2
    dumper._msprobe_last_dump_ts = None
    dumper._dump_cooldown_seconds = 0
    dumper._use_pending_dump_sync = MagicMock(return_value=True)

    with patch("vllm_ascend.dfx.dumper.pending.get_pp_group") as get_pp_group:
        get_pp_group.return_value.is_last_rank = True
        armed = dumper.enable_msprobe_dump_if_needed(
            req_id,
            skip_related_check=True,
        )

    assert armed
    assert dumper._pending_dump
    assert dumper._pending_dump_req_id == req_id
    assert dumper._msprobe_dump_total_count == 0


def test_manual_trigger_skips_quota():
    from vllm_ascend.dfx.input_filters import InputFilterManager
    from vllm_ascend.dfx.manual_trigger import MANUAL_TRIGGER_REQ_ID, TriggerEvent

    InputFilterManager.reset_for_tests()
    # Active filters must not block manual_trigger (manual trigger skips related-check).
    InputFilterManager.get().apply_configs(
        [
            {
                "type": "input_token_id_prefix",
                "mode": "include",
                "prefixes": [[1, 2, 3]],
            }
        ]
    )

    dumper = _make_dumper()
    dumper.runner = SimpleNamespace(tp_rank=0, use_async_scheduling=False)
    dumper._debugger = MagicMock()
    dumper._pending_dump = False
    dumper._pending_dump_req_id = None
    dumper._pending_dump_skip_quota = False
    dumper._msprobe_dump_active = False
    dumper._msprobe_dumped_req_ids = set()
    dumper._msprobe_dump_total_count = 0
    dumper._dump_max_times = 0
    dumper._msprobe_last_dump_ts = time.time()
    dumper._dump_cooldown_seconds = 10_000
    dumper.set_msprobe_dump_state = MagicMock(return_value=True)
    dumper.dfx_config = MagicMock()
    dumper.dfx_config.dump_enabled.return_value = True
    dumper._use_pending_dump_sync = MagicMock(return_value=False)

    trigger = TriggerEvent(
        trigger_type="manual_trigger",
        req_id=MANUAL_TRIGGER_REQ_ID,
        detail={"source": "dump.manual_trigger"},
        consume_quota=False,
    )

    with (
        patch("vllm_ascend.dfx.dumper.pending.get_pp_group") as get_pp_dump,
    ):
        get_pp_dump.return_value.is_last_rank = True
        assert dumper.handle_manual_trigger(trigger) is True

    assert dumper._msprobe_dump_active
    assert dumper._msprobe_dump_total_count == 0
    InputFilterManager.reset_for_tests()


def test_consume_manual_trigger_persists_false(tmp_path: Path):
    cfg = make_dfx_config(tmp_path)
    cfg_path = cfg.config_path
    assert cfg.save({"dump": {"manual_trigger": True}})
    assert cfg.manual_trigger() is True
    assert cfg.consume_manual_trigger() is True
    assert cfg.manual_trigger() is False
    reloaded = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert reloaded["dump"]["manual_trigger"] is False
    assert cfg.consume_manual_trigger() is False


def test_dump_phase_idle_pending_active():
    from vllm_ascend.dfx.dfx_types import DumpPhase

    dumper = _make_dumper()
    dumper._msprobe_dump_active = False
    dumper._pending_dump = False
    assert dumper.dump_phase == DumpPhase.IDLE
    dumper._pending_dump = True
    assert dumper.dump_phase == DumpPhase.PENDING
    dumper._msprobe_dump_active = True
    assert dumper.dump_phase == DumpPhase.ACTIVE


def test_sync_dump_pending_or_skips_or_with_real_dfx_config(tmp_path):
    """Regression: hot_reload_enabled is a property — must not call it as a method."""
    cfg = make_dfx_config(tmp_path)
    assert cfg.hot_reload_enabled is False
    assert cfg.dump_enabled() is False

    dumper = _make_dumper()
    dumper.dfx_config = cfg
    dumper._use_pending_dump_sync = MagicMock(return_value=True)

    with (
        patch("vllm_ascend.dfx.dumper.pending.get_tp_group") as get_tp,
        patch("torch.distributed.all_reduce") as ar,
    ):
        assert dumper.sync_dump_pending_or() is False
        get_tp.assert_not_called()
        ar.assert_not_called()


def test_sync_dump_pending_or_still_ors_when_dump_on_no_reload():
    """Hot-reload off but dump.enabled=true → fast path must NOT trigger."""
    dumper = _make_dumper()
    dumper.dfx_config = MagicMock()
    dumper.dfx_config.hot_reload_enabled = False
    dumper.dfx_config.dump_enabled.return_value = True
    dumper._pending_dump = False
    dumper._pending_dump_req_id = None
    dumper._pending_dump_skip_quota = False
    dumper._activate_msprobe_dump = MagicMock(return_value=True)
    dumper._clear_pending_dump = MagicMock()
    dumper.dump_rank_tag = MagicMock(return_value="tp1")
    dumper._use_pending_dump_sync = MagicMock(return_value=True)

    with (
        patch("vllm_ascend.dfx.dumper.pending.get_pp_group") as get_pp,
        patch("vllm_ascend.dfx.dumper.pending.get_tp_group") as get_tp,
        patch("torch.distributed.all_reduce") as ar,
    ):
        get_pp.return_value.is_last_rank = True
        get_tp.return_value.world_size = 2
        get_tp.return_value.cpu_group = object()

        def _or_sum(t, group=None):
            t[0] = 1
            t[1] = 0

        ar.side_effect = _or_sum
        assert dumper.sync_dump_pending_or(allow_arm=True) is True

    ar.assert_called_once()


def test_sync_dump_pending_or_activates_when_peer_pending():
    dumper = _make_dumper()
    dumper._use_pending_dump_sync = MagicMock(return_value=True)
    dumper._pending_dump = False
    dumper._pending_dump_req_id = None
    dumper._pending_dump_skip_quota = False
    dumper._activate_msprobe_dump = MagicMock(return_value=True)
    dumper._clear_pending_dump = MagicMock()
    dumper.dump_rank_tag = MagicMock(return_value="tp1")

    with (
        patch("vllm_ascend.dfx.dumper.pending.get_pp_group") as get_pp,
        patch("vllm_ascend.dfx.dumper.pending.get_tp_group") as get_tp,
        patch("torch.distributed.all_reduce") as ar,
    ):
        get_pp.return_value.is_last_rank = True
        get_tp.return_value.world_size = 2
        get_tp.return_value.cpu_group = object()

        def _or_sum(t, group=None):
            t[0] = 1
            t[1] = 0

        ar.side_effect = _or_sum
        assert dumper.sync_dump_pending_or(allow_arm=True) is True

    dumper._activate_msprobe_dump.assert_called_once_with(None, consume_quota=True)
    dumper._clear_pending_dump.assert_called_once()


def test_sync_dump_pending_or_propagates_manual_trigger_skip_quota():
    dumper = _make_dumper()
    dumper._use_pending_dump_sync = MagicMock(return_value=True)
    dumper._pending_dump = False
    dumper._pending_dump_req_id = None
    dumper._pending_dump_skip_quota = False
    dumper._activate_msprobe_dump = MagicMock(return_value=True)
    dumper._clear_pending_dump = MagicMock()
    dumper.dump_rank_tag = MagicMock(return_value="tp3")

    with (
        patch("vllm_ascend.dfx.dumper.pending.get_pp_group") as get_pp,
        patch("vllm_ascend.dfx.dumper.pending.get_tp_group") as get_tp,
        patch("torch.distributed.all_reduce") as ar,
    ):
        get_pp.return_value.is_last_rank = True
        get_tp.return_value.world_size = 8
        get_tp.return_value.cpu_group = object()

        def _or_sum(t, group=None):
            t[0] = 1
            t[1] = 1

        ar.side_effect = _or_sum
        assert dumper.sync_dump_pending_or(allow_arm=True) is True

    dumper._activate_msprobe_dump.assert_called_once_with(None, consume_quota=False)


def test_sync_dump_pending_or_allow_arm_false_keeps_pending():
    dumper = _make_dumper()
    dumper._use_pending_dump_sync = MagicMock(return_value=True)
    dumper._pending_dump = True
    dumper._pending_dump_req_id = "r1"
    dumper._pending_dump_skip_quota = False
    dumper._activate_msprobe_dump = MagicMock()
    dumper.dump_rank_tag = MagicMock(return_value="tp0")

    with (
        patch("vllm_ascend.dfx.dumper.pending.get_pp_group") as get_pp,
        patch("vllm_ascend.dfx.dumper.pending.get_tp_group") as get_tp,
        patch("torch.distributed.all_reduce") as ar,
    ):
        get_pp.return_value.is_last_rank = True
        get_tp.return_value.world_size = 2
        get_tp.return_value.cpu_group = object()

        def _or_sum(t, group=None):
            t[0] = 1
            t[1] = 0

        ar.side_effect = _or_sum
        assert dumper.sync_dump_pending_or(allow_arm=False) is False

    dumper._activate_msprobe_dump.assert_not_called()
    assert dumper._pending_dump is True


def test_sync_dump_pending_or_activate_fail_keeps_pending():
    dumper = _make_dumper()
    dumper._use_pending_dump_sync = MagicMock(return_value=True)
    dumper._pending_dump = True
    dumper._pending_dump_req_id = "r1"
    dumper._pending_dump_skip_quota = False
    dumper._activate_msprobe_dump = MagicMock(return_value=False)
    dumper._clear_pending_dump = MagicMock()
    dumper.dump_rank_tag = MagicMock(return_value="tp0")

    with (
        patch("vllm_ascend.dfx.dumper.pending.get_pp_group") as get_pp,
        patch("vllm_ascend.dfx.dumper.pending.get_tp_group") as get_tp,
        patch("torch.distributed.all_reduce") as ar,
    ):
        get_pp.return_value.is_last_rank = True
        get_tp.return_value.world_size = 2
        get_tp.return_value.cpu_group = object()

        def _or_sum(t, group=None):
            t[0] = 1
            t[1] = 0

        ar.side_effect = _or_sum
        assert dumper.sync_dump_pending_or(allow_arm=True) is False

    dumper._activate_msprobe_dump.assert_called_once_with("r1", consume_quota=True)
    dumper._clear_pending_dump.assert_not_called()
    assert dumper._pending_dump is True


def test_set_msprobe_dump_state_skips_reload_for_aclgraph_dumper(tmp_path: Path):
    """AclGraphDumper has no ``_maybe_reload_config``; writing dump_enable must still succeed."""
    cfg = tmp_path / "msprobe_dump_config.json"
    cfg.write_text(json.dumps({"dump_enable": False, "dump_path": str(tmp_path / "out")}), encoding="utf-8")

    dumper = _make_dumper()
    dumper.runner = SimpleNamespace(ascend_config=SimpleNamespace(dump_config_path=str(cfg)))
    # Mimic AclGraphDumper: no hot-reload helper.
    dumper._debugger = SimpleNamespace()
    dumper._uses_aclgraph_dumper = True

    assert dumper.set_msprobe_dump_state(True) is True
    loaded = json.loads(cfg.read_text(encoding="utf-8"))
    assert loaded["dump_enable"] is True


def test_set_msprobe_dump_state_syncs_aclgraph_switch(tmp_path: Path):
    """Disable/enable must update device switch, not only the shared JSON."""
    cfg = tmp_path / "msprobe_dump_config.json"
    cfg.write_text(json.dumps({"dump_enable": True, "dump_path": str(tmp_path / "out")}), encoding="utf-8")

    switch = MagicMock()
    dbg = SimpleNamespace(
        dump_enable=True,
        switch=switch,
        config_path=str(cfg),
        _config_signature=("old", 0),
        _get_config_signature=MagicMock(return_value=("new", 1)),
    )
    dumper = _make_dumper()
    dumper.runner = SimpleNamespace(ascend_config=SimpleNamespace(dump_config_path=str(cfg)))
    dumper._debugger = dbg
    dumper._uses_aclgraph_dumper = True
    dumper.dump_rank_tag = MagicMock(return_value="tp0")

    assert dumper.set_msprobe_dump_state(False) is True
    assert dbg.dump_enable is False
    switch.fill_.assert_called_once_with(0)
    assert dbg._config_signature == ("new", 1)
    assert json.loads(cfg.read_text(encoding="utf-8"))["dump_enable"] is False


def test_set_msprobe_dump_state_atomic_write_leaves_valid_json(tmp_path: Path):
    cfg = tmp_path / "msprobe_dump_config.json"
    cfg.write_text(json.dumps({"dump_enable": False}), encoding="utf-8")
    dumper = _make_dumper()
    dumper.runner = SimpleNamespace(ascend_config=SimpleNamespace(dump_config_path=str(cfg)))
    dumper._debugger = None

    assert dumper.set_msprobe_dump_state(True) is True
    # No leftover tmp from atomic write.
    assert list(tmp_path.glob("*.tmp")) == []
    assert json.loads(cfg.read_text(encoding="utf-8"))["dump_enable"] is True


def test_finalize_dump_data_swallows_step_errors():
    dumper = _make_dumper()
    dumper._debugger = MagicMock()
    dumper._debugger_started = True
    dumper._msprobe_dump_active = True
    dumper._dump_needs_forward = False
    dumper._dump_forward_seen = True
    dumper._debugger.step.side_effect = RuntimeError("Load json file failed.")
    dumper.disable_msprobe_dump_if_needed = MagicMock()
    dumper.dump_rank_tag = MagicMock(return_value="tp0")
    dumper._use_pending_dump_sync = MagicMock(return_value=False)

    dumper.finalize_dump_data()

    dumper.disable_msprobe_dump_if_needed.assert_called_once_with()


def test_dump_count_snapshot_pending_reserves_next_slot():
    dumper = _make_dumper()
    dumper._msprobe_dump_total_count = 1
    dumper._dump_max_times = 5
    dumper._pending_dump = True
    dumper._pending_dump_skip_quota = False
    assert dumper.dump_count_snapshot(dump_armed=True) == (2, 5)
    assert dumper.dump_count_snapshot(dump_armed=False) == (1, 5)

    dumper._pending_dump_skip_quota = True
    assert dumper.dump_count_snapshot(dump_armed=True) == (1, 5)

    dumper._pending_dump = False
    dumper._msprobe_dump_total_count = 2
    assert dumper.dump_count_snapshot(dump_armed=True) == (2, 5)


def test_init_debugger_soft_fails_and_forces_dump_off(tmp_path):
    from vllm.config.compilation import CUDAGraphMode

    cfg = make_dfx_config(tmp_path)
    cfg._data["dump"]["enabled"] = True
    runner = SimpleNamespace(
        ascend_config=SimpleNamespace(dump_config_path="/tmp/msprobe.json"),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
        tp_rank=0,
        dp_rank=0,
        model=None,
        input_batch=None,
        requests=None,
        req_states=None,
        discard_request_mask=None,
        use_async_scheduling=False,
    )

    def _soft_fail(self, mode):
        self._debugger = None
        self._uses_aclgraph_dumper = False
        return None

    with patch.object(Dumper, "_init_debugger", _soft_fail):
        dumper = Dumper(runner, dfx_config=cfg)
    assert dumper._debugger is None
    assert cfg.dump_enabled() is False


def test_apply_dfx_config_lazy_retries_debugger(tmp_path):
    from vllm.config.compilation import CUDAGraphMode

    cfg = make_dfx_config(tmp_path)
    runner = SimpleNamespace(
        ascend_config=SimpleNamespace(dump_config_path="/tmp/msprobe.json"),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
        tp_rank=0,
        dp_rank=0,
        model=None,
        input_batch=None,
        requests=None,
        req_states=None,
        discard_request_mask=None,
        use_async_scheduling=False,
    )
    with patch.object(Dumper, "_init_debugger", return_value=None):
        dumper = Dumper(runner, dfx_config=cfg)
    assert dumper._debugger is None

    fake_dbg = MagicMock()
    cfg._data["dump"]["enabled"] = True

    def _init_ok(self, mode):
        self._debugger = fake_dbg
        self._uses_aclgraph_dumper = False
        return fake_dbg

    with patch.object(Dumper, "_init_debugger", _init_ok):
        dumper.apply_dfx_config()
    assert dumper._debugger is fake_dbg
    assert cfg.dump_enabled() is True


def test_init_debugger_real_import_error_soft_fails():
    """``_init_debugger`` itself must not raise when msprobe import fails."""
    from vllm.config.compilation import CUDAGraphMode

    dumper = _make_dumper()
    dumper.runner = SimpleNamespace(
        ascend_config=SimpleNamespace(dump_config_path="/tmp/msprobe.json"),
    )
    with patch.dict("sys.modules", {"msprobe": None, "msprobe.pytorch": None}):
        # Ensure import of msprobe.pytorch raises.
        import builtins

        real_import = builtins.__import__

        def _import(name, *args, **kwargs):
            if name == "msprobe.pytorch" or name.startswith("msprobe."):
                raise ImportError("No module named msprobe")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_import):
            out = dumper._init_debugger(CUDAGraphMode.NONE)
    assert out is None
    assert dumper._debugger is None
