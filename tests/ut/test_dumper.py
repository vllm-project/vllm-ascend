import json
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm_ascend.dfx.detector.spec_acceptance import SpecAcceptanceDetector
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


def test_spec_check_no_alert_when_thresholds_not_met():
    req_id = "req-1"
    dumper = _make_dumper()
    dumper.runner = SimpleNamespace(
        tp_rank=0,
        input_batch=SimpleNamespace(req_output_token_ids=[[10]]),
    )
    dumper.is_related_local_request = MagicMock(return_value=True)
    dumper.enable_msprobe_dump_if_needed = MagicMock()
    dumper.handle_anomaly_alert = MagicMock()

    detector = SpecAcceptanceDetector.__new__(SpecAcceptanceDetector)
    detector._runner = dumper.runner
    detector._dfx_config = None
    detector._is_related_request = dumper.is_related_local_request
    detector._enabled = True
    detector._history = defaultdict(list)
    detector._window = 1
    detector._low_threshold = 0.1
    detector._len_low_threshold = 0.1
    detector._high_threshold = 2.0
    detector._len_high_threshold = 2.0
    detector._short_log_ts = {}
    detector._short_log_interval_s = 2.0

    with patch("vllm_ascend.dfx.detector.spec_acceptance.get_pp_group") as get_pp_group:
        get_pp_group.return_value.is_last_rank = True
        alert = detector.check_one(
            req_idx=0,
            req_id=req_id,
            req_state=SimpleNamespace(
                prev_num_draft_len=1,
                prompt_token_ids=[],
                output_token_ids=[],
            ),
            accepted_token_num=2,
            sampled_ids=[10, 11],
        )

    assert alert is None
    dumper.handle_anomaly_alert.assert_not_called()
    dumper.enable_msprobe_dump_if_needed.assert_not_called()


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


def test_dump_once_via_manual_detector_skips_quota():
    from vllm_ascend.dfx.detector.manual_dump import MANUAL_DUMP_REQ_ID, ManualDumpDetector
    from vllm_ascend.dfx.input_filters import InputFilterManager

    InputFilterManager.reset_for_tests()
    # Active filters must not block dump_once (ManualDumpDetector skips filters).
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

    detector = ManualDumpDetector(dfx_config=MagicMock(), runner=dumper.runner)
    detector._dfx_config.consume_dump_once.return_value = True

    with (
        patch("vllm_ascend.dfx.dumper.pending.should_run_anomaly_check_on_rank", return_value=True),
        patch("vllm_ascend.dfx.dumper.pending.get_pp_group") as get_pp_dump,
    ):
        get_pp_dump.return_value.is_last_rank = True
        alerts = detector.check_all()
        assert len(alerts) == 1
        assert alerts[0].req_id == MANUAL_DUMP_REQ_ID
        assert alerts[0].consume_quota is False
        assert dumper.handle_anomaly_alert(alerts[0], detector=detector) is True

    assert dumper._msprobe_dump_active
    assert dumper._msprobe_dump_total_count == 0
    InputFilterManager.reset_for_tests()


def test_consume_dump_once_persists_false(tmp_path: Path):
    from vllm_ascend.dfx.runtime_config import DfxRuntimeConfig

    cfg_path = tmp_path / "dfx_config.json"
    cfg = DfxRuntimeConfig(
        cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=0,
    )
    assert cfg.save({"dump": {"dump_once": True}})
    assert cfg.dump_once() is True
    assert cfg.consume_dump_once() is True
    assert cfg.dump_once() is False
    reloaded = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert reloaded["dump"]["dump_once"] is False
    assert cfg.consume_dump_once() is False


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


def test_sync_dump_pending_or_propagates_dump_once_skip_quota():
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


def test_matches_input_token_id_prefixes_or_semantics():
    from vllm_ascend.dfx.input_filters import matches_input_token_id_prefixes

    assert matches_input_token_id_prefixes([1, 2, 3, 4], []) is True
    assert matches_input_token_id_prefixes([1, 2, 3, 4], [[1, 2]]) is True
    assert matches_input_token_id_prefixes([1, 2, 3, 4], [[9], [1, 2]]) is True
    assert matches_input_token_id_prefixes([1, 2, 3, 4], [[1, 9]]) is False
    assert matches_input_token_id_prefixes([1], [[1, 2]]) is False


def test_input_filter_manager_singleton_and_allow():
    from vllm_ascend.dfx.input_filters import InputFilterManager

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
    from vllm_ascend.dfx.input_filters import (
        InputFilterManager,
        InputTokenIdPrefixFilter,
        PromptContainsTokenIdsFilter,
        PromptLengthFilter,
        build_input_filter_chain,
    )

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
    from vllm_ascend.dfx.detector.base import AnomalyDetector
    from vllm_ascend.dfx.input_filters import InputFilterManager

    InputFilterManager.reset_for_tests()
    InputFilterManager.get().apply_configs(
        [
            {
                "type": "input_token_id_prefix",
                "mode": "include",
                "prefixes": [[151644, 872]],
            }
        ]
    )
    det = AnomalyDetector(runner=SimpleNamespace(requests={"req-1": SimpleNamespace(prompt_token_ids=[1, 2, 3])}))
    assert det._passes_input_filter("req-1", log=False) is False
    assert det._passes_input_filter("req-2", prompt_token_ids=[151644, 872, 0], log=False) is True
    InputFilterManager.reset_for_tests()
