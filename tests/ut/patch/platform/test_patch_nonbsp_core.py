from unittest.mock import MagicMock

import numpy as np
import torch
import vllm.v1.request as request_module

import vllm_ascend.patch.platform.patch_nonbsp_core as nonbsp_core

RequestStatus = request_module.RequestStatus


def test_nonbsp_core_uses_extended_request_status():
    assert nonbsp_core.RequestStatus is request_module.RequestStatus
    assert hasattr(nonbsp_core.RequestStatus, "LB_PAUSED")


def test_nonbsp_uses_main_upstream_engine_core_entrypoint():
    assert nonbsp_core._UpstreamRunEngineCore is nonbsp_core._balance_patch._OriginalRunEngineCore


def test_nonbsp_enabled_reads_nested_scheduler_config(monkeypatch):
    monkeypatch.setattr(
        nonbsp_core,
        "get_ascend_config",
        MagicMock(side_effect=RuntimeError("not initialized")),
    )
    vllm_config = MagicMock()
    vllm_config.additional_config = {
        "scheduler_config": {
            "nonbsp_config": {
                "enabled": True,
            }
        }
    }

    assert nonbsp_core._nonbsp_enabled(vllm_config) is True


def test_nonbsp_diagnostics_read_nested_scheduler_config(monkeypatch):
    monkeypatch.setattr(
        nonbsp_core,
        "get_ascend_config",
        MagicMock(side_effect=RuntimeError("not initialized")),
    )
    vllm_config = MagicMock()
    vllm_config.additional_config = {
        "scheduler_config": {
            "nonbsp_config": {
                "enabled": True,
                "enable_diagnostics": True,
            }
        }
    }

    assert nonbsp_core._get_nonbsp_config(vllm_config).enable_diagnostics is True


def test_nonbsp_enabled_ignores_legacy_top_level_config(monkeypatch):
    monkeypatch.setattr(
        nonbsp_core,
        "get_ascend_config",
        MagicMock(side_effect=RuntimeError("not initialized")),
    )
    vllm_config = MagicMock()
    vllm_config.additional_config = {"NONBSP_ENABLE": 1}

    assert nonbsp_core._nonbsp_enabled(vllm_config) is False


def test_default_dp_core_diagnostics_method_is_not_patched():
    assert (
        nonbsp_core.DPEngineCoreProc._has_global_unfinished_reqs
        is not nonbsp_core.NonBSPDPEngineCoreProc._has_global_unfinished_reqs
    )


def test_dp_engine_core_initializes_ascend_config(monkeypatch, capsys):
    vllm_config = MagicMock()
    vllm_config.scheduler_config.max_num_seqs = 4

    ascend_config = MagicMock()
    nonbsp_config = ascend_config.scheduler_config.nonbsp_config
    nonbsp_config.enabled = True
    nonbsp_config.enable_diagnostics = True
    nonbsp_config.mode = "static"
    nonbsp_config.start_step = 0
    nonbsp_config.end_step = -1
    nonbsp_config.bubble_threshold = 5.0
    nonbsp_config.long_req_block_threshold = 700
    nonbsp_config.dynamic_max_step = 256

    init_ascend_config = MagicMock(return_value=ascend_config)
    monkeypatch.setattr(nonbsp_core, "init_ascend_config", init_ascend_config)

    def init_data_parallel(engine_core, _config):
        engine_core.dp_group = MagicMock()
        engine_core.dp_rank = 0

    monkeypatch.setattr(
        nonbsp_core.DPEngineCoreProc,
        "_init_data_parallel",
        init_data_parallel,
    )
    monkeypatch.setattr(nonbsp_core.dist, "get_world_size", lambda group: 2)

    engine_core = object.__new__(nonbsp_core.NonBSPDPEngineCoreProc)
    engine_core._init_data_parallel(vllm_config)

    init_ascend_config.assert_called_once_with(vllm_config)
    assert engine_core._lb_mode == "static"
    assert engine_core._lb_dp_size_cached == 2

    output = capsys.readouterr().out
    assert "nonbsp_config.enabled = True" in output
    assert "nonbsp_config.enable_diagnostics = True" in output
    assert "nonbsp_config.mode = static" in output
    assert "nonbsp_config.start_step = 0" in output
    assert "nonbsp_config.end_step = -1" in output
    assert "nonbsp_config.bubble_threshold = 5.0" in output
    assert "nonbsp_config.long_req_block_threshold = 700" in output
    assert "nonbsp_config.dynamic_max_step = 256" in output


def test_balance_load_runs_before_engine_step(monkeypatch):
    events: list[str] = []

    def process_engine_step(_self: object) -> bool:
        events.append("engine_step")
        return True

    def run_balance_load() -> None:
        events.append("balance_load")

    monkeypatch.setattr(
        nonbsp_core.DPEngineCoreProc,
        "_process_engine_step",
        process_engine_step,
    )

    engine_core = object.__new__(nonbsp_core.NonBSPDPEngineCoreProc)
    engine_core.engines_running = True
    engine_core.scheduler = MagicMock()
    engine_core.scheduler.has_unfinished_requests.return_value = True
    engine_core.run_balance_load = run_balance_load

    assert engine_core._process_engine_step() is True
    assert events == ["balance_load", "engine_step"]


def _make_process_step_test_engine(*, engines_running: bool, local_unfinished: bool):
    engine_core = object.__new__(nonbsp_core.NonBSPDPEngineCoreProc)
    engine_core.engines_running = engines_running
    engine_core.scheduler = MagicMock()
    engine_core.scheduler.has_unfinished_requests.return_value = local_unfinished
    engine_core.run_balance_load = MagicMock()
    return engine_core


def test_idle_connector_maintenance_skips_balance_load(monkeypatch):
    parent_process_step = MagicMock(return_value=False)
    monkeypatch.setattr(
        nonbsp_core.DPEngineCoreProc,
        "_process_engine_step",
        parent_process_step,
    )
    engine_core = _make_process_step_test_engine(
        engines_running=False,
        local_unfinished=False,
    )
    engine_core.scheduler.has_finished_requests.return_value = True

    assert engine_core._process_engine_step() is False

    engine_core.run_balance_load.assert_not_called()
    parent_process_step.assert_called_once_with()


def test_target_rank_with_request_joins_balance_load_before_start_wave(monkeypatch):
    parent_process_step = MagicMock(return_value=True)
    monkeypatch.setattr(
        nonbsp_core.DPEngineCoreProc,
        "_process_engine_step",
        parent_process_step,
    )
    engine_core = _make_process_step_test_engine(
        engines_running=False,
        local_unfinished=True,
    )

    assert engine_core._process_engine_step() is True

    engine_core.run_balance_load.assert_called_once_with()
    parent_process_step.assert_called_once_with()


def test_awake_rank_without_local_request_joins_balance_load(monkeypatch):
    parent_process_step = MagicMock(return_value=False)
    monkeypatch.setattr(
        nonbsp_core.DPEngineCoreProc,
        "_process_engine_step",
        parent_process_step,
    )
    engine_core = _make_process_step_test_engine(
        engines_running=True,
        local_unfinished=False,
    )

    assert engine_core._process_engine_step() is False

    engine_core.run_balance_load.assert_called_once_with()
    parent_process_step.assert_called_once_with()


def test_wave_rank_with_local_request_joins_balance_load(monkeypatch):
    parent_process_step = MagicMock(return_value=True)
    monkeypatch.setattr(
        nonbsp_core.DPEngineCoreProc,
        "_process_engine_step",
        parent_process_step,
    )
    engine_core = _make_process_step_test_engine(
        engines_running=True,
        local_unfinished=True,
    )

    assert engine_core._process_engine_step() is True

    engine_core.run_balance_load.assert_called_once_with()
    parent_process_step.assert_called_once_with()


def test_nonbsp_diagnostics_are_disabled_by_default(monkeypatch, capsys):
    monkeypatch.setattr(
        nonbsp_core.DPEngineCoreProc,
        "_has_global_unfinished_reqs",
        lambda self, local_unfinished: False,
    )

    engine_core = object.__new__(nonbsp_core.NonBSPDPEngineCoreProc)
    engine_core._lb_enable_diagnostics = False
    engine_core.step_counter = 7
    engine_core.scheduler = MagicMock()
    engine_core.scheduler.modifications = {"freeze": True}
    engine_core.scheduler.lb_freeze = True

    assert engine_core._has_global_unfinished_reqs(True) is False
    assert capsys.readouterr().out == ""
    assert engine_core.scheduler.modifications is None
    assert engine_core.scheduler.lb_freeze is False


def test_nonbsp_diagnostics_print_step_counter_once(monkeypatch, capsys):
    monkeypatch.setattr(
        nonbsp_core.DPEngineCoreProc,
        "_has_global_unfinished_reqs",
        lambda self, local_unfinished: True,
    )

    engine_core = object.__new__(nonbsp_core.NonBSPDPEngineCoreProc)
    engine_core._lb_enable_diagnostics = True
    engine_core.step_counter = 7
    engine_core.scheduler = MagicMock()

    assert engine_core._has_global_unfinished_reqs(True) is True
    output = capsys.readouterr().out
    assert output.count("step_counter: 7") == 1


def test_print_balance_summary(capsys):
    nonbsp_core._print_requests_by_rank(
        [([8, 4, 2, 0], 2)],
        dp_rank=0,
        enable_diagnostics=True,
    )
    nonbsp_core._print_modifications(
        [{"out_blk": [8], "in_blk": [2], "freeze": False}],
        dp_rank=0,
        enable_diagnostics=True,
    )

    output = capsys.readouterr().out
    assert "DP0 | Run(2): [  8,   4]" in output
    assert "Wait(1): [  2]" in output
    assert "Out: [  8]" in output
    assert "In: [  2]" in output
    assert "Freeze: False" in output


def test_balance_summary_is_suppressed_on_nonzero_dp_rank(capsys):
    nonbsp_core._print_requests_by_rank(
        [([8, 4, 2, 0], 2)],
        dp_rank=1,
        enable_diagnostics=True,
    )
    nonbsp_core._print_modifications(
        [{"out_blk": [8], "in_blk": [2], "freeze": False}],
        dp_rank=1,
        enable_diagnostics=True,
    )

    assert capsys.readouterr().out == ""


def test_balance_summary_is_suppressed_when_diagnostics_are_disabled(capsys):
    nonbsp_core._print_requests_by_rank(
        [([8, 4, 2, 0], 2)],
        dp_rank=0,
        enable_diagnostics=False,
    )
    nonbsp_core._print_modifications(
        [{"out_blk": [8], "in_blk": [2], "freeze": False}],
        dp_rank=0,
        enable_diagnostics=False,
    )

    assert capsys.readouterr().out == ""


def test_nonbsp_allgather_uses_only_unified_waiting_queue(monkeypatch):
    waiting_request = MagicMock(
        all_token_ids=list(range(17)),
        status=RequestStatus.WAITING,
    )
    skipped_request = MagicMock(
        all_token_ids=list(range(65)),
        status=RequestStatus.WAITING,
    )
    scheduler = MagicMock()
    scheduler.block_size = 8
    scheduler.running = []
    scheduler.waiting = [waiting_request]
    scheduler.skipped_waiting = [skipped_request]

    engine_core = object.__new__(nonbsp_core.NonBSPDPEngineCoreProc)
    engine_core.scheduler = scheduler
    engine_core.dp_group = MagicMock()
    engine_core.dp_rank = 0
    engine_core._lb_enable_diagnostics = False
    engine_core._lb_max_slots_cached = 4
    engine_core._lb_dp_size_cached = 1
    engine_core._lb_max_num_seqs = 2
    engine_core._lb_threshold = 5.0
    engine_core._lb_pending_long_req = False
    engine_core._lb_pending_long_req_blk = 0
    engine_core._lb_data_np = np.zeros(6, dtype=np.int32)
    engine_core._lb_data_t = torch.as_tensor(engine_core._lb_data_np)
    gathered = np.zeros(6, dtype=np.int32)
    engine_core._lb_all_data_np = [gathered]
    engine_core._lb_all_data_t_buf = [torch.as_tensor(gathered)]

    def all_gather(output_tensors, input_tensor, group):
        output_tensors[0].copy_(input_tensor)

    captured = {}

    def balance_load(requests_by_rank, dev_num, max_num_seqs, threshold):
        captured["requests_by_rank"] = requests_by_rank
        return [{"out_blk": [], "in_blk": [], "freeze": True}]

    monkeypatch.setattr(nonbsp_core.dist, "all_gather", all_gather)
    monkeypatch.setattr(nonbsp_core, "balance_load", balance_load)
    monkeypatch.setattr(nonbsp_core, "_print_requests_by_rank", lambda *args: None)
    monkeypatch.setattr(nonbsp_core, "_print_modifications", lambda *args: None)

    engine_core._do_lb_allgather()

    assert captured["requests_by_rank"] == [([3, 0, 0, 0], 0)]
