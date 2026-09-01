import importlib
from unittest.mock import MagicMock

import numpy as np
import torch
import vllm.v1.request as request_module

from vllm_ascend.core.recompute_scheduler import (
    AsyncRecomputeScheduler,
    RecomputeScheduler,
)

NativeRequestStatus = request_module.RequestStatus
dyntra_lb_core = importlib.import_module("vllm_ascend.patch.platform.patch_dyntra_lb_core")

RequestStatus = request_module.RequestStatus


def test_dyntra_lb_core_uses_native_request_status():
    assert request_module.RequestStatus is NativeRequestStatus
    assert dyntra_lb_core.RequestStatus is NativeRequestStatus


def test_dyntra_lb_uses_main_upstream_engine_core_entrypoint():
    assert dyntra_lb_core._UpstreamRunEngineCore is dyntra_lb_core._balance_patch._OriginalRunEngineCore


def test_dyntra_lb_disabled_delegates_to_balance_wrapper(monkeypatch):
    expected = object()
    calls = []

    def balance_wrapper(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(
        dyntra_lb_core,
        "_get_dyntra_lb_config",
        lambda _config: MagicMock(enabled=False),
    )
    monkeypatch.setattr(dyntra_lb_core, "_PreviousRunEngineCore", balance_wrapper)

    vllm_config = object()
    result = dyntra_lb_core._dyntra_lb_run_engine_core(
        "engine-arg",
        vllm_config=vllm_config,
        dp_rank=1,
        local_dp_rank=2,
    )

    assert result is expected
    assert calls == [
        (
            ("engine-arg",),
            {
                "vllm_config": vllm_config,
                "dp_rank": 1,
                "local_dp_rank": 2,
            },
        )
    ]


def test_dyntra_lb_enabled_bypasses_balance_wrapper(monkeypatch):
    expected = object()
    balance_wrapper = MagicMock(side_effect=AssertionError("balance wrapper must be bypassed"))

    def upstream_entrypoint(*args, **kwargs):
        assert dyntra_lb_core._engine_core_mod.DPEngineCoreProc is dyntra_lb_core.DyntraLBDPEngineCoreProc
        return expected

    monkeypatch.setattr(
        dyntra_lb_core,
        "_get_dyntra_lb_config",
        lambda _config: MagicMock(enabled=True, enable_diagnostics=False),
    )
    monkeypatch.setattr(dyntra_lb_core, "_PreviousRunEngineCore", balance_wrapper)
    monkeypatch.setattr(dyntra_lb_core, "_UpstreamRunEngineCore", upstream_entrypoint)

    result = dyntra_lb_core._dyntra_lb_run_engine_core(vllm_config=object())

    assert result is expected
    balance_wrapper.assert_not_called()
    assert dyntra_lb_core._engine_core_mod.DPEngineCoreProc is dyntra_lb_core._OriginalDPEngineCoreProc


def test_dyntra_lb_enabled_reads_nested_scheduler_config(monkeypatch):
    monkeypatch.setattr(
        dyntra_lb_core,
        "get_ascend_config",
        MagicMock(side_effect=RuntimeError("not initialized")),
    )
    vllm_config = MagicMock()
    vllm_config.additional_config = {
        "scheduler_config": {
            "dyntra_lb_config": {
                "enabled": True,
            }
        }
    }

    assert dyntra_lb_core._dyntra_lb_enabled(vllm_config) is True


def test_dyntra_lb_diagnostics_read_nested_scheduler_config(monkeypatch):
    monkeypatch.setattr(
        dyntra_lb_core,
        "get_ascend_config",
        MagicMock(side_effect=RuntimeError("not initialized")),
    )
    vllm_config = MagicMock()
    vllm_config.additional_config = {
        "scheduler_config": {
            "dyntra_lb_config": {
                "enabled": True,
                "enable_diagnostics": True,
            }
        }
    }

    assert dyntra_lb_core._get_dyntra_lb_config(vllm_config).enable_diagnostics is True


def test_dyntra_lb_enabled_ignores_top_level_config(monkeypatch):
    monkeypatch.setattr(
        dyntra_lb_core,
        "get_ascend_config",
        MagicMock(side_effect=RuntimeError("not initialized")),
    )
    vllm_config = MagicMock()
    vllm_config.additional_config = {"DYNTRA_LB_ENABLE": 1}

    assert dyntra_lb_core._dyntra_lb_enabled(vllm_config) is False


def _diagnostics_config(enable_diagnostics: bool):
    vllm_config = MagicMock()
    vllm_config.additional_config = {
        "scheduler_config": {
            "dyntra_lb_config": {
                "enabled": False,
                "enable_diagnostics": enable_diagnostics,
            }
        }
    }
    return vllm_config


def test_dyntra_lb_does_not_patch_default_dp_core_diagnostics_method():
    assert (
        dyntra_lb_core.DPEngineCoreProc._has_global_unfinished_reqs
        is dyntra_lb_core._ORIGINAL_HAS_GLOBAL_UNFINISHED_REQS
    )
    assert AsyncRecomputeScheduler.schedule is RecomputeScheduler.schedule


def test_default_dp_core_diagnostics_are_disabled_by_default(monkeypatch):
    log_info = MagicMock()
    monkeypatch.setattr(dyntra_lb_core.logger, "info", log_info)
    monkeypatch.setattr(
        dyntra_lb_core,
        "_ORIGINAL_HAS_GLOBAL_UNFINISHED_REQS",
        lambda self, local_unfinished: True,
    )
    engine_core = MagicMock()
    engine_core.vllm_config = _diagnostics_config(False)
    engine_core.step_counter = 7

    assert dyntra_lb_core._has_global_unfinished_reqs_with_diagnostics(engine_core, True) is True
    log_info.assert_not_called()


def test_bl_dp_core_diagnostics_can_be_enabled(monkeypatch):
    messages = []
    monkeypatch.setattr(
        dyntra_lb_core.logger,
        "info",
        lambda message, *args: messages.append(message % args if args else message),
    )
    monkeypatch.setattr(
        dyntra_lb_core,
        "_ORIGINAL_HAS_GLOBAL_UNFINISHED_REQS",
        lambda self, local_unfinished: True,
    )
    engine_core = MagicMock()
    engine_core.vllm_config = _diagnostics_config(True)
    engine_core.step_counter = 7

    assert dyntra_lb_core._has_global_unfinished_reqs_with_diagnostics(engine_core, True) is True
    output = "\n".join(messages)
    assert output.count("step_counter: 7") == 1


def test_dp_engine_core_initializes_ascend_config(monkeypatch):
    messages = []
    monkeypatch.setattr(
        dyntra_lb_core.logger,
        "info",
        lambda message, *args: messages.append(message % args if args else message),
    )
    vllm_config = MagicMock()
    vllm_config.scheduler_config.max_num_seqs = 4

    ascend_config = MagicMock()
    dyntra_lb_config = ascend_config.scheduler_config.dyntra_lb_config
    dyntra_lb_config.enabled = True
    dyntra_lb_config.enable_diagnostics = True
    dyntra_lb_config.mode = "static"
    dyntra_lb_config.start_step = 0
    dyntra_lb_config.end_step = -1
    dyntra_lb_config.bubble_threshold = 5.0
    dyntra_lb_config.long_req_block_threshold = 700
    dyntra_lb_config.dynamic_max_step = 256

    init_ascend_config = MagicMock(return_value=ascend_config)
    monkeypatch.setattr(dyntra_lb_core, "init_ascend_config", init_ascend_config)

    def init_data_parallel(engine_core, _config):
        engine_core.dp_group = MagicMock()
        engine_core.dp_rank = 0

    monkeypatch.setattr(
        dyntra_lb_core.DPEngineCoreProc,
        "_init_data_parallel",
        init_data_parallel,
    )
    monkeypatch.setattr(dyntra_lb_core.dist, "get_world_size", lambda group: 2)

    engine_core = object.__new__(dyntra_lb_core.DyntraLBDPEngineCoreProc)
    engine_core._init_data_parallel(vllm_config)

    init_ascend_config.assert_called_once_with(vllm_config)
    assert engine_core._lb_mode == "static"
    assert engine_core._lb_dp_size_cached == 2

    output = "\n".join(messages)
    assert "dyntra_lb_config.enabled = True" in output
    assert "dyntra_lb_config.enable_diagnostics = True" in output
    assert "dyntra_lb_config.mode = static" in output
    assert "dyntra_lb_config.start_step = 0" in output
    assert "dyntra_lb_config.end_step = -1" in output
    assert "dyntra_lb_config.bubble_threshold = 5.0" in output
    assert "dyntra_lb_config.long_req_block_threshold = 700" in output
    assert "dyntra_lb_config.dynamic_max_step = 256" in output


def test_balance_load_runs_after_global_unfinished_sync(monkeypatch):
    events: list[object] = []

    def has_global_unfinished(engine_core, local_unfinished):
        events.append(("global_unfinished", local_unfinished))
        engine_core.step_counter += 1
        return True

    monkeypatch.setattr(
        dyntra_lb_core,
        "_ORIGINAL_HAS_GLOBAL_UNFINISHED_REQS",
        has_global_unfinished,
    )

    engine_core = object.__new__(dyntra_lb_core.DyntraLBDPEngineCoreProc)
    engine_core.vllm_config = _diagnostics_config(False)
    engine_core.step_counter = 7
    engine_core.scheduler = MagicMock()
    engine_core.run_balance_load = lambda: events.append(("balance_load", engine_core.step_counter))

    assert engine_core._has_global_unfinished_reqs(False) is True
    assert events == [
        ("global_unfinished", False),
        ("balance_load", 8),
    ]


def test_run_balance_load_prepares_snapshot_before_allgather():
    events: list[object] = []
    candidates = [MagicMock(request_id="candidate")]
    scheduler = MagicMock()

    def prepare_dyntra_lb_step():
        events.append("prepare")
        return candidates

    scheduler.prepare_dyntra_lb_step.side_effect = prepare_dyntra_lb_step

    engine_core = object.__new__(dyntra_lb_core.DyntraLBDPEngineCoreProc)
    engine_core.scheduler = scheduler
    engine_core._lb_mode = "static"
    engine_core._lb_dynamic_enable = False
    engine_core.step_counter = 0
    engine_core._lb_start_step = 0
    engine_core._lb_end_step = -1

    def do_lb_allgather(snapshot):
        events.append(("allgather", snapshot))
        return False

    engine_core._do_lb_allgather = do_lb_allgather

    engine_core.run_balance_load()

    assert events == ["prepare", ("allgather", candidates)]


def test_dynamic_activation_prepares_next_step_plan_immediately(monkeypatch):
    engine_core = object.__new__(dyntra_lb_core.DyntraLBDPEngineCoreProc)
    engine_core.scheduler = MagicMock()
    engine_core.scheduler.prepare_dyntra_lb_step.return_value = []
    engine_core._lb_mode = "dynamic"
    engine_core._lb_dynamic_enable = False
    engine_core._lb_dynamic_step = 9
    engine_core._lb_pending_long_req = True
    engine_core._lb_pending_long_req_blk = 8
    engine_core._lb_dynamic_flag_np = np.zeros(2, dtype=np.int32)
    engine_core._lb_dynamic_flag_t = torch.as_tensor(engine_core._lb_dynamic_flag_np)
    engine_core.step_counter = 1
    engine_core._lb_start_step = 0
    engine_core._lb_end_step = -1
    engine_core._lb_long_req_threshold = 4
    engine_core.current_wave = 0
    engine_core.dp_rank = 0
    engine_core.dp_group = MagicMock()
    engine_core._lb_enable_diagnostics = False
    engine_core._do_lb_allgather = MagicMock(return_value=False)
    monkeypatch.setattr(dyntra_lb_core.dist, "all_reduce", lambda *args, **kwargs: None)

    engine_core.run_balance_load()

    assert engine_core._lb_dynamic_enable is True
    assert engine_core._lb_dynamic_step == 0
    assert engine_core.scheduler._lb_kv_prefetch_enabled is True
    engine_core._do_lb_allgather.assert_called_once_with([])


def test_dynamic_long_request_uses_effective_attention_block_size(monkeypatch):
    added_requests = []
    monkeypatch.setattr(
        dyntra_lb_core.DPEngineCoreProc,
        "add_request",
        lambda _self, request, request_wave=0: added_requests.append((request, request_wave)),
    )

    request = MagicMock(
        request_id="long-request",
        all_token_ids=list(range(8193)),
    )
    engine_core = object.__new__(dyntra_lb_core.DyntraLBDPEngineCoreProc)
    engine_core.scheduler = MagicMock()
    engine_core.scheduler.block_size = 14080000
    engine_core.scheduler.cache_config.block_size = 2048
    engine_core._lb_mode = "dynamic"
    engine_core._lb_long_req_threshold = 4
    engine_core._lb_pending_long_req = False
    engine_core._lb_pending_long_req_blk = 0

    engine_core.add_request(request, request_wave=3)

    assert engine_core._lb_pending_long_req is True
    assert engine_core._lb_pending_long_req_blk == 5
    assert added_requests == [(request, 3)]


def test_dyntra_lb_diagnostics_are_disabled_by_default(monkeypatch):
    log_info = MagicMock()
    monkeypatch.setattr(dyntra_lb_core.logger, "info", log_info)
    monkeypatch.setattr(
        dyntra_lb_core,
        "_ORIGINAL_HAS_GLOBAL_UNFINISHED_REQS",
        lambda self, local_unfinished: False,
    )

    engine_core = object.__new__(dyntra_lb_core.DyntraLBDPEngineCoreProc)
    engine_core.vllm_config = _diagnostics_config(False)
    engine_core._lb_enable_diagnostics = False
    engine_core.step_counter = 7
    engine_core.scheduler = MagicMock()
    engine_core.scheduler.modifications = {"freeze": True}
    engine_core.scheduler.lb_freeze = True
    engine_core.scheduler._lb_kv_prefetch_enabled = True
    engine_core.run_balance_load = MagicMock()

    assert engine_core._has_global_unfinished_reqs(True) is False
    log_info.assert_not_called()
    assert engine_core.scheduler.modifications is None
    assert engine_core.scheduler.lb_freeze is False
    assert engine_core.scheduler._lb_kv_prefetch_enabled is False
    engine_core.run_balance_load.assert_not_called()


def test_dyntra_lb_diagnostics_print_step_counter_once(monkeypatch):
    messages = []
    monkeypatch.setattr(
        dyntra_lb_core.logger,
        "info",
        lambda message, *args: messages.append(message % args if args else message),
    )
    monkeypatch.setattr(
        dyntra_lb_core,
        "_ORIGINAL_HAS_GLOBAL_UNFINISHED_REQS",
        lambda self, local_unfinished: True,
    )

    engine_core = object.__new__(dyntra_lb_core.DyntraLBDPEngineCoreProc)
    engine_core.vllm_config = _diagnostics_config(True)
    engine_core._lb_enable_diagnostics = True
    engine_core.step_counter = 7
    engine_core.scheduler = MagicMock()
    engine_core.run_balance_load = MagicMock()

    assert engine_core._has_global_unfinished_reqs(True) is True
    output = "\n".join(messages)
    assert output.count("step_counter: 7") == 1
    engine_core.run_balance_load.assert_called_once_with()


def test_print_balance_summary(monkeypatch):
    messages = []
    monkeypatch.setattr(
        dyntra_lb_core.logger,
        "info",
        lambda message, *args: messages.append(message % args if args else message),
    )
    dyntra_lb_core._print_requests_by_rank(
        [([8, 4, 2, 0], 2)],
        dp_rank=0,
        enable_diagnostics=True,
    )
    dyntra_lb_core._print_modifications(
        [{"out_blk": [8], "in_blk": [2], "freeze": False}],
        dp_rank=0,
        enable_diagnostics=True,
    )

    output = "\n".join(messages)
    assert "DP0 | Run(2): [  8,   4]" in output
    assert "Wait(1): [  2]" in output
    assert "Out: [  8]" in output
    assert "In: [  2]" in output
    assert "Freeze: False" in output


def test_balance_load_threshold_one_is_one_absolute_block():
    modifications = dyntra_lb_core.DyntraLBDPEngineCoreProc._balance_load(
        [
            ([3], 1),
            ([1, 2], 1),
        ],
        dev_num=2,
        threshold=1.0,
    )

    assert modifications[1]["out_blk"] == [1]
    assert modifications[1]["in_blk"] == [2]


def test_balance_load_threshold_below_one_is_normalized_ratio():
    modifications = dyntra_lb_core.DyntraLBDPEngineCoreProc._balance_load(
        [
            ([3], 1),
            ([1, 2], 1),
        ],
        dev_num=2,
        threshold=0.5,
    )

    assert modifications == [
        {"out_blk": [], "in_blk": [], "freeze": True},
        {"out_blk": [], "in_blk": [], "freeze": True},
    ]


def test_balance_load_does_not_drop_when_fixed_latency_dominates():
    modifications = dyntra_lb_core.DyntraLBDPEngineCoreProc._balance_load(
        [
            ([60, 40], 2),
            ([60], 1),
        ],
        dev_num=2,
        threshold=5.0,
    )

    assert modifications == [
        {"out_blk": [], "in_blk": [], "freeze": True},
        {"out_blk": [], "in_blk": [], "freeze": True},
    ]


def test_balance_load_drops_request_for_modeled_throughput_gain():
    modifications = dyntra_lb_core.DyntraLBDPEngineCoreProc._balance_load(
        [
            ([4000, 1000], 2),
            ([1000], 1),
        ],
        dev_num=2,
        threshold=1.0,
    )

    assert modifications[0]["out_blk"] == [4000]


def test_balance_summary_is_suppressed_on_nonzero_dp_rank(monkeypatch):
    messages = []
    monkeypatch.setattr(
        dyntra_lb_core.logger,
        "info",
        lambda message, *args: messages.append(message % args if args else message),
    )
    dyntra_lb_core._print_requests_by_rank(
        [([8, 4, 2, 0], 2)],
        dp_rank=1,
        enable_diagnostics=True,
    )
    dyntra_lb_core._print_modifications(
        [{"out_blk": [8], "in_blk": [2], "freeze": False}],
        dp_rank=1,
        enable_diagnostics=True,
    )

    assert messages == []


def test_balance_summary_is_suppressed_when_diagnostics_are_disabled(monkeypatch):
    messages = []
    monkeypatch.setattr(
        dyntra_lb_core.logger,
        "info",
        lambda message, *args: messages.append(message % args if args else message),
    )
    dyntra_lb_core._print_requests_by_rank(
        [([8, 4, 2, 0], 2)],
        dp_rank=0,
        enable_diagnostics=False,
    )
    dyntra_lb_core._print_modifications(
        [{"out_blk": [8], "in_blk": [2], "freeze": False}],
        dp_rank=0,
        enable_diagnostics=False,
    )

    assert messages == []


def test_dyntra_lb_allgather_uses_admission_candidate_snapshot(monkeypatch):
    waiting_request = MagicMock(
        request_id="waiting",
        all_token_ids=list(range(17)),
        status=RequestStatus.WAITING,
    )
    paused_request = MagicMock(
        request_id="paused",
        all_token_ids=list(range(33)),
        status=RequestStatus.PREEMPTED,
    )
    skipped_request = MagicMock(
        request_id="skipped",
        all_token_ids=list(range(65)),
        status=RequestStatus.WAITING,
    )
    scheduler = MagicMock()
    scheduler.block_size = 14080000
    scheduler.cache_config.block_size = 8
    scheduler.running = []
    scheduler._lb_admission_candidates = [
        skipped_request,
        waiting_request,
        paused_request,
    ]

    engine_core = object.__new__(dyntra_lb_core.DyntraLBDPEngineCoreProc)
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
    gathered: np.ndarray = np.zeros(6, dtype=np.int32)
    engine_core._lb_all_data_np = [gathered]
    engine_core._lb_all_data_t_buf = [torch.as_tensor(gathered)]

    def all_gather(output_tensors, input_tensor, group):
        output_tensors[0].copy_(input_tensor)

    captured = {}

    def balance_load(requests_by_rank, dev_num, max_num_seqs, threshold):
        captured["requests_by_rank"] = requests_by_rank
        return [{"out_blk": [], "in_blk": [], "freeze": True}]

    monkeypatch.setattr(dyntra_lb_core.dist, "all_gather", all_gather)
    monkeypatch.setattr(
        dyntra_lb_core.DyntraLBDPEngineCoreProc,
        "_balance_load",
        staticmethod(balance_load),
    )
    monkeypatch.setattr(dyntra_lb_core, "_print_requests_by_rank", lambda *args: None)
    monkeypatch.setattr(dyntra_lb_core, "_print_modifications", lambda *args: None)

    engine_core._do_lb_allgather(scheduler._lb_admission_candidates)

    assert captured["requests_by_rank"] == [([9, 3, 5, 0], 0)]
