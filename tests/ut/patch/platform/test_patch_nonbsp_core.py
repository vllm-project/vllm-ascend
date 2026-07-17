from unittest.mock import MagicMock

import numpy as np
import torch
import vllm_ascend.patch.platform.patch_nonbsp_core as nonbsp_core  # noqa: I001
from vllm.v1.request import RequestStatus


def test_dp_engine_core_initializes_ascend_config(monkeypatch, capsys):
    vllm_config = MagicMock()
    vllm_config.scheduler_config.max_num_seqs = 4

    ascend_config = MagicMock()
    ascend_config.NONBSP_ENABLE = 1
    ascend_config.NONBSP_START_STEP = 0
    ascend_config.NONBSP_END_STEP = -1
    ascend_config.NONBSP_BUBBLE_THRESHOLD = 5.0
    ascend_config.NONBSP_LONG_REQ_BLOCK_THRESHOLD = 700
    ascend_config.NONBSP_DYNAMIC_MAX_STEP = 256

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
    assert engine_core._lb_enable == 1
    assert engine_core._lb_dp_size_cached == 2

    output = capsys.readouterr().out
    assert "NONBSP_ENABLE = 1" in output
    assert "NONBSP_START_STEP = 0" in output
    assert "NONBSP_END_STEP = -1" in output
    assert "NONBSP_BUBBLE_THRESHOLD = 5.0" in output
    assert "NONBSP_LONG_REQ_BLOCK_THRESHOLD = 700" in output
    assert "NONBSP_DYNAMIC_MAX_STEP = 256" in output


def test_balance_load_runs_before_engine_step(monkeypatch):
    events = []

    monkeypatch.setattr(
        nonbsp_core.DPEngineCoreProc,
        "_process_engine_step",
        lambda self: events.append("engine_step") or True,
    )

    engine_core = object.__new__(nonbsp_core.NonBSPDPEngineCoreProc)
    engine_core.run_balance_load = lambda: events.append("balance_load")

    assert engine_core._process_engine_step() is True
    assert events == ["balance_load", "engine_step"]


def test_dp_core_prints_step_counter_without_nonbsp(monkeypatch, capsys):
    monkeypatch.setattr(
        nonbsp_core,
        "_ORIGINAL_HAS_GLOBAL_UNFINISHED_REQS",
        lambda self, local_unfinished: True,
    )

    engine_core = object.__new__(nonbsp_core.DPEngineCoreProc)
    engine_core.step_counter = 7

    assert engine_core._has_global_unfinished_reqs(True) is True
    output = capsys.readouterr().out
    assert output.count("step_counter: 7") == 1


def test_nonbsp_dp_core_does_not_duplicate_step_counter(monkeypatch, capsys):
    monkeypatch.setattr(
        nonbsp_core,
        "_ORIGINAL_HAS_GLOBAL_UNFINISHED_REQS",
        lambda self, local_unfinished: True,
    )

    engine_core = object.__new__(nonbsp_core.NonBSPDPEngineCoreProc)
    engine_core.step_counter = 7
    engine_core.scheduler = MagicMock()

    assert engine_core._has_global_unfinished_reqs(True) is True
    output = capsys.readouterr().out
    assert output.count("step_counter: 7") == 1


def test_print_balance_summary(capsys):
    nonbsp_core._print_requests_by_rank([([8, 4, 2, 0], 2)], dp_rank=0)
    nonbsp_core._print_modifications(
        [{"out_blk": [8], "in_blk": [2], "freeze": False}], dp_rank=0
    )

    output = capsys.readouterr().out
    assert "DP0 | Run(2): [  8,   4]" in output
    assert "Wait(1): [  2]" in output
    assert "Out: [  8]" in output
    assert "In: [  2]" in output
    assert "Freeze: False" in output


def test_balance_summary_is_suppressed_on_nonzero_dp_rank(capsys):
    nonbsp_core._print_requests_by_rank([([8, 4, 2, 0], 2)], dp_rank=1)
    nonbsp_core._print_modifications(
        [{"out_blk": [8], "in_blk": [2], "freeze": False}], dp_rank=1
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
