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

"""Extra DFX unit tests aimed at coverage gaps (core / processor / tokenizer)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from tests.ut.dfx_test_utils import make_dfx_config
from vllm_ascend.dfx.detector.alert import AnomalyAlert
from vllm_ascend.dfx.dfx_types import DumpPhase
from vllm_ascend.dfx.dumper import Dumper
from vllm_ascend.dfx.manual_trigger import ManualTriggerManager, TriggerEvent
from vllm_ascend.dfx.processor import DfxProcessor
from vllm_ascend.dfx.tokenizer import load_model_tokenizer


def _runner_for_dumper(*, dump_config_path=None, cudagraph_none=True):
    from vllm.config.compilation import CUDAGraphMode

    mode = CUDAGraphMode.NONE if cudagraph_none else CUDAGraphMode.FULL
    return SimpleNamespace(
        ascend_config=SimpleNamespace(dump_config_path=dump_config_path),
        compilation_config=SimpleNamespace(cudagraph_mode=mode),
        tp_rank=0,
        dp_rank=1,
        model=MagicMock(),
        input_batch=None,
        requests=None,
        req_states=None,
        discard_request_mask=None,
        use_async_scheduling=False,
    )


def test_load_model_tokenizer_paths():
    assert load_model_tokenizer(None) is None
    assert load_model_tokenizer(SimpleNamespace()) is None
    assert load_model_tokenizer(SimpleNamespace(vllm_config=SimpleNamespace(model_config=None))) is None

    tok = object()
    with patch("vllm.tokenizers.cached_tokenizer_from_config", return_value=tok) as cached:
        out = load_model_tokenizer(SimpleNamespace(vllm_config=SimpleNamespace(model_config=object())))
    assert out is tok
    cached.assert_called_once()


def test_dumper_init_without_dump_config(tmp_path):
    cfg = make_dfx_config(tmp_path)
    runner = _runner_for_dumper(dump_config_path=None)
    dumper = Dumper(runner, dfx_config=cfg)
    assert dumper._debugger is None
    assert dumper.dump_phase == DumpPhase.IDLE
    assert "dp=1" in dumper.dump_rank_tag()
    dumper.apply_dfx_config()  # unchanged limits
    cfg._data["dump"]["max_times"] = 3
    dumper.apply_dfx_config()  # changed limits path
    assert dumper._dump_max_times == 3
    assert dumper.can_run_anomaly_detection() is False  # no detector enabled
    assert "no detector" in (dumper.anomaly_check_skip_reason() or "")


def test_dumper_init_precision_debugger(tmp_path):
    cfg = make_dfx_config(tmp_path)
    runner = _runner_for_dumper(dump_config_path="/tmp/msprobe.json")
    fake_dbg = MagicMock()
    fake_mod = SimpleNamespace(PrecisionDebugger=MagicMock(return_value=fake_dbg))
    with patch.dict("sys.modules", {"msprobe": MagicMock(), "msprobe.pytorch": fake_mod}):
        dumper = Dumper(runner, dfx_config=cfg)
    assert dumper._debugger is fake_dbg
    assert dumper._uses_aclgraph_dumper is False


def test_handle_anomaly_alert_rejects_bad_alerts(tmp_path):
    cfg = make_dfx_config(tmp_path)
    dumper = Dumper(_runner_for_dumper(), dfx_config=cfg)
    assert dumper.handle_anomaly_alert(None) is False
    assert dumper.handle_anomaly_alert(AnomalyAlert(anomaly_type="x", req_id="", is_ill=True)) is False
    assert dumper.handle_anomaly_alert(AnomalyAlert(anomaly_type="x", req_id="r", is_ill=False)) is False
    assert dumper.handle_manual_trigger(None) is False
    assert dumper.handle_manual_trigger(TriggerEvent(trigger_type="manual_trigger", req_id="")) is False


def test_is_related_local_request_batch_idx_path(tmp_path):
    cfg = make_dfx_config(tmp_path)
    dumper = Dumper(_runner_for_dumper(), dfx_config=cfg)
    dumper.runner.input_batch = SimpleNamespace(req_ids=["a", "b"], num_reqs=2, req_id_to_index=None)
    dumper.runner.requests = {"a": object(), "b": object()}
    dumper.runner.req_states = SimpleNamespace(req_id_to_index={"a": 0, "b": 1})
    dumper.runner.discard_request_mask = SimpleNamespace(np=[False, True])

    assert dumper.is_related_local_request("a", 0) is True
    assert dumper.is_related_local_request("a", 1) is False
    assert dumper.is_related_local_request("missing", 0) is False
    assert dumper.is_related_local_request("b", 1) is False  # discarded

    # Fallback map path (no req_idx).
    dumper.runner.input_batch = SimpleNamespace(
        req_ids=["a"],
        num_reqs=1,
        req_id_to_index={"a": 0},
    )
    dumper.runner.discard_request_mask = None
    assert dumper.is_related_local_request("a", None) is True
    assert dumper.is_related_local_request("z", None) is False
    assert dumper.is_related_local_request("a", 5) is False  # idx mismatch


def test_anomaly_check_skip_when_dump_pending(tmp_path):
    cfg = make_dfx_config(tmp_path)
    dumper = Dumper(_runner_for_dumper(), dfx_config=cfg)
    # Mutate after Dumper init: enforce may persist dump.enabled=false and
    # save() reloads disk, which would wipe in-memory-only detector flags.
    cfg._data["detector"]["spec_acceptance"]["enabled"] = True
    cfg._data["dump"]["enabled"] = True
    with patch("vllm_ascend.dfx.dumper.core.anomaly_check_rank_skip_reason", return_value=None):
        dumper._pending_dump = True
        assert "pending_dump" in (dumper.anomaly_check_skip_reason() or "")
        dumper._pending_dump = False
        dumper._msprobe_dump_active = True
        assert "already active" in (dumper.anomaly_check_skip_reason() or "")


def test_manual_trigger_manager_paths(tmp_path):
    cfg = make_dfx_config(tmp_path)
    cfg._data["dump"]["enabled"] = True
    cfg._data["dump"]["manual_trigger"] = True
    runner = SimpleNamespace(
        tp_rank=0,
        use_async_scheduling=False,
        input_batch=SimpleNamespace(req_ids=["r1"]),
    )
    mgr = ManualTriggerManager(dfx_config=cfg, runner=runner)

    assert mgr.consume_once(allow_arm=False) is None
    assert cfg.manual_trigger() is True

    cfg._data["dump"]["enabled"] = False
    assert mgr.consume_once(allow_arm=True) is None

    # Empty batch must not burn remaining count.
    cfg._data["dump"]["enabled"] = True
    cfg._data["dump"]["manual_trigger"] = 2
    empty_mgr = ManualTriggerManager(
        dfx_config=cfg,
        runner=SimpleNamespace(tp_rank=0, use_async_scheduling=False, input_batch=SimpleNamespace(req_ids=[])),
    )
    assert empty_mgr.consume_once(allow_arm=True) is None
    assert cfg.manual_trigger_count() == 2

    cfg._data["dump"]["enabled"] = True
    cfg._data["dump"]["manual_trigger"] = True
    with patch(
        "vllm_ascend.dfx.manual_trigger.should_run_anomaly_check_on_rank",
        return_value=False,
    ):
        # Continuous true: consume path still runs; value stays true.
        assert mgr.consume_once(allow_arm=True) is None
    assert cfg.manual_trigger() is True
    assert cfg.manual_trigger_continuous() is True

    # Int count: consume even when this rank does not arm.
    cfg._data["dump"]["manual_trigger"] = 1
    with patch(
        "vllm_ascend.dfx.manual_trigger.should_run_anomaly_check_on_rank",
        return_value=False,
    ):
        assert mgr.consume_once(allow_arm=True) is None
    assert cfg.manual_trigger() is False

    cfg._data["dump"]["manual_trigger"] = 2
    cfg._data["dump"]["enabled"] = True
    with patch(
        "vllm_ascend.dfx.manual_trigger.should_run_anomaly_check_on_rank",
        return_value=True,
    ):
        ev = mgr.consume_once(allow_arm=True)
    assert ev is not None
    assert ev.trigger_type == "manual_trigger"
    assert ev.to_report_detail()["source"] == "dump.manual_trigger"
    assert cfg.manual_trigger_count() == 1
    assert ev.detail["manual_trigger_remaining_after"] == 1


def test_manual_trigger_v2_req_states_and_scheduler_output(tmp_path):
    """MRV2 has no input_batch; use req_states / scheduler_output instead."""
    from vllm_ascend.dfx.manual_trigger import iter_local_request_rows

    cfg = make_dfx_config(tmp_path)
    cfg._data["dump"]["enabled"] = True
    cfg._data["dump"]["manual_trigger"] = 2

    empty = SimpleNamespace(tp_rank=0, input_batch=None, requests=None, req_states=None)
    assert iter_local_request_rows(empty) == []
    assert ManualTriggerManager(dfx_config=cfg, runner=empty).consume_once(allow_arm=True) is None
    assert cfg.manual_trigger_count() == 2

    v2_runner = SimpleNamespace(
        tp_rank=0,
        input_batch=None,
        requests=None,
        req_states=SimpleNamespace(req_id_to_index={"r_v2": 3}),
    )
    assert iter_local_request_rows(v2_runner) == [("r_v2", 3)]
    with patch(
        "vllm_ascend.dfx.manual_trigger.should_run_anomaly_check_on_rank",
        return_value=True,
    ):
        ev = ManualTriggerManager(dfx_config=cfg, runner=v2_runner).consume_once(allow_arm=True)
    assert ev is not None
    assert cfg.manual_trigger_count() == 1

    cfg._data["dump"]["enabled"] = True
    cfg._data["dump"]["manual_trigger"] = 1
    first_wave = SimpleNamespace(tp_rank=0, input_batch=None, requests=None, req_states=None)
    so = SimpleNamespace(num_scheduled_tokens={"new_req": 11})
    assert iter_local_request_rows(first_wave, so) == [("new_req", -1)]
    with patch(
        "vllm_ascend.dfx.manual_trigger.should_run_anomaly_check_on_rank",
        return_value=True,
    ):
        ev2 = ManualTriggerManager(dfx_config=cfg, runner=first_wave).consume_once(allow_arm=True, scheduler_output=so)
    assert ev2 is not None
    assert cfg.manual_trigger_count() == 0


def test_processor_get_tokenizer_and_save_sample_param(tmp_path):
    proc = DfxProcessor.__new__(DfxProcessor)
    proc.dfx_config = MagicMock()
    proc.dfx_config.report_save_sensitive_info.return_value = True
    proc.dfx_config.report_decode_token_ids.return_value = True
    proc._report_tokenizer = None
    proc._report_tokenizer_failed = False
    proc.runner = SimpleNamespace(tp_rank=0, dp_rank=0)

    with patch("vllm_ascend.dfx.processor.load_model_tokenizer", return_value=None):
        assert proc._get_detector_tokenizer() is None

    with patch("vllm_ascend.dfx.processor.load_model_tokenizer", side_effect=RuntimeError("boom")):
        assert proc._get_detector_tokenizer() is None
        assert proc._report_tokenizer_failed is True

    proc._report_tokenizer_failed = False
    fake_tok = object()
    with patch("vllm_ascend.dfx.processor.load_model_tokenizer", return_value=fake_tok):
        assert proc._get_detector_tokenizer() is fake_tok
        assert proc._get_report_tokenizer() is fake_tok

    proc.dfx_config.report_save_sensitive_info.return_value = False
    assert proc._get_report_tokenizer() is None

    # save_sample_param: non-tp0 skip
    proc.runner = SimpleNamespace(tp_rank=1, dp_rank=0, input_batch=MagicMock())
    with patch("vllm_ascend.dfx.processor.get_pp_group") as pp:
        pp.return_value.is_last_rank = True
        proc.save_sample_param("r1")

    sm = SimpleNamespace(
        temperature=torch.tensor([0.7]),
        top_k=torch.tensor([50]),
        top_p=torch.tensor([0.9]),
        frequency_penalties=torch.tensor([0.0]),
        presence_penalties=torch.tensor([0.0]),
        repetition_penalties=torch.tensor([1.0]),
        bad_words_token_ids={0: []},
        output_token_ids=[[1, 2]],
        spec_token_ids=None,
        logprob_token_ids=None,
        all_greedy=False,
        all_random=True,
        max_num_logprobs=None,
    )
    proc.runner = SimpleNamespace(
        tp_rank=0,
        dp_rank=2,
        input_batch=SimpleNamespace(req_ids=["r1"], sampling_metadata=sm),
    )
    with patch("vllm_ascend.dfx.processor.get_pp_group") as pp:
        pp.return_value.is_last_rank = True
        proc.save_sample_param("r1")
        proc.save_sample_param("missing")


def test_processor_ensure_logprobs_v1_and_v2(tmp_path):
    proc = DfxProcessor.__new__(DfxProcessor)
    proc.detectors = MagicMock()
    proc.detectors.token_logprob_topk_if_enabled.return_value = None
    proc.ensure_logprobs_for_detection()  # no-op

    proc.detectors.token_logprob_topk_if_enabled.return_value = 8
    num_logprobs = {"r1": None, "r2": 2, "r3": 20}
    input_batch = MagicMock()
    input_batch.num_logprobs = num_logprobs
    input_batch.req_ids = ["r1", "r2", "r3"]
    input_batch._make_sampling_metadata = MagicMock(return_value="meta")
    proc.runner = SimpleNamespace(input_batch=input_batch, sampler=None)
    proc.ensure_logprobs_for_detection()
    assert num_logprobs["r1"] == 8
    assert num_logprobs["r2"] == 8
    assert num_logprobs["r3"] == 20
    assert input_batch.sampling_metadata == "meta"

    # v2 path
    states = SimpleNamespace(num_logprobs=[-1, 1, 16])
    sampler = SimpleNamespace(sampling_states=states)
    input_batch2 = SimpleNamespace(
        num_logprobs=None,
        idx_mapping_np=[0, 1, 2],
        num_reqs=3,
    )
    proc.runner = SimpleNamespace(input_batch=input_batch2, sampler=sampler)
    proc.ensure_logprobs_for_detection()
    assert list(states.num_logprobs) == [8, 8, 16]


def test_processor_construct_wires_components(tmp_path):
    cfg = make_dfx_config(tmp_path)
    runner = SimpleNamespace(
        ascend_config=SimpleNamespace(dfx_config=cfg, dump_config_path=None),
        compilation_config=SimpleNamespace(
            cudagraph_mode=__import__("vllm.config.compilation", fromlist=["CUDAGraphMode"]).CUDAGraphMode.NONE
        ),
        tp_rank=0,
        dp_rank=0,
        model=MagicMock(),
        input_batch=None,
        use_async_scheduling=False,
    )
    with patch.object(Dumper, "_init_debugger", return_value=None):
        proc = DfxProcessor(runner)
    assert proc.dumper is not None
    assert proc.manual_triggers is not None
    assert proc.detectors is not None
    proc.sync_for_step(allow_arm=False)


def test_detector_manager_record_spec_step_outputs():
    from vllm_ascend.dfx.detector.manager import DetectorManager
    from vllm_ascend.dfx.io_snapshot import RequestIoSnapshotManager

    RequestIoSnapshotManager.reset_for_tests()
    mgr = DetectorManager.__new__(DetectorManager)
    mgr._runner = SimpleNamespace(input_batch=SimpleNamespace(req_ids=["r1", "r2"]))
    sampled = torch.tensor([[10, 11, 12], [20, 21, 22]])
    accepted = torch.tensor([2, 0])
    mgr._record_spec_step_outputs(sampled, accepted)
    io = RequestIoSnapshotManager.get()
    assert io.cumulative_output_ids("r1") == [10, 11]
    assert io.cumulative_output_ids("r2") == []
    mgr._record_spec_step_outputs(None, accepted)
    RequestIoSnapshotManager.reset_for_tests()


def test_pending_activate_and_clear(tmp_path):
    cfg = make_dfx_config(tmp_path)
    dumper = Dumper(_runner_for_dumper(dump_config_path=None), dfx_config=cfg)
    dumper._debugger = MagicMock()
    dumper._msprobe_dump_active = False
    dumper.set_msprobe_dump_state = MagicMock(return_value=True)
    assert dumper._activate_msprobe_dump("r1", consume_quota=True) is True
    assert dumper._msprobe_dump_active is True
    assert dumper._msprobe_dump_total_count == 1
    assert dumper._activate_msprobe_dump("r2", consume_quota=False) is True  # already active
    dumper._clear_pending_dump()
    assert dumper._pending_dump is False


def test_enable_msprobe_gates(tmp_path):
    cfg = make_dfx_config(tmp_path)
    cfg._data["dump"]["enabled"] = True
    cfg._data["dump"]["max_times"] = 1
    dumper = Dumper(_runner_for_dumper(), dfx_config=cfg)
    dumper._debugger = None
    assert dumper.enable_msprobe_dump_if_needed("r1", skip_related_check=True) is False

    dumper._debugger = MagicMock()
    cfg._data["dump"]["enabled"] = False
    assert dumper.enable_msprobe_dump_if_needed("r1", skip_related_check=True) is False

    cfg._data["dump"]["enabled"] = True
    with patch("vllm_ascend.dfx.dumper.pending.get_pp_group") as pp:
        pp.return_value.is_last_rank = False
        assert dumper.enable_msprobe_dump_if_needed("r1", skip_related_check=True) is False

    with patch("vllm_ascend.dfx.dumper.pending.get_pp_group") as pp:
        pp.return_value.is_last_rank = True
        dumper._use_pending_dump_sync = MagicMock(return_value=False)
        dumper.is_related_local_request = MagicMock(return_value=True)
        dumper.set_msprobe_dump_state = MagicMock(return_value=True)
        dumper._pending_dump = False
        dumper._msprobe_dump_active = False
        dumper._msprobe_dumped_req_ids = set()
        dumper._msprobe_dump_total_count = 0
        dumper._msprobe_last_dump_ts = None
        assert dumper.enable_msprobe_dump_if_needed("r9", skip_related_check=True) is True
