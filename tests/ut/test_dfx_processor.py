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

from unittest.mock import MagicMock, patch

from vllm_ascend.dfx.detector.alert import AnomalyAlert
from vllm_ascend.dfx.io_snapshot import RequestIoSnapshotManager
from vllm_ascend.dfx.manual_trigger import TriggerEvent
from vllm_ascend.dfx.processor import DfxProcessor


def test_dfx_processor_check_after_spec_writes_report_on_arm():
    proc = DfxProcessor.__new__(DfxProcessor)
    proc.runner = MagicMock(tp_rank=0)
    proc.dfx_config = MagicMock()
    proc.dfx_config.dump_enabled.return_value = True
    proc.dfx_config.log_print_sampling_meta.return_value = True
    proc.dfx_config.report_save_sensitive_info.return_value = False
    proc.dumper = MagicMock()
    proc.dumper.dump_count_snapshot.return_value = (0, 3)
    proc.dumper.can_run_anomaly_detection.return_value = True
    proc.dumper.handle_anomaly_alert.return_value = True
    proc.dumper.dump_rank_tag.return_value = "tp0"
    proc.report_writer = MagicMock()
    proc.detectors = MagicMock()
    proc.save_sample_param = MagicMock()
    alert = AnomalyAlert(anomaly_type="spec_acceptance", req_id="r1", detail={"x": 1})
    proc.detectors.check_after_spec.return_value = [alert]
    proc.detectors.get.return_value = MagicMock()

    proc.check_after_spec(sampled_tokens=None, accepted_token_nums=None)

    proc.dumper.handle_anomaly_alert.assert_called_once()
    proc.save_sample_param.assert_called_once_with("r1")
    proc.report_writer.write.assert_called_once()
    assert proc.report_writer.write.call_args.kwargs["req_id"] == "r1"
    assert proc.report_writer.write.call_args.kwargs["dump_attempted"] is True


def test_dfx_processor_refresh_runs_manual_dump_with_batch_report():
    proc = DfxProcessor.__new__(DfxProcessor)
    proc.runner = MagicMock(tp_rank=0)
    proc.runner.input_batch = MagicMock(req_ids=["r1", "r2"])
    proc.dfx_config = MagicMock()
    proc.dfx_config.hot_reload_enabled = True
    proc.dfx_config.sync_dfx_config.return_value = True
    proc.dfx_config.manual_trigger.return_value = True
    proc.dfx_config.dump_enabled.return_value = True
    proc.dfx_config.print_input_token_ids_once.return_value = False
    proc.dfx_config.report_save_sensitive_info.return_value = False
    proc.dfx_config.log_print_sampling_meta.return_value = True
    proc.dfx_config.report_decode_token_ids.return_value = False
    proc.dumper = MagicMock()
    proc.dumper.dump_count_snapshot.return_value = (0, 3)
    proc.dumper.handle_manual_trigger.return_value = True
    proc.dumper.dump_rank_tag.return_value = "tp0"
    proc.report_writer = MagicMock()
    proc.detectors = MagicMock()
    proc.manual_triggers = MagicMock()
    proc.save_sample_param = MagicMock()
    proc._get_report_tokenizer = MagicMock(return_value=None)
    trigger = TriggerEvent(
        trigger_type="manual_trigger",
        req_id="__manual_trigger__",
        consume_quota=False,
        detail={"source": "dump.manual_trigger"},
    )
    proc.manual_triggers.consume_once.return_value = trigger

    with (
        patch("vllm_ascend.dfx.processor.RequestIoSnapshotManager") as mgr_cls,
        patch("vllm_ascend.dfx.processor.InputFilterManager"),
    ):
        mgr = MagicMock()
        mgr_cls.get.return_value = mgr
        snap = MagicMock()
        snap.as_detail_fields.return_value = {"prompt_token_count": 3, "output_token_count": 1}
        mgr.snapshot.return_value = snap
        assert proc.refresh_config() is True

    proc.dumper.apply_dfx_config.assert_called_once()
    proc.detectors.apply_dfx_config.assert_called_once()
    proc.dumper.handle_manual_trigger.assert_called_once_with(trigger, finish_req_ids=["r1", "r2"])
    assert proc.save_sample_param.call_count == 2
    proc.save_sample_param.assert_any_call("r1")
    proc.save_sample_param.assert_any_call("r2")
    proc.report_writer.write.assert_called_once()
    detail = proc.report_writer.write.call_args.kwargs["detail"]
    assert detail["num_requests"] == 2
    assert [r["req_id"] for r in detail["requests"]] == ["r1", "r2"]
    assert proc.report_writer.write.call_args.kwargs["dump_attempted"] is True
    assert proc.report_writer.write.call_args.kwargs["dump_armed"] is True
    assert proc.report_writer.write.call_args.kwargs["dump_count"] == 0
    assert proc.report_writer.write.call_args.kwargs["dump_max_times"] == 3


def test_dfx_processor_refresh_no_change_skips_apply():
    RequestIoSnapshotManager.reset_for_tests()
    proc = DfxProcessor.__new__(DfxProcessor)
    proc.dfx_config = MagicMock()
    proc.dfx_config.hot_reload_enabled = True
    proc.dfx_config.sync_dfx_config.return_value = False
    proc.dfx_config.manual_trigger.return_value = False
    proc.dfx_config.print_input_token_ids_once.return_value = False
    proc.dumper = MagicMock()
    proc.dumper.dump_count_snapshot.return_value = (0, 3)
    proc.detectors = MagicMock()
    proc.manual_triggers = MagicMock()
    proc.manual_triggers.consume_once.return_value = None

    assert proc.refresh_config() is False
    proc.dumper.apply_dfx_config.assert_not_called()
    proc.dumper.sync_dump_limits_from_config.assert_not_called()
    proc.detectors.refresh_all.assert_not_called()
    proc.manual_triggers.consume_once.assert_called_once()


def test_dfx_processor_refresh_drains_manual_trigger_even_when_unchanged():
    RequestIoSnapshotManager.reset_for_tests()
    proc = DfxProcessor.__new__(DfxProcessor)
    proc.runner = MagicMock(tp_rank=0)
    proc.runner.input_batch = MagicMock(req_ids=["r1"])
    proc.dfx_config = MagicMock()
    proc.dfx_config.hot_reload_enabled = True
    proc.dfx_config.sync_dfx_config.return_value = False
    proc.dfx_config.manual_trigger.return_value = True
    proc.dfx_config.dump_enabled.return_value = True
    proc.dfx_config.print_input_token_ids_once.return_value = False
    proc.dfx_config.report_save_sensitive_info.return_value = False
    proc.dfx_config.log_print_sampling_meta.return_value = False
    proc.dfx_config.report_decode_token_ids.return_value = False
    proc.dumper = MagicMock()
    proc.dumper.dump_count_snapshot.return_value = (0, 3)
    proc.dumper.handle_manual_trigger.return_value = True
    proc.dumper.dump_rank_tag.return_value = "tp0"
    proc.report_writer = MagicMock()
    proc.detectors = MagicMock()
    proc.manual_triggers = MagicMock()
    proc._get_report_tokenizer = MagicMock(return_value=None)
    trigger = TriggerEvent(
        trigger_type="manual_trigger",
        req_id="__manual_trigger__",
        consume_quota=False,
        detail={"source": "dump.manual_trigger"},
    )
    proc.manual_triggers.consume_once.return_value = trigger

    with patch("vllm_ascend.dfx.processor.RequestIoSnapshotManager") as mgr_cls:
        mgr = MagicMock()
        mgr_cls.get.return_value = mgr
        snap = MagicMock()
        snap.as_detail_fields.return_value = {"prompt_token_count": 0, "output_token_count": 0}
        mgr.snapshot.return_value = snap
        assert proc.refresh_config() is False

    proc.dumper.apply_dfx_config.assert_not_called()
    proc.manual_triggers.consume_once.assert_called_once()
    proc.dumper.handle_manual_trigger.assert_called_once_with(trigger, finish_req_ids=["r1"])
    proc.report_writer.write.assert_called_once()
    assert proc.report_writer.write.call_args.kwargs["req_id"] == "__manual_trigger__"
    assert proc.report_writer.write.call_args.kwargs["dump_attempted"] is True
    assert proc.report_writer.write.call_args.kwargs["dump_armed"] is True
    assert proc.report_writer.write.call_args.kwargs["dump_count"] == 0
    assert proc.report_writer.write.call_args.kwargs["dump_max_times"] == 3


def test_handle_manual_trigger_still_writes_report_when_arm_fails():
    proc = DfxProcessor.__new__(DfxProcessor)
    proc.runner = MagicMock(tp_rank=0)
    proc.runner.input_batch = MagicMock(req_ids=["r1"])
    proc.dfx_config = MagicMock()
    proc.dfx_config.log_print_sampling_meta.return_value = False
    proc.dfx_config.report_save_sensitive_info.return_value = False
    proc.dumper = MagicMock()
    proc.dumper.dump_count_snapshot.return_value = (2, 5)
    proc.dumper.handle_manual_trigger.return_value = False
    proc.dumper.dump_rank_tag.return_value = "tp0"
    proc.report_writer = MagicMock()
    proc._get_report_tokenizer = MagicMock(return_value=None)

    trigger = TriggerEvent(
        trigger_type="manual_trigger",
        req_id="__manual_trigger__",
        consume_quota=False,
        detail={"source": "dump.manual_trigger"},
    )

    with patch("vllm_ascend.dfx.processor.RequestIoSnapshotManager") as mgr_cls:
        mgr = MagicMock()
        mgr_cls.get.return_value = mgr
        snap = MagicMock()
        snap.as_detail_fields.return_value = {"prompt_token_count": 1, "output_token_count": 0}
        mgr.snapshot.return_value = snap

        proc._handle_manual_trigger(trigger, write_report=True)

    proc.dumper.handle_manual_trigger.assert_called_once_with(trigger, finish_req_ids=["r1"])
    proc.dumper.dump_count_snapshot.assert_called_once_with(dump_armed=False)
    proc.report_writer.write.assert_called_once()
    assert proc.report_writer.write.call_args.kwargs["dump_attempted"] is True
    assert proc.report_writer.write.call_args.kwargs["dump_armed"] is False
    assert proc.report_writer.write.call_args.kwargs["dump_count"] == 2
    assert proc.report_writer.write.call_args.kwargs["dump_max_times"] == 5


def test_handle_alert_calls_save_sample_param_when_print_sampling_meta():
    proc = DfxProcessor.__new__(DfxProcessor)
    proc.runner = MagicMock(tp_rank=0)
    proc.dfx_config = MagicMock()
    proc.dfx_config.dump_enabled.return_value = True
    proc.dfx_config.log_print_sampling_meta.return_value = True
    proc.dfx_config.report_save_sensitive_info.return_value = False
    proc.dumper = MagicMock()
    proc.dumper.dump_count_snapshot.return_value = (0, 3)
    proc.dumper.handle_anomaly_alert.return_value = True
    proc.dumper.dump_rank_tag.return_value = "tp0"
    proc.report_writer = MagicMock()
    proc.save_sample_param = MagicMock()
    alert = AnomalyAlert(
        anomaly_type="token_logprob",
        req_id="r2",
    )
    assert proc._handle_alert(alert, write_report=True) is None
    proc.save_sample_param.assert_called_once_with("r2")
    proc.report_writer.write.assert_called_once()
    assert proc.report_writer.write.call_args.kwargs["req_id"] == "r2"
    assert proc.report_writer.write.call_args.kwargs["dump_attempted"] is True


def test_handle_alert_skips_save_sample_param_when_print_sampling_meta_off():
    proc = DfxProcessor.__new__(DfxProcessor)
    proc.runner = MagicMock(tp_rank=0)
    proc.dfx_config = MagicMock()
    proc.dfx_config.dump_enabled.return_value = True
    proc.dfx_config.log_print_sampling_meta.return_value = False
    proc.dfx_config.report_save_sensitive_info.return_value = False
    proc.dumper = MagicMock()
    proc.dumper.dump_count_snapshot.return_value = (0, 3)
    proc.dumper.handle_anomaly_alert.return_value = True
    proc.dumper.dump_rank_tag.return_value = "tp0"
    proc.report_writer = MagicMock()
    proc.save_sample_param = MagicMock()
    alert = AnomalyAlert(anomaly_type="token_logprob", req_id="r2")
    assert proc._handle_alert(alert, write_report=True) is None
    proc.save_sample_param.assert_not_called()
    proc.report_writer.write.assert_called_once()
    assert proc.report_writer.write.call_args.kwargs["req_id"] == "r2"
    assert proc.report_writer.write.call_args.kwargs["dump_attempted"] is True


def test_handle_alert_detect_only_writes_report_without_dump():
    proc = DfxProcessor.__new__(DfxProcessor)
    proc.runner = MagicMock(tp_rank=0)
    proc.dfx_config = MagicMock()
    proc.dfx_config.dump_enabled.return_value = False
    proc.dfx_config.log_print_sampling_meta.return_value = False
    proc.dfx_config.report_save_sensitive_info.return_value = False
    proc.dumper = MagicMock()
    proc.dumper.dump_count_snapshot.return_value = (0, 3)
    proc.dumper.dump_rank_tag.return_value = "tp0"
    proc.report_writer = MagicMock()
    detector = MagicMock()
    alert = AnomalyAlert(anomaly_type="spec_acceptance", req_id="r3")
    assert proc._handle_alert(alert, detector=detector, write_report=True) is None
    proc.dumper.handle_anomaly_alert.assert_not_called()
    detector.on_alert_armed.assert_called_once_with(alert)
    proc.report_writer.write.assert_called_once()
    assert proc.report_writer.write.call_args.kwargs["dump_attempted"] is False
    assert proc.report_writer.write.call_args.kwargs["dump_armed"] is False


def test_handle_alert_dump_on_still_writes_report_when_dump_fails():
    """Detect evidence is kept even if dump arm fails (quota / cooldown)."""
    proc = DfxProcessor.__new__(DfxProcessor)
    proc.runner = MagicMock(tp_rank=0)
    proc.dfx_config = MagicMock()
    proc.dfx_config.dump_enabled.return_value = True
    proc.dfx_config.log_print_sampling_meta.return_value = False
    proc.dfx_config.report_save_sensitive_info.return_value = False
    proc.dumper = MagicMock()
    proc.dumper.dump_count_snapshot.return_value = (0, 3)
    proc.dumper.handle_anomaly_alert.return_value = False
    proc.dumper.dump_rank_tag.return_value = "tp0"
    proc.report_writer = MagicMock()
    proc.save_sample_param = MagicMock()
    alert = AnomalyAlert(anomaly_type="spec_acceptance", req_id="r4")
    assert proc._handle_alert(alert, write_report=True) is None
    proc.report_writer.write.assert_called_once()
    assert proc.report_writer.write.call_args.kwargs["dump_attempted"] is True
    assert proc.report_writer.write.call_args.kwargs["dump_armed"] is False


def test_handle_alert_passes_dump_armed_true_when_arm_succeeds():
    proc = DfxProcessor.__new__(DfxProcessor)
    proc.runner = MagicMock(tp_rank=0)
    proc.dfx_config = MagicMock()
    proc.dfx_config.dump_enabled.return_value = True
    proc.dfx_config.log_print_sampling_meta.return_value = False
    proc.dfx_config.report_save_sensitive_info.return_value = False
    proc.dumper = MagicMock()
    proc.dumper.dump_count_snapshot.return_value = (0, 3)
    proc.dumper.handle_anomaly_alert.return_value = True
    proc.dumper.dump_rank_tag.return_value = "tp0"
    proc.report_writer = MagicMock()
    alert = AnomalyAlert(anomaly_type="spec_acceptance", req_id="r5")
    with patch("vllm_ascend.dfx.processor.RequestIoSnapshotManager") as mgr_cls:
        mgr = MagicMock()
        mgr_cls.get.return_value = mgr
        snap = MagicMock()
        mgr.snapshot.return_value = snap
        mgr.merge_into_detail.side_effect = lambda detail, _snap: detail
        proc._handle_alert(alert, write_report=True)
    proc.report_writer.write.assert_called_once()
    assert proc.report_writer.write.call_args.kwargs["dump_attempted"] is True
    assert proc.report_writer.write.call_args.kwargs["dump_armed"] is True
    assert proc.report_writer.write.call_args.kwargs["dump_count"] == 0
    assert proc.report_writer.write.call_args.kwargs["dump_max_times"] == 3


def test_save_sample_param_skips_non_tp0():
    proc = DfxProcessor.__new__(DfxProcessor)
    proc.runner = MagicMock(tp_rank=1, input_batch=MagicMock())
    with patch("vllm_ascend.dfx.processor.get_pp_group") as pp:
        pp.return_value.is_last_rank = True
        proc.save_sample_param("r1")
    # No crash; non-TP0 returns before needing sampling_metadata.


def test_ensure_logprobs_for_detection_bumps_v1_num_logprobs():
    proc = DfxProcessor.__new__(DfxProcessor)
    proc.dfx_config = MagicMock()
    proc.dumper = MagicMock()
    proc.dumper.dump_count_snapshot.return_value = (0, 3)
    proc.detectors = MagicMock()
    proc.detectors.token_logprob_topk_if_enabled.return_value = 20

    input_batch = MagicMock()
    input_batch.req_ids = ["r1", "r2"]
    input_batch.num_logprobs = {"r2": 5}  # r1 missing; r2 too small
    input_batch._make_sampling_metadata.return_value = "meta"
    proc.runner = MagicMock(input_batch=input_batch)

    proc.ensure_logprobs_for_detection()

    assert input_batch.num_logprobs["r1"] == 20
    assert input_batch.num_logprobs["r2"] == 20
    input_batch._make_sampling_metadata.assert_called_once()
    assert input_batch.sampling_metadata == "meta"


def test_ensure_logprobs_noop_when_disabled():
    proc = DfxProcessor.__new__(DfxProcessor)
    proc.dfx_config = MagicMock()
    proc.detectors = MagicMock()
    proc.detectors.token_logprob_topk_if_enabled.return_value = None
    proc.runner = MagicMock()
    proc.dumper = MagicMock()
    proc.dumper.dump_count_snapshot.return_value = (0, 3)

    proc.ensure_logprobs_for_detection()

    proc.dfx_config.dump_enabled.assert_not_called()


def test_sync_for_step_calls_refresh_then_dump_or():
    proc = DfxProcessor.__new__(DfxProcessor)
    proc.runner = MagicMock(dp_rank=1, tp_rank=3)
    proc.dumper = MagicMock()
    proc.refresh_config = MagicMock()
    proc.sync_dump_pending_or = MagicMock()

    with patch("vllm_ascend.dfx.processor.get_pp_group") as get_pp:
        get_pp.return_value.rank_in_group = 0
        proc.sync_for_step(allow_arm=False)

    proc.dumper.advance_wave.assert_called_once_with(allow_arm=False)
    proc.refresh_config.assert_called_once_with(allow_arm=False)
    proc.sync_dump_pending_or.assert_called_once_with(allow_arm=False)


def test_refresh_config_skips_manual_trigger_arm_when_not_allow_arm():
    """Dummy wave must not consume manual_trigger (no arm → leave JSON true)."""
    RequestIoSnapshotManager.reset_for_tests()
    proc = DfxProcessor.__new__(DfxProcessor)
    proc.dfx_config = MagicMock()
    proc.dfx_config.hot_reload_enabled = True
    proc.dfx_config.sync_dfx_config.return_value = False
    proc.dfx_config.manual_trigger.return_value = True
    proc.dfx_config.print_input_token_ids_once.return_value = False
    proc.dumper = MagicMock()
    proc.dumper.dump_count_snapshot.return_value = (0, 3)
    proc.report_writer = MagicMock()
    proc.detectors = MagicMock()
    proc.manual_triggers = MagicMock()
    proc.manual_triggers.consume_once.return_value = None
    proc._handle_manual_trigger = MagicMock()

    proc.refresh_config(allow_arm=False)

    proc.manual_triggers.consume_once.assert_called_once_with(allow_arm=False)
    proc._handle_manual_trigger.assert_not_called()
    proc.dfx_config.consume_manual_trigger.assert_not_called()


def test_refresh_config_hot_reload_off_skips_sync_and_filters():
    """Default-off service: no sync_dfx_config / filter apply each step."""
    RequestIoSnapshotManager.reset_for_tests()
    proc = DfxProcessor.__new__(DfxProcessor)
    proc.dfx_config = MagicMock()
    proc.dfx_config.hot_reload_enabled = False
    proc.dfx_config.print_input_token_ids_once.return_value = False
    proc.manual_triggers = MagicMock()
    proc.manual_triggers.consume_once.return_value = None
    proc.maybe_print_input_token_ids_once = MagicMock(return_value=False)  # type: ignore[method-assign]

    with patch("vllm_ascend.dfx.processor.InputFilterManager") as filt_cls:
        assert proc.refresh_config(allow_arm=True) is False
        proc.dfx_config.sync_dfx_config.assert_not_called()
        filt_cls.get.assert_not_called()
    proc.manual_triggers.consume_once.assert_called_once_with(allow_arm=True)


def test_maybe_print_input_token_ids_once_logs_and_consumes():
    from types import SimpleNamespace

    proc = DfxProcessor.__new__(DfxProcessor)
    proc.runner = SimpleNamespace(
        tp_rank=0,
        requests={"r1": SimpleNamespace(prompt_token_ids=[151644, 872, 1, 2])},
        input_batch=None,
    )
    proc.dfx_config = MagicMock()
    proc.dfx_config.print_input_token_ids_once.return_value = True

    assert proc.maybe_print_input_token_ids_once(allow_arm=True) is True
    proc.dfx_config.consume_print_input_token_ids_once.assert_called_once()


def test_maybe_print_input_token_ids_once_defers_without_prompts():
    from types import SimpleNamespace

    proc = DfxProcessor.__new__(DfxProcessor)
    proc.runner = SimpleNamespace(tp_rank=0, requests={}, input_batch=None)
    proc.dfx_config = MagicMock()
    proc.dfx_config.print_input_token_ids_once.return_value = True

    assert proc.maybe_print_input_token_ids_once(allow_arm=True) is False
    proc.dfx_config.consume_print_input_token_ids_once.assert_not_called()


def test_clear_finished_prints_output_when_enabled():
    import logging
    from types import SimpleNamespace

    from vllm_ascend.dfx.input_filters import InputFilterManager
    from vllm_ascend.dfx.io_snapshot import RequestIoSnapshotManager

    RequestIoSnapshotManager.reset_for_tests()
    InputFilterManager.reset_for_tests()
    io = RequestIoSnapshotManager.get()
    io.append_output("r1", [10, 11, 12])

    proc = DfxProcessor.__new__(DfxProcessor)
    proc.runner = SimpleNamespace(tp_rank=0, requests={}, input_batch=None)
    proc.dfx_config = MagicMock()
    proc.dfx_config.log_print_output_on_finish.return_value = True
    proc.dfx_config.report_max_output_token_ids.return_value = 1000
    proc.dfx_config.report_decode_token_ids.return_value = False
    proc.detectors = MagicMock()
    proc.dumper = MagicMock()
    proc.dumper.take_dump_finish_meta.return_value = None
    proc.report_writer = MagicMock()
    proc._get_detector_tokenizer = MagicMock(return_value=None)  # type: ignore[method-assign]

    # DFX loggers disable propagation to root; attach a local handler instead
    # of relying on ``caplog`` to observe ``[DFX print_output]``.
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.setLevel(logging.INFO)
    handler.emit = lambda record: records.append(record)  # type: ignore[method-assign]
    logger = logging.getLogger("vllm_ascend.dfx.processor")
    prev_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        proc.clear_finished(["r1"])
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)

    proc.detectors.clear_finished.assert_called_once_with("r1")
    assert io.cumulative_output_count("r1") == 0  # cleared
    assert any("print_output" in r.getMessage() and "r1" in r.getMessage() for r in records)


def test_clear_finished_skips_print_when_disabled():
    from types import SimpleNamespace

    from vllm_ascend.dfx.input_filters import InputFilterManager
    from vllm_ascend.dfx.io_snapshot import RequestIoSnapshotManager

    RequestIoSnapshotManager.reset_for_tests()
    InputFilterManager.reset_for_tests()

    proc = DfxProcessor.__new__(DfxProcessor)
    proc.runner = SimpleNamespace(tp_rank=0)
    proc.dfx_config = MagicMock()
    proc.dfx_config.log_print_output_on_finish.return_value = False
    proc.dfx_config.report_decode_token_ids.return_value = False
    proc.detectors = MagicMock()
    proc.dumper = MagicMock()
    proc.dumper.take_dump_finish_meta.return_value = None
    proc.report_writer = MagicMock()
    proc._maybe_print_output_on_finish = MagicMock()  # type: ignore[method-assign]

    proc.clear_finished(["r1"])
    proc._maybe_print_output_on_finish.assert_not_called()
    proc.detectors.clear_finished.assert_called_once_with("r1")


def test_consume_print_input_token_ids_once_roundtrip(tmp_path):
    import json
    from pathlib import Path

    from vllm_ascend.dfx.runtime_config import DfxRuntimeConfig

    cfg_path = Path(tmp_path) / "dfx_config.json"
    cfg = DfxRuntimeConfig(
        cfg_path,
        report_dir=Path(tmp_path) / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=0,
    )
    assert cfg.print_input_token_ids_once() is False
    assert cfg.save({"input_filter": {"print_input_token_ids_once": True}})
    assert cfg.print_input_token_ids_once() is True
    assert cfg.consume_print_input_token_ids_once() is True
    assert cfg.print_input_token_ids_once() is False
    saved = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert saved["input_filter"]["print_input_token_ids_once"] is False


def test_clear_finished_writes_dump_finish_when_meta_present():
    from types import SimpleNamespace

    from vllm_ascend.dfx.dfx_types import DumpFinishMeta
    from vllm_ascend.dfx.input_filters import InputFilterManager
    from vllm_ascend.dfx.io_snapshot import RequestIoSnapshotManager

    RequestIoSnapshotManager.reset_for_tests()
    InputFilterManager.reset_for_tests()
    io = RequestIoSnapshotManager.get()
    io.append_output("r1", [1, 2, 3, 4])

    meta = DumpFinishMeta(
        anomaly_type="token_repeat",
        source="anomaly",
        dump_arm_wave=2,
        dump_activate_wave=3,
        dump_waves_after_report=1,
        dump_count=1,
    )
    proc = DfxProcessor.__new__(DfxProcessor)
    proc.runner = SimpleNamespace(tp_rank=0, requests={}, input_batch=None)
    proc.dfx_config = MagicMock()
    proc.dfx_config.log_print_output_on_finish.return_value = False
    proc.dfx_config.report_decode_token_ids.return_value = False
    proc.detectors = MagicMock()
    proc.dumper = MagicMock()
    proc.dumper.take_dump_finish_meta.side_effect = lambda rid: meta if rid == "r1" else None
    proc.dumper.current_wave.return_value = 9
    proc.dumper.dump_rank_tag.return_value = "tp0"
    proc.report_writer = MagicMock()

    proc.clear_finished(["r1"])

    proc.report_writer.write_dump_finish.assert_called_once()
    kwargs = proc.report_writer.write_dump_finish.call_args.kwargs
    assert kwargs["req_id"] == "r1"
    assert kwargs["dump_arm_wave"] == 2
    assert kwargs["dump_activate_wave"] == 3
    assert kwargs["dump_waves_after_report"] == 1
    assert kwargs["finish_wave"] == 9
    assert kwargs["detail"]["output_token_ids"] == [1, 2, 3, 4]
    proc.detectors.clear_finished.assert_called_once_with("r1")
    assert io.cumulative_output_ids("r1") == []
