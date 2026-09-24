from unittest.mock import patch

from vllm_ascend.eplb.timing import EplbCpuTimingWindow, EplbDeviceTimingWindow


class FakeEvent:
    next_id = 0

    def __init__(self, enable_timing: bool):
        assert enable_timing
        self.event_id = FakeEvent.next_id
        FakeEvent.next_id += 1

    def record(self):
        pass

    def synchronize(self):
        pass

    def elapsed_time(self, end):
        return float(end.event_id - self.event_id)


def test_device_timing_reports_one_complete_window():
    FakeEvent.next_id = 0
    timer = EplbDeviceTimingWindow("forward", 2)

    with (
        patch("vllm_ascend.eplb.timing.torch.Event", FakeEvent),
        patch("vllm_ascend.eplb.timing.logger.info") as log_info,
    ):
        with timer.measure():
            pass
        log_info.assert_not_called()
        with timer.measure():
            pass

    assert log_info.call_args.args[1:3] == ("forward", 2)
    assert not timer.samples


def test_cpu_timing_reports_one_complete_window():
    timer = EplbCpuTimingWindow("step", 2)

    with patch("vllm_ascend.eplb.timing.logger.info") as log_info:
        with timer.measure():
            pass
        log_info.assert_not_called()
        with timer.measure():
            pass

    assert log_info.call_args.args[1:3] == ("step", 2)
    assert not timer.samples
