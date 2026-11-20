# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest

from vllm_ascend.ops import moe_stream_events as events


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("capturing", [False, True])
def test_event_protocol(enabled, capturing):
    npu = Mock()
    npu.is_current_stream_capturing.return_value = capturing
    stream = npu.current_stream.return_value
    with (
        patch.object(events, "moe_stream_events_enabled", return_value=enabled),
        patch.object(events.torch, "npu", npu),
    ):
        events.record_capture_boundary()
        events.wait_eager_boundary()
    if enabled:
        stream.record_event.assert_called_once_with()
    else:
        npu.is_current_stream_capturing.assert_not_called()
        npu.current_stream.assert_not_called()
    if enabled and not capturing:
        stream.wait_event.assert_called_once_with(stream.record_event.return_value)
    else:
        stream.wait_event.assert_not_called()
