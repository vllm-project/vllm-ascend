# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_ascend.ascend_config import moe_stream_events_enabled


def wait_eager_boundary() -> None:
    if moe_stream_events_enabled() and not torch.npu.is_current_stream_capturing():
        stream = torch.npu.current_stream()
        stream.wait_event(stream.record_event())


def record_capture_boundary() -> None:
    if moe_stream_events_enabled() and torch.npu.is_current_stream_capturing():
        torch.npu.current_stream().record_event()
