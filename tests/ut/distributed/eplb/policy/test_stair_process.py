# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from io import BytesIO

import pytest

from vllm_ascend.distributed.eplb.policy import _stair_process


def test_stair_process_protocol_round_trip():
    stream = BytesIO()
    request = ("plan", {"layers": 3})

    _stair_process.send_planner_request(stream, request)
    stream.seek(0)

    assert _stair_process._receive(stream) == request


def test_stair_process_protocol_rejects_truncated_message():
    stream = BytesIO(b"\x00")

    with pytest.raises(EOFError, match="socket closed"):
        _stair_process._receive(stream)
