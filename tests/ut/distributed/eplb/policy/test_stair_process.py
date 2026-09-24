# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from io import BytesIO

import numpy as np
import pytest

from vllm_ascend.distributed.eplb.policy import _stair_process


def test_stair_process_protocol_round_trip():
    stream = BytesIO()
    request = (
        np.ones((2, 1, 3)),
        np.array([[[0, 1, 2]]]),
        np.array([1.2]),
        np.array([0]),
        {"load_window_bins": 2},
        [0],
        np.array([1, 1]),
    )

    _stair_process.send_planner_request(stream, request)
    stream.seek(0)
    result = _stair_process._receive_planner_request(stream)

    for actual, expected in zip(result[:4], request[:4]):
        np.testing.assert_array_equal(actual, expected)
    assert result[4] == request[4]
    assert result[5] == request[5]
    np.testing.assert_array_equal(result[6], request[6])


def test_stair_process_protocol_rejects_truncated_message():
    stream = BytesIO(b"\x00")

    with pytest.raises(EOFError, match="socket closed"):
        _stair_process._receive_arrays(stream)


def test_stair_process_response_round_trip():
    stream = BytesIO()
    plan_fields = (
        np.array([[[0, 1]]]),
        np.array([[[0, 0]]]),
        np.array([[[0, 1]]]),
        np.array([1.1]),
    )

    _stair_process._send_planner_response(stream, (None, None, plan_fields))
    stream.seek(0)
    error_type, error, actual_fields = _stair_process.receive_planner_response(stream)

    assert error_type is None
    assert error is None
    assert actual_fields is not None
    for actual, expected in zip(actual_fields, plan_fields):
        np.testing.assert_array_equal(actual, expected)
