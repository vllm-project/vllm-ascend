# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import unittest
from multiprocessing import Pipe

import numpy as np

from vllm_ascend.distributed.eplb.policy import _stair_process


class TestStairProcessProtocol(unittest.TestCase):
    def test_round_trip(self):
        parent, child = Pipe(duplex=True)
        request = (
            np.ones((2, 1, 3)),
            np.array([[[0, 1, 2]]]),
            np.array([1.2]),
            np.array([0]),
            {"load_window_bins": 2},
            np.array([1, 1]),
        )
        _stair_process.send_planner_request(parent, request)
        result = _stair_process._receive_planner_request(child)
        for actual, expected in zip(result[:4], request[:4]):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(result[4], request[4])
        np.testing.assert_array_equal(result[5], request[5])

        plan_fields = (
            np.array([[[0, 1]]]),
            np.array([[[0, 0]]]),
            np.array([[[0, 1]]]),
            np.array([1.1]),
            np.array([[1.2, 1.3, 1.0, 1.1]]),
        )
        _stair_process._send_planner_response(child, (None, None, plan_fields))
        error_type, error, actual_fields = _stair_process.receive_planner_response(parent)
        self.assertIsNone(error_type)
        self.assertIsNone(error)
        self.assertIsNotNone(actual_fields)
        for actual, expected in zip(actual_fields, plan_fields):
            np.testing.assert_array_equal(actual, expected)
        parent.close()
        child.close()
