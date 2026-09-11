# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
import json
import os
import unittest
from unittest.mock import patch

from vllm_ascend.distributed.kv_transfer.utils.utils import (
    PD_QOS_DEFAULT,
    QOS_KEY,
    inject_qos,
)

ENV = "ASCEND_GLOBAL_RESOURCE_CONFIG"


class TestAscendResourceConfig(unittest.TestCase):
    def test_pd_only(self):
        with patch.dict(os.environ, {}, clear=True):
            inject_qos(PD_QOS_DEFAULT)
            self.assertEqual(json.loads(os.environ[ENV]), {QOS_KEY: 1})

    def test_preserves_other_fields(self):
        initial = {QOS_KEY: 3, "other": {"x": [1, 2]}, "store": {QOS_KEY: 4, "pool_other": 9}}
        with patch.dict(os.environ, {ENV: json.dumps(initial)}):
            inject_qos(2)
            self.assertEqual(json.loads(os.environ[ENV]), {**initial, QOS_KEY: 2})

    def test_explicit_pd_qos(self):
        for qos in range(5):
            with self.subTest(qos=qos), patch.dict(os.environ, {}, clear=True):
                inject_qos(qos)
                self.assertEqual(json.loads(os.environ[ENV]), {QOS_KEY: qos})

    def test_invalid_inputs_leave_environment_unchanged(self):
        for qos in (True, False, "1", 1.5, None, -1, 5, 6, 7, 8):
            with self.subTest(qos=qos), patch.dict(os.environ, {ENV: "{}"}):
                with self.assertRaises(ValueError):
                    inject_qos(qos)
                self.assertEqual(os.environ[ENV], "{}")
        for raw in ("{unquoted: 1}", "[]", "null"):
            with self.subTest(raw=raw), patch.dict(os.environ, {ENV: raw}):
                with self.assertRaises(ValueError):
                    inject_qos(1)
                self.assertEqual(os.environ[ENV], raw)

    def test_repeated_injection_is_idempotent(self):
        with patch.dict(os.environ, {}, clear=True):
            inject_qos(1)
            before = os.environ[ENV]
            inject_qos(1)
            self.assertEqual(os.environ[ENV], before)

    def test_injection_logs(self):
        for initial in ({}, {QOS_KEY: 3}, {QOS_KEY: 1}):
            with (
                self.subTest(initial=initial),
                patch.dict(os.environ, {ENV: json.dumps(initial)}),
                patch("vllm_ascend.distributed.kv_transfer.utils.utils.logger") as log,
            ):
                inject_qos(1)
                if QOS_KEY in initial:
                    log.warning.assert_called_once()
                    self.assertEqual(log.warning.call_args.args[1:], (initial[QOS_KEY], 1))
                else:
                    log.warning.assert_not_called()
                log.info.assert_called_once()
                self.assertEqual(log.info.call_args.args[1:], (1,))
                self.assertEqual(json.loads(os.environ[ENV])[QOS_KEY], 1)

    def test_invalid_config_does_not_log_success(self):
        with (
            patch.dict(os.environ, {ENV: "not-json"}),
            patch("vllm_ascend.distributed.kv_transfer.utils.utils.logger") as log,
        ):
            with self.assertRaises(ValueError):
                inject_qos(1)
            log.info.assert_not_called()
