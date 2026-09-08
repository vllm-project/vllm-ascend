# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
import json
import os
import unittest
from unittest.mock import patch

from vllm_ascend.distributed.kv_transfer.utils.ascend_resource_config import (
    PD_QOS_DEFAULT,
    QOS_KEY,
    STORE_QOS_DEFAULT,
    inject_qos,
)

ENV = "ASCEND_GLOBAL_RESOURCE_CONFIG"


class TestAscendResourceConfig(unittest.TestCase):
    def test_pd_only(self):
        with patch.dict(os.environ, {}, clear=True):
            inject_qos(PD_QOS_DEFAULT)
            self.assertEqual(json.loads(os.environ[ENV]), {QOS_KEY: 1})

    def test_store_only(self):
        with patch.dict(os.environ, {}, clear=True):
            inject_qos(STORE_QOS_DEFAULT, store=True)
            self.assertEqual(json.loads(os.environ[ENV]), {"store": {QOS_KEY: 0}})

    def test_both_initialization_orders(self):
        for order in ((False, True), (True, False)):
            for pd_qos, store_qos in ((1, 0), (0, 7), (6, 2)):
                with self.subTest(order=order, pd=pd_qos, pool=store_qos), patch.dict(os.environ, {}, clear=True):
                    for store in order:
                        inject_qos(store_qos if store else pd_qos, store=store)
                    self.assertEqual(json.loads(os.environ[ENV]), {QOS_KEY: pd_qos, "store": {QOS_KEY: store_qos}})

    def test_preserves_other_fields_and_opposite_qos(self):
        initial = {QOS_KEY: 3, "other": {"x": [1, 2]}, "store": {QOS_KEY: 4, "pool_other": 9}}
        for store in (False, True):
            with self.subTest(store=store), patch.dict(os.environ, {ENV: json.dumps(initial)}):
                inject_qos(2, store=store)
                expected = json.loads(json.dumps(initial))
                (expected["store"] if store else expected)[QOS_KEY] = 2
                self.assertEqual(json.loads(os.environ[ENV]), expected)

    def test_store_preserves_legacy_inherited_settings(self):
        initial = {QOS_KEY: 5, "comm_resource_config.protocol_desc": ["roce:device"]}
        with patch.dict(os.environ, {ENV: json.dumps(initial)}):
            inject_qos(0, store=True)
            self.assertEqual(json.loads(os.environ[ENV]), {**initial, "store": {**initial, QOS_KEY: 0}})

    def test_invalid_inputs_leave_environment_unchanged(self):
        for qos in (True, False, "1", 1.5, None, -1, 8):
            with self.subTest(qos=qos), patch.dict(os.environ, {ENV: "{}"}):
                with self.assertRaises(ValueError):
                    inject_qos(qos)
                self.assertEqual(os.environ[ENV], "{}")
        for raw in ("{unquoted: 1}", "[]", "null", '{"store": 3}'):
            with self.subTest(raw=raw), patch.dict(os.environ, {ENV: raw}):
                with self.assertRaises(ValueError):
                    inject_qos(1)
                self.assertEqual(os.environ[ENV], raw)

    def test_repeated_injection_is_idempotent(self):
        with patch.dict(os.environ, {}, clear=True):
            inject_qos(1)
            inject_qos(0, store=True)
            before = os.environ[ENV]
            inject_qos(1)
            inject_qos(0, store=True)
            self.assertEqual(os.environ[ENV], before)
