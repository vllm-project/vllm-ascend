# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.distributed.kv_transfer.utils.utils import (
    PD_QOS_DEFAULT,
    QOS_KEY,
    collect_storage_merged_register_regions,
    get_ascend_store_extra_config,
    inject_qos,
    uses_ascend_store,
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


class TestAscendStoreConfig(unittest.TestCase):
    def test_detects_direct_store_and_preserves_extra_config(self):
        for connector_name in ("AscendStoreConnector", "MooncakeConnectorStoreV1"):
            extra_config = {"backend": "memcache", "use_layerwise": True}
            transfer = SimpleNamespace(
                kv_connector=connector_name,
                kv_connector_extra_config=extra_config,
            )
            self.assertTrue(uses_ascend_store(transfer))
            self.assertIs(get_ascend_store_extra_config(transfer), extra_config)

    def test_detects_nested_store_without_assuming_child_order(self):
        extra_config = {"backend": "memcache", "use_layerwise": True}
        transfer = SimpleNamespace(
            kv_connector="MultiConnector",
            kv_connector_extra_config={
                "connectors": [
                    {"kv_connector": "MooncakeConnectorV2"},
                    {"kv_connector": "AscendStoreConnector", "kv_connector_extra_config": extra_config},
                ]
            },
        )
        self.assertTrue(uses_ascend_store(transfer))
        self.assertIs(get_ascend_store_extra_config(transfer), extra_config)

    def test_ignores_absent_or_malformed_config(self):
        cases = [
            None,
            SimpleNamespace(),
            SimpleNamespace(kv_connector="MooncakeConnectorV2"),
            SimpleNamespace(kv_connector="MultiConnector", kv_connector_extra_config=None),
            SimpleNamespace(kv_connector="MultiConnector", kv_connector_extra_config={"connectors": None}),
            SimpleNamespace(
                kv_connector="MultiConnector",
                kv_connector_extra_config={"connectors": [None, "bad", {"kv_connector": "OtherConnector"}]},
            ),
        ]
        for transfer in cases:
            with self.subTest(transfer=transfer):
                self.assertFalse(uses_ascend_store(transfer))
                self.assertIsNone(get_ascend_store_extra_config(transfer))

    def test_store_without_extra_config_is_still_detected(self):
        transfer = SimpleNamespace(kv_connector="AscendStoreConnector", kv_connector_extra_config=None)
        self.assertTrue(uses_ascend_store(transfer))
        self.assertEqual(get_ascend_store_extra_config(transfer), {})


class TestKVRegisterRegions(unittest.TestCase):
    def test_combined_and_tuple_caches_register_the_same_physical_range(self):
        for num_blocks in (2, 5):
            for scale in (1, 3):
                for block_major in (False, True):
                    with self.subTest(num_blocks=num_blocks, scale=scale, block_major=block_major):
                        count = num_blocks * scale
                        # Keep prefix/suffix bytes outside the view to check offsets.
                        elements = 2 * count * 16 * 2
                        backing = torch.empty(elements + 32, dtype=torch.float32)
                        raw = backing[16 : 16 + elements]
                        if block_major:
                            combined = raw.view(count, 2, 16, 1, 2).permute(1, 0, 2, 3, 4)
                        else:
                            combined = raw.view(2, count, 16, 1, 2)
                        for caches in (combined, tuple(combined.unbind(0))):
                            regions = collect_storage_merged_register_regions({"attn": caches})
                            self.assertEqual(regions.ptrs, [raw.data_ptr()])
                            self.assertEqual(regions.lengths, [raw.nbytes])
                            self.assertEqual(regions.logical_total_bytes, raw.nbytes)
                            # Both components of the final block must be registered.
                            end = regions.ptrs[0] + regions.lengths[0]
                            for component in combined.unbind(0):
                                last_block = component[-1]
                                self.assertLessEqual(last_block.data_ptr() + last_block.nbytes, end)

    def test_padded_subview_registers_span_without_storage_tail(self):
        backing = torch.empty(128, dtype=torch.float32)
        cache = torch.as_strided(backing, size=(3, 4), stride=(16, 1), storage_offset=7)
        regions = collect_storage_merged_register_regions({"attn": (cache, backing[:0])})
        self.assertEqual(regions.ptrs, [cache.data_ptr()])
        self.assertEqual(regions.lengths, [36 * cache.element_size()])
        self.assertEqual(regions.logical_total_bytes, cache.nbytes)
        self.assertEqual(regions.logical_tensor_count, 1)
