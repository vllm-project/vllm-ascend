# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
import json
import os
import unittest
from unittest.mock import patch

import numpy as np
import torch

from vllm_ascend.distributed.kv_transfer.utils.utils import (
    PD_QOS_DEFAULT,
    QOS_KEY,
    as_uint64_transfer_array,
    collect_storage_merged_register_regions,
    concat_uint64_chunks,
    inject_qos,
    validate_uint64_transfer_columns,
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


class TestUint64TransferArrays(unittest.TestCase):
    def test_as_uint64_transfer_array_concat_and_column_validation(self):
        direct = np.array([0x1000, 0xFF00000000200000], dtype=np.uint64)
        self.assertIs(as_uint64_transfer_array(direct), direct)
        self.assertEqual(concat_uint64_chunks([]).size, 0)
        self.assertIs(concat_uint64_chunks([direct]), direct)

        strided = np.array([10, 20, 30, 40], dtype=np.uint64)[::2]
        normalized_strided = as_uint64_transfer_array(strided)
        self.assertTrue(normalized_strided.flags.c_contiguous)
        self.assertEqual(normalized_strided.tolist(), [10, 30])

        swapped = direct.byteswap().view(direct.dtype.newbyteorder())
        normalized_swapped = as_uint64_transfer_array(swapped)
        self.assertEqual(
            (normalized_swapped.dtype, normalized_swapped.tolist()),
            (np.dtype(np.uint64), direct.tolist()),
        )

        merged = concat_uint64_chunks([np.array([(1 << 63) + 7], dtype=np.uint64), np.array([42], dtype=np.int64)])
        self.assertEqual((merged.dtype, [int(x) for x in merged]), (np.uint64, [(1 << 63) + 7, 42]))
        from_list = as_uint64_transfer_array([(1 << 63) + 7, 42])
        self.assertEqual((from_list.dtype, [int(x) for x in from_list]), (np.uint64, [(1 << 63) + 7, 42]))
        self.assertEqual(as_uint64_transfer_array([]).size, 0)

        bad_inputs = (
            [1, -1],
            [1 << 64],
            [True],
            [True, 1],
            [1.0],
            np.zeros((2, 2), dtype=np.uint64),
            np.array([-1], dtype=np.int64),
            np.array([1.0], dtype=np.float64),
            "12",
            123,
        )
        for bad in bad_inputs:
            with self.assertRaises(ValueError):
                as_uint64_transfer_array(bad)

        self.assertEqual(validate_uint64_transfer_columns(direct, direct, direct), 2)
        with self.assertRaisesRegex(ValueError, "contiguous 1D native uint64"):
            validate_uint64_transfer_columns(strided, direct, direct)
        with self.assertRaisesRegex(ValueError, "contiguous 1D native uint64"):
            validate_uint64_transfer_columns(swapped, direct, direct)
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            validate_uint64_transfer_columns(direct, direct[:1], direct)
