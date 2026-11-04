# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest
from unittest.mock import call, patch

import torch

from vllm_ascend.ops.statetree import native
from vllm_ascend.ops.statetree.codec import pack


class TestStateTreeNative(unittest.TestCase):
    def test_prepare_compiles_all_dispatch_tiles(self):
        packed = pack(torch.ones(32, 64, dtype=torch.bfloat16), device="cpu")
        with patch.object(native, "build_kernel") as build:
            native.prepare(packed)
        self.assertEqual(
            build.call_args_list, [call(tile, 32, 64, packed.escape_slots) for tile in (16, 32, 64, 128, 256)]
        )

    def test_rejects_cpu_weights_before_launch(self):
        packed = pack(torch.ones(32, 64, dtype=torch.bfloat16), device="cpu")
        with patch.object(native, "build_kernel") as build:
            with self.assertRaisesRegex(ValueError, "must be on an NPU"):
                native.decode(packed)
            build.assert_not_called()
