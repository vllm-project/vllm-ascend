# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest
from types import SimpleNamespace

import torch

from vllm_ascend.ops.statetree.methods import _unsupported_reason


class TestStateTreeMethods(unittest.TestCase):
    def setUp(self):
        self.weight = SimpleNamespace(
            ndim=2,
            dtype=torch.bfloat16,
            device=SimpleNamespace(type="npu"),
            shape=(2048, 1024),
            numel=lambda: 2048 * 1024,
        )
        self.layer = SimpleNamespace(
            weight=self.weight, bias=None, custom_op=None, prefix="model.layers.0.mlp.down_proj"
        )

    def test_supported_weight(self):
        self.assertIsNone(_unsupported_reason(self.layer))

    def test_unsupported_shape(self):
        self.weight.shape = (2016, 1024)
        self.assertIn("N%128/K%256", _unsupported_reason(self.layer))

    def test_specialized_layers_keep_dense_weights(self):
        for attribute, value in (
            ("bias", object()),
            ("custom_op", object()),
            ("skip_weight_nz_conversion", True),
            ("precast_fp32_weight", True),
            ("_statetree_tied_weight", True),
        ):
            with self.subTest(attribute=attribute):
                layer = SimpleNamespace(**vars(self.layer))
                setattr(layer, attribute, value)
                self.assertIsNotNone(_unsupported_reason(layer))
