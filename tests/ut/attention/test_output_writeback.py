# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl


class _CopyCounter(TorchDispatchMode):
    def __init__(self):
        self.copies = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten.copy_.default:
            self.copies += 1
        return func(*args, **(kwargs or {}))


class TestAttentionOutputWriteback(TestCase):
    def test_output_identity_and_replacement(self):
        # A distinct view must retain write-back even if it shares storage.
        for pooling in (False, True):
            for kind in ("same", "view", "replacement", "padded"):
                with self.subTest(pooling=pooling, kind=kind):
                    query = torch.zeros(3, 2, 4)
                    output = torch.full_like(query, 7.0)
                    result = output
                    if kind == "view":
                        result = output.view_as(output)
                    elif kind == "replacement":
                        result = torch.full_like(output, 5.0)
                    elif kind == "padded":
                        result = torch.full((8, 2, 4), 9.0)
                    impl = object.__new__(AscendAttentionBackendImpl)
                    impl.key_cache = None
                    impl.forward_impl = Mock(return_value=result)
                    impl._forward_encoder_attention = Mock(return_value=result)
                    impl.reshape_and_cache = Mock(return_value=(query, query, query, result))
                    metadata = SimpleNamespace(
                        model_runner_type="pooling" if pooling else "generate",
                        causal=False,
                        uses_pcp_sharded_layout=False,
                    )
                    layer = SimpleNamespace(layer_name="layer", _k_scale_float=1.0, _v_scale_float=1.0)
                    counter = _CopyCounter()
                    with counter:
                        actual = impl.forward(layer, query, query, query, (), metadata, output)
                    self.assertIs(actual, output)
                    self.assertEqual(counter.copies, 0 if kind == "same" else 1)
                    torch.testing.assert_close(output, result[:3])

    def test_profile_output_remains_zero(self):
        impl = object.__new__(AscendAttentionBackendImpl)
        query = torch.ones(3, 2, 4)
        output = torch.full_like(query, float("nan"))
        layer = SimpleNamespace(layer_name="layer", _k_scale_float=1.0, _v_scale_float=1.0)
        result = impl.forward(layer, query, None, None, (), None, output)
        self.assertIs(result, output)
        torch.testing.assert_close(output, torch.zeros_like(output))
