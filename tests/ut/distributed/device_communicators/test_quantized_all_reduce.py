#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
"""Unit tests for the quantized all-reduce wrapper (mocked, no NPU needed).

Covers the same-sign maximum-value overflow regression reported in review:
with ws=4 and all ranks holding x == amax, the headroom scale
(global_amax * ws / 127) makes round(1/scale) == 32 and an unclamped 4-rank
sum is 128, which overflows int8 (HCCL wraps modulo 256).

The wrapper must therefore use the exact-bound scale
global_amax / (127 // ws) instead: |x/scale| <= max_q by construction, so
rounding can never exceed max_q and ws * max_q <= 127 holds for any input.
This needs no clamp kernel at all (int8 clamp_/clamp/hardtanh are ~250x
slower than minimum/maximum on Ascend).
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.distributed.device_communicators import npu_communicator
from vllm_ascend.distributed.device_communicators.quantized_all_reduce import (
    QuantizedAllReduce,
)


def _make(
    world_size: int = 4,
    min_numel: int = 8,
    device_type: AscendDeviceType = AscendDeviceType.A2,
) -> QuantizedAllReduce:
    """Build a wrapper; raises RuntimeError on unvalidated configs."""
    with patch(
        "vllm_ascend.distributed.device_communicators.quantized_all_reduce.get_ascend_device_type",
        return_value=device_type,
    ):
        group = MagicMock()
        with patch(
            "vllm_ascend.distributed.device_communicators.quantized_all_reduce.dist.get_world_size",
            return_value=world_size,
        ):
            return QuantizedAllReduce(group, min_numel=min_numel)


class TestQuantizedAllReduce(unittest.TestCase):
    def test_min_numel_from_config(self):
        wrapper = _make(min_numel=12345)
        self.assertEqual(wrapper.min_numel, 12345)

    def test_accepted_on_validated_device_families(self):
        for dt in (AscendDeviceType.A2, AscendDeviceType.A3):
            wrapper = _make(world_size=4, device_type=dt)
            self.assertEqual(wrapper.max_q, 31)

    def test_raises_on_unvalidated_device_families(self):
        for dt in (AscendDeviceType._310P, AscendDeviceType.A5):
            with self.assertRaisesRegex(RuntimeError, "only validated on"):
                _make(world_size=4, device_type=dt)

    def test_raises_above_max_tp_size(self):
        # ws=16 -> max_q=7 (3 effective int8 levels): refuse instead of
        # silently degrading accuracy.
        with self.assertRaisesRegex(RuntimeError, "supports TP size up to"):
            _make(world_size=16, device_type=AscendDeviceType.A2)
        wrapper = _make(world_size=8, device_type=AscendDeviceType.A2)
        self.assertEqual(wrapper.max_q, 15)

    def test_max_q_integer_safe_for_common_tp_sizes(self):
        for ws in (2, 4, 8):
            self.assertLessEqual(ws * (127 // ws), 127)

    def test_small_message_skips_quantization(self):
        wrapper = _make(min_numel=8)
        x = torch.randn(2, 3, dtype=torch.bfloat16)  # numel=6 < 8
        with (
            patch("torch.distributed.all_reduce") as mock_ar,
            patch("torch_npu.npu_quantize", create=True) as mock_q,
            patch("torch_npu.npu_anti_quant", create=True) as mock_aq,
        ):
            out = wrapper.all_reduce(x)
            # small messages fall back to a plain bf16 all-reduce
            mock_ar.assert_called_once()
            mock_q.assert_not_called()
            mock_aq.assert_not_called()
            self.assertTrue(out is x)

    def test_large_message_uses_quantized_path(self):
        wrapper = _make(min_numel=8)
        x = torch.randn(4, 8, dtype=torch.bfloat16)  # numel=32 >= 8

        q_fake = torch.zeros(4, 8, dtype=torch.int8)
        deq_fake = torch.randn_like(x)
        with (
            patch("torch.distributed.all_reduce") as mock_ar,
            patch("torch_npu.npu_quantize", create=True, return_value=q_fake) as mock_q,
            patch("torch_npu.npu_anti_quant", create=True, return_value=deq_fake) as mock_aq,
        ):
            out = wrapper.all_reduce(x)
            # two all-reduces happen: per-channel amax sync + int8 payload
            self.assertEqual(mock_ar.call_count, 2)
            mock_q.assert_called_once()
            mock_aq.assert_called_once()
            self.assertTrue(torch.equal(out, deq_fake))
            # result is written in place to keep the mutation contract
            self.assertTrue(torch.equal(x, deq_fake))
            # payload all-reduce must be int8
            second_call = mock_ar.call_args_list[1]
            self.assertEqual(second_call.args[0].dtype, torch.int8)

    def test_exact_bound_scale_never_wraps(self):
        """Same-sign worst case: every element equals the per-channel amax.

        With scale = amax / max_q and max_q = 127 // ws, x == amax quantizes
        to exactly max_q on every rank, so the reduced sum is ws * max_q:
        ws=2 -> 126, ws=4 -> 124, ws=8 -> 120. All <= 127, no clamp needed.
        """
        for ws in (2, 4, 8):
            wrapper = _make(world_size=ws)
            amax = torch.ones(8, dtype=torch.float32)
            scale = (amax / float(wrapper.max_q)).clamp_min(1e-8)
            x = torch.ones(ws, 8, dtype=torch.bfloat16) * amax[0]
            q = (x.float() / scale).round().to(torch.int8)
            self.assertLessEqual(q.abs().max().item(), wrapper.max_q)
            total = q.sum(dim=0).abs().max().item()
            self.assertLessEqual(total, 127)

    def test_wrapper_uses_exact_bound_scale_and_does_not_clamp(self):
        """The wrapper must use scale = amax / max_q and skip the clamp."""
        wrapper = _make(min_numel=8)
        x = torch.full((4, 8), 1.0, dtype=torch.bfloat16)  # x == amax
        captured: dict[str, list[torch.Tensor]] = {}

        def fake_ar(t, group=None, op=None):
            captured.setdefault("payload", []).append(t.clone())

        q_fake = torch.full((4, 8), wrapper.max_q, dtype=torch.int8)
        with (
            patch("torch.distributed.all_reduce", side_effect=fake_ar),
            patch("torch_npu.npu_quantize", create=True, return_value=q_fake) as mock_q,
            patch(
                "torch_npu.npu_anti_quant",
                create=True,
                return_value=torch.randn_like(x),
            ) as mock_aq,
        ):
            wrapper.all_reduce(x)
        # scale handed to npu_quantize must be global_amax / max_q = 1 / 31
        scale = mock_q.call_args.args[1]
        self.assertTrue(torch.allclose(scale, torch.full_like(scale, 1.0 / wrapper.max_q)))
        # [0] = amax fp32 sync, [1] = int8 payload; payload is exactly what
        # npu_quantize produced, i.e. nothing clamped it afterwards
        payload = captured["payload"][1]
        self.assertTrue(torch.equal(payload, q_fake))
        mock_aq.assert_called_once()


def _make_communicator(unique_name: str, world_size: int, enable: bool = True):
    """Build an NPUCommunicator with the base init stubbed out."""
    cfg = SimpleNamespace(quantized_all_reduce_config=SimpleNamespace(enable=enable, min_numel=8))

    def fake_base_init(self, cpu_group=None, device=None, device_group=None, unique_name="", **kwargs):
        self.unique_name = unique_name
        self.world_size = world_size
        self.device_group = device_group

    with (
        patch.object(npu_communicator.DeviceCommunicatorBase, "__init__", fake_base_init),
        patch.object(npu_communicator, "get_ascend_config", return_value=cfg),
        patch.object(npu_communicator.torch.npu, "current_device", return_value=0, create=True),
        patch(
            "vllm_ascend.distributed.device_communicators.quantized_all_reduce.get_ascend_device_type",
            return_value=AscendDeviceType.A2,
        ),
    ):
        return npu_communicator.NPUCommunicator(None, unique_name=unique_name)


class TestQuantizedAllReduceGroupScope(unittest.TestCase):
    """The wrapper must only be built for the tensor-parallel group."""

    def test_is_tp_group(self):
        for name in ("tp:0", "tp:1"):
            with self.subTest(name=name):
                self.assertTrue(npu_communicator._is_tp_group(name))
        # TP-like variants and every other parallel group must not match.
        for name in (
            "ep:0",
            "mc2:0",
            "dp:0",
            "dcp:0",
            "pcp:0",
            "pp:0",
            "eplb:0",
            "kvpp:0",
            "p_tp_0:0",
            "mlptp:0",
            "otp:0",
            "emtp:0",
            "etp:0",
            "world:0",
            "anonymous:0",
            "",
        ):
            with self.subTest(name=name):
                self.assertFalse(npu_communicator._is_tp_group(name))

    def test_only_tp_group_builds_the_wrapper(self):
        tp_comm = _make_communicator("tp:0", world_size=4)
        self.assertIsNotNone(tp_comm.quantized_ar)
        self.assertEqual(tp_comm.quantized_ar.world_size, 4)
        self.assertEqual(tp_comm.quantized_ar.max_q, 31)

        # A 16-rank EP/MC2 group must neither build a wrapper nor raise.
        for name in ("ep:0", "mc2:0"):
            with self.subTest(group=name):
                comm = _make_communicator(name, world_size=16)
                self.assertIsNone(comm.quantized_ar)
        comm = _make_communicator("dp:0", world_size=8)
        self.assertIsNone(comm.quantized_ar)

    def test_disabled_builds_nothing(self):
        comm = _make_communicator("tp:0", world_size=4, enable=False)
        self.assertIsNone(comm.quantized_ar)

    def test_tp_group_still_rejects_too_large_tp(self):
        with self.assertRaisesRegex(RuntimeError, "supports TP size up to"):
            _make_communicator("tp:0", world_size=16)


if __name__ == "__main__":
    unittest.main()
