# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import torch
from vllm.lora.layers import ColumnParallelLinearWithShardedLoRA
from vllm.lora.layers.base import BaseLayerWithLoRA

from tests.ut.base import TestBase
from vllm_ascend.attention.dsa_v1 import AscendDSAImpl, _is_lora_wrapped

DSA_MOD = "vllm_ascend.attention.dsa_v1"


class _StubLoRA(BaseLayerWithLoRA):
    """isinstance-only stand-in: the impl keys off the wrapper type."""


class _StubShardedLoRA(ColumnParallelLinearWithShardedLoRA):
    """isinstance-only stand-in for --fully-sharded-loras."""


def _lora_wrapper(delta_fn=None, *, sharded: bool = False, **attrs) -> BaseLayerWithLoRA:
    wrapper = object.__new__(_StubShardedLoRA if sharded else _StubLoRA)
    base = dict(quant_method=None, gather_output=False, _apply_lora_to_output=delta_fn or (lambda _x, out: out))
    for k, v in {**base, **attrs}.items():
        object.__setattr__(wrapper, k, v)
    return wrapper


def _plain_layer() -> MagicMock:
    layer = MagicMock(quant_method=None, gather_output=False)
    layer.__class__ = MagicMock
    return layer


class TestDsaLoRAProjectionRebind(TestBase):
    """Verify DSA projection rebind, overlap disabling, and wo_a LoRA guards."""

    PROJECTIONS = ("wq_a", "wq_b", "wkv", "wo_a", "wo_b")

    def setUp(self):
        super().setUp()
        patcher = patch(f"{DSA_MOD}.oproj_tp_enable", return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _impl(self, n_local_groups=1):
        impl = object.__new__(AscendDSAImpl)
        for name in self.PROJECTIONS:
            setattr(impl, name, _plain_layer())
        impl.cv_wq_a, impl.cv_wkv, impl.cv_wq_b = MagicMock(), MagicMock(), MagicMock()
        impl.n_local_groups, impl.multistream_dsv4_dsa_overlap = n_local_groups, True
        return impl

    def test_rebind_idempotency_and_partial_sync(self):
        self.assertFalse(_is_lora_wrapped(_plain_layer()))
        self.assertTrue(_is_lora_wrapped(_lora_wrapper()))
        impl = self._impl()
        patched = SimpleNamespace(**{n: _lora_wrapper() for n in self.PROJECTIONS})
        impl.sync_projection_layers(patched)
        for n in self.PROJECTIONS:
            self.assertIs(getattr(impl, n), getattr(patched, n))
        for n in ("cv_wq_a", "cv_wkv", "cv_wq_b"):
            self.assertIs(getattr(impl, n).linear, getattr(impl, n.removeprefix("cv_")))
        cv_refs = {n: getattr(impl, n) for n in ("cv_wq_a", "cv_wkv", "cv_wq_b")}
        impl.sync_projection_layers(patched)
        for n, ref in cv_refs.items():
            self.assertIs(getattr(impl, n), ref)

        partial, only_wq_b = self._impl(), SimpleNamespace(wq_b=_lora_wrapper())
        partial.sync_projection_layers(only_wq_b)
        self.assertIs(partial.wq_b, only_wq_b.wq_b)
        self.assertFalse(_is_lora_wrapped(partial.wq_a))

    def test_wo_a_delta_skip_delegate_and_rejects(self):
        x, out = torch.zeros(4, 2, 8), torch.zeros(4, 2, 8)
        self.assertIs(self._impl()._add_wo_a_lora_delta(x, out), out)
        idle, act0, act1 = (
            SimpleNamespace(no_lora=True, active_lora_slots=set()),
            SimpleNamespace(no_lora=False, active_lora_slots={0}),
            SimpleNamespace(no_lora=False, active_lora_slots={1}),
        )
        for attrs, otp in (
            (dict(punica_wrapper=idle), False),
            (dict(punica_wrapper=idle), True),
            (dict(_lora_slots=set(), punica_wrapper=act0), False),
            (dict(_lora_slots=set(), punica_wrapper=act0), True),
            (dict(_lora_slots={0}, punica_wrapper=act1), False),
        ):
            impl = self._impl(n_local_groups=2)
            impl.wo_a = _lora_wrapper(**attrs)
            with patch(f"{DSA_MOD}.oproj_tp_enable", return_value=otp):
                self.assertIs(impl._add_wo_a_lora_delta(x, out), out)

        seen: dict[str, tuple[int, ...]] = {}
        impl = self._impl()
        impl.wo_a = _lora_wrapper(lambda xi, oi: seen.update(x=tuple(xi.shape), out=tuple(oi.shape)) or oi + 1.0)
        res = impl._add_wo_a_lora_delta(x, out)
        self.assertEqual((seen["x"], seen["out"], res.shape), ((4, 16), (4, 16), out.shape))
        self.assertTrue(torch.all(res == 1.0))

        # Small-M single-group decode fast path must also apply the wo_a LoRA delta.
        small_x = torch.ones(4, 1, 8)
        small_impl = self._impl(n_local_groups=1)
        small_impl.support_fp8_attention = False
        small_impl.wo_a = _lora_wrapper(
            lambda _xi, oi: oi + 2.0,
            weight=torch.ones(1, 8, 6),
            weight_scale=None,
        )
        small_res = small_impl._wo_a_bmm(small_x)
        self.assertEqual(small_res.shape, (4, 1, 6))
        torch.testing.assert_close(small_res, torch.full((4, 1, 6), 10.0))

        err_cases: tuple[tuple[int, bool, bool, dict[str, Any], str], ...] = (
            (2, False, False, dict(_lora_slots={0}, punica_wrapper=act0), ""),
            (1, True, False, {}, ""),
            (1, False, True, dict(punica_wrapper=act0), "fully-sharded-loras"),
        )
        for n_g, otp, sharded, attrs, msg in err_cases:
            bad = self._impl(n_local_groups=n_g)
            bad.wo_a = _lora_wrapper(sharded=sharded, **attrs)
            with (
                patch(f"{DSA_MOD}.oproj_tp_enable", return_value=otp),
                self.assertRaisesRegex(NotImplementedError, msg),
            ):
                bad._add_wo_a_lora_delta(x, out)

    def test_prolog_lora_disables_overlaps_and_w8a8_qr_prequant(self):
        from vllm_ascend.quantization.methods.w8a8 import AscendW8A8DynamicLinearMethod

        impl = self._impl()
        self.assertTrue(not impl._prolog_uses_lora() and impl._dsv4_overlap_enabled())
        impl.wkv = _lora_wrapper()
        self.assertTrue(impl._prolog_uses_lora() and not impl._dsv4_overlap_enabled())
        impl = self._impl()
        impl.indexer = SimpleNamespace(wq_b=_lora_wrapper())
        self.assertTrue(impl._prolog_uses_lora() and not impl._dsv4_overlap_enabled())

        impl.wq_a, impl.wkv = (
            MagicMock(return_value=torch.ones(2, 8, dtype=torch.bfloat16), quant_method=None),
            MagicMock(quant_method=None),
        )
        impl.wq_b = MagicMock(
            return_value=torch.ones(2, 16, dtype=torch.bfloat16),
            quant_method=SimpleNamespace(quant_method=object.__new__(AscendW8A8DynamicLinearMethod)),
        )
        impl.q_norm = MagicMock(side_effect=lambda t: t + 1.0)
        (
            impl.eps,
            impl.n_local_heads,
            impl.head_dim,
            impl.nope_head_dim,
            impl.q_norm_without_weight,
            impl.rope_head_dim,
        ) = 1e-6, 2, 8, 4, False, 4
        fake_c = SimpleNamespace(
            npu_rms_norm_dynamic_quant=MagicMock(side_effect=AssertionError("must not pre-quantize qr")),
            inplace_partial_rotary_mul=MagicMock(),
        )
        with (
            patch(f"{DSA_MOD}.DeviceOperator.apply_dsa_q_rms", side_effect=lambda q, *_: q),
            patch.object(torch.ops, "_C_ascend", fake_c, create=True),
        ):
            q, qr, qr_scale = impl._mla_prolog_single_stream(
                torch.ones(2, 8, dtype=torch.bfloat16),
                torch.ones(2, 4, dtype=torch.bfloat16),
                torch.zeros(2, 4, dtype=torch.bfloat16),
                swa_kv_cache=None,
                slot_mapping=None,
                write_swa_cache=False,
            )
        fake_c.npu_rms_norm_dynamic_quant.assert_not_called()
        self.assertIsNone(qr_scale)
        self.assertEqual((qr.dtype, tuple(q.shape)), (torch.bfloat16, (2, 2, 8)))

        attn = SimpleNamespace(wq_a=impl.wq_a, wq_b=impl.wq_b, wkv=impl.wkv, wo_a=impl.wo_a, wo_b=impl.wo_b)
        with patch(f"{DSA_MOD}.CVLinearWrapper", side_effect=lambda mod: mod):
            ov = self._impl()
            ov.pcp_q_projection_overlap = True
            ov.sync_projection_layers(attn)
            self.assertTrue(ov.pcp_q_projection_overlap)
            attn.wq_a = _lora_wrapper()
            ov.sync_projection_layers(attn)
            self.assertFalse(ov.pcp_q_projection_overlap)
            ov.pcp_q_projection_overlap, attn.wq_a, ov.wq_a, ov.indexer = (
                True,
                SimpleNamespace(),
                SimpleNamespace(),
                SimpleNamespace(wq_b=_lora_wrapper()),
            )
            ov.sync_projection_layers(attn)
            self.assertFalse(ov.pcp_q_projection_overlap)
