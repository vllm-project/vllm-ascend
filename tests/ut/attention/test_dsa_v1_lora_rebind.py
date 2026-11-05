#
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
from unittest.mock import MagicMock, patch

import torch
from vllm.lora.layers.base import BaseLayerWithLoRA

from tests.ut.base import TestBase
from vllm_ascend.attention.dsa_v1 import (
    AscendDSAImpl,
    _is_lora_wrapped,
)


class _StubLoRAWrapper(BaseLayerWithLoRA):
    """isinstance-only stand-in: the impl keys off the wrapper type."""


def _lora_wrapper(delta_fn=None) -> BaseLayerWithLoRA:
    wrapper = object.__new__(_StubLoRAWrapper)
    # Bypass nn.Module.__setattr__: these instances are never initialised.
    # The public attributes mirror what a real wrapper forwards to its
    # base_layer, which is what CVLinearWrapper probes at construction time.
    object.__setattr__(wrapper, "quant_method", None)
    object.__setattr__(wrapper, "gather_output", False)
    object.__setattr__(wrapper, "_apply_lora_to_output", delta_fn or (lambda x, out: out))
    return wrapper


def _plain_layer() -> MagicMock:
    layer = MagicMock()
    layer.quant_method = None
    layer.gather_output = False
    # A LoRA-free layer must not be mistaken for a wrapper.
    layer.__class__ = MagicMock
    return layer


class TestDsaLoRAProjectionRebind(TestBase):
    """The DSA impl caches projections at build time.

    Enabling LoRA swaps the model tree's linears for wrappers, so the cached
    references keep pointing at the base layers. Reading ``weight`` off a
    wrapper still works (``__getattr__`` forwards it), which is why the adapter
    used to be dropped without any error instead of failing loudly.
    """

    PROJECTIONS = ("wq_a", "wq_b", "wkv", "wo_a", "wo_b")

    def setUp(self):
        super().setUp()
        # The global ascend config is not initialised in unit tests.
        patcher = patch("vllm_ascend.attention.dsa_v1.oproj_tp_enable", return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _impl(self, n_local_groups=1):
        impl = object.__new__(AscendDSAImpl)
        for name in self.PROJECTIONS:
            setattr(impl, name, _plain_layer())
        impl.cv_wq_a = MagicMock()
        impl.cv_wkv = MagicMock()
        impl.cv_wq_b = MagicMock()
        impl.n_local_groups = n_local_groups
        impl.multistream_dsv4_dsa_overlap = True
        return impl

    def test_plain_layer_is_not_a_wrapper(self):
        self.assertFalse(_is_lora_wrapped(_plain_layer()))
        self.assertTrue(_is_lora_wrapped(_lora_wrapper()))

    def test_rebind_adopts_patched_projections(self):
        impl = self._impl()
        patched = SimpleNamespace(**{name: _lora_wrapper() for name in self.PROJECTIONS})

        impl.sync_projection_layers(patched)

        for name in self.PROJECTIONS:
            self.assertIs(getattr(impl, name), getattr(patched, name))
        # The CV wrappers hold their own reference and must follow.
        for name in ("cv_wq_a", "cv_wkv", "cv_wq_b"):
            self.assertIs(getattr(impl, name).linear, getattr(impl, name.replace("cv_", "")))

    def test_second_sync_is_a_no_op(self):
        impl = self._impl()
        patched = SimpleNamespace(**{name: _lora_wrapper() for name in self.PROJECTIONS})

        impl.sync_projection_layers(patched)
        cv_after_first = {name: getattr(impl, name) for name in ("cv_wq_a", "cv_wkv", "cv_wq_b")}
        impl.sync_projection_layers(patched)

        for name, wrapper in cv_after_first.items():
            # Steady state must not reallocate the CV wrappers on every step.
            self.assertIs(getattr(impl, name), wrapper)

    def test_sync_ignores_absent_projections(self):
        impl = self._impl()
        only_wq_b = SimpleNamespace(wq_b=_lora_wrapper())

        impl.sync_projection_layers(only_wq_b)

        self.assertIs(impl.wq_b, only_wq_b.wq_b)
        self.assertIs(_is_lora_wrapped(impl.wq_a), False)

    def test_overlap_disabled_once_prolog_is_patched(self):
        impl = self._impl()
        self.assertTrue(impl._dsv4_overlap_enabled())

        impl.wkv = _lora_wrapper()
        self.assertTrue(impl._prolog_uses_lora())
        self.assertFalse(impl._dsv4_overlap_enabled())

    def test_wo_a_delta_is_skipped_without_adapter(self):
        impl = self._impl()
        x = torch.randn(4, 2, 8)
        out = torch.randn(4, 2, 8)

        self.assertIs(impl._add_wo_a_lora_delta(x, out), out)

    def test_wo_a_delta_is_delegated_with_flat_layout(self):
        impl = self._impl()
        seen = {}

        def delta(x, out):
            seen["x_shape"] = tuple(x.shape)
            seen["out_shape"] = tuple(out.shape)
            return out + 1.0

        impl.wo_a = _lora_wrapper(delta)
        x = torch.zeros(4, 2, 8)
        out = torch.zeros(4, 2, 8)

        result = impl._add_wo_a_lora_delta(x, out)

        self.assertEqual(seen["x_shape"], (4, 16))
        self.assertEqual(seen["out_shape"], (4, 16))
        self.assertEqual(result.shape, out.shape)
        self.assertTrue(torch.all(result == 1.0))

    def test_wo_a_delta_rejects_multi_group_projection(self):
        impl = self._impl(n_local_groups=2)
        impl.wo_a = _lora_wrapper()

        with self.assertRaises(NotImplementedError):
            impl._add_wo_a_lora_delta(torch.zeros(2, 2, 8), torch.zeros(2, 2, 8))

    def test_wo_a_delta_rejects_fine_grained_oproj_tp(self):
        impl = self._impl()
        impl.wo_a = _lora_wrapper()

        with (
            patch("vllm_ascend.attention.dsa_v1.oproj_tp_enable", return_value=True),
            self.assertRaises(NotImplementedError),
        ):
            impl._add_wo_a_lora_delta(torch.zeros(2, 2, 8), torch.zeros(2, 2, 8))
