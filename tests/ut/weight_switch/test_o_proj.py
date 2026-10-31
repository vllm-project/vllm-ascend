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
#
"""CPU-only unit tests for ``vllm_ascend.weight_switch.o_proj``.

Exercises the single-target O-proj weight-switch lifecycle: initialization,
linear-method resolution (including the nested ``quant_method`` indirection),
idempotent enabling, the not-enabled error, and the full-weight context
manager including its guaranteed restore on exception.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.weight_switch.linear import (
    WeightSwitchConfig,
    WeightSwitchGatherSpec,
    WeightSwitchMixin,
    WeightSwitchRepeatSpec,
)
from vllm_ascend.weight_switch.o_proj import OProjWeightSwitchMixin


class _Switchable(WeightSwitchMixin):
    supports_weight_switch = True
    weight_switch_gather_specs = (WeightSwitchGatherSpec("weight", gather_dim=0),)
    weight_switch_repeat_specs = (WeightSwitchRepeatSpec("bias", repeat_dim=0),)

    def apply(self, layer, x, bias=None):
        raise NotImplementedError


class _NotSwitchable(WeightSwitchMixin):
    supports_weight_switch = False

    def apply(self, layer, x, bias=None):
        raise NotImplementedError


class _Owner(OProjWeightSwitchMixin):
    """Bare owner object hosting the o_proj mixin machinery."""

    def __init__(self, linear_method):
        self.o_proj = _layer_with_weight()
        self.o_proj.quant_method = linear_method
        self.o_proj_full_pools = {}
        self._initialize_o_proj_weight_switch(WeightSwitchConfig(group="grp", world_size=2, rank=0, shard_axis="input"))


def _layer_with_weight():
    layer = torch.nn.Module()
    layer.register_parameter("weight", torch.nn.Parameter(torch.zeros(4, 8)))
    layer.register_parameter("bias", torch.nn.Parameter(torch.ones(8)))
    return layer


def _owner(linear_method=None):
    method = linear_method if linear_method is not None else _Switchable()
    return _Owner(method)


class TestOProjWeightSwitchLifecycle(unittest.TestCase):
    def test_initialize_sets_initial_state(self):
        owner = _owner()
        self.assertIsNotNone(owner.o_proj_weight_switch_config)
        self.assertIsNone(owner.o_proj_weight_state)
        self.assertFalse(owner._o_proj_weight_switch_enabled)

    def test_method_resolution_returns_mixin(self):
        method = _Switchable()
        owner = _owner(method)
        self.assertIs(owner._get_o_proj_weight_switch_method(), method)

    def test_method_resolution_follows_nested_quant_method(self):
        inner = _Switchable()
        quant = MagicMock()
        quant.quant_method = inner
        owner = _owner(quant)
        self.assertIs(owner._get_o_proj_weight_switch_method(), inner)

    def test_method_resolution_rejects_plain_object(self):
        owner = _owner(object())
        with self.assertRaises(RuntimeError) as ctx:
            owner._get_o_proj_weight_switch_method()
        self.assertIn("weight-switch capable", str(ctx.exception))

    def test_method_resolution_rejects_unsupported_mixin(self):
        owner = _owner(_NotSwitchable())
        with self.assertRaises(RuntimeError) as ctx:
            owner._get_o_proj_weight_switch_method()
        self.assertIn("weight-switch capable", str(ctx.exception))

    def test_state_rejected_before_enable(self):
        owner = _owner()
        with self.assertRaises(RuntimeError) as ctx:
            owner._get_o_proj_weight_switch_state()
        self.assertIn("has not been enabled", str(ctx.exception))

    def test_enable_is_idempotent_and_uses_pool(self):
        owner = _owner()
        owner._enable_o_proj_full_weight_switch()
        self.assertTrue(owner._o_proj_weight_switch_enabled)
        first_state = owner.o_proj_weight_state
        # The full-size buffer is registered in the owner-provided pool under
        # the (qualname, pool-key) prefix; the view is not flipped yet.
        self.assertEqual(len(owner.o_proj_full_pools), 1)
        self.assertEqual(next(iter(owner.o_proj_full_pools.values())).shape, (8, 8))
        self.assertEqual(owner.o_proj.weight.shape, (4, 8))
        owner._enable_o_proj_full_weight_switch()
        self.assertIs(owner.o_proj_weight_state, first_state)
        self.assertEqual(owner._get_o_proj_weight_switch_state(), first_state)

    @patch("vllm_ascend.distributed.utils.all_gather_async")
    def test_all_gather_enables_and_collectives(self, mock_gather):
        mock_gather.return_value = (None, MagicMock())
        owner = _owner()
        owner._all_gather_o_proj_full_weight()
        self.assertTrue(owner._o_proj_weight_switch_enabled)
        self.assertEqual(len(owner.o_proj_weight_state.handles), 1)
        mock_gather.assert_called_once()

    def test_context_manager_switches_and_restores(self):
        owner = _owner()
        owner._enable_o_proj_full_weight_switch()
        owner.o_proj_weight_state.gather_parts["weight"].full_tensor.copy_(torch.ones(8, 8))
        with owner._use_full_o_proj_weights():
            self.assertEqual(owner.o_proj.weight.shape, (8, 8))
            self.assertTrue(torch.all(owner.o_proj.weight == 1.0))
        # After the block the local view is back.
        self.assertEqual(owner.o_proj.weight.shape, (4, 8))
        self.assertTrue(torch.all(owner.o_proj.weight == 0.0))

    def test_context_manager_restores_on_exception(self):
        owner = _owner()
        owner._enable_o_proj_full_weight_switch()
        with self.assertRaises(RuntimeError), owner._use_full_o_proj_weights():
            raise RuntimeError("inner")
        self.assertEqual(owner.o_proj.weight.shape, (4, 8))

    def test_context_manager_requires_enable(self):
        owner = _owner()
        with self.assertRaises(RuntimeError), owner._use_full_o_proj_weights():
            pass

    @patch("vllm_ascend.distributed.utils.all_gather_async")
    def test_all_gather_rejects_when_pending(self, mock_gather):
        mock_gather.return_value = (None, MagicMock())
        owner = _owner()
        owner._all_gather_o_proj_full_weight()
        with self.assertRaises(RuntimeError) as ctx:
            owner._all_gather_o_proj_full_weight()
        self.assertIn("still pending", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
