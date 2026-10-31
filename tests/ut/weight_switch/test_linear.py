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
"""CPU-only unit tests for ``vllm_ascend.weight_switch.linear``.

The module is a parallel-domain-agnostic weight-switching toolkit: small
frozen dataclasses for the parallel description, and a mixin that resolves
which axis of a linear layer is sharded, resizes parameters for nested
checkpoint loads, and allocates local/full tensor aliases that can be
swapped with ``set_``.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.distributed import utils as _distributed_utils  # noqa: F401  (register submodule for patching)
from vllm_ascend.weight_switch.linear import (
    WeightLoadPartition,
    WeightSwitchConfig,
    WeightSwitchGatherSpec,
    WeightSwitchLoadState,
    WeightSwitchMixin,
    WeightSwitchRepeatSpec,
)


class _DummyGroup:
    """Minimal stand-in for a vLLM process group."""

    def __init__(self, world_size, rank):
        self.world_size = world_size
        self.rank_in_group = rank


class _Switchable(WeightSwitchMixin):
    """A concrete mixin with declarative input- and output-axis specs."""

    supports_weight_switch = True
    weight_switch_gather_specs = (WeightSwitchGatherSpec("weight", gather_dim=0),)
    weight_switch_repeat_specs = (WeightSwitchRepeatSpec("bias", repeat_dim=0),)
    weight_switch_output_gather_specs = (WeightSwitchGatherSpec("weight", gather_dim=1),)
    weight_switch_output_repeat_specs = (WeightSwitchRepeatSpec("bias", repeat_dim=0),)

    def apply(self, layer, x, bias=None):
        raise NotImplementedError


class _NotSwitchable(WeightSwitchMixin):
    supports_weight_switch = False

    def apply(self, layer, x, bias=None):
        raise NotImplementedError


class TestWeightSwitchConfig(unittest.TestCase):
    def test_valid_config(self):
        config = WeightSwitchConfig(group=object(), world_size=4, rank=3, shard_axis="input")
        self.assertEqual(config.world_size, 4)
        self.assertEqual(config.rank, 3)
        self.assertEqual(config.shard_axis, "input")

    def test_rank_out_of_bounds_raises(self):
        with self.assertRaises(ValueError) as ctx:
            WeightSwitchConfig(group=None, world_size=2, rank=2)
        self.assertIn("rank=2", str(ctx.exception))
        with self.assertRaises(ValueError):
            WeightSwitchConfig(group=None, world_size=2, rank=-1)

    def test_non_positive_world_size_raises(self):
        with self.assertRaises(ValueError):
            WeightSwitchConfig(group=None, world_size=0, rank=0)

    def test_from_group_reads_group_attributes(self):
        config = WeightSwitchConfig.from_group(_DummyGroup(8, 5), shard_axis="output")
        self.assertEqual(config.world_size, 8)
        self.assertEqual(config.rank, 5)
        self.assertEqual(config.shard_axis, "output")


class TestWeightLoadPartition(unittest.TestCase):
    def test_validation(self):
        partition = WeightLoadPartition(world_size=3, rank=2)
        self.assertEqual(partition.world_size, 3)
        self.assertEqual(partition.rank, 2)
        with self.assertRaises(ValueError):
            WeightLoadPartition(world_size=1, rank=1)
        with self.assertRaises(ValueError):
            WeightLoadPartition(world_size=-2, rank=0)

    def test_from_nested_groups_composes_rank(self):
        # Outer (2, rank 1) then inner (3, rank 0): composed rank = 1*3 + 0.
        partition = WeightLoadPartition.from_nested_groups(_DummyGroup(2, 1), _DummyGroup(3, 0))
        self.assertEqual(partition.world_size, 6)
        self.assertEqual(partition.rank, 3)
        # Outer (4, rank 0) then inner (2, rank 1): composed rank = 0*2 + 1.
        partition = WeightLoadPartition.from_nested_groups(_DummyGroup(4, 0), _DummyGroup(2, 1))
        self.assertEqual(partition.world_size, 8)
        self.assertEqual(partition.rank, 1)


class TestSpecResolution(unittest.TestCase):
    def _layer(self, **attrs):
        layer = torch.nn.Module()
        for key, value in attrs.items():
            setattr(layer, key, value)
        return layer

    def test_input_sharded_layer_selects_input_specs(self):
        layer = self._layer(input_size=12, input_size_per_partition=6, output_size=128)
        config = WeightSwitchConfig(group=None, world_size=2, rank=0)
        gather, repeat, axis = _Switchable()._get_weight_switch_specs(layer, config)
        self.assertEqual(axis, "input")
        self.assertIs(gather, _Switchable.weight_switch_gather_specs)
        self.assertIs(repeat, _Switchable.weight_switch_repeat_specs)

    def test_output_sharded_layer_selects_output_specs(self):
        layer = self._layer(input_size=128, output_size=32, output_size_per_partition=16)
        config = WeightSwitchConfig(group=None, world_size=2, rank=1)
        gather, repeat, axis = _Switchable()._get_weight_switch_specs(layer, config)
        self.assertEqual(axis, "output")
        self.assertIs(gather, _Switchable.weight_switch_output_gather_specs)
        self.assertIs(repeat, _Switchable.weight_switch_output_repeat_specs)

    def test_both_axes_sharded_raises(self):
        layer = self._layer(input_size=24, input_size_per_partition=12, output_size=32, output_size_per_partition=16)
        config = WeightSwitchConfig(group=None, world_size=2, rank=0)
        with self.assertRaises(RuntimeError) as ctx:
            _Switchable()._get_weight_switch_specs(layer, config)
        self.assertIn("at most one pre-existing sharded axis", str(ctx.exception))

    def test_unsharded_layer_uses_config_shard_axis(self):
        layer = self._layer(input_size=64, input_size_per_partition=64, output_size=128, output_size_per_partition=128)
        config = WeightSwitchConfig(group=None, world_size=2, rank=0, shard_axis="output")
        _, _, axis = _Switchable()._get_weight_switch_specs(layer, config)
        self.assertEqual(axis, "output")

    def test_unsharded_layer_without_shard_axis_raises(self):
        layer = self._layer(input_size=64, input_size_per_partition=64)
        config = WeightSwitchConfig(group=None, world_size=2, rank=0)
        with self.assertRaises(RuntimeError) as ctx:
            _Switchable()._get_weight_switch_specs(layer, config)
        self.assertIn("shard_axis", str(ctx.exception))


class TestSplitTensorForParallel(unittest.TestCase):
    @staticmethod
    def split(*args, **kwargs):
        return WeightSwitchMixin.split_tensor_for_parallel(*args, **kwargs)

    def test_slices_correct_rank(self):
        tensor = torch.arange(24).reshape(4, 6)
        shard = self.split(tensor, world_size=2, rank=1, dim=0)
        torch.testing.assert_close(shard, tensor[2:4, :])
        shard = self.split(tensor, world_size=3, rank=2, dim=1)
        torch.testing.assert_close(shard, tensor[:, 4:6])

    def test_negative_dim_is_normalized(self):
        tensor = torch.arange(12).reshape(2, 6)
        torch.testing.assert_close(self.split(tensor, world_size=3, rank=1, dim=-1), tensor[:, 2:4])

    def test_non_divisible_dimension_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            self.split(torch.zeros(4, 5), world_size=3, rank=0, dim=1)
        self.assertIn("not divisible", str(ctx.exception))

    def test_contiguity_flag_is_honored(self):
        base = torch.zeros(6, 4).t()  # (4, 6) non-contiguous
        self.assertFalse(base.is_contiguous())
        self.assertFalse(self.split(base, world_size=2, rank=0, dim=0, contiguous=False).is_contiguous())
        self.assertTrue(self.split(base, world_size=2, rank=0, dim=0, contiguous=True).is_contiguous())


class _ShardedLayer:
    """Build a module whose single parameter records its sharded dim."""

    @staticmethod
    def make(param_shape, shard_dim, axis="input", width=8, extra=None):
        layer = torch.nn.Module()
        param = torch.nn.Parameter(torch.zeros(*param_shape))
        setattr(param, f"{axis}_dim", shard_dim)
        param.weight_loader = lambda target, loaded, *a, **kw: target.data.copy_(loaded)
        layer.register_parameter("weight", param)
        setattr(layer, f"{axis}_size_per_partition", width)
        if extra:
            for key, value in extra.items():
                setattr(layer, key, value)
        return layer


class TestNestedWeightLoad(unittest.TestCase):
    def setUp(self):
        self.method = _Switchable()
        self.config = WeightSwitchConfig(group=None, world_size=2, rank=0)

    def _partition(self, world_size, rank):
        return WeightLoadPartition(world_size=world_size, rank=rank)

    def test_unsupported_method_raises(self):
        layer = _ShardedLayer.make((4, 8), shard_dim=0, width=8)
        with self.assertRaises(RuntimeError) as ctx:
            _NotSwitchable().prepare_layer_for_parallel_weight_load(layer, self.config, self._partition(2, 0))
        self.assertIn("does not support weight switching", str(ctx.exception))

    def test_missing_width_attribute_raises(self):
        layer = torch.nn.Module()
        layer.register_parameter("weight", torch.nn.Parameter(torch.zeros(4, 8)))
        with self.assertRaises(RuntimeError) as ctx:
            self.method.prepare_layer_for_parallel_weight_load(layer, self.config, self._partition(2, 0))
        self.assertIn("input_size_per_partition", str(ctx.exception))

    def test_non_divisible_width_raises(self):
        # width 7 is not divisible by world_size 2.
        layer = _ShardedLayer.make((4, 8), shard_dim=0, width=7)
        with self.assertRaises(RuntimeError) as ctx:
            self.method.prepare_layer_for_parallel_weight_load(layer, self.config, self._partition(2, 0))
        self.assertIn("Cannot split", str(ctx.exception))

    def test_non_divisible_physical_shape_raises(self):
        # Width 8 splits fine, but the parameter's sharded dim (size 3) does not.
        layer = _ShardedLayer.make((3, 8), shard_dim=0, width=8)
        with self.assertRaises(RuntimeError) as ctx:
            self.method.prepare_layer_for_parallel_weight_load(layer, self.config, self._partition(2, 0))
        self.assertIn("Cannot resize", str(ctx.exception))

    def test_sharded_parameter_without_loader_raises(self):
        layer = _ShardedLayer.make((4, 8), shard_dim=0, width=8)
        layer.weight.weight_loader = None
        with self.assertRaises(RuntimeError) as ctx:
            self.method.prepare_layer_for_parallel_weight_load(layer, self.config, self._partition(2, 0))
        self.assertIn("requires weight_loader", str(ctx.exception))

    def test_no_sharded_parameters_raises(self):
        layer = _ShardedLayer.make((4, 8), shard_dim=0, width=8)
        del layer.weight.input_dim  # the param then no longer declares a shard dim
        with self.assertRaises(RuntimeError) as ctx:
            self.method.prepare_layer_for_parallel_weight_load(layer, self.config, self._partition(2, 0))
        self.assertIn("found no", str(ctx.exception))

    def test_resizes_parameter_and_wraps_loader(self):
        layer = _ShardedLayer.make((8, 16), shard_dim=0, width=8)
        full = torch.arange(8 * 16, dtype=torch.float32).reshape(8, 16)
        result = self.method.prepare_layer_for_parallel_weight_load(layer, self.config, self._partition(2, 1))
        self.assertIsInstance(result, WeightSwitchLoadState)
        self.assertEqual(result.input_size_per_partition_before, 8)
        self.assertEqual(result.input_size_per_partition_after, 4)
        self.assertEqual(layer.input_size_per_partition, 4)
        self.assertEqual(layer.weight.shape, (4, 16))
        # The wrapped loader selects load_partition.rank's slice of the full weight.
        layer.weight.weight_loader(layer.weight, full.clone())
        torch.testing.assert_close(layer.weight.data, full[4:8, :])

    def test_output_axis_uses_physical_shape_for_shard_size(self):
        layer = _ShardedLayer.make(
            (8, 8), shard_dim=1, axis="output", width=8, extra={"output_partition_sizes": [8, 8]}
        )
        self.method.prepare_layer_for_parallel_output_weight_load(layer, self.config, self._partition(2, 1))
        self.assertEqual(layer.weight.shape, (8, 4))
        # The partition sizes are rescaled to the TP-local width as well.
        self.assertEqual(layer.output_partition_sizes, [4, 4])
        full = torch.arange(8 * 8, dtype=torch.float32).reshape(8, 8)
        layer.weight.weight_loader(layer.weight, full.clone())
        torch.testing.assert_close(layer.weight.data, full[:, 4:8])

    def test_sharded_weight_skips_internal_narrow(self):
        # is_sharded_weight=True and the loaded weight already matches the
        # resized parameter: the value is copied through untouched.
        layer = _ShardedLayer.make((4, 16), shard_dim=0, width=8)
        layer.weight.is_sharded_weight = True
        self.method.prepare_layer_for_parallel_weight_load(layer, self.config, self._partition(2, 1))
        self.assertEqual(layer.weight.shape, (2, 16))
        loaded = torch.full((2, 16), 7.0)
        layer.weight.weight_loader(layer.weight, loaded)
        torch.testing.assert_close(layer.weight.data, loaded)

    def test_v2_loader_swaps_and_restores_tp_rank(self):
        calls = []

        def v2_loader(target_param, loaded_weight, *args, **kwargs):
            calls.append((target_param.tp_rank, target_param.tp_size))
            target_param.data.copy_(loaded_weight)
            return None

        layer = _ShardedLayer.make((4, 16), shard_dim=0, width=8)
        param = layer.weight
        param.weight_loader = v2_loader
        param.load_input_parallel_weight = v2_loader
        param.tp_rank = 0
        param.tp_size = 4
        layer.weight_loader_v2 = v2_loader

        self.method.prepare_layer_for_parallel_weight_load(layer, self.config, self._partition(2, 1))
        param.weight_loader(param, torch.zeros(2, 16))
        # The V2 implementation must have observed the composed checkpoint rank.
        self.assertEqual(calls, [(1, 2)])
        # ...and the forward TP identity must be restored afterwards.
        self.assertEqual(param.tp_rank, 0)
        self.assertEqual(param.tp_size, 4)

    def test_v1_loader_path_copies_narrowed_shard_directly(self):
        layer = _ShardedLayer.make((4, 16), shard_dim=0, width=8)
        param = layer.weight
        param.load_input_parallel_weight = param.weight_loader
        param.tp_rank = 0
        param.tp_size = 4
        # Without a layer-level weight_loader_v2 the wrapper performs the
        # narrow and copy itself; the forward TP identity must be untouched.
        full = torch.arange(8 * 16, dtype=torch.float32).reshape(8, 16)
        self.method.prepare_layer_for_parallel_weight_load(layer, self.config, self._partition(2, 1))
        param.weight_loader(param, full.clone())
        self.assertEqual(param.tp_rank, 0)
        self.assertEqual(param.tp_size, 4)
        torch.testing.assert_close(param.data, full[2:4, :])


class TestEnableWeightSwitch(unittest.TestCase):
    def _layer(self, weight_shape=(4, 8)):
        layer = torch.nn.Module()
        layer.register_parameter(
            "weight",
            torch.nn.Parameter(
                torch.arange(weight_shape[0] * weight_shape[1], dtype=torch.float32).reshape(*weight_shape)
            ),
        )
        if len(weight_shape) == 1:
            layer.register_parameter("bias", torch.nn.Parameter(torch.ones(weight_shape[0])))
        else:
            layer.register_parameter("bias", torch.nn.Parameter(torch.ones(weight_shape[1])))
        return layer

    def _method(self):
        return _Switchable()

    def _config(self, world_size=2):
        # The bare test layer is not pre-sharded, so the config must name the axis.
        return WeightSwitchConfig(group=None, world_size=world_size, rank=0, shard_axis="input")

    def test_unsupported_method_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            _NotSwitchable().enable_weight_switch(self._layer(), self._config())
        self.assertIn("does not support weight switching", str(ctx.exception))

    def test_missing_declared_attribute_raises(self):
        class _MissingBias(WeightSwitchMixin):
            supports_weight_switch = True
            weight_switch_gather_specs = (WeightSwitchGatherSpec("weight"),)
            weight_switch_repeat_specs = (WeightSwitchRepeatSpec("bias"),)

            def apply(self, layer, x, bias=None):
                raise NotImplementedError

        layer = torch.nn.Module()
        layer.register_parameter("weight", torch.nn.Parameter(torch.zeros(4, 8)))
        with self.assertRaises(RuntimeError) as ctx:
            _MissingBias().enable_weight_switch(layer, self._config())
        self.assertIn("does not have it", str(ctx.exception))

    def test_invalid_gather_dim_raises(self):
        class _BadDim(WeightSwitchMixin):
            supports_weight_switch = True
            weight_switch_gather_specs = (WeightSwitchGatherSpec("weight", gather_dim=3),)

            def apply(self, layer, x, bias=None):
                raise NotImplementedError

        with self.assertRaises(RuntimeError) as ctx:
            _BadDim().enable_weight_switch(self._layer(), self._config())
        self.assertIn("Invalid gather dim", str(ctx.exception))

    def test_gather_dim_zero_aliases(self):
        layer = self._layer()
        state = self._method().enable_weight_switch(layer, self._config())
        part = state.gather_parts["weight"]
        self.assertEqual(part.gather_output.shape, (8, 8))
        # dim 0: the local tensor is the gather input as-is.
        self.assertIs(part.gather_input, part.local_tensor)
        self.assertIs(part.full_tensor, part.gather_output)
        # dim 1: gather_input moves the gather axis to the front.
        state1 = _BadDim1().enable_weight_switch(self._layer(), self._config())
        part1 = state1.gather_parts["weight"]
        self.assertEqual(part1.gather_input.shape, (8, 4))
        self.assertEqual(part1.gather_output.shape, (16, 4))
        self.assertEqual(part1.full_tensor.shape, (4, 16))

    def test_repeat_part_repeats_locally(self):
        layer = self._layer()
        state = self._method().enable_weight_switch(layer, self._config())
        part = state.repeat_parts["bias"]
        self.assertEqual(part.full_tensor.shape, (16,))
        torch.testing.assert_close(part.full_tensor, torch.cat([part.local_tensor, part.local_tensor]))
        self.assertIsNot(part.full_tensor, part.local_tensor)

    def test_pool_reuses_buffers_across_calls(self):
        layer = self._layer()
        pool = {}
        first = self._method().enable_weight_switch(layer, self._config(), pool=pool, pool_key_prefix="k")
        self.assertEqual(len(pool), 1)
        second = self._method().enable_weight_switch(layer, self._config(), pool=pool, pool_key_prefix="k")
        # Same prefix/attr/device/data_ptr key -> same pooled buffer object.
        self.assertIs(first.gather_parts["weight"].gather_output, second.gather_parts["weight"].gather_output)

    def test_clone_local_tensors_rebinds_layer_weight(self):
        layer = self._layer()
        before = layer.weight.data_ptr()
        state = self._method().enable_weight_switch(layer, self._config(), clone_local_tensors=True)
        local = state.gather_parts["weight"].local_tensor
        # The layer attribute now shares storage with the clone; values intact.
        self.assertEqual(local.data_ptr(), layer.weight.data_ptr())
        self.assertNotEqual(before, layer.weight.data_ptr())
        torch.testing.assert_close(layer.weight, torch.arange(32, dtype=torch.float32).reshape(4, 8))


class _BadDim1(WeightSwitchMixin):
    supports_weight_switch = True
    weight_switch_gather_specs = (WeightSwitchGatherSpec("weight", gather_dim=1),)

    def apply(self, layer, x, bias=None):
        raise NotImplementedError


class TestAllGatherAndWait(unittest.TestCase):
    @staticmethod
    def _state_with_part():
        layer = torch.nn.Module()
        layer.register_parameter("weight", torch.nn.Parameter(torch.zeros(4, 8)))
        layer.register_parameter("bias", torch.nn.Parameter(torch.ones(8)))
        config = WeightSwitchConfig(group="grp", world_size=2, rank=0, shard_axis="input")
        state = _Switchable().enable_weight_switch(layer, config)
        return layer, state, config

    @patch("vllm_ascend.distributed.utils.all_gather_async")
    def test_all_gather_records_handle(self, mock_gather):
        handle = MagicMock()
        mock_gather.return_value = (None, handle)
        layer, state, config = self._state_with_part()
        _Switchable().all_gather_weight(state, config)
        part = state.gather_parts["weight"]
        mock_gather.assert_called_once_with(part.gather_input, "grp", output=part.gather_output, async_op=True)
        self.assertIs(state.handles[0], handle)

    @patch("vllm_ascend.distributed.utils.all_gather_async")
    def test_pending_handle_rejected(self, mock_gather):
        handle = MagicMock()
        mock_gather.return_value = (None, handle)
        layer, state, config = self._state_with_part()
        _Switchable().all_gather_weight(state, config)
        with self.assertRaises(RuntimeError) as ctx:
            _Switchable().all_gather_weight(state, config)
        self.assertIn("still pending", str(ctx.exception))

    def test_wait_clears_handles(self):
        layer, state, config = self._state_with_part()
        handle = MagicMock()
        state.handles = [handle]
        _Switchable().wait_weight_all_gather(state)
        self.assertTrue(handle.wait.called)
        self.assertEqual(state.handles, [])

    def test_wait_clears_handles_on_error(self):
        layer, state, config = self._state_with_part()
        handle = MagicMock()
        handle.wait.side_effect = RuntimeError("collective failed")
        state.handles = [handle]
        with self.assertRaises(RuntimeError):
            _Switchable().wait_weight_all_gather(state)
        self.assertEqual(state.handles, [])

    def test_switch_weight_flips_views(self):
        layer, state, config = self._state_with_part()
        part = state.gather_parts["weight"]
        part.full_tensor.copy_(torch.full(part.full_tensor.shape, 9.0))
        method = _Switchable()
        method.switch_weight(layer, state, use_full_weight=True)
        self.assertEqual(layer.weight.shape, (8, 8))
        self.assertTrue(torch.all(layer.weight == 9.0))
        method.switch_weight(layer, state, use_full_weight=False)
        self.assertEqual(layer.weight.shape, (4, 8))
        self.assertTrue(torch.all(layer.weight == 0.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
