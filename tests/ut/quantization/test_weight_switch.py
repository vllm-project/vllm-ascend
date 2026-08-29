# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.weight_switch import (
    WeightLoadPartition,
    WeightSwitchConfig,
    WeightSwitchGatherSpec,
    WeightSwitchRepeatSpec,
    WeightSwitchMixin,
)


def _weight_switch_config() -> WeightSwitchConfig:
    return WeightSwitchConfig(group=object(), world_size=2, rank=0)


def test_weight_switch_config_uses_the_caller_selected_parallel_group() -> None:
    group = SimpleNamespace(world_size=4, rank_in_group=2)

    config = WeightSwitchConfig.from_group(group)

    assert config.group is group
    assert config.world_size == 4
    assert config.rank == 2


class _TestLinearMethod(WeightSwitchMixin):
    supports_weight_switch = True
    weight_switch_gather_specs = (
        WeightSwitchGatherSpec("weight"),
        WeightSwitchGatherSpec("weight_scale"),
    )
    weight_switch_repeat_specs = (WeightSwitchRepeatSpec("input_scale"),)
    weight_switch_output_gather_specs = (
        WeightSwitchGatherSpec("weight", gather_dim=1),
        WeightSwitchGatherSpec("weight_scale", gather_dim=1),
    )


def _input_sharded_layer() -> SimpleNamespace:
    return SimpleNamespace(
        input_size=8,
        input_size_per_partition=4,
        output_size=3,
        output_size_per_partition=3,
        weight=torch.arange(12, dtype=torch.float32).reshape(4, 3),
        weight_scale=torch.arange(6, dtype=torch.float32).reshape(2, 3),
        input_scale=torch.tensor([2.0]),
    )


def _output_sharded_layer() -> SimpleNamespace:
    return SimpleNamespace(
        input_size=4,
        input_size_per_partition=4,
        output_size=6,
        output_size_per_partition=3,
        weight=torch.arange(12, dtype=torch.float32).reshape(4, 3),
        weight_scale=torch.arange(12, dtype=torch.float32).reshape(2, 3, 2),
        input_scale=torch.tensor([2.0]),
    )


def test_split_tensor_for_parallel_supports_nonzero_and_negative_dims() -> None:
    tensor = torch.arange(24).reshape(4, 6)

    shard = WeightSwitchMixin.split_tensor_for_parallel(tensor, world_size=3, rank=1, dim=-1)

    torch.testing.assert_close(shard, tensor[:, 2:4])
    assert shard.is_contiguous()


def test_split_tensor_for_parallel_rejects_non_divisible_dimension() -> None:
    with pytest.raises(RuntimeError, match="not divisible"):
        WeightSwitchMixin.split_tensor_for_parallel(torch.empty(3, 5), world_size=2, rank=0, dim=1)


def test_enable_input_sharded_state_builds_gather_and_repeat_tensors() -> None:
    layer = _input_sharded_layer()
    method = _TestLinearMethod()

    state = method.enable_weight_switch(layer, _weight_switch_config())

    assert set(state.gather_parts) == {"weight", "weight_scale"}
    assert state.gather_parts["weight"].full_tensor.shape == (8, 3)
    assert state.gather_parts["weight_scale"].full_tensor.shape == (4, 3)
    assert state.gather_parts["weight"].local_tensor.data_ptr() == layer.weight.data_ptr()
    torch.testing.assert_close(state.repeat_parts["input_scale"].full_tensor, torch.tensor([2.0, 2.0]))


def test_prepare_layer_for_parallel_weight_load_uses_one_composed_checkpoint_slice() -> None:
    class _LoadLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_size = 8
            self.input_size_per_partition = 4
            self.output_size = 3
            self.output_size_per_partition = 3
            self.weight = torch.nn.Parameter(torch.empty(3, 4), requires_grad=False)
            self.weight.input_dim = 1

            def tp_weight_loader(*args, **kwargs):
                raise AssertionError("direct loading must not call the original TP loader")

            self.weight.weight_loader = tp_weight_loader

    class _LoadMethod(WeightSwitchMixin):
        supports_weight_switch = True
        weight_switch_gather_specs = (WeightSwitchGatherSpec("weight", gather_dim=1),)

    layer = _LoadLayer()
    method = _LoadMethod()
    config = WeightSwitchConfig(group=object(), world_size=2, rank=1)

    load_state = method.prepare_layer_for_parallel_weight_load(
        layer,
        config,
        WeightLoadPartition(world_size=4, rank=1),
    )
    layer.weight.weight_loader(layer.weight, torch.arange(24, dtype=torch.float32).reshape(3, 8))

    assert load_state.input_size_per_partition_before == 4
    assert load_state.input_size_per_partition_after == 2
    assert layer.input_size_per_partition == 2
    torch.testing.assert_close(layer.weight, torch.tensor([[2.0, 3.0], [10.0, 11.0], [18.0, 19.0]]))

    state = method.enable_weight_switch(layer, config)
    assert state.gather_parts["weight"].full_tensor.shape == (3, 4)


def test_enable_output_sharded_state_moves_gather_dim_and_reuses_pool() -> None:
    method = _TestLinearMethod()
    pool: dict[object, torch.Tensor] = {}

    first = method.enable_weight_switch(
        _output_sharded_layer(),
        _weight_switch_config(),
        pool=pool,
        pool_key_prefix="shared",
    )
    second = method.enable_weight_switch(
        _output_sharded_layer(),
        _weight_switch_config(),
        pool=pool,
        pool_key_prefix="shared",
    )

    weight_part = first.gather_parts["weight"]
    scale_part = first.gather_parts["weight_scale"]
    assert weight_part.gather_input.shape == (3, 4)
    assert weight_part.full_tensor.shape == (4, 6)
    assert scale_part.gather_input.shape == (3, 2, 2)
    assert scale_part.full_tensor.shape == (2, 6, 2)
    assert second.gather_parts["weight"].gather_output.data_ptr() == weight_part.gather_output.data_ptr()
    assert second.gather_parts["weight_scale"].gather_output.data_ptr() == scale_part.gather_output.data_ptr()


def test_all_gather_wait_and_switch_restore_local_storage() -> None:
    layer = _input_sharded_layer()
    method = _TestLinearMethod()
    config = _weight_switch_config()
    state = method.enable_weight_switch(layer, config)
    original_ptrs = {name: getattr(layer, name).data_ptr() for name in ("weight", "weight_scale", "input_scale")}
    handles: list[MagicMock] = []

    def fake_all_gather_async(gather_input, group, *, output, async_op):
        del group
        assert async_op
        output.copy_(torch.cat((gather_input, gather_input + 100), dim=0))
        handle = MagicMock()
        handles.append(handle)
        return output, handle

    with patch(
        "vllm_ascend.distributed.utils.all_gather_async",
        side_effect=fake_all_gather_async,
    ):
        method.all_gather_weight(state, config)

    assert state.handles == handles
    method.wait_weight_all_gather(state)
    assert not state.handles
    for handle in handles:
        handle.wait.assert_called_once_with()

    method.switch_weight(layer, state, use_full_weight=True)
    assert layer.weight.data_ptr() == state.gather_parts["weight"].full_tensor.data_ptr()
    assert layer.weight_scale.data_ptr() == state.gather_parts["weight_scale"].full_tensor.data_ptr()
    assert layer.input_scale.data_ptr() == state.repeat_parts["input_scale"].full_tensor.data_ptr()

    method.switch_weight(layer, state, use_full_weight=False)
    for name, ptr in original_ptrs.items():
        assert getattr(layer, name).data_ptr() == ptr


@pytest.mark.parametrize(
    ("layer", "message"),
    [
        (
            SimpleNamespace(
                input_size=4,
                input_size_per_partition=4,
                output_size=3,
                output_size_per_partition=3,
            ),
            "shard_axis",
        ),
        (
            SimpleNamespace(
                input_size=8,
                input_size_per_partition=4,
                output_size=6,
                output_size_per_partition=3,
            ),
            "at most one pre-existing sharded axis",
        ),
    ],
)
def test_enable_rejects_zero_or_two_sharded_axes(layer: SimpleNamespace, message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        _TestLinearMethod().enable_weight_switch(layer, _weight_switch_config())


def test_enable_rejects_unsupported_method_and_missing_attribute() -> None:
    with pytest.raises(RuntimeError, match="does not support"):
        WeightSwitchMixin().enable_weight_switch(_input_sharded_layer(), _weight_switch_config())

    layer = _input_sharded_layer()
    del layer.weight_scale
    with pytest.raises(RuntimeError, match="weight_scale"):
        _TestLinearMethod().enable_weight_switch(layer, _weight_switch_config())


def test_all_gather_rejects_a_second_launch_while_pending() -> None:
    config = _weight_switch_config()
    state = _TestLinearMethod().enable_weight_switch(_input_sharded_layer(), config)
    state.handles.append(MagicMock())

    with pytest.raises(RuntimeError, match="still pending"):
        _TestLinearMethod().all_gather_weight(state, config)
