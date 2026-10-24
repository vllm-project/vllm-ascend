# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.weight_switch import (
    OProjectionMode,
    ParallelOProjection,
    WeightLoadPartition,
    WeightSwitchConfig,
    WeightSwitchGatherSpec,
    WeightSwitchMixin,
    WeightSwitchRepeatSpec,
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


def test_parallel_projection_prepares_pcp_load_shard_before_runtime_switching() -> None:
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
    layer.quant_method = _LoadMethod()
    config = WeightSwitchConfig.from_group(SimpleNamespace(world_size=2, rank_in_group=1), shard_axis="input")
    adapter = ParallelOProjection(
        4,
        layer,
        config=config,
        tp_group=SimpleNamespace(world_size=2, rank_in_group=0),
        pool={},
        pool_key="load_test",
        project_full=MagicMock(),
        decode_sharded=False,
    )
    adapter.prepare_weight_load(None)
    assert adapter._load_prepared
    assert layer.input_size_per_partition == 2
    loader = layer.weight.weight_loader
    adapter.prepare_weight_load(None)
    assert layer.weight.weight_loader is loader
    assert layer.input_size_per_partition == 2

    layer.weight.weight_loader(layer.weight, torch.arange(24, dtype=torch.float32).reshape(3, 8))

    torch.testing.assert_close(
        layer.weight,
        torch.tensor([[2.0, 3.0], [10.0, 11.0], [18.0, 19.0]]),
    )


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


class _ProjectionLinearMethod(WeightSwitchMixin):
    supports_weight_switch = True
    weight_switch_gather_specs = (WeightSwitchGatherSpec("weight", gather_dim=1),)
    weight_switch_output_gather_specs = (WeightSwitchGatherSpec("weight", gather_dim=0),)

    def apply(self, layer, inputs, bias=None):
        return torch.nn.functional.linear(inputs, layer.weight, bias)


def _projection_config():
    return SimpleNamespace(
        num_speculative_tokens=1,
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=4, cudagraph_capture_sizes=(1, 4)),
        model_config=SimpleNamespace(dtype=torch.float32),
    )


def _parallel_projection(*, pool=None, decode_sharded=True, tp_size=1, tp_rank=0, pcp_rank=0, prepare=True):
    layer = torch.nn.Module()
    layer.input_size = 4 * tp_size
    layer.input_size_per_partition = 4
    layer.output_size = layer.output_size_per_partition = 3
    # Both PCP shards have the same coefficients, allowing the fake
    # collectives to reproduce a complete global linear numerically.
    layer.weight = torch.nn.Parameter(torch.arange(6.0).reshape(3, 2).repeat(1, 2), requires_grad=False)
    layer.weight.input_dim = 1
    layer.weight.weight_loader = MagicMock()
    layer.quant_method = _ProjectionLinearMethod()
    layer.reduce_results = True
    layer.skip_bias_add = False
    layer.bias = None
    pcp_group = SimpleNamespace(
        world_size=2,
        rank_in_group=pcp_rank,
        device_group=object(),
        unique_name="pcp",
        all_reduce=lambda value: value * 2,
    )
    tp_group = SimpleNamespace(world_size=tp_size, rank_in_group=tp_rank, all_reduce=lambda value: value * tp_size)

    def project_full(inputs, output):
        projected = torch.nn.functional.linear(inputs.flatten(1), layer.weight)
        if layer.reduce_results:
            projected = tp_group.all_reduce(projected)
        if not layer.skip_bias_add and layer.bias is not None:
            projected = projected + layer.bias
        output.copy_(projected)
        return output

    projection = ParallelOProjection(
        4,
        layer,
        config=WeightSwitchConfig.from_group(pcp_group, shard_axis="input"),
        tp_group=tp_group,
        pool={} if pool is None else pool,
        pool_key="test_o_proj",
        project_full=project_full,
        decode_sharded=decode_sharded,
    )
    if prepare:
        projection.prepare_weight_load(_projection_config())
        # Load preparation allocates the resident shard; simulate checkpoint
        # loading before building the runtime aliases.
        with torch.no_grad():
            layer.weight.copy_(torch.arange(6.0).reshape(3, 2))
        projection.enable()
    return projection


def test_parallel_projection_reuses_graph_stable_buffers_and_separates_groups() -> None:
    pool: dict[Any, torch.Tensor] = {}
    first = _parallel_projection(pool=pool)
    second = _parallel_projection(pool=pool)
    assert first.token_capacity == 4
    for name, shape in (
        ("all_gather_input", (4, 4)),
        ("all_gather_output", (8, 4)),
        ("reduce_scatter_input", (8, 3)),
        ("reduce_scatter_output", (4, 3)),
    ):
        assert getattr(first, name).shape == shape
        assert getattr(first, name).data_ptr() == getattr(second, name).data_ptr()
    second.config.group.unique_name = "another_pcp"
    second._prepare_exchange_buffers(_projection_config())
    assert first.all_gather_input.data_ptr() != second.all_gather_input.data_ptr()


@pytest.mark.parametrize("pcp_rank", [0, 1])
@pytest.mark.parametrize("exchange_tokens", [False, True])
def test_parallel_projection_local_and_exchanged_tokens_match_global_linear(pcp_rank, exchange_tokens) -> None:
    projection = _parallel_projection(pcp_rank=pcp_rank)
    inputs = torch.tensor([[1.0, 2.0, 1.0, 2.0], [3.0, 4.0, 3.0, 4.0]])
    full_weight = torch.arange(6.0).reshape(3, 2).repeat(1, 2)
    expected = torch.nn.functional.linear(inputs, full_weight)
    output = torch.empty(2, 3)
    pointers = []

    def gather(destination, source, *, group):
        assert group is projection.config.group.device_group
        remote = source + 10
        destination.copy_(torch.cat((source, remote) if pcp_rank == 0 else (remote, source)))
        pointers.append((source.data_ptr(), destination.data_ptr()))

    def scatter(destination, source, *, group):
        assert group is projection.config.group.device_group
        destination.copy_(source.chunk(2)[pcp_rank] * 2)

    with (
        projection.batch(needs_global_metadata=False, has_prefill=False),
        patch("vllm_ascend.weight_switch.o_proj.dist.all_gather_into_tensor", side_effect=gather),
        patch("vllm_ascend.weight_switch.o_proj.dist.reduce_scatter_tensor", side_effect=scatter),
    ):
        projection.set_mode(use_full_weight=False, exchange_tokens=exchange_tokens)
        assert projection.forward(inputs, output) is output
        torch.testing.assert_close(output, expected)
        projection.forward(inputs, output)
        empty = projection.forward(torch.empty(0, 4), torch.empty(0, 3))
        assert empty.shape == (0, 3)
        if exchange_tokens:
            assert pointers[0] == pointers[1]
            with pytest.raises(ValueError, match="capacity"):
                projection.forward(torch.empty(5, 4), torch.empty(5, 3))
    assert projection.mode is OProjectionMode.LOCAL_WEIGHT
    assert not projection.uses_full_weight


@pytest.mark.parametrize("failure_stage", [None, "attention", "projection"])
def test_parallel_projection_full_weights_and_failure_cleanup(failure_stage) -> None:
    projection = _parallel_projection()
    layer = projection.o_proj
    parameter = layer.weight
    address = parameter.data_ptr()
    local_weight = parameter.detach().clone()
    state = projection._weight_switch_states[0][2]
    handle = MagicMock()
    callback = projection.project_full

    def gather(gather_state, config):
        assert config is projection.config
        part = gather_state.gather_parts["weight"]
        part.gather_output.copy_(torch.cat((part.gather_input, part.gather_input)))
        gather_state.handles.append(handle)

    def project_full(inputs, output):
        assert layer.weight.shape == (3, 4)
        if failure_stage == "projection":
            raise RuntimeError("injected projection failure")
        return callback(inputs, output)

    projection.project_full = project_full
    inputs = torch.arange(8.0).reshape(2, 4)
    output = torch.empty(2, 3)
    with patch.object(layer.quant_method, "all_gather_weight", side_effect=gather):
        if failure_stage is None:
            with projection.batch(needs_global_metadata=True, has_prefill=True):
                assert projection.uses_full_weight
                projection.all_gather_weights()
                assert projection.forward(inputs, output) is output
            torch.testing.assert_close(output, torch.nn.functional.linear(inputs, local_weight.repeat(1, 2)))
        else:
            with (
                pytest.raises(RuntimeError, match="injected"),
                projection.batch(needs_global_metadata=True, has_prefill=True),
            ):
                projection.all_gather_weights()
                if failure_stage == "attention":
                    raise RuntimeError("injected attention failure")
                projection.forward(inputs, output)
    assert layer.weight is parameter
    assert layer.weight.data_ptr() == address
    torch.testing.assert_close(layer.weight, local_weight)
    assert not state.handles
    handle.wait.assert_called_once_with()
    assert projection.mode is OProjectionMode.LOCAL_WEIGHT
    assert not projection.uses_full_weight


def test_parallel_projection_disabled_path_does_not_create_switches() -> None:
    projection = _parallel_projection(prepare=False)
    inputs = torch.arange(8.0).reshape(2, 4)
    output = torch.empty(2, 3)
    with projection.batch(needs_global_metadata=True, has_prefill=True):
        projection.all_gather_weights()
        assert not projection.uses_full_weight
        assert projection.forward(inputs, output) is output
    torch.testing.assert_close(output, torch.nn.functional.linear(inputs, projection.o_proj.weight))
    assert projection._weight_switch_states is None
    assert projection.o_proj_full_pools == {}


@pytest.mark.parametrize("reduce_results", [False, True])
@pytest.mark.parametrize("skip_bias_add", [False, True])
def test_parallel_projection_adds_ordinary_bias_once_after_internal_reductions(reduce_results, skip_bias_add) -> None:
    inputs = torch.tensor([[1.0, 2.0, 1.0, 2.0]])
    bias = torch.tensor([7.0, 8.0, 9.0])
    outputs = []
    for tp_rank in range(2):
        projection = _parallel_projection(tp_size=2, tp_rank=tp_rank, decode_sharded=False)
        projection.o_proj.reduce_results = reduce_results
        projection.o_proj.skip_bias_add = skip_bias_add
        projection.o_proj.bias = torch.nn.Parameter(bias.clone(), requires_grad=False)
        with projection.batch(needs_global_metadata=False, has_prefill=False):
            outputs.append(projection.forward(inputs, torch.empty(1, 3)))
    full_weight = torch.arange(6.0).reshape(3, 2).repeat(1, 2)
    expected = torch.nn.functional.linear(inputs, full_weight) * 2
    if not skip_bias_add:
        expected = expected + bias
    if reduce_results:
        for actual in outputs:
            torch.testing.assert_close(actual, expected)
    else:
        torch.testing.assert_close(sum(outputs), expected)


def test_parallel_projection_keeps_pcp_before_tp_and_reads_current_bias() -> None:
    projection = _parallel_projection(tp_size=2, decode_sharded=False)
    events: list[str] = []

    def pcp_reduce(value: torch.Tensor) -> torch.Tensor:
        events.append("pcp")
        return value * 2

    def tp_reduce(value: torch.Tensor) -> torch.Tensor:
        events.append("tp")
        return value * 2

    projection.config.group.all_reduce = pcp_reduce
    projection.tp_group.all_reduce = tp_reduce
    inputs = torch.tensor([[1.0, 2.0, 1.0, 2.0]])
    baseline = torch.nn.functional.linear(inputs, projection.o_proj.weight.repeat(1, 2)) * 2
    for bias in (torch.ones(3), torch.full((3,), 5.0)):
        projection.o_proj.bias = torch.nn.Parameter(bias, requires_grad=False)
        with projection.batch(needs_global_metadata=False, has_prefill=False):
            output = projection.forward(inputs, torch.empty(1, 3))
        torch.testing.assert_close(output, baseline + bias)
    assert events == ["pcp", "tp", "pcp", "tp"]


@pytest.mark.parametrize(
    ("needs_global_metadata", "has_prefill", "decode_sharded", "mode"),
    [
        (True, True, True, OProjectionMode.FULL_WEIGHT),
        (True, False, True, OProjectionMode.EXCHANGE_TOKENS),
        (True, False, False, OProjectionMode.FULL_WEIGHT),
        (False, False, True, OProjectionMode.LOCAL_WEIGHT),
    ],
)
def test_parallel_projection_batch_selects_one_collective_mode(
    needs_global_metadata, has_prefill, decode_sharded, mode
):
    projection = _parallel_projection(decode_sharded=decode_sharded)
    with projection.batch(needs_global_metadata=needs_global_metadata, has_prefill=has_prefill):
        assert projection.mode is mode
        assert projection.uses_full_weight == (mode is OProjectionMode.FULL_WEIGHT)
        assert projection.exchanges_tokens == (mode is OProjectionMode.EXCHANGE_TOKENS)
    assert projection.mode is OProjectionMode.LOCAL_WEIGHT
    assert not projection.uses_full_weight


def _grouped_parallel_projection(pcp_rank=0):
    base = _parallel_projection(pcp_rank=pcp_rank, prepare=False, decode_sharded=False)
    input_layer = torch.nn.Module()
    input_layer.input_size = input_layer.input_size_per_partition = 2
    input_layer.output_size = input_layer.output_size_per_partition = 4
    input_layer.output_partition_sizes = [4]
    input_layer.weight = torch.nn.Parameter(torch.empty(4, 2), requires_grad=False)
    input_layer.weight.output_dim = 0
    input_layer.weight.weight_loader = MagicMock()
    input_layer.quant_method = _ProjectionLinearMethod()
    input_layer.quant_method.n_local_groups = 4

    def input_projection(grouped):
        return torch.einsum("tgi,gi->tg", grouped, input_layer.weight)

    projection = ParallelOProjection(
        8,
        base.o_proj,
        config=base.config,
        tp_group=base.tp_group,
        pool={},
        pool_key="grouped_o_proj",
        project_full=MagicMock(side_effect=AssertionError("local projection must not invoke the full callback")),
        decode_sharded=False,
        input_layer=input_layer,
        input_projection=input_projection,
        num_input_groups=4,
    )
    projection.prepare_weight_load(_projection_config())
    with torch.no_grad():
        input_layer.weight.copy_(torch.tensor([[1.0, 2.0], [1.0, 2.0]]))
        projection.o_proj.weight.copy_(torch.arange(6.0).reshape(3, 2))
    projection.enable()
    return projection


@pytest.mark.parametrize("pcp_rank", [0, 1])
def test_grouped_parallel_projection_matches_two_factor_reference_and_empty_rows(pcp_rank) -> None:
    projection = _grouped_parallel_projection(pcp_rank)
    assert projection.input_layer.n_local_groups == 2
    assert projection.input_layer.quant_method.n_local_groups == 2
    assert projection.input_layer.weight.shape == (2, 2)
    assert projection.o_proj.weight.shape == (3, 2)
    inputs = torch.tensor([[[1.0, 2.0]] * 4, [[3.0, 4.0]] * 4])
    first_factor = torch.tensor([[1.0, 2.0]] * 4)
    second_factor = torch.arange(6.0).reshape(3, 2).repeat(1, 2)
    expected = torch.nn.functional.linear(torch.einsum("tgi,gi->tg", inputs, first_factor), second_factor)
    with projection.batch(needs_global_metadata=False, has_prefill=False):
        output = projection.forward(inputs, torch.empty(2, 3))
        torch.testing.assert_close(output, expected)
        assert projection.forward(torch.empty(0, 4, 2), torch.empty(0, 3)).shape == (0, 3)


def test_parallel_projection_cleanup_waits_all_factors_without_hiding_attention_error() -> None:
    projection = _grouped_parallel_projection()
    first = projection._weight_switch_states[0][2]
    second = projection._weight_switch_states[1][2]
    first.handles.append(SimpleNamespace(wait=MagicMock(side_effect=RuntimeError("wait failure"))))
    second_wait = MagicMock()
    second.handles.append(SimpleNamespace(wait=second_wait))
    attention_error = RuntimeError("attention failed")
    with pytest.raises(RuntimeError) as caught, projection.batch(needs_global_metadata=True, has_prefill=True):
        raise attention_error
    assert caught.value is attention_error
    second_wait.assert_called_once_with()
    assert not first.handles and not second.handles
    assert projection.mode is OProjectionMode.LOCAL_WEIGHT
    assert not projection.uses_full_weight


def test_parallel_projection_rejects_wrong_input_width_before_communication() -> None:
    projection = _parallel_projection()
    with projection.batch(needs_global_metadata=False, has_prefill=False):
        projection.set_mode(use_full_weight=False, exchange_tokens=True)
        with patch("vllm_ascend.weight_switch.o_proj.dist.all_gather_into_tensor") as gather:
            with pytest.raises(RuntimeError):
                projection.forward(torch.empty(2, 3), torch.empty(2, 3))
            gather.assert_not_called()


def test_parallel_projection_can_enable_cloned_legacy_weights_without_load_sharding() -> None:
    base = _parallel_projection(prepare=False, decode_sharded=False)
    projection = ParallelOProjection(
        4,
        base.o_proj,
        config=base.config,
        tp_group=base.tp_group,
        pool={},
        pool_key="legacy_test",
        project_full=base.project_full,
        decode_sharded=False,
        clone_local_tensors=True,
    )
    address = base.o_proj.weight.data_ptr()
    projection.enable()
    initial_states = projection._weight_switch_states
    assert initial_states is None
    projection.enabled = True
    projection.enable()
    states = projection._weight_switch_states
    assert states is not None
    state = states[0][2]
    assert not projection._load_prepared
    assert base.o_proj.weight.data_ptr() != address
    assert base.o_proj.weight.data_ptr() == state.gather_parts["weight"].local_tensor.data_ptr()
    projection.enable()
    assert projection._weight_switch_states[0][2] is state
