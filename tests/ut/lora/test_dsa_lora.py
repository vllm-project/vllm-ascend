from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.lora import utils as lora_utils

from vllm_ascend.lora.dsa import (
    DSA_LORA_CLASSES,
    DSA_LORA_CONTEXT_ATTR,
    AscendDSAReplicatedLinearWithLoRA,
    DSALoRAContext,
    DSALoRARouting,
    apply_grouped_dsa_lora,
    apply_prepared_dsa_lora,
    forward_with_dsa_lora,
    prepare_dsa_lora,
)
from vllm_ascend.lora.punica_npu import PunicaWrapperNPU
from vllm_ascend.lora.utils import refresh_all_lora_classes


class FakePunica:
    def __init__(self, token_lora_indices: torch.Tensor):
        self.token_lora_indices = token_lora_indices

    def get_token_lora_indices(self, num_tokens: int) -> torch.Tensor:
        return self.token_lora_indices[:num_tokens]

    @staticmethod
    def bgmv_shrink(x, weights, output, indices, scale):
        for row, index in enumerate(indices.tolist()):
            if index >= 0:
                output[row].copy_(torch.mv(weights[index], x[row]) * scale)

    @staticmethod
    def bgmv_expand(x, weights, output, indices, add_inputs):
        for row, index in enumerate(indices.tolist()):
            if index >= 0:
                value = torch.mv(weights[index], x[row])
                if add_inputs:
                    output[row].add_(value)
                else:
                    output[row].copy_(value)

    @classmethod
    def bgmv_expand_slice(
        cls,
        x,
        weights,
        output,
        indices,
        offset,
        output_size,
        add_inputs,
    ):
        target = output[:, offset : offset + output_size]
        cls.bgmv_expand(x, weights, target, indices, add_inputs)


def _fake_grouped_matmul(*, x, weight, group_list, **kwargs):
    del kwargs
    inputs = x[0]
    weights = weight[0]
    outputs = []
    offset = 0
    for group, length in enumerate(group_list.tolist()):
        outputs.append(inputs[offset : offset + length] @ weights[group])
        offset += length
    return [torch.cat(outputs)]


def _make_grouped_routing(slots: list[int], num_groups: int = 1) -> DSALoRARouting:
    group_lengths: list[int] = []
    segment_slots: list[int] = []
    segment_starts: list[int] = []
    for row, slot in enumerate(slots):
        if row == 0 or slot != slots[row - 1]:
            segment_starts.append(row)
            group_lengths.append(1)
            segment_slots.append(max(slot, 0))
        else:
            group_lengths[-1] += 1

    grouped_lengths = [length for length in group_lengths for _ in range(num_groups)]
    grouped_slots = [slot * num_groups + group for slot in segment_slots for group in range(num_groups)]
    grouped_rows = [
        token * num_groups + group
        for start, length in zip(segment_starts, group_lengths)
        for group in range(num_groups)
        for token in range(start, start + length)
    ]
    has_base = any(slot < 0 for slot in slots)
    return DSALoRARouting(
        token_lora_indices=torch.tensor(slots),
        num_tokens=len(slots),
        prefer_grouped_matmul=True,
        has_lora=any(slot >= 0 for slot in slots),
        has_base=has_base,
        segment_lora_indices_cpu=tuple(segment_slots),
        group_list=torch.tensor(group_lengths),
        segment_lora_indices=torch.tensor(segment_slots),
        active_mask=torch.tensor([slot >= 0 for slot in slots]).unsqueeze(1) if has_base else None,
        expanded_group_list=torch.tensor([length * num_groups for length in group_lengths]),
        grouped_group_list=torch.tensor(grouped_lengths) if num_groups > 1 else None,
        segment_group_lora_indices=torch.tensor(grouped_slots) if num_groups > 1 else None,
        segment_group_lora_indices_cpu=tuple(grouped_slots) if num_groups > 1 else (),
        grouped_row_indices=torch.tensor(grouped_rows) if num_groups > 1 else None,
        expanded_active_mask=(
            torch.tensor([slot >= 0 for slot in slots for _ in range(num_groups)]).unsqueeze(1) if has_base else None
        ),
    )


def _attach_context(
    linear,
    punica,
    lora_a,
    lora_b,
    *,
    parallel_mode="replicated",
    fully_sharded=False,
    tp_size=1,
    tp_rank=0,
):
    context = DSALoRAContext(
        punica_wrapper=punica,
        lora_a_stacked=(lora_a,),
        lora_b_stacked=(lora_b,),
        output_slices=(lora_b.shape[-2],),
        parallel_mode=parallel_mode,
        fully_sharded=fully_sharded,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )
    setattr(linear, DSA_LORA_CONTEXT_ATTR, context)
    return context


def test_punica_caches_dsa_segment_and_group_routing() -> None:
    wrapper = object.__new__(PunicaWrapperNPU)
    wrapper.device = torch.device("cpu")
    wrapper._token_lora_indices = torch.full((8,), -1, dtype=torch.long)
    wrapper._token_lora_indices[:5].copy_(torch.tensor([0, 0, -1, 1, 1]))
    wrapper._dsa_lora_indices_cpu = (0, 0, -1, 1, 1)
    wrapper._dsa_lora_routing_cache = {}

    routing = wrapper.get_dsa_lora_routing(
        0,
        5,
        num_rows=8,
        prefer_grouped_matmul=True,
        group_multiplier=2,
    )
    cached = wrapper.get_dsa_lora_routing(
        0,
        5,
        num_rows=8,
        prefer_grouped_matmul=True,
        group_multiplier=2,
    )

    assert cached is routing
    assert routing.segment_lora_indices_cpu == (0, 0, 1)
    assert routing.group_list.tolist() == [2, 1, 2]
    assert routing.segment_lora_indices.tolist() == [0, 0, 1]
    assert routing.active_mask.squeeze(1).tolist() == [True, True, False, True, True]
    assert routing.expanded_group_list.tolist() == [4, 2, 4]
    assert routing.grouped_group_list.tolist() == [2, 2, 1, 1, 2, 2]
    assert routing.segment_group_lora_indices.tolist() == [0, 1, 0, 1, 2, 3]
    assert routing.grouped_row_indices.tolist() == [0, 2, 1, 3, 4, 5, 6, 8, 7, 9]
    assert routing.expanded_token_lora_indices.tolist() == [0, 0, 0, 0, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1]
    assert routing.expanded_combined_indices.tolist() == [0, 1, 0, 1, -1, -1, 2, 3, 2, 3, -1, -1, -1, -1, -1, -1]


def test_replicated_projection_publishes_unsharded_context() -> None:
    wrapper = object.__new__(AscendDSAReplicatedLinearWithLoRA)
    torch.nn.Module.__init__(wrapper)
    wrapper.base_layer = torch.nn.Module()
    wrapper.lora_config = SimpleNamespace(fully_sharded_loras=True)
    wrapper.lora_a_stacked = (torch.zeros(2, 1, 4, 8),)
    wrapper.lora_b_stacked = (torch.zeros(2, 1, 16, 4),)
    wrapper.output_slices = (16,)
    wrapper.tp_size = 8
    wrapper.tp_rank = 3

    punica = FakePunica(torch.tensor([0]))
    wrapper.set_mapping(punica)

    context = getattr(wrapper.base_layer, DSA_LORA_CONTEXT_ATTR)
    assert context.punica_wrapper is punica
    assert context.parallel_mode == "replicated"
    assert context.fully_sharded is False


def test_dsa_lora_registry_refresh_is_idempotent() -> None:
    class DummyLoRAClass:
        pass

    initial_registry = (DummyLoRAClass, *DSA_LORA_CLASSES)
    with patch.object(lora_utils, "_all_lora_classes", initial_registry):
        refresh_all_lora_classes()
        refresh_all_lora_classes()
        registry = lora_utils._all_lora_classes

    for dsa_lora_class in DSA_LORA_CLASSES:
        assert registry.count(dsa_lora_class) == 1
    assert registry[-1] is DummyLoRAClass


def test_cv_lora_uses_explicit_multi_adapter_token_indices() -> None:
    linear = torch.nn.Module()
    token_indices = torch.tensor([0, -1, 1])
    punica = FakePunica(token_indices)
    lora_a = torch.tensor(
        [
            [[[1.0, 0.0]]],
            [[[0.0, 1.0]]],
        ]
    )
    lora_b = torch.tensor(
        [
            [[[2.0], [3.0]]],
            [[[4.0], [5.0]]],
        ]
    )
    _attach_context(linear, punica, lora_a, lora_b)

    x = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 4.0]])
    output = torch.ones(3, 2)
    intermediate = prepare_dsa_lora(linear, x, token_indices)
    actual = apply_prepared_dsa_lora(linear, output, intermediate)

    expected = torch.tensor([[3.0, 4.0], [1.0, 1.0], [17.0, 21.0]])
    torch.testing.assert_close(actual, expected)


def test_cv_lora_uses_cube_gmm_for_long_prefill_segments() -> None:
    linear = torch.nn.Module()
    slots = [0] * 400 + [-1] * 200 + [1] * 424
    routing = _make_grouped_routing(slots)
    punica = FakePunica(routing.token_lora_indices)
    lora_a = torch.tensor(
        [
            [[[1.0, 0.0]]],
            [[[0.0, 1.0]]],
        ],
        dtype=torch.bfloat16,
    )
    lora_b = torch.tensor(
        [
            [[[2.0], [3.0]]],
            [[[4.0], [5.0]]],
        ],
        dtype=torch.bfloat16,
    )
    _attach_context(linear, punica, lora_a, lora_b)

    x = torch.arange(2048, dtype=torch.bfloat16).view(1024, 2) / 128
    output = torch.ones(1024, 2, dtype=torch.bfloat16)
    with (
        patch("vllm_ascend.lora.dsa._DSA_LORA_GMM_MIN_CUBE_RANK", 0),
        patch("vllm_ascend.lora.dsa._DSA_LORA_GMM_MIN_TOKENS", 0),
        patch(
            "vllm_ascend.lora.dsa.torch_npu.npu_grouped_matmul",
            side_effect=_fake_grouped_matmul,
        ) as grouped_matmul,
    ):
        intermediate = prepare_dsa_lora(linear, x, routing)
        actual = apply_prepared_dsa_lora(linear, output, intermediate)

    assert intermediate is not None
    assert intermediate.used_grouped_matmul
    assert grouped_matmul.call_count == 2
    expected = torch.ones_like(output)
    expected[:400] += x[:400, :1] * torch.tensor([2.0, 3.0], dtype=torch.bfloat16)
    expected[600:] += x[600:, 1:] * torch.tensor([4.0, 5.0], dtype=torch.bfloat16)
    torch.testing.assert_close(actual, expected)


def test_cv_lora_segment_mapping_does_not_restart_from_batch_zero() -> None:
    linear = torch.nn.Module()
    punica = FakePunica(torch.tensor([0]))
    lora_a = torch.tensor(
        [
            [[[1.0, 0.0]]],
            [[[0.0, 1.0]]],
        ]
    )
    lora_b = torch.tensor(
        [
            [[[2.0]]],
            [[[5.0]]],
        ]
    )
    _attach_context(linear, punica, lora_a, lora_b)

    # This represents a prefill segment following a decode token. The global
    # mapping starts with slot 0, while this segment must use slot 1.
    segment_indices = torch.tensor([1])
    intermediate = prepare_dsa_lora(
        linear,
        torch.tensor([[3.0, 4.0]]),
        segment_indices,
    )
    actual = apply_prepared_dsa_lora(linear, torch.zeros(1, 1), intermediate)

    torch.testing.assert_close(actual, torch.tensor([[20.0]]))


def test_fully_sharded_column_gathers_lora_rank_before_expand() -> None:
    linear = torch.nn.Module()
    token_indices = torch.tensor([0])
    punica = FakePunica(token_indices)
    lora_a = torch.tensor([[[[1.0, 1.0]]]])
    lora_b = torch.tensor([[[[1.0, 1.0], [2.0, 0.0]]]])
    _attach_context(
        linear,
        punica,
        lora_a,
        lora_b,
        parallel_mode="column",
        fully_sharded=True,
        tp_size=2,
    )

    with patch(
        "vllm_ascend.lora.dsa.tensor_model_parallel_all_gather",
        side_effect=lambda value: torch.cat((value, value * 2), dim=-1),
    ) as all_gather:
        intermediate = prepare_dsa_lora(
            linear,
            torch.tensor([[1.0, 2.0]]),
            token_indices,
        )
        actual = apply_prepared_dsa_lora(linear, torch.zeros(1, 2), intermediate)

    all_gather.assert_called_once()
    torch.testing.assert_close(actual, torch.tensor([[9.0, 6.0]]))


def test_grouped_wo_a_routes_by_adapter_and_group() -> None:
    linear = torch.nn.Module()
    token_indices = torch.tensor([0, -1, 1])
    punica = FakePunica(token_indices)
    lora_a = torch.tensor(
        [
            [[[1.0, 0.0]]],
            [[[0.0, 1.0]]],
        ]
    )
    lora_b = torch.tensor(
        [
            [[[2.0], [3.0]]],
            [[[4.0], [5.0]]],
        ]
    )
    _attach_context(
        linear,
        punica,
        lora_a,
        lora_b,
        parallel_mode="grouped_column",
    )

    x = torch.tensor(
        [
            [[1.0, 10.0], [2.0, 20.0]],
            [[7.0, 8.0], [9.0, 10.0]],
            [[3.0, 4.0], [5.0, 6.0]],
        ]
    )
    actual = apply_grouped_dsa_lora(
        linear,
        torch.ones(3, 2, 1),
        x,
        token_indices,
    )

    expected = torch.tensor([[[3.0], [7.0]], [[1.0], [1.0]], [[17.0], [31.0]]])
    torch.testing.assert_close(actual, expected)


def test_grouped_wo_a_uses_cached_group_major_cube_routing() -> None:
    linear = torch.nn.Module()
    slots = [0] * 512 + [1] * 512
    routing = _make_grouped_routing(slots, num_groups=2)
    punica = FakePunica(routing.token_lora_indices)
    lora_a = torch.tensor(
        [
            [[[1.0, 0.0]]],
            [[[0.0, 1.0]]],
        ],
        dtype=torch.bfloat16,
    )
    lora_b = torch.tensor(
        [
            [[[2.0], [3.0]]],
            [[[4.0], [5.0]]],
        ],
        dtype=torch.bfloat16,
    )
    _attach_context(
        linear,
        punica,
        lora_a,
        lora_b,
        parallel_mode="grouped_column",
    )

    x = torch.arange(4096, dtype=torch.bfloat16).view(1024, 2, 2) / 256
    output = torch.ones(1024, 2, 1, dtype=torch.bfloat16)
    with (
        patch("vllm_ascend.lora.dsa._DSA_LORA_GMM_MIN_CUBE_RANK", 0),
        patch("vllm_ascend.lora.dsa._DSA_LORA_GMM_MIN_TOKENS", 0),
        patch(
            "vllm_ascend.lora.dsa.torch_npu.npu_grouped_matmul",
            side_effect=_fake_grouped_matmul,
        ) as grouped_matmul,
    ):
        actual = apply_grouped_dsa_lora(linear, output, x, routing)

    assert grouped_matmul.call_count == 2
    expected = torch.ones_like(output)
    expected[:512, 0, 0] += x[:512, 0, 0] * 2
    expected[:512, 1, 0] += x[:512, 1, 0] * 3
    expected[512:, 0, 0] += x[512:, 0, 1] * 4
    expected[512:, 1, 0] += x[512:, 1, 1] * 5
    torch.testing.assert_close(actual, expected)


def test_fully_sharded_wo_b_adds_local_b_slice_before_output_reduce() -> None:
    class FakeQuantMethod:
        @staticmethod
        def apply(layer, x, bias):
            del layer, bias
            return torch.ones(x.shape[0], 4)

    linear = torch.nn.Module()
    linear.input_is_parallel = True
    linear.tp_rank = 1
    linear.tp_size = 2
    linear.skip_bias_add = False
    linear.bias = None
    linear.quant_method = FakeQuantMethod()
    linear.reduce_results = True

    token_indices = torch.tensor([0, -1])
    punica = FakePunica(token_indices)
    lora_a = torch.tensor([[[[1.0, 1.0]]]])
    lora_b = torch.tensor([[[[2.0], [3.0]]]])
    _attach_context(
        linear,
        punica,
        lora_a,
        lora_b,
        parallel_mode="row",
        fully_sharded=True,
        tp_size=2,
        tp_rank=1,
    )

    with patch(
        "vllm_ascend.lora.dsa.tensor_model_parallel_all_reduce",
        side_effect=lambda value: value,
    ) as all_reduce:
        actual = forward_with_dsa_lora(
            linear,
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            token_indices,
        )

    assert all_reduce.call_count == 2
    expected = torch.tensor([[1.0, 1.0, 7.0, 10.0], [1.0, 1.0, 1.0, 1.0]])
    torch.testing.assert_close(actual, expected)
