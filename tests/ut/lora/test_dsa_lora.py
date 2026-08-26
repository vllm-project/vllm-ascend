from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.lora import utils as lora_utils

from vllm_ascend.lora.dsa import (
    DSA_LORA_CLASSES,
    DSA_LORA_CONTEXT_ATTR,
    AscendDSAReplicatedLinearWithLoRA,
    DSALoRAContext,
    apply_grouped_dsa_lora,
    apply_prepared_dsa_lora,
    forward_with_dsa_lora,
    prepare_dsa_lora,
)
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
