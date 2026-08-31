from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.lora import utils as lora_utils

from vllm_ascend.lora.dsa import (
    DSA_LORA_CLASSES,
    DSA_LORA_CONTEXT_ATTR,
    AscendDSAReplicatedLinearWithLoRA,
    DSALoRAContext,
    apply_prepared_dsa_lora,
    prepare_dsa_lora,
)
from vllm_ascend.lora.utils import refresh_all_lora_classes


class FakePunica:
    def __init__(self, token_lora_indices: torch.Tensor):
        self.token_lora_indices = token_lora_indices

    def get_token_lora_indices(self, num_tokens: int) -> torch.Tensor:
        return self.token_lora_indices[:num_tokens]

    @staticmethod
    def get_dsa_sgmv_metadata(token_lora_indices: torch.Tensor):
        lora_indices, seq_lengths = torch.unique_consecutive(
            token_lora_indices,
            return_counts=True,
        )
        seq_start_locs = torch.zeros_like(seq_lengths)
        if seq_lengths.shape[0] > 1:
            seq_start_locs[1:] = torch.cumsum(seq_lengths, dim=0)[:-1]
        return SimpleNamespace(
            no_lora=bool(torch.all(lora_indices < 0)),
            op_args=(
                seq_start_locs,
                seq_lengths,
                lora_indices,
                lora_indices.shape[0],
                int(seq_lengths.max()) if seq_lengths.numel() else 0,
                token_lora_indices.shape[0],
            ),
        )

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

    @classmethod
    def sgmv_shrink(
        cls,
        x,
        weights,
        output,
        seq_start_locs,
        seq_lengths,
        lora_indices,
        batches,
        max_seq_length,
        token_nums,
        scale,
    ):
        del seq_start_locs, batches, max_seq_length, token_nums
        indices = torch.repeat_interleave(lora_indices, seq_lengths)
        cls.bgmv_shrink(x, weights, output, indices, scale)

    @classmethod
    def sgmv_expand_slice(
        cls,
        x,
        weights,
        output,
        seq_start_locs,
        seq_lengths,
        lora_indices,
        batches,
        max_seq_length,
        token_nums,
        offset,
        output_size,
        add_inputs,
    ):
        del seq_start_locs, batches, max_seq_length, token_nums
        indices = torch.repeat_interleave(lora_indices, seq_lengths)
        cls.bgmv_expand_slice(
            x,
            weights,
            output,
            indices,
            offset,
            output_size,
            add_inputs,
        )

    def add_shrink(
        self,
        y,
        x,
        lora_a_stacked,
        scale,
        *,
        sgmv_metadata=None,
        **kwargs,
    ):
        del kwargs
        assert sgmv_metadata is not None
        for output, lora_a in zip(y, lora_a_stacked):
            update = torch.zeros_like(output)
            self.sgmv_shrink(
                x,
                lora_a[:, 0].contiguous(),
                update,
                *sgmv_metadata.op_args,
                scale,
            )
            output.add_(update)

    def add_expand(
        self,
        y,
        x,
        lora_b_stacked,
        output_slices,
        offset_start=0,
        add_inputs=True,
        *,
        sgmv_metadata=None,
        **kwargs,
    ):
        del kwargs
        assert sgmv_metadata is not None
        offset = offset_start
        for inputs, lora_b, output_slice in zip(x, lora_b_stacked, output_slices):
            self.sgmv_expand_slice(
                inputs,
                lora_b[:, 0].contiguous(),
                y,
                *sgmv_metadata.op_args,
                offset,
                output_slice,
                add_inputs,
            )
            offset += output_slice


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
    wrapper.base_layer.prefix = "model.layers.0.self_attn.wq_a"
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
    assert punica.has_dsa_qkv_lora


def test_dsa_lora_wrappers_exclude_wo_projections() -> None:
    for projection_name in ("wo_a", "wo_b"):
        source_layer = torch.nn.Module()
        source_layer.prefix = f"model.layers.0.self_attn.{projection_name}"
        for wrapper_class in DSA_LORA_CLASSES:
            assert not wrapper_class.can_replace_layer(
                source_layer,
                SimpleNamespace(),
                [],
            )


def test_equal_projections_allocate_fresh_shrink_buffers() -> None:
    punica = FakePunica(torch.tensor([0, 0]))

    def make_wrapper(layer_index: int, projection_name: str):
        wrapper = object.__new__(AscendDSAReplicatedLinearWithLoRA)
        torch.nn.Module.__init__(wrapper)
        wrapper.base_layer = torch.nn.Module()
        wrapper.base_layer.prefix = f"model.layers.{layer_index}.self_attn.{projection_name}"
        wrapper.lora_config = SimpleNamespace(fully_sharded_loras=False)
        wrapper.lora_a_stacked = (torch.tensor([[[[1.0, 1.0]]]]),)
        wrapper.lora_b_stacked = (torch.tensor([[[[2.0], [3.0]]]]),)
        wrapper.output_slices = (2,)
        wrapper.tp_size = 1
        wrapper.tp_rank = 0
        wrapper.set_mapping(punica)
        return wrapper.base_layer

    first_wq_a = make_wrapper(0, "wq_a")
    second_wq_a = make_wrapper(1, "wq_a")

    with patch("vllm_ascend.lora.dsa.torch.zeros", wraps=torch.zeros) as zeros:
        first = prepare_dsa_lora(first_wq_a, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        assert first is not None
        first_values = first.buffers[0].clone()
        first.buffers[0].fill_(123.0)
        second = prepare_dsa_lora(second_wq_a, torch.tensor([[5.0, 6.0], [7.0, 8.0]]))

    assert second is not None
    assert first.buffers[0].data_ptr() != second.buffers[0].data_ptr()
    assert second.buffers[0].shape == (2, 1)
    assert zeros.call_count == 2
    torch.testing.assert_close(first_values, torch.tensor([[3.0], [7.0]]))
    torch.testing.assert_close(second.buffers[0], torch.tensor([[11.0], [15.0]]))


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
    with (
        patch.object(punica, "add_shrink", wraps=punica.add_shrink) as add_shrink,
        patch.object(punica, "add_expand", wraps=punica.add_expand) as add_expand,
        patch.object(punica, "sgmv_shrink", wraps=punica.sgmv_shrink) as sgmv_shrink,
        patch.object(punica, "sgmv_expand_slice", wraps=punica.sgmv_expand_slice) as sgmv_expand_slice,
    ):
        intermediate = prepare_dsa_lora(linear, x, token_indices)
        actual = apply_prepared_dsa_lora(linear, output, intermediate)

    expected = torch.tensor([[3.0, 4.0], [1.0, 1.0], [17.0, 21.0]])
    torch.testing.assert_close(actual, expected)
    add_shrink.assert_called_once()
    add_expand.assert_called_once()
    assert add_shrink.call_args.kwargs["sgmv_metadata"] is intermediate.sgmv_metadata
    assert add_expand.call_args.kwargs["sgmv_metadata"] is intermediate.sgmv_metadata
    sgmv_shrink.assert_called_once()
    sgmv_expand_slice.assert_called_once()


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


def test_cv_lora_skips_sgmv_for_base_only_segment() -> None:
    linear = torch.nn.Module()
    token_indices = torch.tensor([-1, -1])
    punica = FakePunica(token_indices)
    lora_a = torch.tensor([[[[1.0, 0.0]]]])
    lora_b = torch.tensor([[[[2.0], [3.0]]]])
    _attach_context(linear, punica, lora_a, lora_b)
    output = torch.ones(2, 2)

    with (
        patch.object(punica, "sgmv_shrink", wraps=punica.sgmv_shrink) as sgmv_shrink,
        patch.object(punica, "sgmv_expand_slice", wraps=punica.sgmv_expand_slice) as sgmv_expand_slice,
    ):
        intermediate = prepare_dsa_lora(
            linear,
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            token_indices,
        )
        actual = apply_prepared_dsa_lora(linear, output, intermediate)

    torch.testing.assert_close(actual, output)
    sgmv_shrink.assert_not_called()
    sgmv_expand_slice.assert_not_called()


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
