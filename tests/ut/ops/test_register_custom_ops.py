# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops import register_custom_ops


def test_pad_and_reduce_uses_tp_reduce_scatter(monkeypatch):
    reduce_scatter = MagicMock(side_effect=lambda tensor, dim: tensor[::2])

    monkeypatch.setattr(register_custom_ops, "tensor_model_parallel_reduce_scatter", reduce_scatter)
    monkeypatch.setattr(register_custom_ops, "_EXTRA_CTX", SimpleNamespace(pad_size=0))

    input_tensor = torch.arange(8).view(4, 2)
    output = register_custom_ops._pad_and_reduce_impl(input_tensor)

    reduce_scatter.assert_called_once_with(input_tensor, 0)
    assert torch.equal(output, input_tensor[::2])


def test_maybe_all_reduce_with_prefetch_keeps_existing_collective(monkeypatch):
    input_tensor = torch.ones((2, 4))
    controller = MagicMock()
    controller.begin_next.return_value = True
    prepared = object()
    controller.prepare_next_gdn_out.return_value = prepared
    all_reduce = MagicMock(side_effect=lambda tensor: tensor)

    monkeypatch.setattr(
        "vllm_ascend.ops.weight_prefetch_qwen35.current_controller",
        lambda: controller,
    )
    monkeypatch.setattr(register_custom_ops, "tensor_model_parallel_all_reduce", all_reduce)
    monkeypatch.setattr(
        register_custom_ops,
        "_EXTRA_CTX",
        SimpleNamespace(moe_comm_type=MoECommType.ALLGATHER, flash_comm_v1_enabled=False),
    )

    output = register_custom_ops._maybe_all_reduce_tensor_model_parallel_impl(input_tensor, "layer.0")

    assert output is input_tensor
    controller.begin_next.assert_called_once_with("layer.0", input_tensor)
    controller.prepare_next_gdn_out.assert_called_once_with("layer.0", input_tensor)
    controller.commit_next_gdn_out.assert_called_once_with(prepared)
    controller.settle_next_input.assert_called_once_with()
    all_reduce.assert_called_once_with(input_tensor)


def test_gdn_prefetch_hooks_preserve_dependency_tensors(monkeypatch):
    dependency = torch.ones((2, 4))
    consumer_input = torch.zeros((2, 4))
    controller = MagicMock()
    monkeypatch.setattr(
        "vllm_ascend.ops.weight_prefetch_qwen35.current_controller",
        lambda: controller,
    )

    commit_output = register_custom_ops._qwen35_gdn_out_prefetch_commit_impl(dependency, "layer.0")
    join_output = register_custom_ops._qwen35_gdn_out_prefetch_join_impl(consumer_input, "layer.0")

    assert commit_output is dependency
    assert join_output is consumer_input
    controller.begin_gdn_out.assert_called_once_with("layer.0", dependency)
    controller.join_gdn_out.assert_called_once_with("layer.0")
