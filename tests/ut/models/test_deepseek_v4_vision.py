# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from vllm.model_executor.models.interfaces import supports_eagle3

from vllm_ascend.models.deepseek_v4 import vision_dp
from vllm_ascend.models.deepseek_v4.vision import DeepseekV4Aligner, DeepseekV4ViT
from vllm_ascend.models.deepseek_v4.vl_model import (
    AscendDeepseekV4ForConditionalGeneration,
)


def test_vision_wrapper_exposes_dspark_aux_hidden_state_interface():
    model = AscendDeepseekV4ForConditionalGeneration.__new__(AscendDeepseekV4ForConditionalGeneration)
    nn.Module.__init__(model)
    language_model = MagicMock()
    model.language_model = language_model

    assert supports_eagle3(model)

    model.set_aux_hidden_state_layers((41, 42, 43))
    language_model.set_aux_hidden_state_layers.assert_called_once_with((41, 42, 43))


@pytest.mark.parametrize("enabled", [False, True])
def test_image_encoder_dp_is_opt_in(enabled):
    model = AscendDeepseekV4ForConditionalGeneration.__new__(AscendDeepseekV4ForConditionalGeneration)
    nn.Module.__init__(model)
    model.use_data_parallel = enabled
    model.vision = MagicMock()
    model.aligner = MagicMock()
    model.aligner.w1.weight = torch.ones(1)
    features = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    model.aligner.return_value = features
    patches = torch.ones(2, 3, 2, 2)
    grid = torch.tensor([[1, 2]])
    perm = torch.tensor([1, 0])
    with patch(
        "vllm_ascend.models.deepseek_v4.vl_model.run_dp_sharded_deepseek_v4_vision",
        return_value=(features,),
    ) as dp:
        result = model._process_image_input(patches, grid, grid, perm)
    assert model.supports_encoder_tp_data
    if enabled:
        dp.assert_called_once_with(model.vision, model.aligner, patches, grid, grid, perm)
        model.vision.assert_not_called()
        assert result[0] is features
    else:
        dp.assert_not_called()
        model.vision.assert_called_once()
        torch.testing.assert_close(result[0], features[perm])


def make_models(dtype=torch.float32):
    torch.manual_seed(17)
    config = SimpleNamespace(
        vision_patch_size=2,
        vision_dim=8,
        vision_n_heads=2,
        vision_inter_dim=12,
        vision_n_layers=2,
        vision_rope_theta=10000.0,
        vision_downsample_ratio=3,
        hidden_size=12,
    )
    return DeepseekV4ViT(config).to(dtype).eval(), DeepseekV4Aligner(config).to(dtype).eval()


def make_inputs(grids, dtype=torch.float32):
    generator = torch.Generator().manual_seed(23)
    vit_grid = torch.tensor(grids, dtype=torch.int64).reshape(-1, 2)
    llm_grid = (vit_grid + 2) // 3
    patches = torch.randn(sum(h * w for h, w in grids), 3, 2, 2, generator=generator).to(dtype)
    lengths = llm_grid.prod(dim=1).tolist()
    perm = torch.cat([torch.randperm(n, generator=generator) for n in lengths]) if lengths else torch.empty(0).long()
    return patches, vit_grid, llm_grid, perm


def reference_embeddings(vision, aligner, inputs):
    patches, vit_grid, llm_grid, perm = inputs
    outputs = []
    patch_offset = output_offset = 0
    for (h, w), (lh, lw) in zip(vit_grid.tolist(), llm_grid.tolist(), strict=True):
        image = patches[patch_offset : patch_offset + h * w].to(aligner.w1.weight.dtype)
        features = aligner(vision(image, h, w), h, w)
        outputs.append(features[perm[output_offset : output_offset + lh * lw].to(features.device)])
        patch_offset += h * w
        output_offset += lh * lw
    return tuple(outputs)


def assignments(inputs, tp_size):
    order, counts, _ = vision_dp.get_load_balance_assignment(inputs[1].prod(dim=1).tolist(), tp_size)
    result = []
    offset = 0
    for count in counts:
        result.append(order[offset : offset + count])
        offset += count
    return result


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("grids", [[(1, 1)], [(4, 4), (3, 5)], [(3, 3), (8, 7), (1, 1), (4, 5), (6, 6)]])
@torch.inference_mode()
def test_encoder_dp_matches_replicated_path(monkeypatch, dtype, tp_size, grids):
    vision, aligner = make_models(dtype)
    # The processor's dtype need not match the model's dtype.
    inputs = make_inputs(grids, torch.float32)
    reference = reference_embeddings(vision, aligner, inputs)
    rank_images = assignments(inputs, tp_size)
    max_rows = max(sum(reference[i].shape[0] for i in indices) for indices in rank_images)
    rank_buffers = []
    for indices in rank_images:
        buffer = aligner.w2.weight.new_zeros((max_rows, aligner.w2.out_features))
        offset = 0
        for i in indices:
            rows = reference[i].shape[0]
            buffer[offset : offset + rows].copy_(reference[i])
            offset += rows
        rank_buffers.append(buffer)
    expected_gather = torch.cat(rank_buffers)
    monkeypatch.setattr(vision_dp, "get_tensor_model_parallel_world_size", lambda: tp_size)
    call_count = 0
    for rank in range(tp_size):
        monkeypatch.setattr(vision_dp, "get_tensor_model_parallel_rank", lambda rank=rank: rank)
        spy = MagicMock(wraps=vision.forward)
        with monkeypatch.context() as context:
            context.setattr(vision, "forward", spy)

            def gather(local, dim, rank=rank):
                assert dim == 0
                torch.testing.assert_close(local, rank_buffers[rank], rtol=0, atol=0)
                return expected_gather

            gather_mock = MagicMock(side_effect=gather)
            context.setattr(vision_dp, "tensor_model_parallel_all_gather", gather_mock)
            output = vision_dp.run_dp_sharded_deepseek_v4_vision(vision, aligner, *inputs)
        assert len(output) == len(reference)
        for actual, expected in zip(output, reference, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert actual.dtype == dtype
            if tp_size > 1:
                assert actual.untyped_storage().data_ptr() != expected_gather.untyped_storage().data_ptr()
                assert actual.untyped_storage().nbytes() == actual.numel() * actual.element_size()
        assert spy.call_count == len(rank_images[rank])
        assert gather_mock.call_count == int(tp_size > 1)
        call_count += spy.call_count
    # Every image is computed exactly once across the whole TP group.
    assert call_count == len(grids)


@torch.inference_mode()
def test_empty_batch_has_no_encoder_or_collective_calls(monkeypatch):
    vision, aligner = make_models()
    monkeypatch.setattr(vision, "forward", MagicMock(side_effect=AssertionError("Unexpected encoder call")))
    gather = MagicMock(side_effect=AssertionError("Unexpected collective"))
    monkeypatch.setattr(vision_dp, "tensor_model_parallel_all_gather", gather)
    assert vision_dp.run_dp_sharded_deepseek_v4_vision(vision, aligner, *make_inputs([])) == ()


@pytest.mark.parametrize(
    "case", ["grid_shape", "grid_dtype", "nonpositive", "merge", "patch_count", "perm_count", "perm_dtype"]
)
def test_invalid_metadata_rejected_before_collective(monkeypatch, case):
    vision, aligner = make_models()
    inputs = list(make_inputs([(4, 5)]))
    if case == "grid_shape":
        inputs[1] = inputs[1].flatten()
    elif case == "grid_dtype":
        inputs[1] = inputs[1].float()
    elif case == "nonpositive":
        inputs[1][0, 0] = 0
    elif case == "merge":
        inputs[2][0, 0] += 1
    elif case == "patch_count":
        inputs[0] = inputs[0][:-1]
    elif case == "perm_count":
        inputs[3] = inputs[3][:-1]
    elif case == "perm_dtype":
        inputs[3] = inputs[3].float()
    gather = MagicMock()
    monkeypatch.setattr(vision_dp, "tensor_model_parallel_all_gather", gather)
    with pytest.raises(ValueError):
        vision_dp.run_dp_sharded_deepseek_v4_vision(vision, aligner, *inputs)
    gather.assert_not_called()


def test_assignment_uses_patch_load_not_image_count():
    inputs = make_inputs([(10, 10), (3, 3), (2, 2), (1, 1)])
    assert assignments(inputs, 2) == [[0], [1, 2, 3]]
