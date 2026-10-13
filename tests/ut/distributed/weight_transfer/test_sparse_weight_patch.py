# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.distributed.weight_transfer.sparse_weight_patch import SparseWeightPatch


@pytest.mark.parametrize("projection", ["gate_proj", "up_proj", "down_proj"])
@pytest.mark.parametrize("expert", [0, 2, 3, 4])
@pytest.mark.parametrize("device", ["cpu", "npu"])
@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ep_high_global_expert_uses_local_slot_without_fallback(
    monkeypatch, projection, expert, device, tp_size, tp_rank, dtype
):
    if device == "npu" and (not hasattr(torch, "npu") or not torch.npu.is_available()):
        pytest.skip("NPU regression requires available NPU hardware")
    from vllm_ascend.distributed.weight_transfer import sparse_weight_patch as patcher
    from vllm_ascend.ops.fused_moe import routed_experts

    class FakeMethod:
        pass

    class FakeExperts:
        pass

    monkeypatch.setattr(routed_experts, "AscendRoutedExperts", FakeExperts)
    monkeypatch.setattr(routed_experts, "AscendUnquantizedFusedMoEMethod", FakeMethod)
    monkeypatch.setattr(patcher, "_supports_logical_flat_index_copy", lambda tensor: True)
    layer = FakeExperts()
    layer.quant_method = FakeMethod()
    layer.moe_config = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(tp_size=tp_size, ep_size=2),
        tp_rank=tp_rank,
        enable_eplb=False,
        is_act_and_mul=True,
    )
    layer.ascend_expert_map = torch.tensor([-1, -1, 0, 1])
    layer.w13_weight = torch.nn.Parameter(torch.arange(32.0, device=device, dtype=dtype).reshape(2, 4, 4))
    layer.w2_weight = torch.nn.Parameter(torch.arange(16.0, device=device, dtype=dtype).reshape(2, 2, 4))
    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_moe"), get_submodule=lambda name: layer)
    down = projection == "down_proj"
    shape = [4, 2 * tp_size] if down else [2 * tp_size, 4]
    param = layer.w2_weight if down else layer.w13_weight
    expected = param.detach().clone()
    target = 4 if down else 6 if projection == "up_proj" else 4
    if expert in (2, 3) and tp_rank == 0:
        target += (expert - 2) * param[0].numel()
        expected.view(-1)[target] = 99.0
    patch = SparseWeightPatch(
        f"model.layers.0.mlp.experts.{expert}.{projection}.weight",
        torch.tensor([1], dtype=torch.int32, device=device),
        torch.tensor([99.0], device=device, dtype=dtype),
        full_shape=tuple(shape),
    )
    if expert == 4:
        with pytest.raises(IndexError, match="outside expert map"):
            patcher._apply_direct_moe_patches(model, [(patch, shape)], verify=True)
    else:
        assert patcher._apply_direct_moe_patches(model, [(patch, shape)], verify=True) == []
    assert torch.equal(param, expected)


@pytest.mark.parametrize("bad_index", [-1, 8])
def test_batched_bounds_validates_entire_flush_before_loading(bad_index):
    from vllm_ascend.distributed.weight_transfer.sparse_weight_patch import apply_sparse_hf_patches_with_loader

    class NeverLoad:
        def load_weights(self, weights):
            pytest.fail("A malformed flush must not partially apply")

    patches = [
        (
            SparseWeightPatch(str(i), torch.tensor([0, 7], dtype=torch.int32), torch.ones(2), full_shape=tuple([2, 4])),
            [2, 4],
        )
        for i in range(20)
    ]
    patches.append(
        (
            SparseWeightPatch(
                "invalid", torch.tensor([bad_index], dtype=torch.int32), torch.ones(1), full_shape=tuple([4, 2])
            ),
            [4, 2],
        )
    )
    with pytest.raises(ValueError):
        apply_sparse_hf_patches_with_loader(NeverLoad(), patches)


def test_dynamic_eplb_falls_back_before_cached_nonlocal_filter(monkeypatch):
    from vllm_ascend.distributed.weight_transfer import sparse_weight_patch as patcher
    from vllm_ascend.ops.fused_moe import routed_experts

    class FakeMethod:
        dynamic_eplb = True

    class FakeExperts:
        pass

    monkeypatch.setattr(routed_experts, "AscendRoutedExperts", FakeExperts)
    monkeypatch.setattr(routed_experts, "AscendUnquantizedFusedMoEMethod", FakeMethod)
    layer = FakeExperts()
    layer.quant_method = FakeMethod()
    layer.moe_config = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(tp_size=1, ep_size=2), tp_rank=0, enable_eplb=False, is_act_and_mul=True
    )
    layer.ascend_expert_map = torch.tensor([-1, 0])
    layer.w13_weight = torch.nn.Parameter(torch.arange(16.0).reshape(1, 4, 4))
    original = layer.w13_weight.detach().clone()
    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_moe"), get_submodule=lambda name: layer)
    patch = SparseWeightPatch(
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([99.0]),
        full_shape=(2, 4),
    )
    patches = [(patch, [2, 4])]
    assert patcher._apply_direct_moe_patches(model, patches) == patches
    assert not hasattr(layer, "_sparse_expert_map_cpu")
    assert torch.equal(layer.w13_weight, original)


def test_batched_bounds_keeps_distinct_shape_limits():
    from vllm_ascend.distributed.weight_transfer.sparse_weight_patch import _validate_sparse_hf_patches

    patches = [
        (
            SparseWeightPatch("valid", torch.tensor([15], dtype=torch.int32), torch.ones(1), full_shape=tuple([4, 4])),
            [4, 4],
        ),
        (
            SparseWeightPatch("invalid", torch.tensor([8], dtype=torch.int32), torch.ones(1), full_shape=tuple([2, 4])),
            [2, 4],
        ),
    ]
    with pytest.raises(ValueError):
        _validate_sparse_hf_patches(patches)


@pytest.mark.parametrize("down", [False, True])
@pytest.mark.parametrize("tp_rank", [0, 1, 2, 3])
def test_batched_moe_group_matches_nan_masked_tp_loader(down, tp_rank):
    from vllm_ascend.distributed.weight_transfer.sparse_weight_patch import _apply_moe_group

    tp, experts, hidden, intermediate = (4, 3, 8, 12)
    shape = (hidden, intermediate) if down else (intermediate, hidden)
    runtime_shape = (experts, intermediate // tp, hidden) if down else (experts, hidden, 2 * intermediate // tp)
    actual = torch.randn(runtime_shape, dtype=torch.bfloat16)
    expected = actual.clone()
    entries = []
    for expert in range(experts):
        for projection in ["down_proj"] if down else ["gate_proj", "up_proj"]:
            indices = torch.arange(expert, shape[0] * shape[1], 7, dtype=torch.int32)
            values = torch.arange(indices.numel(), dtype=torch.bfloat16) - 5
            values[::5] = float("nan")
            values[1::5] = -0.0
            patch = SparseWeightPatch(f"{expert}.{projection}", indices, values, full_shape=tuple(shape))
            entries.append((patch, expert, projection))
            dense = torch.full(shape, float("nan"), dtype=torch.bfloat16)
            dense.view(-1)[indices.long()] = values
            width = intermediate // tp
            if down:
                source = dense[:, tp_rank * width : (tp_rank + 1) * width].t()
                destination = expected[expert]
            else:
                source = dense[tp_rank * width : (tp_rank + 1) * width].t()
                offset = width if projection == "up_proj" else 0
                destination = expected[expert, :, offset : offset + width]
            destination.copy_(torch.where(torch.isnan(source), destination, source))
    _apply_moe_group(actual, entries, shape, tp_rank, tp, down)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
