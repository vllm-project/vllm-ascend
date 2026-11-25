# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import vllm.lora.utils as lora_utils
from torch import nn
from vllm.config.lora import LoRAConfig
from vllm.lora.layers import ColumnParallelLinearWithLoRA, DeepseekV4GroupedWoAWithLoRA
from vllm.lora.lora_model import LoRAModel
from vllm.lora.lora_weights import LoRALayerWeights
from vllm.lora.model_manager import LoRAModelManager

from vllm_ascend.attention.dsa_v1 import AscendDSAImpl
from vllm_ascend.lora.deepseek_v4 import AscendDeepseekV4GroupedWoAWithLoRA
from vllm_ascend.lora.utils import refresh_all_lora_classes
from vllm_ascend.models.deepseek_v4.grouped_wo_a import AscendDeepseekV4GroupedWoA
from vllm_ascend.ops.dsa import AscendDeepseekSparseAttention, DSAModules, dsa_forward


class _ReferencePunica:
    def __init__(self, indices):
        self.indices = indices

    def add_lora_linear(self, output, x, lora_a, lora_b, scale, output_slices):
        assert x.is_contiguous() and output.is_contiguous()
        assert len(lora_a) == len(lora_b) == len(output_slices) == 1
        for token, adapter in enumerate(self.indices):
            if adapter >= 0:
                output[token] += x[token] @ lora_a[0][adapter, 0].T @ lora_b[0][adapter, 0].T * scale


def _make_grouped_projection(tp_size, tp_rank):
    base = AscendDeepseekV4GroupedWoA.__new__(AscendDeepseekV4GroupedWoA)
    nn.Module.__init__(base)
    base.input_size = 4
    base.output_size = 12
    base.output_size_per_partition = base.output_size // tp_size
    base.tp_size, base.tp_rank = tp_size, tp_rank
    base.n_local_groups = 4 // tp_size
    base.support_fp8_attention = False
    full_weight = torch.arange(48, dtype=torch.float32).reshape(4, 4, 3) / 32
    start = tp_rank * base.n_local_groups
    base.weight = nn.Parameter(full_weight[start : start + base.n_local_groups].clone(), requires_grad=False)
    return base


def test_grouped_projection_constructor_inherits_tp_group_metadata():
    config = SimpleNamespace(
        lora_config=None,
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(o_groups=4, o_lora_rank=3)),
    )
    with (
        patch("vllm_ascend.models.deepseek_v4.grouped_wo_a.get_current_vllm_config", return_value=config),
        patch("vllm_ascend.ops.linear.get_current_vllm_config", return_value=config),
        patch("vllm_ascend.ops.linear.get_parallel_op", return_value=(None, 1, 2)),
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=1),
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=2),
        patch("vllm_ascend.models.deepseek_v4.grouped_wo_a.get_current_hardware_profile") as profile,
    ):
        profile.return_value.supports.return_value = False
        layer = AscendDeepseekV4GroupedWoA(
            4, 12, bias=False, params_dtype=torch.bfloat16, prefix="model.layers.2.self_attn.wo_a", return_bias=False
        )
    assert layer.n_local_groups == 2
    assert layer.o_lora_rank == 3
    assert layer.tp_size == 2 and layer.tp_rank == 1
    assert layer.weight.shape == (6, 4)


@pytest.mark.parametrize("fully_sharded", [False, True])
def test_grouped_projection_selects_specialized_lora_wrapper(monkeypatch, fully_sharded):
    monkeypatch.setattr(lora_utils, "_all_lora_classes", lora_utils._all_lora_classes)
    refresh_all_lora_classes()
    config = SimpleNamespace(max_lora_rank=2, max_loras=2, lora_dtype=torch.float32, fully_sharded_loras=fully_sharded)
    layer = lora_utils.from_layer(_make_grouped_projection(2, 1), config.max_loras, config, [])
    assert isinstance(layer, AscendDeepseekV4GroupedWoAWithLoRA)
    assert isinstance(layer, DeepseekV4GroupedWoAWithLoRA)
    assert len(layer.lora_b_stacked) == 2
    assert layer.lora_a_stacked[0].shape[-2] == config.max_lora_rank


@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 1), (4, 3)])
@pytest.mark.parametrize("fully_sharded", [False, True])
def test_grouped_adapter_reload_clears_unused_rank_without_changing_other_slots(tp_size, tp_rank, fully_sharded):
    layer = AscendDeepseekV4GroupedWoAWithLoRA(_make_grouped_projection(tp_size, tp_rank))
    config = SimpleNamespace(max_lora_rank=4, lora_dtype=torch.float32, fully_sharded_loras=fully_sharded)
    layer.create_lora_weights(2, config)
    for slot in range(2):
        layer.set_lora(slot, torch.ones(4, 4), torch.ones(12, 4))

    layer.set_lora(0, torch.full((1, 4), 2.0), torch.full((12, 1), 3.0))
    expected_a = torch.zeros_like(layer.lora_a_stacked[0][0])
    expected_a[:, 0] = 2
    torch.testing.assert_close(layer.lora_a_stacked[0][0], expected_a)
    for group_b in layer.lora_b_stacked:
        expected_b = torch.zeros_like(group_b[0])
        expected_b[..., 0] = 3
        torch.testing.assert_close(group_b[0], expected_b)

    layer.reset_lora(0)
    for weights in (*layer.lora_a_stacked, *layer.lora_b_stacked):
        torch.testing.assert_close(weights[0], torch.zeros_like(weights[0]))
        torch.testing.assert_close(weights[1], torch.ones_like(weights[1]))


def test_shared_dsa_generic_lora_wrapper_keeps_existing_bmm_path():
    base = _make_grouped_projection(1, 0)
    generic_wrapper = ColumnParallelLinearWithLoRA(base)
    generic_wrapper.forward = MagicMock(side_effect=AssertionError("Generic linear cannot consume grouped input"))
    impl = AscendDSAImpl.__new__(AscendDSAImpl)
    impl.wo_a = generic_wrapper
    impl.support_fp8_attention = False
    x = torch.randn(3, 4, 4)
    expected = torch.einsum("tgi,gio->tgo", x, base.weight)
    with patch("torch_npu.npu_transpose_batchmatmul", return_value=expected, create=True) as bmm:
        actual = impl._wo_a_bmm(x)
    generic_wrapper.forward.assert_not_called()
    assert bmm.call_args.args[0] is x
    assert bmm.call_args.args[1] is base.weight
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 0), (2, 1), (4, 3)])
@pytest.mark.parametrize("fully_sharded", [False, True])
@pytest.mark.parametrize("quantized", [False, True])
def test_grouped_wo_a_mixed_adapters_preserves_tp_groups(tp_size, tp_rank, fully_sharded, quantized):
    base = _make_grouped_projection(tp_size, tp_rank)
    base.support_fp8_attention = quantized
    if quantized:
        base.weight_scale = nn.Parameter(torch.zeros(1, dtype=torch.uint8), requires_grad=False)
    layer = AscendDeepseekV4GroupedWoAWithLoRA(base)
    config = SimpleNamespace(max_lora_rank=4, lora_dtype=torch.float32, fully_sharded_loras=fully_sharded)
    layer.create_lora_weights(2, config)
    adapter_a = [torch.arange(8, dtype=torch.float32).reshape(2, 4) / 16 + i for i in range(2)]
    adapter_b = [torch.arange(24, dtype=torch.float32).reshape(12, 2) / 16 + i for i in range(2)]
    for index in range(2):
        layer.set_lora(index, adapter_a[index], adapter_b[index])
    indices = [0, -1, 1, 0]
    layer.punica_wrapper = _ReferencePunica(indices)
    x = torch.arange(4 * base.n_local_groups * 4, dtype=torch.float32).reshape(4, base.n_local_groups, 4) / 16
    expected = torch.einsum("tgi,gio->tgo", x, base.weight)
    for token, adapter in enumerate(indices):
        if adapter >= 0:
            for group in range(base.n_local_groups):
                global_group = tp_rank * base.n_local_groups + group
                rows = adapter_b[adapter][global_group * 3 : (global_group + 1) * 3]
                expected[token, group] += x[token, group] @ adapter_a[adapter].T @ rows.T

    def bmm(inputs, weight, **kwargs):
        # Match the contiguous token-major output of the real NPU operator.
        return torch.einsum("tgi,gio->tgo", inputs, weight).contiguous()

    impl = AscendDSAImpl.__new__(AscendDSAImpl)
    impl.wo_a = layer
    with (
        patch("torch_npu.npu_transpose_batchmatmul", side_effect=bmm, create=True),
        patch("torch_npu.npu_transpose_quant_batchmatmul", side_effect=bmm, create=True),
        patch(
            "torch_npu.npu_dynamic_mx_quant",
            side_effect=lambda value, **kwargs: (value.clone(), torch.zeros(1, dtype=torch.uint8)),
            create=True,
        ),
    ):
        actual = impl._wo_a_bmm(x)

    torch.testing.assert_close(actual, expected)
    assert actual.is_contiguous()
    assert layer.lora_a_stacked[0].shape[-2] == config.max_lora_rank
    assert all(weight.is_contiguous() for weight in layer.lora_b_stacked)


@pytest.mark.parametrize("prefill_size,decode_size,otp", [(2, 1, False), (1, 2, False), (1, 1, True)])
def test_grouped_wo_a_rejects_unmapped_token_exchange(prefill_size, decode_size, otp):
    config = SimpleNamespace(
        lora_config=object(),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=prefill_size, decode_context_parallel_size=decode_size
        ),
    )
    with (
        patch("vllm_ascend.models.deepseek_v4.grouped_wo_a.get_current_vllm_config", return_value=config),
        patch("vllm_ascend.models.deepseek_v4.grouped_wo_a.oproj_tp_enable", return_value=otp),
        pytest.raises(NotImplementedError, match="adapter mappings"),
    ):
        AscendDeepseekV4GroupedWoA(4, 12)


def test_dsa_forward_refreshes_projection_aliases_after_lora_wrapping():
    impl = SimpleNamespace(wo_a=object(), wo_b=object(), forward=MagicMock())
    owner = SimpleNamespace(wo_a=object(), wo_b=object(), dsa_attn=SimpleNamespace(impl=impl, layer_name="attn"))
    context = SimpleNamespace(no_compile_layers={"layer": owner}, attn_metadata=None)
    hidden_states = torch.ones(2, 4)
    output = torch.empty_like(hidden_states)
    with patch("vllm_ascend.ops.dsa.get_forward_context", return_value=context):
        dsa_forward(hidden_states, output, "layer")
    assert impl.wo_a is owner.wo_a
    assert impl.wo_b is owner.wo_b
    impl.forward.assert_called_once_with("attn", hidden_states, None, None, output)


def _make_dsa_projection_model():
    owner = nn.Module()
    owner.wo_a = _make_grouped_projection(1, 0)
    owner.wo_b = nn.Identity()
    impl = AscendDSAImpl.__new__(AscendDSAImpl)
    impl.support_fp8_attention = False
    impl.forward = lambda name, x, cache, metadata, output: output.copy_(impl._wo_a_bmm(x))
    backend = nn.Module()
    backend.impl, backend.layer_name = impl, "attn"
    modules = DSAModules(
        wq_a=nn.Identity(),
        q_norm=nn.Identity(),
        q_norm_without_weight=nn.Identity(),
        wq_b=nn.Identity(),
        wkv=nn.Identity(),
        kv_norm=nn.Identity(),
        wo_a=owner.wo_a,
        wo_b=owner.wo_b,
        attn_sink=nn.Identity(),
        indexer=None,
        compressor=None,
        swa_cache_layer=nn.Identity(),
    )
    config = SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context={}))
    with (
        patch("vllm_ascend.ops.dsa.DSAAttention", return_value=backend),
        patch("vllm_ascend.ops.dsa.get_current_vllm_config", return_value=config),
    ):
        owner.dsa_attn = AscendDeepseekSparseAttention(
            dim=12,
            n_heads=4,
            scale=1.0,
            n_local_heads=4,
            q_lora_rank=2,
            o_lora_rank=3,
            head_dim=4,
            rope_head_dim=0,
            nope_head_dim=4,
            eps=1e-6,
            n_groups=4,
            n_local_groups=4,
            window_size=8,
            compress_ratio=1,
            dsa_modules=modules,
            prefix="layer",
            output_projection_owner=owner,
        )
    assert all(module is not owner for module in owner.dsa_attn.modules())
    model = nn.Module()
    model.config, model.packed_modules_mapping = SimpleNamespace(), {}
    model.embed_input_ids = lambda input_ids: input_ids
    model.forward = lambda input_ids, positions: input_ids
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module() for _ in range(3)])
    model.model.layers[2].self_attn = owner
    return model, owner, config.compilation_config.static_forward_context


@pytest.mark.parametrize(
    "target", [None, "wo_a", "model.layers.2.self_attn.wo_a", "model.layers.2.self_attn.dsa_attn.wo_a"]
)
@pytest.mark.parametrize("is_profiling", [False, True])
def test_dsa_applies_adapter_loaded_with_target_allowlist(monkeypatch, target, is_profiling):
    monkeypatch.setattr(lora_utils, "_all_lora_classes", lora_utils._all_lora_classes)
    refresh_all_lora_classes()
    model, owner, layers = _make_dsa_projection_model()
    base = owner.wo_a
    x = torch.arange(48, dtype=torch.float32).reshape(3, 4, 4) / 16
    base_output = torch.einsum("tgi,gio->tgo", x, base.weight)
    a = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 8
    b = torch.arange(24, dtype=torch.float32).reshape(12, 2) / 16
    targets_alias = target is not None and ".dsa_attn." in target
    module_name = target if targets_alias else "model.layers.2.self_attn.wo_a"
    lora_config = LoRAConfig(
        max_loras=1,
        max_lora_rank=8,
        lora_dtype=torch.float32,
        target_modules=None if target is None else [target],
    )
    vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(enable_expert_parallel=False))
    with patch("vllm.lora.model_manager.get_punica_wrapper", return_value=_ReferencePunica([0, -1, 0])):
        manager = LoRAModelManager(model, 3, 8, 16, lora_config, torch.device("cpu"), vllm_config)
    projection = owner.dsa_attn.wo_a if targets_alias else owner.wo_a
    assert isinstance(projection, AscendDeepseekV4GroupedWoAWithLoRA)
    assert set(manager.modules) == {module_name}
    if targets_alias:
        assert owner.wo_a is base
    elif target == module_name:
        # The real manager intentionally leaves the non-matching alias unwrapped.
        assert owner.dsa_attn.wo_a is base
    manager.add_adapter(LoRAModel(1, 2, {module_name: LoRALayerWeights(module_name, 2, 2, a, b)}))
    manager.activate_adapter(1)

    context = SimpleNamespace(no_compile_layers=layers, attn_metadata=None if is_profiling else {})
    actual = torch.empty_like(base_output)
    with (
        patch("vllm_ascend.ops.dsa.get_forward_context", return_value=context),
        patch("vllm_ascend.ops.dsa._build_kv_cache", return_value=None),
        patch(
            "torch_npu.npu_transpose_batchmatmul",
            side_effect=lambda inputs, weight, **kwargs: torch.einsum("tgi,gio->tgo", inputs, weight).contiguous(),
            create=True,
        ),
    ):
        dsa_forward(x, actual, "layer")
    delta = torch.einsum("tgi,ri,gor->tgo", x, a, b.reshape(4, 3, 2))
    delta[1].zero_()
    torch.testing.assert_close(actual, base_output + delta)
    torch.testing.assert_close(actual[1], base_output[1])
    assert not torch.allclose(actual[0], base_output[0])
    assert owner.dsa_attn.dsa_attn.impl.wo_a is projection
