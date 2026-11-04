#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
"""Unit tests for the Mistral Large 4 Ascend adapter.

The upstream model package is not in a vLLM release yet
(vllm-project/vllm#64534), so every test here skips where it is absent.
"""

import importlib.util
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip(
    "vllm.models.mistral.mistral_large4",
    reason="upstream vLLM does not ship Mistral Large 4 (vllm-project/vllm#64534)",
)

from vllm_ascend.models import mistral_large4 as ml4  # noqa: E402


def _spec(layer_idx: int, kv_source_layer: int, has_indexer: bool = True):
    return SimpleNamespace(
        layer_idx=layer_idx,
        kv_source_layer=kv_source_layer,
        shares_kv=kv_source_layer != layer_idx,
        use_rope=layer_idx % 4 != 3,
        use_sparse=True,
        has_indexer=has_indexer,
        indexer_use_rope=True,
        is_moe=layer_idx > 0,
    )


class _Impl:
    """Stand-in for AscendSFAImpl: only the one hook the mixin overrides."""

    def _get_sfa_kv_slot_mapping(self, attn_metadata):
        return attn_metadata.slot_mapping


def _attn(spec):
    module = ml4.MistralMLAAttention.__new__(ml4.MistralMLAAttention)
    torch.nn.Module.__init__(module)
    module.spec = spec
    module.attn = SimpleNamespace(impl=_Impl())
    module.indexer_stream = object()
    module.indexer_events = [object(), object()]
    return module


class _Model(torch.nn.Module):
    def __init__(self, specs):
        super().__init__()
        self.layers = torch.nn.ModuleList(_attn(spec) for spec in specs)


# --------------------------------------------------------------- KV sharing


def test_shared_kv_mixin_narrows_the_slot_mapping_to_nothing():
    """A borrowing layer must write no latent rows at all."""
    impl = _Impl()
    impl.__class__ = ml4._shared_kv_impl_class(_Impl)
    metadata = SimpleNamespace(slot_mapping=torch.arange(8, dtype=torch.int32))

    mapped = impl._get_sfa_kv_slot_mapping(metadata)

    assert mapped.numel() == 0
    # A zero-row view, not a sentinel: -1 would scatter onto the last slot.
    assert mapped.dtype == torch.int32
    assert not (mapped == -1).any()


def test_a_layer_that_owns_its_kv_still_writes_every_row():
    impl = _Impl()
    metadata = SimpleNamespace(slot_mapping=torch.arange(8, dtype=torch.int32))

    assert impl._get_sfa_kv_slot_mapping(metadata).numel() == 8


def test_adopt_installs_the_skip_on_borrowing_layers_only():
    # Layer 35 owns its cache; 39 and 43 read it. Mirrors the real pattern.
    model = _Model([_spec(35, 35), _spec(38, 38), _spec(39, 35), _spec(43, 35)])

    shared = ml4.adopt_ascend_modules(model)

    assert shared == 2
    installed = [isinstance(layer.attn.impl, ml4.AscendSharedKVSlotMappingMixin) for layer in model.layers]
    assert installed == [False, False, True, True]


def test_adopt_is_idempotent():
    model = _Model([_spec(39, 35)])

    ml4.adopt_ascend_modules(model)
    first = type(model.layers[0].attn.impl)
    ml4.adopt_ascend_modules(model)

    assert type(model.layers[0].attn.impl) is first


def test_the_skip_refuses_a_layer_with_no_impl():
    attn = _attn(_spec(39, 35))
    attn.attn = SimpleNamespace(impl=None)

    with pytest.raises(ValueError, match="overwrite the latent cache"):
        ml4.adopt_shared_kv_skip(attn)


# -------------------------------------------------------------- CUDA paths


def test_adopt_switches_the_attention_class_and_drops_the_cuda_stream():
    model = _Model([_spec(0, 0), _spec(39, 35)])

    ml4.adopt_ascend_modules(model)

    for layer in model.layers:
        assert isinstance(layer, ml4.AscendMistralMLAAttention)
        # Built by upstream's constructor for the aux-stream overlap; unused
        # once the forward is serial, and torch.cuda has no NPU counterpart.
        assert layer.indexer_stream is None
        assert layer.indexer_events == []


def test_forward_runs_the_indexer_before_the_projections_and_skips_the_hook():
    """Serial order, and no call to a method the Ascend SFA impl lacks."""
    calls = []
    attn = _attn(_spec(0, 0))
    attn.__class__ = ml4.AscendMistralMLAAttention
    attn.num_local_heads = 2
    attn.v_head_dim = 4
    attn.indexer = lambda *a, **k: calls.append("indexer")

    def _project(*_args):
        calls.append("project")
        return torch.zeros(3, 2, 8), torch.zeros(3, 16), torch.zeros(3, 1, 4)

    attn._project_qkv = _project
    attn.attn = SimpleNamespace(
        impl=_Impl(),  # no record_logical_topk_ready, as on Ascend
        __call__=None,
    )
    attn.attn = lambda *a, **k: (calls.append("attn"), torch.zeros(3, 8))[1]
    attn.attn.impl = _Impl()
    attn.o_proj = lambda x: (calls.append("o_proj"), x)[1:]

    attn.forward(torch.zeros(3), torch.zeros(3, 16), torch.zeros(3), None)

    assert calls == ["indexer", "project", "attn", "o_proj"]


def test_forward_calls_the_topk_hook_when_the_impl_defines_it():
    calls = []
    attn = _attn(_spec(0, 0))
    attn.__class__ = ml4.AscendMistralMLAAttention
    attn.num_local_heads = 2
    attn.v_head_dim = 4
    attn.indexer = None
    attn._project_qkv = lambda *_a: (
        torch.zeros(3, 2, 8),
        torch.zeros(3, 16),
        torch.zeros(3, 1, 4),
    )
    impl = _Impl()
    impl.record_logical_topk_ready = lambda: calls.append("topk_ready")
    attn.attn = lambda *a, **k: torch.zeros(3, 8)
    attn.attn.impl = impl
    attn.o_proj = lambda x: (x,)

    attn.forward(torch.zeros(3), torch.zeros(3, 16), torch.zeros(3), None)

    assert calls == ["topk_ready"]


# ------------------------------------------------------------- rejections


def _vllm_config(**overrides):
    parallel = SimpleNamespace(
        enable_eplb=False,
        use_sequence_parallel_moe=False,
        enable_expert_parallel=False,
    )
    compilation = SimpleNamespace(pass_config=SimpleNamespace(fuse_allreduce_rms=False))
    for key, value in overrides.items():
        target = parallel if hasattr(parallel, key) else compilation.pass_config
        setattr(target, key, value)
    return SimpleNamespace(parallel_config=parallel, compilation_config=compilation)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"enable_eplb": True}, "EPLB"),
        ({"fuse_allreduce_rms": True}, "fuse_allreduce_rms"),
        ({"use_sequence_parallel_moe": True}, "Sequence-parallel MoE"),
    ],
)
def test_unsupported_deployments_are_rejected(overrides, match):
    with pytest.raises(NotImplementedError, match=match):
        ml4._reject_unsupported(_vllm_config(**overrides))


def test_a_supported_deployment_is_accepted():
    ml4._reject_unsupported(_vllm_config())


def test_fused_mc2_is_not_rejected():
    """Unlike Muse Spark: no per-route norm needs an apply_gmm2 boundary."""
    ml4._reject_unsupported(_vllm_config())


# ------------------------------------------------------------- MoE layout


def _moe_config(enable_ep=False, n_routed_experts=128, moe_intermediate_size=6144):
    parallel = SimpleNamespace(enable_expert_parallel=enable_ep)
    hf = SimpleNamespace(
        n_routed_experts=n_routed_experts,
        moe_intermediate_size=moe_intermediate_size,
    )
    return SimpleNamespace(
        parallel_config=parallel,
        model_config=SimpleNamespace(hf_config=hf),
    )


def test_experts_must_divide_across_the_expert_parallel_group(monkeypatch):
    monkeypatch.setattr(ml4, "get_ep_group", lambda: SimpleNamespace(world_size=5))

    with pytest.raises(ValueError, match="divide evenly"):
        ml4.validate_moe_parallelism(_moe_config(enable_ep=True))


@pytest.mark.parametrize("ep_size", [1, 2, 4, 8, 16])
def test_a_divisible_expert_parallel_group_is_accepted(monkeypatch, ep_size):
    monkeypatch.setattr(ml4, "get_ep_group", lambda: SimpleNamespace(world_size=ep_size))

    ml4.validate_moe_parallelism(_moe_config(enable_ep=True))


def test_the_expert_group_is_not_consulted_when_expert_parallel_is_off(monkeypatch):
    """vLLM builds the EP group either way; only the flag shards the experts."""

    def _boom():
        raise AssertionError("EP group must not be consulted with EP disabled")

    monkeypatch.setattr(ml4, "get_ep_group", _boom)

    ml4.validate_moe_parallelism(_moe_config(enable_ep=False, n_routed_experts=127))


def test_the_megamoe_fallback_is_reported_once(caplog):
    with caplog.at_level("INFO"):
        ml4.validate_moe_parallelism(_moe_config(moe_intermediate_size=6144))

    assert "grouped-matmul" in caplog.text


def test_an_expert_width_inside_the_window_reports_nothing(caplog):
    with caplog.at_level("INFO"):
        ml4.validate_moe_parallelism(_moe_config(moe_intermediate_size=2048))

    assert "grouped-matmul" not in caplog.text


def test_a_dense_config_is_left_alone(monkeypatch):
    monkeypatch.setattr(ml4, "get_ep_group", lambda: SimpleNamespace(world_size=7))

    ml4.validate_moe_parallelism(_moe_config(enable_ep=True, n_routed_experts=0))


# ------------------------------------------------------------ registration


def test_registration_is_guarded_on_the_upstream_package():
    from vllm_ascend.models import _upstream_has_module

    assert _upstream_has_module("vllm.models.mistral.mistral_large4") is True
    assert _upstream_has_module("vllm.models.not_a_real_package") is False
    # A dotted name whose parent is missing raises inside find_spec.
    assert _upstream_has_module("no_such_root.and_a_child") is False


def test_the_adapter_module_is_importable_without_registering():
    assert importlib.util.find_spec("vllm_ascend.models.mistral_large4") is not None
