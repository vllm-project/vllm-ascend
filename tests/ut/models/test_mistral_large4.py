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

import abc
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


# B024: an ABC with no abstract methods is the point. The stand-in has to
# inherit from abc.ABC to get the layout AscendSFAImpl has -- a metaclass=
# ABCMeta class does not reproduce it, and neither does a plain class -- and
# it has to be instantiable.
class _Impl(abc.ABC):  # noqa: B024
    """Stand-in for AscendSFAImpl, with its object layout.

    The base matters. ``AscendSFAImpl`` descends from ``MLAAttentionImpl`` and
    ``AttentionImplBase(ABC, Generic[T])``, none of which declare
    ``__slots__``, so ``AttentionImplBase`` is the class that adds the instance
    dict. Putting a mixin in front of that layout is refused by CPython, which
    is why the write skip is installed on the instance instead. A stand-in
    without the ABC accepts the class switch and would let that regression
    back in; ``Generic`` is left out because it declares empty slots and so
    does not change the layout.
    """

    def _get_sfa_kv_slot_mapping(self, attn_metadata):
        return attn_metadata.slot_mapping


class _ImplWithTopkHook(_Impl):
    """An impl that defines the top-k hook AscendSFAImpl does not."""

    def __init__(self, calls: list[str]):
        self._calls = calls

    def record_logical_topk_ready(self) -> None:
        self._calls.append("topk_ready")


class _Attn:
    """Stand-in for the outer Ascend MLA wrapper."""

    def __init__(self, impl: _Impl | None = None, on_call=None):
        self.impl = impl if impl is not None else _Impl()
        self.mla_attn = SimpleNamespace(impl=self.impl)
        self._on_call = on_call

    def __call__(self, *_args, **_kwargs):
        if self._on_call is not None:
            self._on_call()
        return torch.zeros(3, 8)


def _attn(spec):
    module = ml4.MistralMLAAttention.__new__(ml4.MistralMLAAttention)
    torch.nn.Module.__init__(module)
    module.spec = spec
    module.attn = _Attn()
    module.indexer_stream = object()
    module.indexer_events = [object(), object()]
    return module


class _Model(torch.nn.Module):
    def __init__(self, specs):
        super().__init__()
        self.layers = torch.nn.ModuleList(_attn(spec) for spec in specs)


# --------------------------------------------------------------- KV sharing


def test_shared_kv_skip_narrows_the_slot_mapping_to_nothing():
    """A borrowing layer must write no latent rows at all."""
    impl = _Impl()
    ml4.adopt_shared_kv_skip(SimpleNamespace(attn=SimpleNamespace(impl=impl)))
    assert ml4.has_shared_kv_skip(impl)
    metadata = SimpleNamespace(slot_mapping=torch.arange(8, dtype=torch.int32))

    mapped = impl._get_sfa_kv_slot_mapping(metadata)

    assert mapped.numel() == 0
    # A zero-row view, not a sentinel: -1 would scatter onto the last slot.
    assert mapped.dtype == torch.int32
    assert not (mapped == -1).any()


def test_shared_kv_skip_finds_the_nested_ascend_impl():
    impl = _Impl()
    inner = SimpleNamespace(impl=None, mla_attn=SimpleNamespace(impl=impl))

    ml4.adopt_shared_kv_skip(SimpleNamespace(attn=inner))

    assert ml4.has_shared_kv_skip(impl)


def test_a_layer_that_owns_its_kv_still_writes_every_row():
    impl = _Impl()
    metadata = SimpleNamespace(slot_mapping=torch.arange(8, dtype=torch.int32))

    assert impl._get_sfa_kv_slot_mapping(metadata).numel() == 8


def test_adopt_installs_the_skip_on_borrowing_layers_only():
    # Layer 35 owns its cache; 39 and 43 read it. Mirrors the real pattern.
    model = _Model([_spec(35, 35), _spec(38, 38), _spec(39, 35), _spec(43, 35)])

    shared = ml4.adopt_ascend_modules(model)

    assert shared == 2
    installed = [ml4.has_shared_kv_skip(layer.attn.impl) for layer in model.layers]
    assert installed == [False, False, True, True]


def test_adopt_is_idempotent():
    """A second walk must leave the installed skip alone.

    Comparing the impl's type would pass for free, since the skip no longer
    changes it. What has to hold is that the override itself is not replaced:
    reinstalling would wrap the function already in the instance dict.
    """
    model = _Model([_spec(39, 35)])

    ml4.adopt_ascend_modules(model)
    impl = model.layers[0].attn.impl
    first = impl._get_sfa_kv_slot_mapping
    ml4.adopt_ascend_modules(model)

    assert model.layers[0].attn.impl is impl
    assert impl._get_sfa_kv_slot_mapping is first


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


def test_forward_delegates_hidden_states_to_the_outer_ascend_mla():
    calls: list[str] = []
    attn = _attn(_spec(0, 0))
    attn.__class__ = ml4.AscendMistralMLAAttention
    attn.num_local_heads = 2
    attn.v_head_dim = 4
    # Driving the indexer is AscendSFAImpl's job, not this module's.
    attn.indexer = lambda *a, **k: pytest.fail("the model must not run the indexer")

    attn._project_qkv = lambda *_a: pytest.fail("the outer Ascend MLA owns projection")
    attn.attn = _Attn(on_call=lambda: calls.append("attn"))
    attn.o_proj = lambda *_a: pytest.fail("the outer Ascend MLA owns o_proj")

    attn.forward(torch.zeros(3), torch.zeros(3, 16), torch.zeros(3), None)

    assert calls == ["attn"]


def test_forward_does_not_call_the_upstream_topk_hook():
    calls: list[str] = []
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
    attn.attn = _Attn(impl=_ImplWithTopkHook(calls))
    attn.o_proj = lambda x: (x,)

    attn.forward(torch.zeros(3), torch.zeros(3, 16), torch.zeros(3), None)

    assert calls == []


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
        ({"fuse_allreduce_rms": True}, "fuse_allreduce_rms"),
        ({"use_sequence_parallel_moe": True}, "Sequence-parallel MoE"),
    ],
)
def test_unsupported_deployments_are_rejected(overrides, match):
    with pytest.raises(NotImplementedError, match=match):
        ml4._reject_unsupported(_vllm_config(**overrides))


def test_a_supported_deployment_is_accepted():
    ml4._reject_unsupported(_vllm_config())


def test_eplb_is_accepted():
    """EPLB is supported; vLLM itself refuses it without expert parallelism."""
    ml4._reject_unsupported(_vllm_config(enable_eplb=True, enable_expert_parallel=True))


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


# ------------------------------------------------------------------- EPLB


class _FakeMoE:
    """Stands in for upstream's MistralMoE, counting expert-map rewrites."""

    def __init__(self):
        self.experts = SimpleNamespace(remaps=0)
        self.experts.update_expert_map = self._remap

    def _remap(self):
        self.experts.remaps += 1


def _eplb_config(num_redundant_experts=0, n_shared_experts=1, n_routed_experts=128):
    hf_config = SimpleNamespace(
        n_routed_experts=n_routed_experts,
        n_shared_experts=n_shared_experts,
        n_group=1,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config),
        parallel_config=SimpleNamespace(
            eplb_config=SimpleNamespace(num_redundant_experts=num_redundant_experts),
        ),
    )


def _moe_view(monkeypatch, num_moe=3, num_dense=1, ep_size=8, **config_kwargs):
    """Build the mixin over a stage of ``num_moe`` MoE and ``num_dense`` dense layers."""
    monkeypatch.setattr(ml4, "MistralMoE", _FakeMoE)
    monkeypatch.setattr(ml4, "get_ep_group", lambda: SimpleNamespace(world_size=ep_size))

    layers = [SimpleNamespace(mlp=object()) for _ in range(num_dense)]
    layers += [SimpleNamespace(mlp=_FakeMoE()) for _ in range(num_moe)]

    view = ml4.MistralLarge4MixtureOfExperts()
    view.model = SimpleNamespace(layers=layers)
    view.extract_moe_parameters(_eplb_config(**config_kwargs))
    return view


def test_only_moe_layers_of_this_stage_are_collected(monkeypatch):
    view = _moe_view(monkeypatch, num_moe=3, num_dense=2)

    assert view.num_moe_layers == 3
    assert len(view.moe_layers) == 3
    assert view.expert_weights == []


def test_counts_come_from_the_config_and_the_expert_parallel_group(monkeypatch):
    view = _moe_view(monkeypatch, ep_size=8, n_shared_experts=1)

    assert view.num_routed_experts == 128
    assert view.num_logical_experts == 128
    assert view.num_physical_experts == 128
    assert view.num_local_physical_experts == 16
    assert view.num_shared_experts == 1
    assert view.num_redundant_experts == 0
    assert view.num_expert_groups == 1


def test_redundant_experts_widen_the_physical_placement(monkeypatch):
    view = _moe_view(monkeypatch, ep_size=8, num_redundant_experts=16)

    assert view.num_logical_experts == 128
    assert view.num_physical_experts == 144
    assert view.num_local_physical_experts == 18


def test_a_stage_without_moe_layers_reports_no_experts(monkeypatch):
    view = _moe_view(monkeypatch, num_moe=0, num_dense=2)

    # is_mixture_of_experts() keys on num_moe_layers, so a dense stage must
    # report zero rather than the config's expert count.
    assert view.num_moe_layers == 0
    assert view.num_logical_experts == 0
    assert view.num_physical_experts == 0
    assert view.num_local_physical_experts == 0


def test_rebalancing_rewrites_the_counts_and_remaps_every_layer(monkeypatch):
    view = _moe_view(monkeypatch, num_moe=3, ep_size=8, num_redundant_experts=16)

    view.update_physical_experts_metadata(144, 18)

    assert view.num_physical_experts == 144
    assert view.num_redundant_experts == 16
    assert [moe.experts.remaps for moe in view.moe_mlp_layers] == [1, 1, 1]


def test_rebalancing_refuses_to_resize_the_local_expert_slots(monkeypatch):
    view = _moe_view(monkeypatch, ep_size=8)

    with pytest.raises(AssertionError):
        view.update_physical_experts_metadata(128, 8)


# ---------------------------------------------------------------- indexer


@pytest.mark.parametrize(
    "layer_idx,expected",
    [(0, False), (16, True), (17, False), (18, False), (19, False), (20, True)],
)
def test_only_the_producer_before_shared_layers_caches_topk(layer_idx, expected):
    config = SimpleNamespace(index_sharing=["full"] * 17 + ["shared", "shared", "full", "full", "shared"])
    spec = _spec(layer_idx, layer_idx, has_indexer=config.index_sharing[layer_idx] == "full")

    assert ml4._produces_topk_for_shared_layers(config, spec) is expected


def test_no_index_sharing_needs_no_topk_cache_copy():
    assert not ml4._produces_topk_for_shared_layers(SimpleNamespace(index_sharing=None), _spec(0, 0))


def test_prepare_supplies_the_three_scalars_the_backend_reads():
    """MistralIndexer defines none of them; the Ascend backend reads all three."""
    indexer = SimpleNamespace(head_dim=128)

    ml4.prepare_indexer_for_ascend(indexer, SimpleNamespace(index_topk=2048))

    assert indexer.topk_tokens == 2048
    # No q-LoRA: the value AscendSFAImpl already branches on.
    assert indexer.q_lora_rank is None
    assert indexer.softmax_scale == pytest.approx(128**-0.5)


def test_prepare_leaves_the_shared_module_names_alone():
    """The backend borrows these by name; the shim must not rebind them."""
    wq_b, wk, k_norm = object(), object(), object()
    indexer = SimpleNamespace(head_dim=128, n_head=32, wq_b=wq_b, wk_weights_proj=wk, k_norm=k_norm)

    ml4.prepare_indexer_for_ascend(indexer, SimpleNamespace(index_topk=2048))

    assert indexer.wq_b is wq_b
    assert indexer.wk_weights_proj is wk
    assert indexer.k_norm is k_norm
    assert indexer.n_head == 32


def test_ssmax_refuses_to_run_unscaled():
    """Without log positions the query would go unscaled, which is silent."""
    backend = ml4.AscendMistralIndexerBackend.__new__(ml4.AscendMistralIndexerBackend)
    backend.n_head, backend.head_dim = 2, 4
    backend.q_input_norm = lambda x: x
    backend.wq_b = lambda x: (torch.zeros(x.shape[0], 8),)
    backend.ssmax = lambda q, log_pos: q
    backend.log_positions = None

    with pytest.raises(RuntimeError, match="log"):
        backend.mistral_q(torch.zeros(3, 16))


def test_ssmax_is_applied_after_wq_b_with_the_step_log_positions():
    seen = {}

    def _ssmax(q, log_positions):
        seen["log_positions"] = log_positions
        return q * 2

    backend = ml4.AscendMistralIndexerBackend.__new__(ml4.AscendMistralIndexerBackend)
    backend.n_head, backend.head_dim = 2, 4
    backend.q_input_norm = lambda x: x + 1
    backend.wq_b = lambda x: (torch.ones(x.shape[0], 8),)
    backend.ssmax = _ssmax
    log_positions = torch.tensor([0.0, 0.7, 1.1])
    backend.log_positions = log_positions

    q = backend.mistral_q(torch.zeros(3, 16))

    assert q.shape == (3, 2, 4)
    assert torch.equal(q, torch.full((3, 2, 4), 2.0))
    assert seen["log_positions"] is log_positions


def test_a_model_without_ssmax_needs_no_log_positions():
    backend = ml4.AscendMistralIndexerBackend.__new__(ml4.AscendMistralIndexerBackend)
    backend.n_head, backend.head_dim = 2, 4
    backend.q_input_norm = lambda x: x
    backend.wq_b = lambda x: (torch.ones(x.shape[0], 8),)
    backend.ssmax = None
    backend.log_positions = None

    assert backend.mistral_q(torch.zeros(3, 16)).shape == (3, 2, 4)


@pytest.mark.parametrize("use_rope", [True, False])
def test_indexer_uses_interleaved_rope_or_bypasses_it_for_both_q_and_k(monkeypatch, use_rope):
    from vllm_ascend.attention import indexer

    def init_backend(self, _indexer, rope_dim):
        torch.nn.Module.__init__(self)
        self.qk_rope_head_dim = rope_dim
        self.is_rope_neox_style = True
        self.head_dim = 8
        self.n_head = 1
        self.wq_b = self.wk_weights_proj = lambda x: (x, None)
        self.k_norm = lambda x: x
        self.enable_sparse_li_c8 = self.enable_sparse_li_c4 = False

    calls = []

    def fake_rope(x, cos, sin, *, rope_dim, is_neox_style):
        calls.append((rope_dim, is_neox_style))
        return torch.cat((x[..., :rope_dim] + 1, x[..., rope_dim:]), dim=-1)

    monkeypatch.setattr(indexer.AscendSFAIndexerBackend, "__init__", init_backend)
    monkeypatch.setattr(indexer, "HAS_TRITON", True)
    monkeypatch.setattr(indexer, "rope_forward_triton_siso", fake_rope, raising=False)
    upstream = SimpleNamespace(indexer_use_rope=use_rope, q_input_norm=lambda x: x, ssmax=None)
    backend = ml4.AscendMistralIndexerBackend(upstream, 4)
    x = torch.arange(16).reshape(2, 8)
    cos = sin = torch.ones(2, 1, 1, 4)

    q = backend.project_query(x, x, cos, sin)
    k, _, _ = backend.forward_k(x, cos, sin)
    assert calls == ([(4, False), (4, False)] if use_rope else [])
    expected = x.clone()
    if use_rope:
        expected[:, :4] += 1
    torch.testing.assert_close(q, expected.unsqueeze(1))
    torch.testing.assert_close(k, q)


def test_nontriton_indexer_restores_interleaved_cache_layout(monkeypatch):
    from vllm_ascend.attention import indexer

    def split_half_rope(x, cos, sin):
        return torch.cat((-x[..., 1::2], x[..., 0::2]), dim=-1)

    monkeypatch.setattr(indexer, "HAS_TRITON", False)
    monkeypatch.setattr(indexer.torch_npu, "npu_interleave_rope", split_half_rope)
    backend = SimpleNamespace(qk_rope_head_dim=4, head_dim=6, is_rope_neox_style=False)
    x = torch.tensor([[[1, 2, 3, 4, 5, 6]]])
    cos, sin = torch.zeros(1, 1, 1, 4), torch.ones(1, 1, 1, 4)

    out = indexer.AscendSFAIndexerBackend.apply_indexer_rope(backend, x, cos, sin)

    torch.testing.assert_close(out, torch.tensor([[[-2, 1, -4, 3, 5, 6]]]))


def test_dspark_constructor_enables_local_rope(monkeypatch):
    rotary_emb = object()
    impls = [SimpleNamespace(use_mla_rope=False) for _ in range(3)]

    def init_draft(self, *, vllm_config, prefix):
        torch.nn.Module.__init__(self)
        layers = [SimpleNamespace(self_attn=SimpleNamespace(attn=SimpleNamespace(impl=impl))) for impl in impls]
        self.model = SimpleNamespace(rotary_emb=rotary_emb, layers=layers)

    monkeypatch.setattr(ml4.MistralLarge4DSparkForCausalLM, "__init__", init_draft)
    ml4.AscendMistralLarge4DSparkForCausalLM(vllm_config=SimpleNamespace())

    for impl in impls:
        assert impl.use_mla_rope
        assert impl.rope_metadata_source is rotary_emb


def test_dspark_draft_uses_local_rope_and_stable_metadata_buffers(monkeypatch):
    from vllm_ascend.attention import mla_v1

    def target_rope(*args, **kwargs):
        pytest.fail("the draft must use its own table")

    monkeypatch.setattr(mla_v1, "get_cos_and_sin_mla", target_rope)
    builder = mla_v1.AscendMLAMetadataBuilder.__new__(mla_v1.AscendMLAMetadataBuilder)
    table = torch.arange(24).view(6, 4)
    builder.rope_metadata_source = SimpleNamespace(cos_sin_cache=table)
    builder.rope_buffer_capacity = 4
    builder.rope_buffers = {}
    positions = torch.tensor([1, 3])

    prefill = builder._get_metadata_rope(positions, kind="prefill")
    decode = builder._get_metadata_rope(torch.tensor([2]), kind="decode", use_cache=True)
    updated = builder._get_metadata_rope(torch.tensor([4, 5]), kind="decode", use_cache=True)

    for actual, expected in zip(prefill, ml4.gather_mla_rope(table, positions)):
        torch.testing.assert_close(actual, expected)
    for actual, expected, old in zip(updated, ml4.gather_mla_rope(table, torch.tensor([4, 5])), decode):
        torch.testing.assert_close(actual, expected)
        assert actual.data_ptr() == old.data_ptr()


# ----------------------------------------------------- graph and DSpark


def test_target_uses_the_ascend_model_state_for_mrv2():
    from vllm_ascend.worker.v2.model_states.default import AscendModelState

    model_cls = ml4.AscendMistralLarge4ForCausalLM

    assert model_cls.get_model_state_cls(None) is AscendModelState


def test_outer_mla_exposes_the_backend_cache_contract():
    outer = ml4.AscendMultiHeadLatentAttention.__new__(ml4.AscendMultiHeadLatentAttention)
    torch.nn.Module.__init__(outer)
    original_cache = torch.zeros(2, 6)
    impl = object()
    outer.mla_attn = SimpleNamespace(
        impl=impl,
        kv_cache=original_cache,
        kv_cache_dtype="auto",
        _k_scale=torch.tensor(1.0),
    )

    assert outer.impl is impl
    assert outer.kv_cache is original_cache
    assert outer.kv_cache_dtype == "auto"
    assert outer._k_scale.item() == 1.0

    replacement = torch.ones(2, 6)
    outer.kv_cache = replacement
    assert outer.mla_attn.kv_cache is replacement


def test_dspark_splits_a_fused_mla_cache_into_component_views():
    fused = torch.arange(24).view(2, 12)

    nope, rope = ml4.AscendMistralLarge4DSparkForCausalLM._normalize_draft_kv_cache(fused, kv_lora_rank=8)

    torch.testing.assert_close(nope, fused[..., :8])
    torch.testing.assert_close(rope, fused[..., 8:])
    assert nope.untyped_storage().data_ptr() == fused.untyped_storage().data_ptr()
    assert rope.untyped_storage().data_ptr() == fused.untyped_storage().data_ptr()


def test_dspark_accepts_nested_component_cache_views():
    nope, rope = torch.zeros(2, 8), torch.zeros(2, 4)

    actual = ml4.AscendMistralLarge4DSparkForCausalLM._normalize_draft_kv_cache(((nope, rope),), kv_lora_rank=8)

    assert actual[0] is nope
    assert actual[1] is rope


@pytest.mark.parametrize("cache", [torch.zeros(2, 8), object()])
def test_dspark_rejects_an_invalid_cache_layout(cache):
    with pytest.raises(RuntimeError, match="Mistral DSpark"):
        ml4.AscendMistralLarge4DSparkForCausalLM._normalize_draft_kv_cache(cache, kv_lora_rank=8)


def test_the_model_does_not_call_the_indexer_itself():
    """AscendSFAImpl runs it; calling it here too would write the cache twice."""
    attn = _attn(_spec(0, 0))
    attn.__class__ = ml4.AscendMistralMLAAttention
    attn.num_local_heads, attn.v_head_dim = 2, 4
    attn.indexer = lambda *a, **k: pytest.fail("the model must not run the indexer")
    attn._project_qkv = lambda *_a: (
        torch.zeros(3, 2, 8),
        torch.zeros(3, 16),
        torch.zeros(3, 1, 4),
    )
    attn.attn = _Attn()
    attn.o_proj = lambda x: (x,)

    attn.forward(torch.zeros(3), torch.zeros(3, 16), torch.zeros(3), None)


# ------------------------------------------------------------ registration


def test_registration_is_guarded_on_the_upstream_package():
    from vllm_ascend.models import _upstream_has_module

    assert _upstream_has_module("vllm.models.mistral.mistral_large4") is True
    assert _upstream_has_module("vllm.models.not_a_real_package") is False
    # A dotted name whose parent is missing raises inside find_spec.
    assert _upstream_has_module("no_such_root.and_a_child") is False


def test_the_adapter_module_is_importable_without_registering():
    assert importlib.util.find_spec("vllm_ascend.models.mistral_large4") is not None
