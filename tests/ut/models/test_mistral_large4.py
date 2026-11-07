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
