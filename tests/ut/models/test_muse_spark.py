# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU contracts of the Muse Spark Ascend adapter.

The NPU kernels are replaced by PyTorch references here; the end-to-end run
on NPU lives in ``tests/e2e/pull_request/one_card/test_muse_spark.py``.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch import nn

pytest.importorskip("vllm.models.muse_spark.amd.model", reason="vLLM without Muse Spark")

import vllm.models.muse_spark.common.layers as upstream_layers  # noqa: E402
from vllm.distributed import parallel_state  # noqa: E402
from vllm.models.muse_spark.amd.model import MuseSpark1xAttention  # noqa: E402
from vllm.models.muse_spark.common.layers import (  # noqa: E402
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)

import vllm_ascend.models.muse_spark as muse_spark  # noqa: E402
from vllm_ascend import models  # noqa: E402
from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner  # noqa: E402

NUM_TOKENS = 5
NUM_HEADS = 2
HEAD_DIM = 8
EPS = 1e-5


def _registered_architectures() -> dict[str, str]:
    registered: dict[str, str] = {}
    with patch.object(
        models.ModelRegistry,
        "register_model",
        side_effect=lambda name, target: registered.setdefault(name, target),
    ):
        models.register_model()
    return registered


def test_both_entry_points_are_registered():
    registered = _registered_architectures()

    assert registered["MuseSpark1xForCausalLM"] == "vllm_ascend.models.muse_spark:AscendMuseSpark1xForCausalLM"
    assert (
        registered["MuseSparkForConditionalGeneration"]
        == "vllm_ascend.models.muse_spark:AscendMuseSparkForConditionalGeneration"
    )


def test_registration_is_skipped_without_upstream_package():
    with patch.object(models, "_upstream_has_module", return_value=False):
        registered = _registered_architectures()

    assert "MuseSpark1xForCausalLM" not in registered
    assert "MuseSparkForConditionalGeneration" not in registered


def _post_expert_norm(hidden_size: int) -> nn.Module:
    norm = muse_spark.AscendMuseSpark1xRMSNormWithInputScale(
        hidden_size=hidden_size,
        eps=EPS,
        has_weight=True,
        is_post_norm=True,
        zero_centered_gamma=True,
        post_norm_gain_center_type="one",
    )
    with torch.no_grad():
        norm.weight.normal_(0, 0.1)
    return norm


def test_reference_experts_normalize_each_route_before_the_weighted_sum():
    torch.manual_seed(0)
    num_experts, hidden_size, intermediate_size, top_k = 4, 6, 5, 2
    experts = muse_spark.MuseSparkReferenceExperts(num_experts, hidden_size, intermediate_size, torch.float32)
    with torch.no_grad():
        experts.w13_weight.normal_()
        experts.w2_weight.normal_()
    norm = _post_expert_norm(hidden_size)
    hidden_states = torch.randn(NUM_TOKENS, hidden_size)
    topk_ids = torch.stack([torch.randperm(num_experts)[:top_k] for _ in range(NUM_TOKENS)]).to(torch.int32)
    topk_weights = torch.rand(NUM_TOKENS, top_k)

    output = experts(hidden_states, topk_weights, topk_ids, norm)

    expected = torch.zeros(NUM_TOKENS, hidden_size)
    unnormalized = torch.zeros(NUM_TOKENS, hidden_size)
    for token in range(NUM_TOKENS):
        for route in range(top_k):
            expert = int(topk_ids[token, route])
            gate, up = (experts.w13_weight[expert] @ hidden_states[token]).chunk(2)
            row = experts.w2_weight[expert] @ (F.silu(gate) * up)
            expected[token] += topk_weights[token, route] * norm.forward_native(row, out_dtype=torch.float32)
            unnormalized[token] += topk_weights[token, route] * row
    torch.testing.assert_close(output, expected)

    # A standard MoE combine followed by one norm is a different function.
    norm_after_sum = norm.forward_native(unnormalized, out_dtype=torch.float32)
    assert not torch.allclose(output, norm_after_sum)


def test_a_row_wise_norm_before_the_combine_is_the_per_route_norm():
    """``npu_moe_token_unpermute`` weights and sums the routes in one op, so the
    norm cannot sit inside the combine. It does not have to: the norm reads one
    row at a time, so it commutes with the permutation the combine undoes, and
    normalizing the grouped rows is the same function as normalizing per route.
    """
    torch.manual_seed(0)
    top_k, hidden_size = 3, 8
    norm = _post_expert_norm(hidden_size)
    expert_rows = torch.randn(NUM_TOKENS, top_k, hidden_size)
    topk_weights = torch.rand(NUM_TOKENS, top_k)

    reference = (norm.forward_native(expert_rows, out_dtype=torch.float32) * topk_weights.unsqueeze(-1)).sum(dim=1)

    # The dispatcher hands the experts their rows grouped by expert and the
    # combine scatters them back; any permutation stands in for that round trip.
    permutation = torch.randperm(NUM_TOKENS * top_k)
    normalized = norm.forward_native(expert_rows.reshape(-1, hidden_size)[permutation], out_dtype=torch.float32)
    restored = torch.empty_like(normalized)
    restored[permutation] = normalized
    combined = (restored.view(NUM_TOKENS, top_k, hidden_size) * topk_weights.unsqueeze(-1)).sum(dim=1)

    torch.testing.assert_close(combined, reference)


class _StubMlpStage(nn.Module):
    """An Ascend MoE method reduced to the one call the norm wraps."""

    rows: torch.Tensor

    def apply_gmm2(self, mlp_compute_input, hidden_states, act_out_scale):
        return self.rows


class _StubAdapter(nn.Module):
    """``AscendFusedMoEMethod``'s shape: an adapter holding the live scheme."""

    def __init__(self, scheme: nn.Module) -> None:
        super().__init__()
        self.quant_method = scheme


def _bare_moe_method(rows: torch.Tensor | None = None) -> nn.Module:
    method = _StubMlpStage()
    if rows is not None:
        method.rows = rows
    muse_spark.adopt_post_expert_norm(method)
    return method


def test_fused_moe_method_normalizes_the_rows_gmm2_returns():
    """The norm is installed on the down-projection output, one row per route."""
    torch.manual_seed(0)
    hidden_size = 8
    routed_rows = torch.randn(NUM_TOKENS * 3, hidden_size)
    method = _bare_moe_method(routed_rows)
    norm = _post_expert_norm(hidden_size)
    method.post_expert_norm_fn = norm.forward

    with patch.object(muse_spark, "tensor_model_parallel_all_reduce", side_effect=AssertionError("all-reduce")):
        # Complete rows, as on one card or under expert parallelism.
        output = method.apply_gmm2(None, None, None)

    expected = norm.forward_native(routed_rows, out_dtype=routed_rows.dtype)
    torch.testing.assert_close(output, expected)
    assert output.dtype == routed_rows.dtype


@pytest.mark.parametrize(("tp_size", "pre_norm_all_reduce"), [(1, False), (2, True)])
def test_routed_experts_set_the_pre_norm_all_reduce_from_the_moe_config(tp_size, pre_norm_all_reduce):
    """Expert parallelism leaves the MoE TP size at 1; plain TP does not.

    The flag is set by the routed experts, not by the method's constructor:
    the norm is mixed in by reassigning ``__class__`` on a method the factory
    already built, so there is no constructor of ours to read the config in.
    """
    experts = object.__new__(muse_spark.AscendMuseSparkRoutedExperts)
    nn.Module.__init__(experts)
    experts.moe_config = SimpleNamespace(moe_parallel_config=SimpleNamespace(tp_size=tp_size))
    experts.quant_method = method = _StubMlpStage()

    experts.adopt_mlp_stage()

    assert method.pre_norm_all_reduce is pre_norm_all_reduce


def test_the_pre_norm_all_reduce_reaches_a_quantized_checkpoints_scheme():
    """A quantized checkpoint's MLP stage is the scheme the adapter wraps, and
    the TP all-reduce has to land there too.

    This is what a subclass of the unquantized MoE method cannot do: the
    quantized path never constructs one, so the partial rows would be
    normalized unreduced and every TP rank would return a different answer.
    """
    experts = object.__new__(muse_spark.AscendMuseSparkRoutedExperts)
    nn.Module.__init__(experts)
    experts.moe_config = SimpleNamespace(moe_parallel_config=SimpleNamespace(tp_size=2))
    scheme = _StubMlpStage()
    adapter = _StubAdapter(scheme)
    experts.quant_method = adapter

    assert experts.adopt_mlp_stage() is scheme

    assert scheme.pre_norm_all_reduce is True
    assert not isinstance(adapter, muse_spark.MuseSparkPostExpertNormMixin)


def test_tp_partial_rows_are_summed_before_the_norm():
    """Without expert parallelism each TP rank holds half of every expert's
    intermediate size, so its down-projection rows are partial sums. Summing
    them before the norm gives the unsharded rows' norm on every rank;
    normalizing the partials and summing afterwards does not.
    """
    torch.manual_seed(0)
    hidden_size, intermediate_size, num_rows = 8, 6, NUM_TOKENS * 3
    norm = _post_expert_norm(hidden_size)
    activated = torch.randn(num_rows, intermediate_size)
    w2 = torch.randn(hidden_size, intermediate_size)
    half = intermediate_size // 2
    partial_rows = [activated[:, :half] @ w2[:, :half].T, activated[:, half:] @ w2[:, half:].T]

    outputs = []
    for rank_rows in partial_rows:
        method = _bare_moe_method(rank_rows)
        method.post_expert_norm_fn = norm.forward
        method.pre_norm_all_reduce = True
        with patch.object(
            muse_spark,
            "tensor_model_parallel_all_reduce",
            side_effect=lambda rows: partial_rows[0] + partial_rows[1],
        ) as all_reduce:
            outputs.append(method.apply_gmm2(None, None, None))
        assert all_reduce.call_count == 1
        assert all_reduce.call_args.args[0] is rank_rows

    expected = norm.forward_native(activated @ w2.T, out_dtype=torch.float32)
    for output in outputs:
        torch.testing.assert_close(output, expected)
    normalized_partials = sum(norm.forward_native(rows, out_dtype=torch.float32) for rows in partial_rows)
    assert not torch.allclose(normalized_partials, expected)


def test_the_mlp_stage_raises_when_the_norm_was_never_attached():
    """Silently skipping the norm would be a wrong model, not a slower one."""
    method = _bare_moe_method(torch.zeros(2, 4))

    with pytest.raises(RuntimeError, match="post-expert norm"):
        method.apply_gmm2(None, None, None)


def test_the_norm_lands_on_the_scheme_a_quantized_checkpoint_installs():
    """Quantized, ``apply_gmm2`` runs on the scheme the adapter wraps, not the
    adapter, because that is the object handed to ``fused_experts``."""
    torch.manual_seed(0)
    hidden_size = 8
    routed_rows = torch.randn(NUM_TOKENS, hidden_size)
    scheme = _StubMlpStage()
    scheme.rows = routed_rows
    adapter = _StubAdapter(scheme)

    target = muse_spark.adopt_post_expert_norm(adapter)

    assert target is scheme
    assert not isinstance(adapter, muse_spark.MuseSparkPostExpertNormMixin)
    norm = _post_expert_norm(hidden_size)
    target.post_expert_norm_fn = norm.forward
    torch.testing.assert_close(
        scheme.apply_gmm2(None, None, None),
        norm.forward_native(routed_rows, out_dtype=routed_rows.dtype),
    )


def test_a_method_without_an_mlp_stage_is_rejected_rather_than_left_unnormalized():
    with pytest.raises(RuntimeError, match="MLP stage"):
        muse_spark.adopt_post_expert_norm(nn.Module())


class _Container(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = MuseSpark1xRMSNorm(HEAD_DIM, eps=EPS, has_weight=True, zero_centered_gamma=True)
        self.scaled_norm = MuseSpark1xRMSNormWithInputScale(HEAD_DIM, eps=EPS, has_weight=True)


def test_adopt_ascend_modules_moves_norms_to_the_torch_reference():
    container = _Container()
    with torch.no_grad():
        container.norm.weight.normal_()
        container.scaled_norm.weight.normal_()
    weights = (container.norm.weight, container.scaled_norm.weight)

    muse_spark.adopt_ascend_modules(container)

    assert type(container.norm) is muse_spark.AscendMuseSpark1xRMSNorm
    assert type(container.scaled_norm) is muse_spark.AscendMuseSpark1xRMSNormWithInputScale
    assert (container.norm.weight, container.scaled_norm.weight) == weights
    x = torch.randn(NUM_TOKENS, HEAD_DIM)
    with patch.object(upstream_layers, "fused_rmsnorm_fwd", side_effect=AssertionError("fused kernel called")):
        for norm in (container.norm, container.scaled_norm):
            assert not norm.supports_fused_kernel(x)
            torch.testing.assert_close(norm(x), norm.forward_native(x))


def _attention(rotary_emb=None) -> muse_spark.AscendMuseSpark1xAttention:
    attention = object.__new__(MuseSpark1xAttention)
    nn.Module.__init__(attention)
    attention.num_heads = NUM_HEADS
    attention.num_kv_heads = NUM_HEADS
    attention.head_dim = HEAD_DIM
    attention.q_size = NUM_HEADS * HEAD_DIM
    attention.kv_size = NUM_HEADS * HEAD_DIM
    attention.rms_norm_eps = EPS
    attention.use_o_gate = True
    attention.fuse_o_norm_gate = True
    attention.use_attn_temperature_tuning = False
    attention.qk_norm = MuseSpark1xRMSNorm(HEAD_DIM, eps=EPS, has_weight=False)
    attention.rotary_emb = rotary_emb
    attention.fused_qk_norm_rope_kv_insert = True
    attention.attn = SimpleNamespace(kv_cache_update_bypass=True)
    muse_spark.adopt_ascend_modules(attention)
    return attention


def test_attention_hands_the_kv_cache_write_back_to_the_backend():
    attention = _attention()

    assert type(attention) is muse_spark.AscendMuseSpark1xAttention
    assert type(attention.qk_norm) is muse_spark.AscendMuseSpark1xRMSNorm
    assert attention.fused_qk_norm_rope_kv_insert is False
    assert attention.attn.kv_cache_update_bypass is False


def test_qk_norm_and_rope_run_in_fp32_and_round_once():
    rope_inputs: list[torch.Tensor] = []

    def forward_native(positions, query, key):
        rope_inputs.extend((query, key))
        return query + 1, key + 1

    attention = _attention(rotary_emb=SimpleNamespace(forward_native=forward_native))
    qkv = torch.randn(NUM_TOKENS, 4 * NUM_HEADS * HEAD_DIM, dtype=torch.bfloat16)

    q, k, v, og = attention._qk_norm_rope_kv_insert(qkv, torch.arange(NUM_TOKENS))

    # The upstream kernel rounds once at the store, so RoPE must see the fp32
    # norm output itself, not a bf16 copy of it.
    q_in, k_in, _, _ = qkv.split(NUM_HEADS * HEAD_DIM, dim=-1)
    q_norm, k_norm = (
        attention.qk_norm.forward_native(x.reshape(-1, HEAD_DIM), out_dtype=torch.float32).reshape(NUM_TOKENS, -1)
        for x in (q_in, k_in)
    )
    assert len(rope_inputs) == 2
    assert torch.equal(rope_inputs[0], q_norm)
    assert torch.equal(rope_inputs[1], k_norm)
    assert torch.equal(q, (q_norm + 1).to(torch.bfloat16))
    assert torch.equal(k, (k_norm + 1).to(torch.bfloat16))
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16
    assert og is not None and og.shape == (NUM_TOKENS, NUM_HEADS * HEAD_DIM)


def _torch_rms_norm_gated(x, g, weight, bias, activation, eps):
    x32 = x.float()
    normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (normed * torch.sigmoid(g.float())).to(x.dtype)


def test_output_gate_uses_the_weightless_ascend_gated_norm():
    attention = _attention()
    attention.qkv_proj = lambda hidden: (hidden, None)
    attention.attn = lambda q, k, v: q
    attention.o_proj = lambda x: (x, None)
    hidden = torch.randn(NUM_TOKENS, 4 * NUM_HEADS * HEAD_DIM)
    calls = []

    def recording_gated_norm(x, g, weight, bias, activation, eps):
        calls.append((weight, bias, activation, g.is_contiguous()))
        return _torch_rms_norm_gated(x, g, weight, bias, activation, eps)

    with patch.object(muse_spark, "rms_norm_gated", side_effect=recording_gated_norm):
        output = attention(torch.arange(NUM_TOKENS), hidden)

    assert calls == [(None, None, "sigmoid", True)]
    q, _, _, og = attention._qk_norm_rope_kv_insert(hidden, torch.arange(NUM_TOKENS))
    assert og is not None
    shape = (NUM_TOKENS, NUM_HEADS, HEAD_DIM)
    expected = _torch_rms_norm_gated(q.reshape(shape), og.reshape(shape), None, None, "sigmoid", EPS)
    torch.testing.assert_close(output, expected.reshape(NUM_TOKENS, -1))


def _init_with_quantization(name: str | None):
    model = object.__new__(muse_spark.AscendMuseSpark1xForCausalLM)
    quant_config = None if name is None else SimpleNamespace(get_name=lambda: name)
    with (
        patch.object(muse_spark.MuseSpark1xForCausalLM, "__init__", return_value=None) as upstream_init,
        patch.object(muse_spark, "adopt_ascend_modules") as adopt,
    ):
        muse_spark.AscendMuseSpark1xForCausalLM.__init__(model, vllm_config=SimpleNamespace(quant_config=quant_config))
    return upstream_init, adopt


def test_quantization_configs_other_than_modelslim_are_rejected():
    """Their schemes reach method classes the per-route norm was never put on."""
    model = object.__new__(muse_spark.AscendMuseSpark1xForCausalLM)
    vllm_config = SimpleNamespace(quant_config=SimpleNamespace(get_name=lambda: "compressed-tensors"))

    with pytest.raises(NotImplementedError, match="ModelSlim"):
        muse_spark.AscendMuseSpark1xForCausalLM.__init__(model, vllm_config=vllm_config)


@pytest.mark.parametrize("name", [None, "ascend"])
def test_unquantized_and_modelslim_checkpoints_pass_the_quantization_gate(name):
    upstream_init, adopt = _init_with_quantization(name)

    assert upstream_init.called
    assert adopt.called


def test_the_fused_qkv_o_gate_projection_is_declared_for_the_quant_lookup():
    """Upstream drops ``qkv_proj`` from its own mapping, so the quantization
    lookup cannot resolve the fused module without this entry."""
    from vllm_ascend.quantization.configs.modelslim_config import UPDATED_PACKED_MODULES_MAPPING

    for model_type in ("muse_spark", "muse_spark_text", "musespark1x_text"):
        mapping = UPDATED_PACKED_MODULES_MAPPING[model_type]
        # The output gate is the fourth shard the stock q/k/v packing misses.
        assert mapping["qkv_proj"] == ["q_proj", "k_proj", "v_proj", "gate_proj"]


def _moe_vllm_config(
    *,
    enable_expert_parallel: bool,
    enable_eplb: bool = False,
    data_parallel_size: int = 1,
    prefill_context_parallel_size: int = 1,
) -> SimpleNamespace:
    text_config = SimpleNamespace(
        hidden_size=16,
        num_local_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=0,
        routed_scaling_factor=1.0,
        output_multiplier=1.0,
        moe_hidden_size=8,
        moe_intermediate_size=8,
        rms_norm_eps=EPS,
        post_norm_eps=EPS,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=text_config, dtype=torch.float32),
        parallel_config=SimpleNamespace(
            enable_expert_parallel=enable_expert_parallel,
            enable_eplb=enable_eplb,
            use_sequence_parallel_moe=False,
            data_parallel_size=data_parallel_size,
            prefill_context_parallel_size=prefill_context_parallel_size,
        ),
    )


class _StubLinear(nn.Module):
    """Stands in for the vLLM linear layers, which need an initialized TP group."""

    def __init__(self, input_size: int, output_size: int, **kwargs) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_size, input_size), requires_grad=False)


class _FakeRoutedExperts(nn.Module):
    """The container the factory returns, with the real norm plumbing on it.

    The methods are taken from the class under test rather than reimplemented,
    so a change to how the norm is installed shows up here.
    """

    mlp_stage_method = muse_spark.AscendMuseSparkRoutedExperts.mlp_stage_method
    attach_post_expert_norm = muse_spark.AscendMuseSparkRoutedExperts.attach_post_expert_norm


class _FakeRunner(nn.Module):
    """Stands in for the runner the factory returns."""

    def __init__(self) -> None:
        super().__init__()
        self.routed_experts = _FakeRoutedExperts()
        self.routed_experts.quant_method = _bare_moe_method()


def _ep_group(world_size: int, rank: int = 0) -> SimpleNamespace:
    return SimpleNamespace(device_group=SimpleNamespace(size=lambda: world_size), rank_in_group=rank)


def _build_moe(*, vllm_config=None, ep_world_size: int = 1, ep_rank: int = 0, enable_fused_mc2: int = 0):
    """Build the MoE with the factory, the vLLM linears and the groups stubbed.

    Returns the layer and the keyword arguments the factory was called with.
    """
    if vllm_config is None:
        vllm_config = _moe_vllm_config(enable_expert_parallel=False)
    factory = patch.object(muse_spark, "FusedMoEFactory", side_effect=lambda **kwargs: _FakeRunner())
    with (
        factory as fused_moe,
        patch.object(muse_spark, "GateLinear", _StubLinear),
        patch.object(muse_spark, "ReplicatedLinear", _StubLinear),
        patch.object(parallel_state, "_EP", _ep_group(ep_world_size, ep_rank)),
        patch.object(
            muse_spark,
            "get_ascend_config",
            return_value=SimpleNamespace(enable_fused_mc2=enable_fused_mc2),
        ),
    ):
        moe = muse_spark.AscendMuseSpark1xMoE(vllm_config, layer_idx=1, prefix="model.layers.1.mlp")
    return moe, fused_moe.call_args.kwargs


def test_moe_builds_under_tensor_parallelism_without_ep():
    # vLLM builds the EP group over the TP ranks of every MoE model, so TP=2
    # yields a two-rank EP group even with expert parallelism disabled.
    moe, _ = _build_moe(ep_world_size=2)

    assert moe.ep_size == 1
    assert moe.n_local_physical_experts == moe.n_physical_experts == 4


def test_moe_shards_the_experts_across_the_expert_parallel_group():
    """The per-route norm is row-wise, so a rank normalizing its own rows is
    still normalizing per route. Only the expert bookkeeping changes."""
    moe, _ = _build_moe(
        vllm_config=_moe_vllm_config(enable_expert_parallel=True),
        ep_world_size=2,
        ep_rank=1,
    )

    assert moe.use_ep
    assert moe.ep_size == 2
    assert moe.n_physical_experts == 4
    assert moe.n_local_physical_experts == 2
    assert (moe.physical_expert_start, moe.physical_expert_end) == (2, 4)


def test_moe_rejects_an_expert_count_that_does_not_divide_across_the_ep_group():
    with pytest.raises(ValueError, match="divide"):
        _build_moe(vllm_config=_moe_vllm_config(enable_expert_parallel=True), ep_world_size=3)


def test_moe_rejects_the_fused_dispatch_mlp_combine_path():
    """That operator has no MLP-stage boundary, so it would drop the norm."""
    with pytest.raises(NotImplementedError, match="enable_fused_mc2"):
        _build_moe(vllm_config=_moe_vllm_config(enable_expert_parallel=True), ep_world_size=2, enable_fused_mc2=1)


def test_moe_rejects_eplb():
    """Redundant experts would break the fixed n_redundant_experts = 0 bookkeeping."""
    with pytest.raises(NotImplementedError, match="EPLB"):
        _build_moe(vllm_config=_moe_vllm_config(enable_expert_parallel=True, enable_eplb=True), ep_world_size=2)


@pytest.mark.parametrize("parallel_sizes", [{"data_parallel_size": 2}, {"prefill_context_parallel_size": 2}])
def test_moe_rejects_data_or_context_parallelism_without_ep(parallel_sizes):
    """Without EP the experts are sliced over DP and PCP too, but the rows are
    summed before the norm over TP only."""
    with pytest.raises(NotImplementedError, match="data or prefill context parallelism"):
        _build_moe(
            vllm_config=_moe_vllm_config(enable_expert_parallel=False, **parallel_sizes),
            ep_world_size=2,
        )


def test_moe_builds_under_data_parallelism_with_ep():
    """Under EP every rank holds whole experts, so its rows are complete."""
    moe, _ = _build_moe(
        vllm_config=_moe_vllm_config(enable_expert_parallel=True, data_parallel_size=2),
        ep_world_size=2,
    )

    assert moe.ep_size == 2


def test_moe_builds_the_experts_on_the_ascend_fused_path():
    """The expert container and the runner's final reduction are all that differ."""
    moe, kwargs = _build_moe()

    assert kwargs["routed_experts_cls"] is muse_spark.AscendMuseSparkRoutedExperts
    # A subclass of the Ascend runner keeps its comm methods and combine;
    # upstream's runner would bypass them.
    assert kwargs["runner_cls"] is muse_spark.AscendMuseSparkMoERunner
    assert issubclass(kwargs["runner_cls"], AscendMoERunner)
    assert "moe_config_cls" not in kwargs
    assert kwargs["router"] is moe.moe_router


def _runner(tp_size: int) -> muse_spark.AscendMuseSparkMoERunner:
    runner = muse_spark.AscendMuseSparkMoERunner.__new__(muse_spark.AscendMuseSparkMoERunner)
    runner.moe_config = SimpleNamespace(moe_parallel_config=SimpleNamespace(tp_size=tp_size))
    return runner


def test_runner_skips_the_final_all_reduce_after_the_pre_norm_one():
    """The rows were summed before the norm; reducing again would multiply by TP."""
    runner = _runner(tp_size=2)
    states = torch.randn(NUM_TOKENS, 4)

    with patch.object(AscendMoERunner, "_maybe_reduce_final_output", side_effect=AssertionError("final all-reduce")):
        assert runner._maybe_reduce_final_output(states, trunc_size=None) is states
        torch.testing.assert_close(runner._maybe_reduce_final_output(states, trunc_size=2), states[..., :2])
    assert runner._fused_output_is_reduced


def test_runner_keeps_the_ascend_final_reduction_without_tp_partials():
    """Under EP, or on one card, the Ascend runner reduces as it always does."""
    runner = _runner(tp_size=1)
    states = torch.randn(NUM_TOKENS, 4)
    reduced = states + 1

    with patch.object(AscendMoERunner, "_maybe_reduce_final_output", return_value=reduced) as final_reduce:
        assert runner._maybe_reduce_final_output(states, trunc_size=None) is reduced
    final_reduce.assert_called_once_with(states, None, None)


def test_moe_sizes_the_experts_to_the_pooled_width():
    """The experts live after the pooler, so they take the latent width."""
    config = _moe_vllm_config(enable_expert_parallel=False).model_config.hf_text_config
    _, kwargs = _build_moe()

    assert kwargs["hidden_size"] == config.moe_hidden_size
    assert kwargs["hidden_size"] != config.hidden_size
    assert kwargs["intermediate_size"] == config.moe_intermediate_size
    assert kwargs["num_experts"] == config.num_local_experts


def test_moe_hands_its_post_expert_norm_to_the_quant_method():
    """Without this the MLP stage raises rather than skipping the norm."""
    moe, _ = _build_moe()

    attached = moe.experts.routed_experts.quant_method.post_expert_norm_fn
    # A bound method, not the module: holding the module would register the
    # norm's weight a second time under the quant method.
    assert attached.__self__ is moe.post_expert_norm


def _bare_routed_experts(num_experts: int, hidden_size: int, intermediate_size: int):
    """The routed experts without their constructor, which needs a live vLLM config.

    Each expert parameter records what reaches its weight loader. The real
    loader is upstream's: it keeps this rank's experts and TP slice.
    """
    experts = object.__new__(muse_spark.AscendMuseSparkRoutedExperts)
    nn.Module.__init__(experts)
    experts.layer_name = "model.layers.1.feed_forward.experts"
    calls: list[tuple[str, str, int, torch.Tensor]] = []

    def record(param, loaded_weight, weight_name, shard_id, expert_id):
        calls.append((weight_name, shard_id, expert_id, loaded_weight))

    shapes = {
        "w13_weight": (num_experts, 2 * intermediate_size, hidden_size),
        "w2_weight": (num_experts, hidden_size, intermediate_size),
    }
    for name, shape in shapes.items():
        param = nn.Parameter(torch.empty(shape), requires_grad=False)
        param.weight_loader = record
        setattr(experts, name, param)
    return experts, calls


def test_routed_experts_load_the_consolidated_checkpoint_tensors():
    """The checkpoint stacks every expert into one tensor per projection.

    The generic ``RoutedExperts.load_weights`` matches neither ``w13_weight``
    nor ``w2_weight`` and skips them without an error, which the loader's
    unloaded-weight check does not catch for MoE parameters.
    """
    num_experts, hidden_size, intermediate_size = 3, 4, 2
    experts, calls = _bare_routed_experts(num_experts, hidden_size, intermediate_size)
    w13 = torch.randn(num_experts, 2 * intermediate_size, hidden_size)
    w2 = torch.randn(num_experts, hidden_size, intermediate_size)

    loaded = list(experts.load_weights([("w13_weight", w13), ("w2_weight", w2)]))

    assert isinstance(experts, muse_spark.AscendRoutedExperts)
    assert loaded == ["w13_weight", "w2_weight"]
    # Global expert ids: the expert weight loader maps them to this rank.
    expected = [
        *(("w13_weight", "w1", expert, w13[expert, :intermediate_size]) for expert in range(num_experts)),
        *(("w13_weight", "w3", expert, w13[expert, intermediate_size:]) for expert in range(num_experts)),
        *(("w2_weight", "w2", expert, w2[expert]) for expert in range(num_experts)),
    ]
    assert [call[:3] for call in calls] == [call[:3] for call in expected]
    for (*_, actual), (*_, wanted) in zip(calls, expected):
        assert torch.equal(actual, wanted)


def test_routed_experts_reject_expert_tensors_they_cannot_place():
    """An unknown expert tensor fails the load instead of being skipped."""
    experts, calls = _bare_routed_experts(num_experts=2, hidden_size=4, intermediate_size=2)

    with pytest.raises(AttributeError, match="unexpected expert weight"):
        list(experts.load_weights([("0.gate_proj.weight", torch.randn(2, 4))]))
    assert calls == []
