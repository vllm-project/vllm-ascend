# SPDX-License-Identifier: Apache-2.0
"""Generic dense sequence parallelism (FlashComm SP switch, all dense families).

Covers the three layers of the generic path:

- the ``enable_dense_sp`` predicate (structural gates, no model whitelist);
- the projection-name selection in ``linear_op``;
- numerical equivalence of a full decoder block built from the real
  ``SequenceColumnParallelOp`` / ``SequenceRowParallelOp`` against the
  all-reduce (non-SP) reference, parametrized over decoder skeletons
  (merged-qkv Llama-style, split q/k/v, GDN in-projections), TP sizes and
  unaligned token counts.

The embedding entry (reduce-scatter) is exercised through a placement test;
the runner-side TP alignment (V2 ``_prepare_inputs_impl`` / V1
``_pad_for_sequence_parallelism``) and the sample-time exit need NPU e2e runs.
"""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import vllm_ascend.utils as utils_module
from vllm_ascend.ops import linear_op
from vllm_ascend.ops.linear_op import (
    _DENSE_SP_COLUMN_PROJECTIONS,
    _DENSE_SP_ROW_PROJECTIONS,
    SequenceColumnParallelOp,
    SequenceRowParallelOp,
    _get_column_parallel_op,
    _get_row_parallel_op,
    _is_dense_sp_decoder_projection,
)
from vllm_ascend.ops.vocab_parallel_embedding import (
    AscendVocabParallelEmbedding,
    VocabParallelMode,
)
from vllm_ascend.utils import enable_dense_sp, enable_sp


def _make_vllm_config(**overrides):
    # Dense default: hf_config is the same object as hf_text_config (no
    # separate vision sub-config), which is vLLM's structural criterion for
    # a pure-text model. Dense Qwen3.5-family checkpoints also resolve this
    # way even though their architecture name is multimodal-registered.
    text_config = SimpleNamespace(model_type="llama", to_dict=lambda: {"hidden_size": 8})
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            use_sequence_parallel_moe=False,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            hf_config=text_config,
            hf_text_config=text_config,
            use_mla=False,
        ),
        speculative_config=None,
        lora_config=None,
        additional_config={"enable_flashcomm1": True},
    )
    for key, value in overrides.items():
        if "." in key:
            parent, leaf = key.split(".", 1)
            setattr(getattr(config, parent), leaf, value)
        else:
            setattr(config, key, value)
    return config


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", raising=False)
    # is_moe_model / is_vl_model cache into module globals; reset them so
    # parametrized configs with different hf configs cannot pollute each
    # other.
    monkeypatch.setattr(utils_module, "_IS_MOE_MODEL", None, raising=False)
    monkeypatch.setattr(utils_module, "_IS_VL_MODEL", None, raising=False)
    # Default to the V2 model runner (the production default); the V1 spec
    # case below opts out.
    import vllm_ascend.mrv2_utils as mrv2_module

    monkeypatch.setattr(mrv2_module, "use_v2_model_runner", lambda vllm_config: True)


class TestEnableDenseSpPredicate:
    def test_enabled_by_flashcomm_switch(self):
        assert enable_dense_sp(_make_vllm_config())

    def test_enabled_by_env_var(self, monkeypatch):
        monkeypatch.setenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", "1")
        config = _make_vllm_config()
        config.additional_config = {}
        assert enable_dense_sp(config)

    def test_disabled_without_switch(self):
        assert not enable_dense_sp(_make_vllm_config(additional_config={}))

    @pytest.mark.parametrize("switch", [False, "false", "0"])
    def test_falsey_switch_stays_disabled(self, switch):
        assert not enable_dense_sp(_make_vllm_config(additional_config={"enable_flashcomm1": switch}))

    @pytest.mark.parametrize(
        "override",
        [
            {"parallel_config.tensor_parallel_size": 1},
            {"parallel_config.pipeline_parallel_size": 2},
            {"parallel_config.use_sequence_parallel_moe": True},
            {"parallel_config.prefill_context_parallel_size": 2},
            {"parallel_config.decode_context_parallel_size": 2},
            {"parallel_config.enable_dbo": True},
            {"speculative_config": SimpleNamespace(method="eagle3")},
            {"lora_config": SimpleNamespace(max_lora_rank=16)},
            # Structural VL gate: a separate vision sub-config
            # (hf_config is not hf_text_config), not the registry-based
            # is_multimodal_model (dense Qwen3.5 checkpoints carry the
            # multimodal-registered architecture name).
            {
                "model_config.hf_config": SimpleNamespace(
                    vision_config=SimpleNamespace(),
                    to_dict=lambda: {"vision_config": {}},
                )
            },
            {"model_config.use_mla": True},
            {"model_config": None},
        ],
    )
    def test_structural_gates_disable_dense_sp(self, override):
        assert not enable_dense_sp(_make_vllm_config(**override))

    def test_mtp_spec_enabled_under_v2(self):
        # The patched Qwen3.5 MTP forward gathers the shared embedding's
        # token shard back to full rows, so MTP drafts coexist with dense SP.
        config = _make_vllm_config(speculative_config=SimpleNamespace(method="mtp"))
        assert enable_dense_sp(config)
        assert enable_sp(config)

    def test_mtp_spec_disabled_under_v1(self, monkeypatch):
        # The V1 spec flow carries its own SP handling (e.g. the draft RoPE
        # positions all-gather) that assumes the MoE-SP layout.
        import vllm_ascend.mrv2_utils as mrv2_module

        monkeypatch.setattr(mrv2_module, "use_v2_model_runner", lambda vllm_config: False)
        config = _make_vllm_config(speculative_config=SimpleNamespace(method="mtp"))
        assert not enable_dense_sp(config)

    def test_vl_gate_is_structural_not_registry_based(self):
        # Dense Qwen3.5-family checkpoints: multimodal-registered
        # architecture name, but hf_config is hf_text_config (pure text).
        text_config = SimpleNamespace(model_type="qwen3_5_text", to_dict=lambda: {"hidden_size": 8})
        config = _make_vllm_config(
            model_config=SimpleNamespace(
                hf_config=text_config,
                hf_text_config=text_config,
                is_multimodal_model=True,
                use_mla=False,
            ),
        )
        assert enable_dense_sp(config)

    def test_mm_checkpoint_enabled_with_wired_model_class(self, monkeypatch):
        # mm-capable Qwen3.5 checkpoints (separate vision sub-config) run
        # dense SP when the registered model class declares the wiring: the
        # multimodal merge happens full-token side in the runner's
        # inputs_embeds preparation, and the wrapper's patch restores the
        # text shard and shards the merged embeds back.
        monkeypatch.setattr(utils_module, "_mm_arch_supports_dense_sp", lambda vllm_config: True)
        vision_config = SimpleNamespace(to_dict=lambda: {"vision_config": {}})
        text_config = SimpleNamespace(model_type="qwen3_5_text", to_dict=lambda: {"hidden_size": 8})
        config = _make_vllm_config(
            model_config=SimpleNamespace(
                hf_config=vision_config,
                hf_text_config=text_config,
                use_mla=False,
            ),
        )
        assert enable_dense_sp(config)

    def test_mm_checkpoint_disabled_without_wired_model_class(self):
        # The default test config doubles (no architectures) resolve to no
        # wired model class: the VL override in the parametrized gates above
        # covers this; assert the reason names the wiring, not the gate.
        vision_config = SimpleNamespace(to_dict=lambda: {"vision_config": {}})
        text_config = SimpleNamespace(model_type="qwen3_5_text", to_dict=lambda: {"hidden_size": 8})
        config = _make_vllm_config(
            model_config=SimpleNamespace(
                hf_config=vision_config,
                hf_text_config=text_config,
                use_mla=False,
            ),
        )
        assert not enable_dense_sp(config)
        from vllm_ascend.utils import dense_sp_unavailable_reason

        assert "dense-SP-wired model class" in dense_sp_unavailable_reason(config)

    def test_moe_model_excluded_even_with_switch(self):
        moe_config = SimpleNamespace(model_type="qwen3_moe", to_dict=lambda: {"num_experts": 8})
        config = _make_vllm_config(
            model_config=SimpleNamespace(
                hf_config=moe_config,
                hf_text_config=moe_config,
                use_mla=False,
            ),
        )
        assert not enable_dense_sp(config)

    def test_no_model_family_whitelist(self):
        # Any dense family qualifies: the point of the generic path.
        for model_type in ("llama", "qwen2", "qwen3", "glm4", "qwen3_5_text", "internlm2"):
            text_config = SimpleNamespace(model_type=model_type, to_dict=lambda: {})
            config = _make_vllm_config(
                model_config=SimpleNamespace(
                    hf_config=text_config,
                    hf_text_config=text_config,
                    use_mla=False,
                ),
            )
            assert enable_dense_sp(config), model_type

    def test_runner_choice_is_not_a_gate(self):
        # Both runners are supported: V2 rounds the executed token count in
        # _prepare_inputs_impl, V1 pads in _pad_for_sequence_parallelism.
        assert enable_dense_sp(_make_vllm_config())

    def test_enable_sp_covers_dense_models(self):
        assert enable_sp(_make_vllm_config())

    def test_enable_sp_stays_off_without_dense_sp(self):
        assert not enable_sp(_make_vllm_config(additional_config={}))

    def test_enable_sp_covers_sp_moe_models(self):
        config = _make_vllm_config(additional_config={})
        config.parallel_config.use_sequence_parallel_moe = True
        assert enable_sp(config)
        assert not enable_dense_sp(config)

    def test_no_config_falls_back_to_false(self):
        assert not enable_dense_sp(None)


class TestProjectionSelection:
    @pytest.fixture(autouse=True)
    def _selection_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(linear_op, "mlp_tp_enable", lambda: False)
        monkeypatch.setattr(linear_op, "oproj_tp_enable", lambda: False)
        monkeypatch.setattr(linear_op, "enable_dense_sp", lambda: True)

    @pytest.mark.parametrize("prefix", sorted(_DENSE_SP_COLUMN_PROJECTIONS))
    def test_decoder_column_projections_are_wrapped(self, prefix):
        op = _get_column_parallel_op(f"model.layers.0.self_attn.{prefix}", MagicMock())
        assert isinstance(op, SequenceColumnParallelOp)

    @pytest.mark.parametrize("prefix", sorted(_DENSE_SP_ROW_PROJECTIONS))
    def test_decoder_row_projections_are_wrapped(self, prefix):
        op = _get_row_parallel_op(f"model.layers.0.mlp.{prefix}", MagicMock())
        assert isinstance(op, SequenceRowParallelOp)

    @pytest.mark.parametrize(
        "prefix",
        [
            # Substring trap: only the final prefix component is matched.
            "model.layers.0.mlp.fc1",
            "model.layers.0.mlp.fc2",
            # GDN's conv1d runs inside the already-gathered mixer core.
            "model.layers.0.linear_attn.conv1d",
            # Outside the decoder stack (the .layers. requirement).
            "model.embed_tokens",
            "lm_head",
            "vision.mlp.gate_up_proj",
        ],
    )
    def test_non_residual_stream_projections_stay_stock(self, prefix):
        assert _get_column_parallel_op(prefix, MagicMock()) is None
        assert _get_row_parallel_op(prefix, MagicMock()) is None

    def test_gate_up_outside_layers_is_rejected_by_name(self):
        assert not _is_dense_sp_decoder_projection("vision.mlp.gate_up_proj", _DENSE_SP_COLUMN_PROJECTIONS)

    def test_selection_off_without_dense_sp(self, monkeypatch):
        monkeypatch.setattr(linear_op, "enable_dense_sp", lambda: False)
        assert _get_column_parallel_op("model.layers.0.self_attn.qkv_proj", MagicMock()) is None
        assert _get_row_parallel_op("model.layers.0.self_attn.o_proj", MagicMock()) is None

    def test_selection_off_for_draft_head_builds(self, monkeypatch):
        # load_eagle_model builds MTP draft heads under the "eagle_head"
        # model tag; drafts run the stock full-token layout.
        monkeypatch.setattr(linear_op, "_building_draft_model", lambda: True)
        assert _get_column_parallel_op("model.layers.0.self_attn.qkv_proj", MagicMock()) is None
        assert _get_row_parallel_op("model.layers.0.self_attn.o_proj", MagicMock()) is None


class _FakeQuantMethod:
    """Plain GEMM stand-in for ``quant_method.apply``."""

    def __init__(self, weight: torch.Tensor):
        self.weight = weight

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        out = x @ self.weight.t()
        return out if bias is None else out + bias


def _make_column_op(weight: torch.Tensor, prefix: str):
    layer = SimpleNamespace(
        prefix=prefix,
        bias=None,
        skip_bias_add=False,
        return_bias=True,
        quant_method=_FakeQuantMethod(weight),
        gather_output=False,
    )
    op = SequenceColumnParallelOp(layer)
    op.update_attrs()
    return op


def _make_row_op(weight: torch.Tensor, prefix: str, bias: torch.Tensor | None = None):
    layer = SimpleNamespace(
        prefix=prefix,
        bias=bias,
        skip_bias_add=False,
        return_bias=True,
        quant_method=_FakeQuantMethod(weight),
        input_is_parallel=True,
        reduce_results=True,
        input_size_per_partition=weight.shape[1],
    )
    op = SequenceRowParallelOp(layer)
    op.update_attrs()
    return op


class _SimTP:
    """Single-process simulation of the TP collectives the dense-SP ops use.

    ``apply_impl`` is deterministic, so the driver stages every collective
    round: an all-gather round consumes the list of per-rank shards (each
    rank's call is checked against its staged shard, then returns the
    concatenation); a reduce-scatter round consumes the per-rank partials
    expected from the reference layout (each rank's call is checked against
    its staged partial, then returns that rank's chunk of the TP sum).
    """

    def __init__(self, tp_size: int):
        self.tp_size = tp_size
        self.gather_stage: list[torch.Tensor] | None = None
        self.rs_stage: list[torch.Tensor] | None = None
        self._gather_calls = 0
        self._rs_calls = 0

    def fake_all_gather(self, x: torch.Tensor) -> torch.Tensor:
        assert self.gather_stage is not None
        rank = self._gather_calls
        assert rank < self.tp_size
        assert torch.equal(x, self.gather_stage[rank]), (
            f"all-gather round got rank {rank}'s input out of order or from the wrong stage"
        )
        self._gather_calls += 1
        if self._gather_calls == self.tp_size:
            self._gather_calls = 0
        return torch.cat(self.gather_stage, dim=0)

    def fake_reduce_scatter(self, x: torch.Tensor) -> torch.Tensor:
        assert self.rs_stage is not None
        rank = self._rs_calls
        assert rank < self.tp_size
        assert torch.equal(x, self.rs_stage[rank]), "reduce-scatter partial does not match the reference-layout partial"
        self._rs_calls += 1
        if self._rs_calls == self.tp_size:
            self._rs_calls = 0
        chunk_sum = None
        for partial in self.rs_stage:
            pad = (-partial.shape[0]) % self.tp_size
            padded = partial if pad == 0 else torch.nn.functional.pad(partial, (0, 0, 0, pad))
            chunk = padded.chunk(self.tp_size, dim=0)[rank]
            chunk_sum = chunk if chunk_sum is None else chunk_sum + chunk
        return chunk_sum


def _silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _pad_tokens(x: torch.Tensor, tp_size: int) -> torch.Tensor:
    pad = (-x.shape[0]) % tp_size
    return x if pad == 0 else torch.nn.functional.pad(x, (0, 0, 0, pad))


def _make_weights(tp_size: int, hidden: int, head_dim: int, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    return {
        "qkv": [torch.randn(3 * head_dim, hidden, generator=gen, dtype=torch.float64) for _ in range(tp_size)],
        "o": [torch.randn(hidden, head_dim, generator=gen, dtype=torch.float64) for _ in range(tp_size)],
        "gate_up": [torch.randn(2 * (hidden + 2), hidden, generator=gen, dtype=torch.float64) for _ in range(tp_size)],
        "down": [torch.randn(hidden, hidden + 2, generator=gen, dtype=torch.float64) for _ in range(tp_size)],
        "q": [torch.randn(head_dim, hidden, generator=gen, dtype=torch.float64) for _ in range(tp_size)],
        "k": [torch.randn(head_dim, hidden, generator=gen, dtype=torch.float64) for _ in range(tp_size)],
        "v": [torch.randn(head_dim, hidden, generator=gen, dtype=torch.float64) for _ in range(tp_size)],
        "qkvz": [torch.randn(4 * head_dim, hidden, generator=gen, dtype=torch.float64) for _ in range(tp_size)],
        "ba": [torch.randn(2 * head_dim, hidden, generator=gen, dtype=torch.float64) for _ in range(tp_size)],
        "out": [torch.randn(hidden, head_dim, generator=gen, dtype=torch.float64) for _ in range(tp_size)],
    }


def _apply_per_rank(ops, inputs, fake_group):
    """Call each rank's op with its input, exposing its TP rank to the op."""
    outputs = []
    for rank, (op, x) in enumerate(zip(ops, inputs)):
        fake_group.rank_in_group = rank
        outputs.append(op.apply(x)[0])
    fake_group.rank_in_group = 0
    return outputs


def _run_merged_qkv_block(x, weights, tp_size, sim, fake_group, o_bias=None):
    """Llama/Qwen2-style block: merged qkv + o_proj + merged gate_up + down."""
    ranks = range(tp_size)
    qkv_ops = [_make_column_op(weights["qkv"][r], "model.layers.0.self_attn.qkv_proj") for r in ranks]
    o_ops = [_make_row_op(weights["o"][r], "model.layers.0.self_attn.o_proj", bias=o_bias) for r in ranks]
    gu_ops = [_make_column_op(weights["gate_up"][r], "model.layers.0.mlp.gate_up_proj") for r in ranks]
    down_ops = [_make_row_op(weights["down"][r], "model.layers.0.mlp.down_proj") for r in ranks]

    shards = list(_pad_tokens(x, tp_size).chunk(tp_size, dim=0))

    sim.gather_stage = shards
    qkv_outs = _apply_per_rank(qkv_ops, shards, fake_group)

    # Token-wise attention stand-in on each rank's feature shard.
    cores = [sum(qkv_outs[r].chunk(3, dim=-1)) for r in ranks]

    # Reference-layout partials: what a non-SP rank would feed its all-reduce.
    o_partials = [cores[r] @ weights["o"][r].t() for r in ranks]
    if o_bias is not None:
        o_partials[0] = o_partials[0] + o_bias
    sim.rs_stage = o_partials
    o_outs = _apply_per_rank(o_ops, cores, fake_group)

    sim.gather_stage = o_outs
    gu_outs = _apply_per_rank(gu_ops, o_outs, fake_group)
    hiddens = [_silu(g) * u for g, u in (gu_outs[r].chunk(2, dim=-1) for r in ranks)]

    down_partials = [hiddens[r] @ weights["down"][r].t() for r in ranks]
    sim.rs_stage = down_partials
    down_outs = _apply_per_rank(down_ops, hiddens, fake_group)

    # Residual add runs on the shard.
    return [shards[r] + down_outs[r] for r in ranks]


def _run_split_qkv_block(x, weights, tp_size, sim, fake_group):
    """Skeleton with separate q/k/v projections: each gathers on its own."""
    ranks = range(tp_size)
    q_ops = [_make_column_op(weights["q"][r], "model.layers.0.self_attn.q_proj") for r in ranks]
    k_ops = [_make_column_op(weights["k"][r], "model.layers.0.self_attn.k_proj") for r in ranks]
    v_ops = [_make_column_op(weights["v"][r], "model.layers.0.self_attn.v_proj") for r in ranks]
    o_ops = [_make_row_op(weights["o"][r], "model.layers.0.self_attn.o_proj") for r in ranks]

    shards = list(_pad_tokens(x, tp_size).chunk(tp_size, dim=0))

    sim.gather_stage = shards
    q_outs = _apply_per_rank(q_ops, shards, fake_group)
    sim.gather_stage = shards
    k_outs = _apply_per_rank(k_ops, shards, fake_group)
    sim.gather_stage = shards
    v_outs = _apply_per_rank(v_ops, shards, fake_group)

    cores = [q_outs[r] + k_outs[r] + v_outs[r] for r in ranks]

    o_partials = [cores[r] @ weights["o"][r].t() for r in ranks]
    sim.rs_stage = o_partials
    o_outs = _apply_per_rank(o_ops, cores, fake_group)
    return [shards[r] + o_outs[r] for r in ranks]


def _run_gdn_block(x, weights, tp_size, sim, fake_group):
    """GDN skeleton: in_proj_qkvz + in_proj_ba columns, out_proj row."""
    ranks = range(tp_size)
    qkvz_ops = [_make_column_op(weights["qkvz"][r], "model.layers.0.linear_attn.in_proj_qkvz") for r in ranks]
    ba_ops = [_make_column_op(weights["ba"][r], "model.layers.0.linear_attn.in_proj_ba") for r in ranks]
    out_ops = [_make_row_op(weights["out"][r], "model.layers.0.linear_attn.out_proj") for r in ranks]

    shards = list(_pad_tokens(x, tp_size).chunk(tp_size, dim=0))

    sim.gather_stage = shards
    qkvz_outs = _apply_per_rank(qkvz_ops, shards, fake_group)
    sim.gather_stage = shards
    ba_outs = _apply_per_rank(ba_ops, shards, fake_group)

    # Token-wise gated stand-in: q, k, v, z from qkvz; b, a from ba.
    cores = []
    for r in ranks:
        q, k, v, z = qkvz_outs[r].chunk(4, dim=-1)
        b, a = ba_outs[r].chunk(2, dim=-1)
        cores.append((q + k + v) * torch.sigmoid(z) + b * a)

    out_partials = [cores[r] @ weights["out"][r].t() for r in ranks]
    sim.rs_stage = out_partials
    out_outs = _apply_per_rank(out_ops, cores, fake_group)
    return [shards[r] + out_outs[r] for r in ranks]


def _reference_merged_qkv(x_pad, weights, tp_size, o_bias=None):
    """All-reduce reference: full tokens, per-rank TP partials summed."""
    o_full = None
    for r in range(tp_size):
        q, k, v = (x_pad @ weights["qkv"][r].t()).chunk(3, dim=-1)
        partial = (q + k + v) @ weights["o"][r].t()
        o_full = partial if o_full is None else o_full + partial
    if o_bias is not None:
        o_full = o_full + o_bias

    mlp_full = None
    for r in range(tp_size):
        weight = weights["gate_up"][r]
        half = weight.shape[0] // 2
        g = o_full @ weight[:half].t()
        u = o_full @ weight[half:].t()
        partial = (_silu(g) * u) @ weights["down"][r].t()
        mlp_full = partial if mlp_full is None else mlp_full + partial
    return x_pad + mlp_full


def _reference_split_qkv(x_pad, weights, tp_size):
    o_full = None
    for r in range(tp_size):
        core = x_pad @ weights["q"][r].t() + x_pad @ weights["k"][r].t() + x_pad @ weights["v"][r].t()
        partial = core @ weights["o"][r].t()
        o_full = partial if o_full is None else o_full + partial
    return x_pad + o_full


def _reference_gdn(x_pad, weights, tp_size):
    out_full = None
    for r in range(tp_size):
        q, k, v, z = (x_pad @ weights["qkvz"][r].t()).chunk(4, dim=-1)
        b, a = (x_pad @ weights["ba"][r].t()).chunk(2, dim=-1)
        core = (q + k + v) * torch.sigmoid(z) + b * a
        partial = core @ weights["out"][r].t()
        out_full = partial if out_full is None else out_full + partial
    return x_pad + out_full


def _assert_shards_match_reference(sp_outs, reference, num_tokens, tp_size):
    chunk = math.ceil(num_tokens / tp_size)
    for rank, shard in enumerate(sp_outs):
        ref_rows = reference[rank * chunk : (rank + 1) * chunk]
        real_rows = min(shard.shape[0], ref_rows.shape[0])
        torch.testing.assert_close(shard[:real_rows], ref_rows[:real_rows], rtol=1e-9, atol=1e-9)


class TestDenseSpBlockEquivalence:
    @pytest.fixture(autouse=True)
    def _tp_env(self, monkeypatch: pytest.MonkeyPatch):
        self.fake_group = MagicMock()
        self.fake_group.world_size = 2
        self.fake_group.rank_in_group = 0
        monkeypatch.setattr(linear_op, "get_tp_group", lambda: self.fake_group)
        self.sim = _SimTP(tp_size=2)
        monkeypatch.setattr(linear_op, "dense_sp_all_gather", self.sim.fake_all_gather)
        monkeypatch.setattr(linear_op, "dense_sp_reduce_scatter", self.sim.fake_reduce_scatter)

    def _configure_tp(self, tp_size):
        self.fake_group.world_size = tp_size
        self.sim.tp_size = tp_size

    @pytest.mark.parametrize("tp_size", [2, 4])
    @pytest.mark.parametrize("num_tokens", [8, 5])
    def test_merged_qkv_block_matches_all_reduce_reference(self, tp_size, num_tokens):
        self._configure_tp(tp_size)
        gen = torch.Generator().manual_seed(42)
        hidden, head_dim = 6, 4
        x = torch.randn(num_tokens, hidden, generator=gen, dtype=torch.float64)
        weights = _make_weights(tp_size, hidden, head_dim, seed=7)
        o_bias = torch.randn(hidden, generator=gen, dtype=torch.float64)

        sp_outs = _run_merged_qkv_block(x, weights, tp_size, self.sim, self.fake_group, o_bias=o_bias)
        reference = _reference_merged_qkv(_pad_tokens(x, tp_size), weights, tp_size, o_bias=o_bias)
        _assert_shards_match_reference(sp_outs, reference, num_tokens, tp_size)

    @pytest.mark.parametrize("tp_size", [2, 4])
    def test_split_qkv_block_matches_all_reduce_reference(self, tp_size):
        self._configure_tp(tp_size)
        gen = torch.Generator().manual_seed(1)
        hidden, head_dim = 6, 4
        num_tokens = 7  # 7 % tp != 0 for tp in {2, 4}
        x = torch.randn(num_tokens, hidden, generator=gen, dtype=torch.float64)
        weights = _make_weights(tp_size, hidden, head_dim, seed=3)

        sp_outs = _run_split_qkv_block(x, weights, tp_size, self.sim, self.fake_group)
        reference = _reference_split_qkv(_pad_tokens(x, tp_size), weights, tp_size)
        _assert_shards_match_reference(sp_outs, reference, num_tokens, tp_size)

    @pytest.mark.parametrize("tp_size", [2, 4])
    def test_gdn_block_matches_all_reduce_reference(self, tp_size):
        self._configure_tp(tp_size)
        gen = torch.Generator().manual_seed(2)
        hidden, head_dim = 6, 4
        num_tokens = 8
        x = torch.randn(num_tokens, hidden, generator=gen, dtype=torch.float64)
        weights = _make_weights(tp_size, hidden, head_dim, seed=5)

        sp_outs = _run_gdn_block(x, weights, tp_size, self.sim, self.fake_group)
        reference = _reference_gdn(_pad_tokens(x, tp_size), weights, tp_size)
        _assert_shards_match_reference(sp_outs, reference, num_tokens, tp_size)

    def test_row_op_asserts_on_short_input(self, monkeypatch):
        """A missed upstream gather must fail loudly, not compute garbage."""
        weight = torch.randn(6, 4, dtype=torch.float64)
        op = _make_row_op(weight, "model.layers.0.self_attn.o_proj")
        monkeypatch.setattr(linear_op, "is_forward_context_available", lambda: True)
        monkeypatch.setattr(linear_op, "_EXTRA_CTX", SimpleNamespace(num_tokens=8))

        short = torch.randn(4, 4, dtype=torch.float64)  # 4 != 8: the token layout broke upstream
        with pytest.raises(AssertionError, match="Sequence-parallel row projection"):
            op.apply(short)

        good = torch.randn(8, 4, dtype=torch.float64)
        self.sim.rs_stage = [good, good]
        out, _ = op.apply(good)
        assert out.shape[0] == 4  # reduce-scatter semantics: 8 / tp(2) rows


class TestEmbeddingEntryShard:
    def _make_embedding(self, embedding_dim=4, dense_sp_entry=True):
        gen = torch.Generator().manual_seed(0)
        table = torch.randn(8, embedding_dim, generator=gen, dtype=torch.float64)
        emb = object.__new__(AscendVocabParallelEmbedding)
        emb.tp_size = 2
        emb.embedding_dim = embedding_dim
        # Construction-time capture: the V2 runner does not pin the current
        # vllm config during forwards, so the gate is a plain instance flag.
        emb.dense_sp_entry = dense_sp_entry
        emb.shard_indices = SimpleNamespace(
            org_vocab_start_index=0,
            org_vocab_end_index=4,
            num_org_vocab_padding=0,
            added_vocab_start_index=4,
            added_vocab_end_index=4,
        )
        emb.quant_method = SimpleNamespace(embedding=lambda layer, ids: torch.nn.functional.embedding(ids, table))
        emb.output_reduce_group = SimpleNamespace(world_size=2, unique_name="tp")
        emb.parallel_mode = VocabParallelMode.STANDARD
        return emb

    def test_dense_sp_reduce_scatters_the_embedding_output(self, monkeypatch):
        import vllm_ascend.ops.vocab_parallel_embedding as vocab_module

        emb = self._make_embedding()
        calls = []

        def fake_rs(x):
            calls.append(x)
            return x.chunk(2, dim=0)[0]

        monkeypatch.setattr(vocab_module, "sp_reduce_scatter", fake_rs)

        input_ids = torch.tensor([0, 1, 2, 3])
        out = emb._forward_standard(input_ids)

        assert len(calls) == 1
        # The reduce-scatter input is the full-token vocab-shard contribution.
        assert calls[0].shape == (4, 4)
        assert out.shape == (2, 4)

    def test_standard_layout_keeps_all_reduce_without_dense_sp(self, monkeypatch):
        import vllm_ascend.ops.vocab_parallel_embedding as vocab_module

        emb = self._make_embedding(dense_sp_entry=False)

        def fail_rs(x):
            raise AssertionError("sp_reduce_scatter must not run without dense SP")

        monkeypatch.setattr(vocab_module, "sp_reduce_scatter", fail_rs)
        # Keep the stock path off the real all_reduce custom op (no NPU here).
        monkeypatch.setattr("torch.ops.vllm.all_reduce", lambda x, group: x, raising=False)

        input_ids = torch.tensor([0, 1, 2, 3])
        out = emb._forward_standard(input_ids)
        # Without dense SP the embedding returns full-token rows.
        assert out.shape == (4, 4)


class _FakeMtpLayer:
    use_attn_reduce_scatter_for_moe = False

    def __init__(self):
        self.rows_seen: list[int] = []

    def __call__(self, positions, hidden_states, residual):
        self.rows_seen.append(hidden_states.shape[0])
        return hidden_states, residual


class TestMtpDraftFullTokenLayout:
    """The drafter's own embedding seam restores full rows generically: the
    speculator installs a gather on the core predictor's ``embed_input_ids``
    instance (flag-gated on the shared embedding's ``dense_sp_entry``), so
    the patched Qwen3.5 MTP forward itself stays stock — no family wiring —
    and future MTP dense families work without any patch."""

    def _make_predictor(self, layer):
        return SimpleNamespace(
            embed_tokens=SimpleNamespace(dense_sp_entry=True),
            embed_input_ids=lambda input_ids: torch.arange(3 * 2, dtype=torch.float64).reshape(3, 2),
            pre_fc_norm_embedding=lambda x: x,
            pre_fc_norm_hidden=lambda x: x,
            fc=lambda x: x,
            num_mtp_layers=1,
            layers=[layer],
            norm=lambda h, r: (h, r),
        )

    def _run(self, monkeypatch):
        from vllm_ascend.patch.worker import patch_qwen3_5

        monkeypatch.setattr(patch_qwen3_5, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))
        layer = _FakeMtpLayer()
        predictor = self._make_predictor(layer)
        hidden_states = torch.arange(5 * 2, dtype=torch.float64).reshape(5, 2) + 10
        out = patch_qwen3_5.qwen3_5_mtp_forward(
            predictor,
            input_ids=torch.zeros(5, dtype=torch.int64),
            positions=torch.arange(5, dtype=torch.int64),
            hidden_states=hidden_states,
            inputs_embeds=None,
            spec_step_idx=0,
        )
        return out, layer, predictor

    def _install(self, monkeypatch, dense_sp_entry):
        from vllm_ascend.worker.v2.spec_decode import target_dependent_ar as spec_module

        def fake_gather(x):
            # TP=2 simulation: duplicate rows; the rows beyond the true count
            # are the reduce-scatter zero padding.
            return torch.cat([x, x], dim=0)

        monkeypatch.setattr(spec_module, "sp_all_gather", fake_gather)
        draft = SimpleNamespace(
            embed_input_ids=lambda input_ids: torch.arange(3 * 2, dtype=torch.float64).reshape(3, 2),
            embed_tokens=SimpleNamespace(dense_sp_entry=dense_sp_entry),
        )
        wrapper = SimpleNamespace(model=draft)
        spec_module._install_full_token_draft_embedding(wrapper)
        return draft

    def test_installed_wrap_gathers_and_trims_the_shard(self, monkeypatch):
        draft = self._install(monkeypatch, dense_sp_entry=True)
        input_ids = torch.zeros(5, dtype=torch.int64)
        embeds = draft.embed_input_ids(input_ids)
        # The 3-row shard was gathered to 6 and trimmed to the true 5.
        assert embeds.shape[0] == 5

    def test_no_wrap_without_shared_sequence_parallel_embedding(self, monkeypatch):
        draft = self._install(monkeypatch, dense_sp_entry=False)
        input_ids = torch.zeros(5, dtype=torch.int64)
        # The draft kept its own full-token embedding: stock path, no gather.
        assert draft.embed_input_ids(input_ids).shape[0] == 3

    def test_no_wrap_without_core_model(self, monkeypatch):
        from vllm_ascend.worker.v2.spec_decode import target_dependent_ar as spec_module

        spec_module._install_full_token_draft_embedding(SimpleNamespace())  # no .model

    def test_forward_runs_stock_on_installed_full_rows(self, monkeypatch):
        # With the wrap installed on the core's embed_input_ids, the patched
        # forward consumes its output unchanged: no gather inside the forward.
        self._install(monkeypatch, dense_sp_entry=True)
        out, layer, predictor = self._run(monkeypatch)
        assert layer.rows_seen == [5]
        assert out.shape[0] == 5


class TestModelStateInputEmbedsShard:
    """The runner-precomputed inputs_embeds (multimodal merge done full-token
    side, only produced for mm-capable / prompt-embeds models) are sharded to
    the rank-local token chunk at the model-state seam, generically for every
    dense family; the self-embedding path is already sharded by the
    embedding's dense_sp_entry gate."""

    def _make_state(self, dense_sp):
        from vllm_ascend.worker.v2.model_states.default import AscendModelState

        state = AscendModelState.__new__(AscendModelState)
        state.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace())
        state._dense_sp = dense_sp
        return state

    def _run(self, monkeypatch, dense_sp, method, *args):
        import vllm_ascend.worker.v2.model_states.default as state_module

        def fake_shard(x):
            return x.chunk(2, dim=0)[0]

        monkeypatch.setattr(state_module, "sp_shard", fake_shard)
        monkeypatch.setattr(state_module, "enable_dense_sp", lambda vllm_config=None: dense_sp)
        state = self._make_state(dense_sp)
        embeds = torch.zeros(8, 2, dtype=torch.float64)
        if method == "prepare_inputs_embeds":
            return state.prepare_inputs_embeds(None, SimpleNamespace(num_tokens_after_padding=8), None)
        if method == "dummy_inputs_embeds":
            return state.dummy_inputs_embeds(8)
        if method == "super_prepare_inputs":
            # super().prepare_inputs_embeds path: patch the upstream base.
            monkeypatch.setattr(
                "vllm.v1.worker.gpu.model_states.default.DefaultModelState.prepare_inputs_embeds",
                lambda self, a, b, c: embeds,
                raising=True,
            )
            return state.prepare_inputs_embeds(None, SimpleNamespace(num_tokens_after_padding=8), None)
        raise ValueError(method)

    def test_prepare_inputs_embeds_shards_under_dense_sp(self, monkeypatch):
        out = self._run(monkeypatch, dense_sp=True, method="super_prepare_inputs")
        assert out.shape[0] == 4

    def test_prepare_inputs_embeds_stays_full_without_dense_sp(self, monkeypatch):
        out = self._run(monkeypatch, dense_sp=False, method="super_prepare_inputs")
        assert out.shape[0] == 8

    def test_dummy_inputs_embeds_shards_under_dense_sp(self, monkeypatch):
        embeds = torch.zeros(8, 2, dtype=torch.float64)
        monkeypatch.setattr(
            "vllm.v1.worker.gpu.model_states.default.DefaultModelState.dummy_inputs_embeds",
            lambda self, num_tokens: embeds,
            raising=True,
        )
        out = self._run(monkeypatch, dense_sp=True, method="dummy_inputs_embeds")
        assert out.shape[0] == 4

    def test_dummy_inputs_embeds_stays_full_without_dense_sp(self, monkeypatch):
        embeds = torch.zeros(8, 2, dtype=torch.float64)
        monkeypatch.setattr(
            "vllm.v1.worker.gpu.model_states.default.DefaultModelState.dummy_inputs_embeds",
            lambda self, num_tokens: embeds,
            raising=True,
        )
        out = self._run(monkeypatch, dense_sp=False, method="dummy_inputs_embeds")
        assert out.shape[0] == 8


class TestCaptureWrapperGather:
    """The FULL-graph capture path calls the model through ModelWithContext,
    bypassing execute_model's exit gather, so the wrapper restores the
    sharded output to full rows (capture sizes are TP-aligned)."""

    def _make_wrapper(self, dense_sp, is_draft_model=False):
        from vllm.config.compilation import CUDAGraphMode

        from vllm_ascend.worker.v2.aclgraph_utils import ModelWithContext

        original = SimpleNamespace(forward=lambda *a, **k: torch.arange(3 * 2, dtype=torch.float64).reshape(3, 2))
        wrapper = ModelWithContext.__new__(ModelWithContext)
        wrapper.original_model = original
        wrapper.is_draft_model = is_draft_model
        wrapper.is_draft_model_prefill = False
        wrapper.device_metadata = None
        wrapper.dense_sp = dense_sp
        return wrapper, CUDAGraphMode

    def _run(self, monkeypatch, dense_sp, mode, is_draft_model=False):
        from vllm_ascend.worker.v2 import aclgraph_utils

        def fake_gather(x):
            # TP=2: duplicate the 3-row shard -> 6 full rows.
            return torch.cat([x, x], dim=0)

        monkeypatch.setattr(aclgraph_utils, "sp_all_gather", fake_gather)
        monkeypatch.setattr(aclgraph_utils, "get_forward_context", lambda: SimpleNamespace(cudagraph_runtime_mode=mode))
        wrapper, CUDAGraphMode = self._make_wrapper(dense_sp, is_draft_model)
        monkeypatch.setattr(aclgraph_utils, "_EXTRA_CTX", SimpleNamespace(capturing=None))
        out = wrapper.forward()
        return out

    def test_full_capture_gathers_the_shard(self, monkeypatch):
        from vllm.config.compilation import CUDAGraphMode

        out = self._run(monkeypatch, dense_sp=True, mode=CUDAGraphMode.FULL)
        assert out.shape[0] == 6

    def test_warmup_mode_gathers_the_shard(self, monkeypatch):
        from vllm.config.compilation import CUDAGraphMode

        out = self._run(monkeypatch, dense_sp=True, mode=CUDAGraphMode.NONE)
        assert out.shape[0] == 6

    def test_piecewise_capture_stays_sharded(self, monkeypatch):
        from vllm.config.compilation import CUDAGraphMode

        out = self._run(monkeypatch, dense_sp=True, mode=CUDAGraphMode.PIECEWISE)
        assert out.shape[0] == 3

    def test_draft_wrapper_stays_full_token(self, monkeypatch):
        from vllm.config.compilation import CUDAGraphMode

        out = self._run(monkeypatch, dense_sp=True, mode=CUDAGraphMode.FULL, is_draft_model=True)
        assert out.shape[0] == 3

    def test_stock_layout_without_dense_sp(self, monkeypatch):
        from vllm.config.compilation import CUDAGraphMode

        out = self._run(monkeypatch, dense_sp=False, mode=CUDAGraphMode.FULL)
        assert out.shape[0] == 3
