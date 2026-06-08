# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for VWN-Eagle3 model components.

Tests cover PreVwnLayerV1, VwnLlamaDecoderLayer, VwnLlamaModel, and
Eagle3VwnLlamaForCausalLM using CPU-only execution with mocked VllmConfig.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from vllm.config import CacheConfig, CompilationMode, VllmConfig, set_current_vllm_config

from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.models.llama_eagle3_vwn import (
    Eagle3VwnLlamaForCausalLM,
    PreVwnLayerV1,
    VwnLlamaDecoderLayer,
    VwnLlamaModel,
)

# ---------------------------------------------------------------------------
# Helper: nn.Module that returns a fixed-shape tensor (replaces NPU-only ops)
# ---------------------------------------------------------------------------


class _PassthroughAttn(nn.Module):
    """Replaces self_attn for CPU tests — returns input-like tensor."""

    def __init__(self, hs):
        super().__init__()
        self.hs = hs

    def forward(self, *, positions, hidden_states, **kwargs):
        return hidden_states


class _PassthroughMLP(nn.Module):
    """Replaces mlp for CPU tests — returns input-like tensor."""

    def __init__(self, hs):
        super().__init__()
        self.hs = hs

    def forward(self, hidden_states):
        return hidden_states


def _mock_npu_ops_on_layer(layer, hs, num_tokens):
    """Replace self_attn and mlp with passthrough modules for CPU testing."""
    layer.self_attn = _PassthroughAttn(hs)
    layer.mlp = _PassthroughMLP(hs)


# ---------------------------------------------------------------------------
# Autouse fixture: mock TP group so ReplicatedLinear works on CPU
# ---------------------------------------------------------------------------


class _MockTPGroup:
    """Minimal mock for get_tp_group() when TP=1."""

    rank_in_group = 0
    world_size = 1

    def all_reduce(self, *args, **kwargs):
        pass

    def all_gather(self, x, *args, **kwargs):
        return x.unsqueeze(0)

    def reduce_scatter(self, x, *args, **kwargs):
        return x


@pytest.fixture(autouse=True)
def _mock_tp_group():
    """Patch get_tp_group so CustomLinearOp.tp_rank/tp_size work on CPU."""
    _mock = _MockTPGroup()
    with patch("vllm_ascend.ops.linear_op.get_tp_group", return_value=_mock), \
         patch("vllm.distributed.parallel_state.get_tp_group", return_value=_mock), \
         patch("vllm_ascend.ops.vocab_parallel_embedding.get_tp_group", return_value=_mock):
        yield


@pytest.fixture(autouse=True)
def _mock_ascend_config():
    """Patch get_ascend_config so Ascend linear ops work on CPU."""
    mock_cfg = MagicMock()
    mock_cfg.enable_flashcomm2_parallel_size = 0
    mock_cfg.enable_context_parallel = False
    mock_cfg.enable_flashcomm1 = False
    mock_cfg.enable_matmul_allreduce = False
    mock_cfg.weight_nz_mode = 1
    mock_cfg.enable_mlapo = True
    mock_cfg.enable_fused_mc2 = 0
    mock_cfg.msmonitor_use_daemon = False
    mock_cfg.enable_transpose_kv_cache_by_block = True
    mock_cfg.finegrained_tp_config = MagicMock()
    # All finegrained_tp fields default to 0 (disabled)
    mock_cfg.finegrained_tp_config.lmhead_tensor_parallel_size = 0
    mock_cfg.finegrained_tp_config.embedding_tensor_parallel_size = 0
    mock_cfg.finegrained_tp_config.oproj_tensor_parallel_size = 0
    mock_cfg.finegrained_tp_config.olora_tensor_parallel_size = 0
    mock_cfg.finegrained_tp_config.mlp_tensor_parallel_size = 0
    with patch("vllm_ascend.utils.get_ascend_config", return_value=mock_cfg):
        yield


@pytest.fixture(autouse=True)
def _mock_gemm_op():
    """Patch NPU-only vllm custom ops to work on CPU."""
    import torch.nn.functional as F

    def _cpu_gemm(input, weight, bias=None):
        return F.linear(input, weight, bias)

    def _cpu_maybe_calc_kv_scales(*args, **kwargs):
        return None

    def _cpu_maybe_pad_and_reduce(x, *args, **kwargs):
        return x

    with patch.object(torch.ops.vllm, "unquantized_gemm", _cpu_gemm), \
         patch.object(torch.ops.vllm, "maybe_calc_kv_scales", _cpu_maybe_calc_kv_scales), \
         patch.object(torch.ops.vllm, "maybe_pad_and_reduce", _cpu_maybe_pad_and_reduce):
        yield

# ---------------------------------------------------------------------------
# Helper: build a VllmConfig suitable for instantiating VWN components on CPU
# ---------------------------------------------------------------------------

# Default dimensions matching the real VWN checkpoint config.json
_HIDDEN = 2048
_INTERMEDIATE = 6144
_VOCAB = 151936
_DRAFT_VOCAB = 35000
_NUM_HEADS = 32
_NUM_KV_HEADS = 4
_HEAD_DIM = 128
_RMS_EPS = 1e-6


def _make_hf_config(
    hidden_size=_HIDDEN,
    intermediate_size=_INTERMEDIATE,
    vocab_size=_VOCAB,
    draft_vocab_size=_DRAFT_VOCAB,
    num_hidden_layers=1,
    num_attention_heads=_NUM_HEADS,
    num_key_value_heads=_NUM_KV_HEADS,
    head_dim=_HEAD_DIM,
    rms_norm_eps=_RMS_EPS,
    vwn_m=4,
    vwn_r=1.5,
    **extra,
):
    """Create a real LlamaConfig with VWN attributes.

    Using a real config object instead of MagicMock avoids whack-a-mole
    with missing attributes that the deep init chain of LlamaDecoderLayer
    expects (rope_parameters, max_position_embeddings, etc.).
    """
    from transformers import LlamaConfig

    cfg = LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_hidden_layers=num_hidden_layers,
        vocab_size=vocab_size,
        rms_norm_eps=rms_norm_eps,
        max_position_embeddings=40960,
    )
    # VWN-specific attributes (not in LlamaConfig schema)
    cfg.vwn_m = vwn_m
    cfg.vwn_r = vwn_r
    cfg.draft_vocab_size = draft_vocab_size
    for k, v in extra.items():
        setattr(cfg, k, v)
    return cfg


def _create_vllm_config_for_vwn(
    vwn_m=4,
    vwn_r=1.5,
    hidden_size=_HIDDEN,
    num_hidden_layers=1,
    draft_vocab_size=_DRAFT_VOCAB,
    vocab_size=_VOCAB,
    num_target_layers=48,
):
    """Create a fully mocked VllmConfig for VWN model instantiation.

    Pattern follows test_eagle_proposer.py and
    test_extract_hidden_states_proposer.py.
    """
    hf_config = _make_hf_config(
        hidden_size=hidden_size,
        vwn_m=vwn_m,
        vwn_r=vwn_r,
        num_hidden_layers=num_hidden_layers,
        draft_vocab_size=draft_vocab_size,
        vocab_size=vocab_size,
    )

    vllm_config = MagicMock(spec=VllmConfig)

    # speculative_config
    vllm_config.speculative_config = MagicMock()
    vllm_config.speculative_config.method = "eagle3"
    vllm_config.speculative_config.num_speculative_tokens = 3
    vllm_config.speculative_config.draft_tensor_parallel_size = 1
    vllm_config.speculative_config.draft_model_config = MagicMock()
    vllm_config.speculative_config.draft_model_config.hf_config = hf_config
    vllm_config.speculative_config.draft_model_config.get_hidden_size = MagicMock(
        return_value=hidden_size,
    )
    vllm_config.speculative_config.draft_model_config.get_inputs_embeds_size = MagicMock(
        return_value=hidden_size,
    )
    vllm_config.speculative_config.draft_model_config.uses_mrope = False
    vllm_config.speculative_config.draft_model_config.uses_xdrope_dim = 0
    vllm_config.speculative_config.draft_model_config.quantization = None
    vllm_config.speculative_config.draft_model_config.load_config = MagicMock()
    vllm_config.speculative_config.disable_padded_drafter_batch = False
    vllm_config.speculative_config.parallel_drafting = False
    vllm_config.speculative_config.speculative_token_tree = str(
        [(i + 1) * (0,) for i in range(3)],
    )

    # cache_config
    vllm_config.cache_config = MagicMock(spec=CacheConfig)
    vllm_config.cache_config.block_size = 16
    vllm_config.cache_config.kv_cache_dtype_skip_layers = None
    vllm_config.cache_config.cache_dtype = "auto"

    # scheduler_config
    vllm_config.scheduler_config = MagicMock()
    vllm_config.scheduler_config.max_num_batched_tokens = 1024
    vllm_config.scheduler_config.max_num_seqs = 32
    vllm_config.scheduler_config.async_scheduling = True

    # model_config — use float32 for CPU testing
    vllm_config.model_config = MagicMock()
    vllm_config.model_config.dtype = torch.float32
    vllm_config.model_config.max_model_len = 2048
    vllm_config.model_config.uses_mrope = False
    vllm_config.model_config.uses_xdrope_dim = 0
    vllm_config.model_config.enforce_eager = True
    vllm_config.model_config.hf_text_config = MagicMock(spec=[])
    vllm_config.model_config.hf_text_config.to_dict = MagicMock(return_value={})
    vllm_config.model_config.hf_config = hf_config
    vllm_config.model_config.get_num_layers = MagicMock(return_value=num_target_layers)

    # compilation_config
    vllm_config.compilation_config = MagicMock()
    vllm_config.compilation_config.mode = CompilationMode.NONE
    vllm_config.compilation_config.pass_config = MagicMock()
    vllm_config.compilation_config.pass_config.enable_sp = False
    vllm_config.compilation_config.custom_ops = ["none"]  # Required by CustomOp.default_on()

    # parallel_config
    vllm_config.parallel_config = MagicMock()
    vllm_config.parallel_config.tensor_parallel_size = 1
    vllm_config.parallel_config.data_parallel_rank = 0
    vllm_config.parallel_config.data_parallel_size = 1
    vllm_config.parallel_config.prefill_context_parallel_size = 1
    vllm_config.parallel_config.decode_context_parallel_size = 1
    vllm_config.parallel_config.enable_expert_parallel = False

    vllm_config.additional_config = None

    init_ascend_config(vllm_config)
    return vllm_config


# ---------------------------------------------------------------------------
# Helper: create a full model with mocked NPU ops, ready for forward pass
# ---------------------------------------------------------------------------


def _make_model_with_mocked_ops(**kwargs):
    """Create Eagle3VwnLlamaForCausalLM with mocked attention/MLP for CPU."""
    vllm_config = _create_vllm_config_for_vwn(**kwargs)
    hs = vllm_config.speculative_config.draft_model_config.hf_config.hidden_size
    ctx = set_current_vllm_config(vllm_config)
    ctx.__enter__()
    try:
        model = Eagle3VwnLlamaForCausalLM(vllm_config=vllm_config, prefix="")
        for layer in model.model.layers:
            _mock_npu_ops_on_layer(layer, hs, 1)
        return model, vllm_config, hs, ctx
    except Exception:
        ctx.__exit__(None, None, None)
        raise


# ---------------------------------------------------------------------------
# Test: PreVwnLayerV1
# ---------------------------------------------------------------------------


class TestPreVwnLayerV1:
    """Tests for the pre-VWN projection layer — init + forward merged."""

    @pytest.mark.parametrize("vwn_m,vwn_r", [(1, 1.5), (4, 1.5), (4, 1.0), (1, 1.0)])
    def test_init_and_forward(self, vwn_m, vwn_r):
        """Verify layer init (dims, submodules) and forward output shape."""
        vllm_config = _create_vllm_config_for_vwn(vwn_m=vwn_m, vwn_r=vwn_r)
        hs = _HIDDEN
        wd = int(hs * vwn_r)
        batch = 4

        with set_current_vllm_config(vllm_config):
            layer = PreVwnLayerV1(
                vllm_config=vllm_config,
                prefix="test_prevwn",
                config=vllm_config.speculative_config.draft_model_config.hf_config,
            )
            # Init checks
            assert layer.hidden_size == hs
            assert layer.wider_dim == wd
            assert layer.m == vwn_m
            assert hasattr(layer, "input_layernorm")
            assert hasattr(layer, "hidden_norm")
            assert hasattr(layer, "fc")
            assert hasattr(layer, "upward")

            # Forward check
            out = layer(torch.randn(batch, hs), torch.randn(batch, hs))

        assert out.shape == (batch, wd)


# ---------------------------------------------------------------------------
# Test: VwnLlamaDecoderLayer
# ---------------------------------------------------------------------------


class TestVwnLlamaDecoderLayer:
    """Tests for the VWN-augmented Llama decoder layer."""

    @pytest.mark.parametrize("vwn_m,vwn_r", [(4, 1.5), (4, 1.0), (1, 1.5)])
    def test_init_vwn_projections(self, vwn_m, vwn_r):
        """VWN-specific submodules and dimension bookkeeping."""
        vllm_config = _create_vllm_config_for_vwn(vwn_m=vwn_m, vwn_r=vwn_r)

        with set_current_vllm_config(vllm_config):
            layer = VwnLlamaDecoderLayer(
                vllm_config=vllm_config,
                prefix="model.layers.48",
                config=vllm_config.speculative_config.draft_model_config.hf_config,
                layer_idx=0,
            )

        hs = _HIDDEN
        wd = int(hs * vwn_r)
        assert isinstance(layer.pre_vwn_layer, PreVwnLayerV1)
        assert hasattr(layer, "downward_and_forgot")
        assert hasattr(layer, "upward_after_attn")
        assert hasattr(layer, "upward_after_mlp")
        assert hasattr(layer, "downward")
        assert layer.m == vwn_m
        assert layer.wider_dim == wd

    @pytest.mark.parametrize("vwn_m,vwn_r,batch", [
        (4, 1.5, 4), (4, 1.0, 4), (1, 1.5, 4), (4, 1.5, 1),
    ])
    def test_forward_layer0(self, vwn_m, vwn_r, batch):
        """VWN forward with various m/r configs and batch sizes."""
        vllm_config = _create_vllm_config_for_vwn(vwn_m=vwn_m, vwn_r=vwn_r)
        hs = _HIDDEN

        with set_current_vllm_config(vllm_config):
            layer = VwnLlamaDecoderLayer(
                vllm_config=vllm_config,
                prefix="model.layers.48",
                config=vllm_config.speculative_config.draft_model_config.hf_config,
                layer_idx=0,
            )
            _mock_npu_ops_on_layer(layer, hs, batch)

            embeds = torch.randn(batch, hs)
            hidden = torch.randn(batch, hs)
            positions = torch.arange(batch, dtype=torch.long)

            out_hidden, out_residual = layer(positions, embeds, hidden, None)

        assert out_hidden.shape == (batch, hs)

    def test_forward_layer_nonzero_passthrough(self):
        """Non-zero layer_idx returns hidden_states unchanged."""
        vllm_config = _create_vllm_config_for_vwn()
        hs = _HIDDEN
        batch = 2

        with set_current_vllm_config(vllm_config):
            layer = VwnLlamaDecoderLayer(
                vllm_config=vllm_config,
                prefix="model.layers.48",
                config=vllm_config.speculative_config.draft_model_config.hf_config,
                layer_idx=1,
            )
            embeds = torch.randn(batch, hs)
            hidden = torch.randn(batch, hs)

            out_hidden, out_residual = layer(
                torch.arange(batch, dtype=torch.long), embeds, hidden, None,
            )

        assert out_hidden.shape == (batch, hs)

    def test_qkv_proj_input_size_layer0(self):
        """VWN layer 0 restores qkv_proj input to hidden_size (not 2*hs).

        This is the critical VWN-vs-Eagle3 difference: the parent class
        overrides qkv_proj to accept 2*hidden_size (concatenating embeds +
        hidden), but VWN feeds hidden_size into attention via the
        downward_and_forgot projection instead.
        """
        vllm_config = _create_vllm_config_for_vwn()

        with set_current_vllm_config(vllm_config):
            layer = VwnLlamaDecoderLayer(
                vllm_config=vllm_config,
                prefix="model.layers.48",
                config=vllm_config.speculative_config.draft_model_config.hf_config,
                layer_idx=0,
            )

        assert layer.self_attn.qkv_proj.input_size == _HIDDEN


# ---------------------------------------------------------------------------
# Test: VwnLlamaModel
# ---------------------------------------------------------------------------


class TestVwnLlamaModel:
    """Tests for the full VWN model body."""

    @pytest.mark.parametrize("num_hidden_layers", [1, 2])
    def test_init_and_forward(self, num_hidden_layers):
        """Verify layer count/type and forward output shapes."""
        vllm_config = _create_vllm_config_for_vwn(num_hidden_layers=num_hidden_layers)
        hs = _HIDDEN
        num_tokens = 4

        with set_current_vllm_config(vllm_config):
            model = VwnLlamaModel(
                vllm_config=vllm_config,
                prefix="model",
                start_layer_id=48,
            )

            assert len(model.layers) == num_hidden_layers
            for i, layer in enumerate(model.layers):
                assert isinstance(layer, VwnLlamaDecoderLayer)
                assert layer.layer_idx == i

            for layer in model.layers:
                _mock_npu_ops_on_layer(layer, hs, num_tokens)

            input_ids = torch.randint(0, _VOCAB, (num_tokens,))
            positions = torch.arange(num_tokens, dtype=torch.long)
            hidden_states = torch.randn(num_tokens, hs)

            postnorm, prenorm = model(input_ids, positions, hidden_states)

        assert postnorm.shape == (num_tokens, hs)
        assert prenorm.shape == (num_tokens, hs)

    def test_forward_with_input_embeds(self):
        """Forward with explicit input_embeds bypasses embedding lookup."""
        vllm_config = _create_vllm_config_for_vwn(num_hidden_layers=1)
        hs = _HIDDEN
        num_tokens = 3

        with set_current_vllm_config(vllm_config):
            model = VwnLlamaModel(
                vllm_config=vllm_config,
                prefix="model",
                start_layer_id=48,
            )
            for layer in model.layers:
                _mock_npu_ops_on_layer(layer, hs, num_tokens)

            input_ids = torch.randint(0, _VOCAB, (num_tokens,))
            positions = torch.arange(num_tokens, dtype=torch.long)
            hidden_states = torch.randn(num_tokens, hs)
            input_embeds = torch.randn(num_tokens, hs)

            postnorm, prenorm = model(
                input_ids, positions, hidden_states, input_embeds=input_embeds,
            )

        assert postnorm.shape == (num_tokens, hs)
        assert prenorm.shape == (num_tokens, hs)


# ---------------------------------------------------------------------------
# Test: Eagle3VwnLlamaForCausalLM
# ---------------------------------------------------------------------------


class TestEagle3VwnLlamaForCausalLM:
    """Tests for the top-level VWN-Eagle3 CausalLM model."""

    @pytest.mark.parametrize("vwn_m", [4, 1])
    def test_init_and_forward(self, vwn_m):
        """Init creates VwnLlamaModel; forward returns (postnorm, prenorm)."""
        model, vllm_config, hs, ctx = _make_model_with_mocked_ops(vwn_m=vwn_m)
        try:
            assert isinstance(model.model, VwnLlamaModel)
            num_tokens = 3

            input_ids = torch.randint(0, _VOCAB, (num_tokens,))
            positions = torch.arange(num_tokens, dtype=torch.long)
            hidden_states = torch.randn(num_tokens, hs)

            postnorm, prenorm = model(input_ids, positions, hidden_states)

            assert postnorm.shape == (num_tokens, hs)
            assert prenorm.shape == (num_tokens, hs)
        finally:
            ctx.__exit__(None, None, None)

    def test_embed_input_ids(self):
        vllm_config = _create_vllm_config_for_vwn()
        num_tokens = 3

        with set_current_vllm_config(vllm_config):
            model = Eagle3VwnLlamaForCausalLM(vllm_config=vllm_config, prefix="")
            input_ids = torch.randint(0, _VOCAB, (num_tokens,))
            embeds = model.embed_input_ids(input_ids)

        assert embeds.shape == (num_tokens, _HIDDEN)

    def test_compute_logits_output_shape_and_mapping(self):
        """compute_logits maps draft vocab logits to target vocab positions."""
        vllm_config = _create_vllm_config_for_vwn()
        hs = _HIDDEN
        batch = 2

        with set_current_vllm_config(vllm_config):
            model = Eagle3VwnLlamaForCausalLM(vllm_config=vllm_config, prefix="")
            model.draft_id_to_target_id.data.copy_(
                torch.arange(_DRAFT_VOCAB, dtype=torch.long),
            )

            # Patch logits_processor.forward to bypass ParallelLMHead (needs
            # real TP group coordinator).
            draft_logits = torch.randn(batch, _DRAFT_VOCAB)
            with patch.object(type(model.logits_processor), "forward",
                              return_value=draft_logits):
                logits = model.compute_logits(torch.randn(batch, hs))

        assert logits.shape == (batch, _VOCAB)
        # Mapped positions should have finite values
        mapped = torch.arange(_DRAFT_VOCAB) + model.draft_id_to_target_id
        assert logits[0, mapped[0]].item() != float("-inf")
        # Unmapped positions should be -inf
        assert logits[0, _VOCAB - 1].item() == float("-inf")

    @pytest.mark.parametrize("use_aux", [True, False])
    def test_combine_hidden_states(self, use_aux):
        """combine_hidden_states: FC projection when aux=True, identity when False."""
        vllm_config = _create_vllm_config_for_vwn()
        hs = _HIDDEN
        batch = 2

        if not use_aux:
            hf_config = vllm_config.speculative_config.draft_model_config.hf_config
            hf_config.eagle_config = {"use_aux_hidden_state": False}

        with set_current_vllm_config(vllm_config):
            model = Eagle3VwnLlamaForCausalLM(vllm_config=vllm_config, prefix="")
            assert model.model.use_aux_hidden_state == use_aux

            if use_aux:
                hidden = torch.randn(batch, hs * 3)
                combined = model.combine_hidden_states(hidden)
                assert combined.shape == (batch, hs)
            else:
                hidden = torch.randn(batch, hs)
                combined = model.combine_hidden_states(hidden)
                assert torch.equal(combined, hidden)

    def test_vwn_parameters_and_count(self):
        """Verify VWN parameters exist and total count matches expectation."""
        vllm_config = _create_vllm_config_for_vwn(vwn_m=4, vwn_r=1.5)

        with set_current_vllm_config(vllm_config):
            model = Eagle3VwnLlamaForCausalLM(vllm_config=vllm_config, prefix="")

        param_names = {name for name, _ in model.named_parameters()}

        # VWN-specific prefixes must all have registered parameters
        for prefix in ("pre_vwn_layer", "downward_and_forgot", "upward_after_attn",
                       "upward_after_mlp", "downward"):
            assert any(prefix in n for n in param_names), (
                f"No parameters found for VWN prefix '{prefix}'"
            )

        # draft_id_to_target_id exists and does not require grad
        assert "draft_id_to_target_id" in param_names
        assert not dict(model.named_parameters())["draft_id_to_target_id"].requires_grad

        # Total parameter count (update if architecture changes)
        total = sum(p.numel() for p in model.parameters())
        assert total == 454_607_032, (
            f"Parameter count mismatch: got {total}, expected 454_607_032. "
            "Update if model architecture changed."
        )


# ---------------------------------------------------------------------------
# Test: Weight loading integration
# ---------------------------------------------------------------------------


def _make_synthetic_weights_m4_r15():
    """Synthetic weights matching vwn_eagle3_full_m_4_r_1_5 checkpoint shapes."""
    hs, wd, m = _HIDDEN, int(_HIDDEN * 1.5), 4
    dv = _DRAFT_VOCAB
    v = _VOCAB
    inter = _INTERMEDIATE
    nkv = _NUM_KV_HEADS
    nq = _NUM_HEADS
    hd = _HEAD_DIM
    return {
        "d2t": torch.zeros(dv, dtype=torch.long),
        "embed_tokens.weight": torch.randn(v, hs, dtype=torch.float32),
        "fc.weight": torch.randn(hs, hs * 3, dtype=torch.float32),
        "layers.0.downward.weight": torch.randn(hs // m, wd // m, dtype=torch.float32),
        "layers.0.downward_and_forgot.weight": torch.randn(
            (hs + wd) // m, wd // m, dtype=torch.float32,
        ),
        "layers.0.downward_and_forgot_after_attn.weight": torch.randn(
            (hs + wd) // m, wd // m, dtype=torch.float32,
        ),
        "layers.0.mlp.gate_proj.weight": torch.randn(inter, hs, dtype=torch.float32),
        "layers.0.mlp.up_proj.weight": torch.randn(inter, hs, dtype=torch.float32),
        "layers.0.mlp.down_proj.weight": torch.randn(hs, inter, dtype=torch.float32),
        "layers.0.post_attention_layernorm.weight": torch.randn(hs, dtype=torch.float32),
        "layers.0.pre_attention_layernorm.weight": torch.randn(hs, dtype=torch.float32),
        "layers.0.pre_vwn_layer.fc.weight": torch.randn(hs, 2 * hs, dtype=torch.float32),
        "layers.0.pre_vwn_layer.hidden_norm.weight": torch.randn(hs, dtype=torch.float32),
        "layers.0.pre_vwn_layer.input_layernorm.weight": torch.randn(hs, dtype=torch.float32),
        "layers.0.pre_vwn_layer.upward.weight": torch.randn(
            wd // m, hs // m, dtype=torch.float32,
        ),
        "layers.0.self_attn.q_proj.weight": torch.randn(nq * hd, hs, dtype=torch.float32),
        "layers.0.self_attn.k_proj.weight": torch.randn(nkv * hd, hs, dtype=torch.float32),
        "layers.0.self_attn.v_proj.weight": torch.randn(nkv * hd, hs, dtype=torch.float32),
        "layers.0.self_attn.o_proj.weight": torch.randn(hs, nq * hd, dtype=torch.float32),
        "layers.0.upward_after_attn.weight": torch.randn(wd // m, hs // m, dtype=torch.float32),
        "layers.0.upward_after_mlp.weight": torch.randn(wd // m, hs // m, dtype=torch.float32),
        "lm_head.weight": torch.randn(dv, hs, dtype=torch.float32),
        "norm.weight": torch.randn(hs, dtype=torch.float32),
    }


def _make_synthetic_weights_m4_r10():
    """Synthetic weights for m=4, r=1.0 (no downward.weight, no pre_vwn upward)."""
    hs, m = _HIDDEN, 4
    wd = hs  # r=1.0 => wider_dim == hidden_size
    dv = _DRAFT_VOCAB
    v = _VOCAB
    inter = _INTERMEDIATE
    nkv = _NUM_KV_HEADS
    nq = _NUM_HEADS
    hd = _HEAD_DIM
    return {
        "d2t": torch.zeros(dv, dtype=torch.long),
        "embed_tokens.weight": torch.randn(v, hs, dtype=torch.float32),
        "fc.weight": torch.randn(hs, hs * 3, dtype=torch.float32),
        "layers.0.downward_and_forgot.weight": torch.randn(
            (hs + wd) // m, wd // m, dtype=torch.float32,
        ),
        "layers.0.downward_and_forgot_after_attn.weight": torch.randn(
            (hs + wd) // m, wd // m, dtype=torch.float32,
        ),
        "layers.0.mlp.gate_proj.weight": torch.randn(inter, hs, dtype=torch.float32),
        "layers.0.mlp.up_proj.weight": torch.randn(inter, hs, dtype=torch.float32),
        "layers.0.mlp.down_proj.weight": torch.randn(hs, inter, dtype=torch.float32),
        "layers.0.post_attention_layernorm.weight": torch.randn(hs, dtype=torch.float32),
        "layers.0.pre_attention_layernorm.weight": torch.randn(hs, dtype=torch.float32),
        "layers.0.pre_vwn_layer.fc.weight": torch.randn(hs, 2 * hs, dtype=torch.float32),
        "layers.0.pre_vwn_layer.hidden_norm.weight": torch.randn(hs, dtype=torch.float32),
        "layers.0.pre_vwn_layer.input_layernorm.weight": torch.randn(hs, dtype=torch.float32),
        "layers.0.self_attn.q_proj.weight": torch.randn(nq * hd, hs, dtype=torch.float32),
        "layers.0.self_attn.k_proj.weight": torch.randn(nkv * hd, hs, dtype=torch.float32),
        "layers.0.self_attn.v_proj.weight": torch.randn(nkv * hd, hs, dtype=torch.float32),
        "layers.0.self_attn.o_proj.weight": torch.randn(hs, nq * hd, dtype=torch.float32),
        "layers.0.upward_after_attn.weight": torch.randn(wd // m, hs // m, dtype=torch.float32),
        "layers.0.upward_after_mlp.weight": torch.randn(wd // m, hs // m, dtype=torch.float32),
        "lm_head.weight": torch.randn(dv, hs, dtype=torch.float32),
        "norm.weight": torch.randn(hs, dtype=torch.float32),
    }


class TestWeightLoadingIntegration:
    """Weight loading tests using synthetic weight dicts with real shapes."""

    def test_load_weights_r15(self):
        """Load all weights for m=4, r=1.5: d2t remap, t2d skip, fc loaded."""
        vllm_config = _create_vllm_config_for_vwn(vwn_m=4, vwn_r=1.5)
        weights = _make_synthetic_weights_m4_r15()
        # Set d2t to non-trivial values
        weights["d2t"] = torch.arange(_DRAFT_VOCAB, dtype=torch.long)
        # Add a t2d entry — should be skipped without error
        weights["t2d"] = torch.zeros(_VOCAB, dtype=torch.bool)
        # Use a distinctive fc.weight
        weights["fc.weight"] = torch.ones(_HIDDEN, _HIDDEN * 3, dtype=torch.float32)

        with set_current_vllm_config(vllm_config):
            model = Eagle3VwnLlamaForCausalLM(vllm_config=vllm_config, prefix="")
            model.load_weights(weights.items())

        # d2t remapped to draft_id_to_target_id
        dit = dict(model.named_parameters())["draft_id_to_target_id"]
        assert torch.equal(dit.data, torch.arange(_DRAFT_VOCAB, dtype=torch.long))

        # fc weight loaded
        assert model.model.fc.weight.data.abs().sum() > 0
        assert model.model.fc.weight.shape == (_HIDDEN, _HIDDEN * 3)

        # Key parameters loaded (not all-zero)
        for name, param in model.named_parameters():
            if "draft_id_to_target_id" in name:
                continue
            if "embed_tokens" in name or "lm_head" in name:
                assert param.data.abs().sum() > 0, f"{name} not loaded"

    def test_load_weights_r10(self):
        """Load weights for m=4, r=1.0: no downward/pre_vwn upward in checkpoint.

        When vwn_r=1.0, wider_dim == hidden_size, so downward and pre_vwn
        projections are identity-sized (hs//m, hs//m).
        """
        vllm_config = _create_vllm_config_for_vwn(vwn_m=4, vwn_r=1.0)
        weights = _make_synthetic_weights_m4_r10()
        hs, m = _HIDDEN, 4

        with set_current_vllm_config(vllm_config):
            model = Eagle3VwnLlamaForCausalLM(vllm_config=vllm_config, prefix="")
            model.load_weights(weights.items())

        layer = model.model.layers[0]
        # wd == hs when r=1.0, so dimensions are identity-sized
        assert layer.downward.weight.shape == (hs // m, hs // m)
        assert layer.pre_vwn_layer.upward.weight.shape == (hs // m, hs // m)
