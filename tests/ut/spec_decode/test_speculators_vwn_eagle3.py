#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
"""Unit tests for VWN-Eagle3 model components.

Tests cover PreVwnLayerV1, VwnLlamaDecoderLayer, VwnLlamaModel, and
Eagle3VwnLlamaForCausalLM using CPU-only execution with mocked VllmConfig.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import CacheConfig, CompilationMode, VllmConfig, set_current_vllm_config

from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.models.llama_eagle3_vwn import (
    Eagle3VwnLlamaForCausalLM,
    PreVwnLayerV1,
    VwnLlamaDecoderLayer,
    VwnLlamaModel,
)

_HIDDEN = 2048
_INTERMEDIATE = 6144
_VOCAB = 151936
_DRAFT_VOCAB = 35000
_NUM_HEADS = 32
_NUM_KV_HEADS = 4
_RMS_EPS = 1e-6


class _PassthroughAttn(nn.Module):
    """Replaces self_attn for CPU tests — returns input unchanged."""

    def forward(self, *, positions, hidden_states, **kwargs):
        return hidden_states


class _PassthroughMLP(nn.Module):
    """Replaces mlp for CPU tests — returns input unchanged."""

    def forward(self, hidden_states):
        return hidden_states


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


def _mock_npu_ops_on_layer(layer):
    """Replace self_attn and mlp with passthrough modules for CPU testing."""
    layer.self_attn = _PassthroughAttn()
    layer.mlp = _PassthroughMLP()


def _cpu_rms_norm(x, weight, eps):
    """CPU fallback for torch_npu.npu_rms_norm.

    Returns the normalized tensor and a placeholder rstd (None), matching the
    2-tuple shape the production op yields so callers can unpack it.
    """
    orig_dtype = x.dtype
    x32 = x.float()
    var = x32.pow(2).mean(-1, keepdim=True)
    x32 = x32 * torch.rsqrt(var + eps)
    out = (x32 * weight.float()).to(orig_dtype)
    return out, None


def _cpu_add_rms_norm(x, residual, weight, eps):
    """CPU fallback for torch_npu.npu_add_rms_norm (returns 3-tuple)."""
    x_plus_res = x + residual
    out, _ = _cpu_rms_norm(x_plus_res, weight, eps)
    return out, None, x_plus_res


def _cpu_add_rms_norm_bias(x, residual, weight, bias, eps):
    """CPU fallback for torch.ops._C_ascend.npu_add_rms_norm_bias."""
    out, _, new_residual = _cpu_add_rms_norm(x, residual, weight, eps)
    if bias is not None:
        out = out + bias
    return out, _, new_residual


@pytest.fixture(autouse=True)
def _mock_npu_env():
    """Patch TP group, Ascend config, and NPU ops so all tests run on CPU.

    conftest.py stubs ``torch_npu.npu_rms_norm`` with a bare ``MagicMock()``,
    which yields an empty iterator and breaks tuple unpacking. We override it
    here (plus the related add/bias variants and the weight-prefetch hook) with
    pure-torch CPU implementations so AscendRMSNorm can run on CPU runners.
    """
    import torch_npu

    _mock = _MockTPGroup()
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
    mock_cfg.finegrained_tp_config = MagicMock(
        lmhead_tensor_parallel_size=0,
        embedding_tensor_parallel_size=0,
        oproj_tensor_parallel_size=0,
        olora_tensor_parallel_size=0,
        mlp_tensor_parallel_size=0,
    )

    _prefetch_mock = MagicMock()

    with (
        patch("vllm_ascend.ops.linear_op.get_tp_group", return_value=_mock),
        patch("vllm.distributed.parallel_state.get_tp_group", return_value=_mock),
        patch("vllm_ascend.ops.vocab_parallel_embedding.get_tp_group", return_value=_mock),
        patch("vllm_ascend.utils.get_ascend_config", return_value=mock_cfg),
        patch.object(torch.ops.vllm, "unquantized_gemm", F.linear),
        patch.object(torch.ops.vllm, "maybe_calc_kv_scales", lambda *a, **kw: None),
        patch.object(torch.ops.vllm, "maybe_pad_and_reduce", lambda x, *a, **kw: x),
        patch("vllm.model_executor.layers.logits_processor.tensor_model_parallel_all_gather", lambda x, *a, **kw: x),
        patch.object(torch_npu, "npu_rms_norm", side_effect=_cpu_rms_norm, create=True),
        patch.object(torch_npu, "npu_add_rms_norm", side_effect=_cpu_add_rms_norm, create=True),
        patch.object(
            torch.ops._C_ascend,
            "npu_add_rms_norm_bias",
            side_effect=_cpu_add_rms_norm_bias,
            create=True,
        ),
        patch("vllm_ascend.ops.layernorm.get_weight_prefetch_method", return_value=_prefetch_mock),
        # enable_cp() reads parallel_config.*_context_parallel_size and runs `> 1`.
        # On MagicMock these fields yield TypeError on Python 3.12, so short-circuit
        # the check everywhere it's imported.
        patch("vllm_ascend.attention.attention_v1.enable_cp", return_value=False),
        patch("vllm_ascend.attention.sfa_v1.enable_cp", return_value=False, create=True),
        patch("vllm_ascend.attention.mla_v1.enable_cp", return_value=False, create=True),
    ):
        yield


def _make_hf_config(
    hidden_size=_HIDDEN,
    vwn_m=4,
    vwn_r=1.5,
    num_hidden_layers=1,
    draft_vocab_size=_DRAFT_VOCAB,
    **extra,
):
    """Create a real LlamaConfig with VWN attributes.

    Using a real config object avoids whack-a-mole with missing attributes
    that LlamaDecoderLayer's deep init chain expects.
    """
    from transformers import LlamaConfig

    cfg = LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=_INTERMEDIATE,
        num_attention_heads=_NUM_HEADS,
        num_key_value_heads=_NUM_KV_HEADS,
        num_hidden_layers=num_hidden_layers,
        vocab_size=_VOCAB,
        rms_norm_eps=_RMS_EPS,
        max_position_embeddings=40960,
    )
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
    num_target_layers=48,
):
    """Create a mocked VllmConfig for VWN model instantiation on CPU."""
    hf_config = _make_hf_config(
        hidden_size=hidden_size,
        vwn_m=vwn_m,
        vwn_r=vwn_r,
        num_hidden_layers=num_hidden_layers,
    )

    vllm_config = MagicMock(spec=VllmConfig)

    # speculative_config
    vllm_config.speculative_config = MagicMock()
    vllm_config.speculative_config.num_speculative_tokens = 3
    vllm_config.speculative_config.draft_tensor_parallel_size = 1
    vllm_config.speculative_config.parallel_drafting = False
    vllm_config.speculative_config.disable_padded_drafter_batch = False
    vllm_config.speculative_config.draft_model_config = MagicMock(
        hf_config=hf_config,
        uses_mrope=False,
        uses_xdrope_dim=0,
        quantization=None,
        load_config=MagicMock(),
        get_hidden_size=MagicMock(return_value=hidden_size),
        get_inputs_embeds_size=MagicMock(return_value=hidden_size),
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

    # model_config
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
    vllm_config.compilation_config.pass_config = MagicMock(enable_sp=False)
    vllm_config.compilation_config.custom_ops = ["none"]

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


@contextmanager
def _make_model_with_mocked_ops(**kwargs):
    """Create Eagle3VwnLlamaForCausalLM with mocked attention/MLP for CPU."""
    vllm_config = _create_vllm_config_for_vwn(**kwargs)
    hs = vllm_config.speculative_config.draft_model_config.hf_config.hidden_size
    with set_current_vllm_config(vllm_config):
        model = Eagle3VwnLlamaForCausalLM(vllm_config=vllm_config, prefix="")
        for layer in model.model.layers:
            _mock_npu_ops_on_layer(layer)
        yield model, vllm_config, hs


class TestPreVwnLayerV1:
    @pytest.mark.parametrize("vwn_m,vwn_r", [(4, 1.5), (1, 1.0)])
    def test_init_and_forward(self, vwn_m, vwn_r):
        """Verify layer init and forward output shape."""
        vllm_config = _create_vllm_config_for_vwn(vwn_m=vwn_m, vwn_r=vwn_r)
        hs, batch = _HIDDEN, 4
        wd = int(hs * vwn_r)

        with set_current_vllm_config(vllm_config):
            layer = PreVwnLayerV1(
                vllm_config=vllm_config,
                prefix="test_prevwn",
                config=vllm_config.speculative_config.draft_model_config.hf_config,
            )
            assert layer.wider_dim == wd
            out = layer(torch.randn(batch, hs), torch.randn(batch, hs))

        assert out.shape == (batch, wd)


class TestVwnLlamaDecoderLayer:
    @pytest.mark.parametrize("vwn_m,vwn_r", [(4, 1.5), (4, 1.0)])
    def test_forward_layer0(self, vwn_m, vwn_r):
        """VWN forward with various m/r configs — init + shape check."""
        vllm_config = _create_vllm_config_for_vwn(vwn_m=vwn_m, vwn_r=vwn_r)
        hs, batch = _HIDDEN, 4

        with set_current_vllm_config(vllm_config):
            layer = VwnLlamaDecoderLayer(
                vllm_config=vllm_config,
                prefix="model.layers.48",
                config=vllm_config.speculative_config.draft_model_config.hf_config,
                layer_idx=0,
            )
            _mock_npu_ops_on_layer(layer)
            out_hidden, _ = layer(
                torch.arange(batch, dtype=torch.long),
                torch.randn(batch, hs),
                torch.randn(batch, hs),
                None,
            )

        assert out_hidden.shape == (batch, hs)

    def test_qkv_proj_input_size_layer0(self):
        """VWN layer 0 qkv_proj input is hidden_size (not 2*hidden_size)."""
        vllm_config = _create_vllm_config_for_vwn()

        with set_current_vllm_config(vllm_config):
            layer = VwnLlamaDecoderLayer(
                vllm_config=vllm_config,
                prefix="model.layers.48",
                config=vllm_config.speculative_config.draft_model_config.hf_config,
                layer_idx=0,
            )

        assert layer.self_attn.qkv_proj.input_size == _HIDDEN


class TestVwnLlamaModel:
    @pytest.mark.parametrize(
        "num_hidden_layers,use_input_embeds",
        [
            (1, False),
            (1, True),
        ],
    )
    def test_forward(self, num_hidden_layers, use_input_embeds):
        """Verify layer count, type, and forward output shapes."""
        vllm_config = _create_vllm_config_for_vwn(num_hidden_layers=num_hidden_layers)
        hs, num_tokens = _HIDDEN, 4

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
                _mock_npu_ops_on_layer(layer)

            input_ids = torch.randint(0, _VOCAB, (num_tokens,))
            positions = torch.arange(num_tokens, dtype=torch.long)
            hidden_states = torch.randn(num_tokens, hs)
            input_embeds = torch.randn(num_tokens, hs) if use_input_embeds else None

            postnorm, prenorm = model(
                input_ids,
                positions,
                hidden_states,
                input_embeds=input_embeds,
            )

        assert postnorm.shape == (num_tokens, hs)
        assert prenorm.shape == (num_tokens, hs)


class TestEagle3VwnLlamaForCausalLM:
    def test_init_and_forward(self):
        with _make_model_with_mocked_ops(vwn_m=4) as (model, _, hs):
            assert isinstance(model.model, VwnLlamaModel)
            num_tokens = 3

            input_ids = torch.randint(0, _VOCAB, (num_tokens,))
            positions = torch.arange(num_tokens, dtype=torch.long)

            postnorm, prenorm = model(
                input_ids,
                positions,
                torch.randn(num_tokens, hs),
            )

            assert postnorm.shape == (num_tokens, hs)
            assert prenorm.shape == (num_tokens, hs)

    def test_embed_input_ids(self):
        vllm_config = _create_vllm_config_for_vwn()
        num_tokens = 3

        with set_current_vllm_config(vllm_config):
            model = Eagle3VwnLlamaForCausalLM(vllm_config=vllm_config, prefix="")
            embeds = model.embed_input_ids(torch.randint(0, _VOCAB, (num_tokens,)))

        assert embeds.shape == (num_tokens, _HIDDEN)
