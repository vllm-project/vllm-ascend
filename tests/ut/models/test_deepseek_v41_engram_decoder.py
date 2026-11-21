# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""CPU constructor coverage for decoder-to-Engram sequence-parallel wiring."""

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch")

pytest.importorskip(
    "vllm.models.deepseek_v41",
    reason="DeepSeek V4.1 is unavailable on this vLLM release",
)

from vllm_ascend.models.deepseek_v41 import model as model_mod


@pytest.mark.parametrize("sp", [False, True])
@pytest.mark.parametrize("supply_config", [False, True], ids=["normalized", "explicit"])
@pytest.mark.parametrize(
    "enabled,layer_id,is_draft,expected",
    [
        (True, 1, False, True),
        (True, 14, False, True),
        (True, 0, False, False),
        (True, 1, True, False),
        (False, 1, False, False),
    ],
    ids=["first-engram", "second-engram", "other-layer", "draft", "disabled"],
)
def test_decoder_threads_sp_only_into_enabled_target_engram(
    monkeypatch, sp, supply_config, enabled, layer_id, is_draft, expected
):
    config = SimpleNamespace(
        hidden_size=32,
        rope_parameters={"original_max_position_embeddings": 4096},
        rms_norm_eps=1e-6,
        hc_mult=4,
        hc_sinkhorn_iters=3,
        hc_eps=1e-6,
        engram_layer_ids=[1, 14],
    )
    quant_config = object()
    hf_config = object()
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config),
        cache_config=object(),
        quant_config=quant_config,
        parallel_config=SimpleNamespace(use_sequence_parallel_moe=sp),
    )

    class Attention(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.wo_b = SimpleNamespace(reduce_results=True)

    monkeypatch.setattr(model_mod.DeepseekV41DecoderLayer, "attention_cls", Attention)
    monkeypatch.setattr(model_mod, "DeepseekV41MoE", lambda **kwargs: torch.nn.Identity())
    monkeypatch.setattr(model_mod, "RMSNorm", lambda *args, **kwargs: torch.nn.Identity())
    monkeypatch.setattr(model_mod, "enable_dsa_cp", lambda: False)
    eligibility = Mock(return_value=enabled)
    monkeypatch.setattr(model_mod, "_engram_enabled_for_runtime", eligibility)
    normalize = Mock(return_value=config)
    monkeypatch.setattr(model_mod, "normalize_deepseek_v41_config", normalize)
    engram_factory = Mock(return_value=torch.nn.Identity())
    monkeypatch.setattr(model_mod, "AscendEngram", engram_factory)
    prefix = f"model.layers.{layer_id}"

    layer = model_mod.DeepseekV41DecoderLayer(
        vllm_config,
        prefix,
        config=config if supply_config else None,
        is_draft_layer=is_draft,
    )

    assert layer.use_sequence_parallel is sp
    assert layer.self_attn.wo_b.reduce_results is (not sp)
    eligibility.assert_called_once_with(config, vllm_config)
    if supply_config:
        normalize.assert_not_called()
    else:
        normalize.assert_called_once_with(hf_config)
    if expected:
        engram_factory.assert_called_once_with(
            config,
            quant_config,
            f"{prefix}.engram",
            use_sequence_parallel=sp,
        )
        assert layer.engram is engram_factory.return_value
    else:
        engram_factory.assert_not_called()
        assert layer.engram is None
