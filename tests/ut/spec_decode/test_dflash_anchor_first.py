# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
"""Unit tests for anchor-first (`sample_from_anchor`) Qwen3 DFlash2 query layout."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import vllm.model_executor.models.qwen3_dflash2 as upstream_qwen3_dflash2_module
from torch import nn
from vllm.config import SpeculativeConfig
from vllm.v1.worker.gpu.spec_decode.dflash.config import DFlashSpeculatorConfig

from vllm_ascend.models.qwen3_dflash2 import DFlash2Qwen3DecoderLayer, DFlashQwen3DecoderLayer, _grouped_conv

_SPEC_PKG = "vllm.v1.worker.gpu.spec_decode"
HAS_NUM_QUERY_TOKENS = hasattr(DFlashSpeculatorConfig, "num_query_tokens")


class _LinearProjStub(nn.Linear):
    def __init__(
        self, input_size: int, output_size: int, bias: bool = False, params_dtype: Any = None, **_: Any
    ) -> None:
        super().__init__(input_size, output_size, bias=bias, dtype=params_dtype)


def _patch_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    def _init_base(self: Any, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        nn.Module.__init__(self)
        norm = lambda x, residual=None: x if residual is None else (x, residual)  # noqa: E731
        self.input_layernorm, self.post_attention_layernorm = norm, norm
        self.self_attn = lambda positions, hidden_states: hidden_states
        self.mlp = nn.Identity()

    monkeypatch.setattr(DFlashQwen3DecoderLayer, "__init__", _init_base)
    monkeypatch.setattr(upstream_qwen3_dflash2_module, "ReplicatedLinear", _LinearProjStub)


def _make_cfg(
    num_spec: Any,
    dflash_flag: bool | None = None,
    top_flag: bool | None = None,
    method: str = "dflash",
    arch: str = "DFlash2DraftModel",
    subpath: str | None = None,
    cls_name: str | None = None,
    has_draft: bool = True,
) -> tuple[Any, Any]:
    dflash_cfg = {"conv_kernel_size": 4, "conv_group_size": 4}
    if dflash_flag is not None:
        dflash_cfg["sample_from_anchor"] = dflash_flag
    hf_cfg = SimpleNamespace(
        hidden_size=8, dflash_config=dflash_cfg, **({} if top_flag is None else {"sample_from_anchor": top_flag})
    )
    declared = (
        f"{_SPEC_PKG}.{subpath}.{cls_name}"
        if subpath and cls_name and cls_name not in ("DFlashSpeculatorConfig", "DSparkSpeculatorConfig")
        else None
    )
    draft = (
        SimpleNamespace(model="test/draft", architectures=[arch], hf_config=hf_cfg, speculator_config_class=declared)
        if has_draft
        else None
    )
    spec: Any = object.__new__(SpeculativeConfig)
    spec.custom_method, spec.draft_model_config, spec.prefill_context_kv_only = None, draft, False
    spec.method, spec.num_speculative_tokens = method, num_spec
    return SimpleNamespace(speculative_config=spec, model_config=SimpleNamespace(dtype=torch.float32)), hf_cfg


@pytest.mark.parametrize(
    "case",
    [
        ("dflash.config", "DFlashSpeculatorConfig", "dflash", "DFlashDraftModel", None, None, 7, 8, True),
        ("dflash.config", "DFlashSpeculatorConfig", "dflash", "DFlashDraftModel", None, True, 7, 7, True),
        ("dflash2.config", "DFlash2SpeculatorConfig", "dflash", "DFlash2DraftModel", False, None, 7, 8, True),
        ("dflash2.config", "DFlash2SpeculatorConfig", "dflash", "DFlash2DraftModel", True, None, 7, 7, True),
        ("dflash2.config", "DFlash2SpeculatorConfig", "dflash", "DFlash2DraftModel", True, None, 1, 1, True),
        ("lilicorr.config", "LiLiCorrSpeculatorConfig", "dflash", "LiLiCorrDraftModel", True, None, 7, 7, True),
        ("dspark.config", "DSparkSpeculatorConfig", "dspark", "Qwen3DSparkForCausalLM", None, None, 7, 7, True),
        ("dspark.config", "DSparkSpeculatorConfig", "dspark", "Qwen3DSparkForCausalLM", None, False, 7, 8, True),
        ("dflash.config", "DFlashSpeculatorConfig", "dflash", "DFlashDraftModel", True, None, 7, 8, False),
    ],
)
def test_decoder_layer_block_size_resolution(monkeypatch: pytest.MonkeyPatch, case: tuple[Any, ...]) -> None:
    _patch_deps(monkeypatch)
    subpath, cls_name, method, arch, dflash_flag, top_flag, num_spec, expected_bs, has_draft = case
    vllm_cfg, layer_cfg = _make_cfg(num_spec, dflash_flag, top_flag, method, arch, subpath, cls_name, has_draft)
    expected_cls = getattr(importlib.import_module(f"{_SPEC_PKG}.{subpath}"), cls_name)
    spec_cfg = vllm_cfg.speculative_config
    assert type(spec_cfg.speculator_config) is expected_cls
    if HAS_NUM_QUERY_TOKENS:
        assert spec_cfg.speculator_config.num_query_tokens == expected_bs
    layer = DFlash2Qwen3DecoderLayer(vllm_cfg, config=layer_cfg, layer_idx=0)
    assert layer.attention_conv.block_size == layer.mlp_conv.block_size == expected_bs


def test_decoder_layer_delegation_and_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_deps(monkeypatch)
    dflash2_cls = importlib.import_module(f"{_SPEC_PKG}.dflash2.config").DFlash2SpeculatorConfig
    with monkeypatch.context() as mp:
        mp.setattr(dflash2_cls, "num_query_tokens", property(lambda self: 5), raising=False)
        vllm_cfg, layer_cfg = _make_cfg(7, subpath="dflash2.config", cls_name="DFlash2SpeculatorConfig")
        assert DFlash2Qwen3DecoderLayer(vllm_cfg, config=layer_cfg, layer_idx=0).attention_conv.block_size == 5
    c_vllm, c_cfg = _make_cfg(7, True, False, subpath="dflash2.config", cls_name="DFlash2SpeculatorConfig")
    with pytest.raises(ValueError, match="Conflicting sample_from_anchor"):
        DFlash2Qwen3DecoderLayer(c_vllm, config=c_cfg, layer_idx=0)
    eagle_vllm, eagle_cfg = _make_cfg(7, method="eagle", arch="LlamaForCausalLM")
    with pytest.raises(TypeError, match="DFlashSpeculatorConfig"):
        DFlash2Qwen3DecoderLayer(eagle_vllm, config=eagle_cfg, layer_idx=0)
    bad_vllm, bad_cfg = _make_cfg(0, True, subpath="dflash2.config", cls_name="DFlash2SpeculatorConfig")
    with pytest.raises(ValueError, match="must be a positive integer"):
        DFlash2Qwen3DecoderLayer(bad_vllm, config=bad_cfg, layer_idx=0)
    with monkeypatch.context() as mp:
        if HAS_NUM_QUERY_TOKENS:
            mp.setattr(dflash2_cls, "num_query_tokens", property(lambda self: True), raising=False)
        else:
            bad_vllm.speculative_config.num_speculative_tokens = True
        with pytest.raises(ValueError, match="must be a positive integer"):
            DFlash2Qwen3DecoderLayer(bad_vllm, config=bad_cfg, layer_idx=0)


@pytest.mark.parametrize("block_size", [1, 7])
def test_grouped_conv_anchor_first_isolation_and_wrong_width_regression(
    monkeypatch: pytest.MonkeyPatch, block_size: int
) -> None:
    _patch_deps(monkeypatch)
    vllm_cfg, layer_cfg = _make_cfg(block_size, True, subpath="dflash2.config", cls_name="DFlash2SpeculatorConfig")
    layer = DFlash2Qwen3DecoderLayer(vllm_cfg, config=layer_cfg, layer_idx=0)
    torch.manual_seed(42 + block_size)
    with torch.no_grad():
        for conv in (layer.attention_conv, layer.mlp_conv):
            conv.base_kernel.normal_(std=0.125)
            conv.kernel_projection.weight.normal_(std=0.125)
        per_req = [torch.randn(block_size, 8, dtype=torch.float32) for _ in range(2)]
        out, _ = layer(torch.arange(2 * block_size), torch.cat(per_req, dim=0), None)
        expected = torch.cat([layer(torch.arange(block_size), h, None)[0] for h in per_req], dim=0)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
    if block_size == 7:
        hidden = torch.arange(14 * 8, dtype=torch.float32).reshape(14, 8) / 10.0
        delta, base = torch.full((14, 4, 2), 0.25), torch.ones((4, 2, 4))
        req0 = _grouped_conv(hidden[:7], delta[:7], base, block_size=7, num_groups=2, group_size=4, taps=4)
        req1 = _grouped_conv(hidden[7:], delta[7:], base, block_size=7, num_groups=2, group_size=4, taps=4)
        ok = _grouped_conv(hidden, delta, base, block_size=7, num_groups=2, group_size=4, taps=4)
        bad = _grouped_conv(hidden, delta, base, block_size=8, num_groups=2, group_size=4, taps=4)
        torch.testing.assert_close(ok, torch.cat([req0, req1], dim=0), atol=1e-6, rtol=1e-6)
        assert not torch.allclose(bad[7], req1[0])
