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

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch_npu

from vllm_ascend.patch.worker import patch_qwen3_asr


@pytest.fixture
def layers_wrapper(monkeypatch):
    """Give the graph body a stand-in FIA.

    conftest already provides a torch_npu stand-in on a CPU runner; the graph
    body additionally calls npu_fused_infer_attention_score, so return what it
    returns -- this rank's heads, BSND -- which is the geometry the reshape
    under test has to respect. The patch looks the symbol up at call time, so
    no reimport is needed.
    """

    def _fia(q, k, v, **kwargs):
        return (q,)

    monkeypatch.setattr(torch_npu, "npu_fused_infer_attention_score", _fia, raising=False)
    return patch_qwen3_asr._LayersWrapper


class _FakeAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, tp_size: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_local_heads = num_heads // tp_size
        self.scaling = self.head_dim**-0.5
        local = self.num_local_heads * self.head_dim
        self.qkv = nn.Linear(embed_dim, 3 * local, bias=False)
        self.out_proj = nn.Linear(local, embed_dim, bias=False)


class _FakeLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, tp_size: int):
        super().__init__()
        self.self_attn = _FakeAttention(embed_dim, num_heads, tp_size)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = nn.GELU()


class _FakeTower(nn.Module):
    def __init__(self, embed_dim: int = 32, num_heads: int = 4, tp_size: int = 1):
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer(embed_dim, num_heads, tp_size)])
        self.ln_post = nn.LayerNorm(embed_dim)
        self.proj1 = nn.Linear(embed_dim, embed_dim)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(embed_dim, embed_dim)


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_layers_wrapper_keeps_local_head_geometry(layers_wrapper, tp_size):
    """The attention output carries this rank's heads, not the global width.

    FIA returns `num_local_heads * head_dim` channels per rank, which equals
    `embed_dim` only at TP1. Reshaping to the global `embed_dim` instead makes
    capture raise, and `_precompile_graphs` swallows that per bin, so the graph
    silently never runs.
    """
    embed_dim, num_heads = 32, 4
    tower = _FakeTower(embed_dim, num_heads, tp_size)
    wrapper = layers_wrapper(tower)

    num_windows, ws_grid = 2, 8
    grid = torch.randn(num_windows, ws_grid, embed_dim)
    mask = torch.zeros(num_windows, 1, ws_grid, ws_grid, dtype=torch.bool)

    with torch.no_grad():
        out = wrapper(grid, mask)

    assert out.shape == (num_windows, ws_grid, embed_dim)


def test_layers_wrapper_reads_the_local_head_count(layers_wrapper):
    """The wrapper takes its head count from the sharded attention layer."""
    tower = _FakeTower(embed_dim=32, num_heads=4, tp_size=2)
    wrapper = layers_wrapper(tower)

    assert wrapper._num_heads == 2
    assert wrapper._head_dim == 8
    # The global width stays available for the captured grid, which holds
    # d_model-wide hidden states rather than attention output.
    assert wrapper._embed_dim == 32
