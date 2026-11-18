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
# Adapted from https://gitcode.com/Ascend-SACT/Qwen3-ASR-1.7B
# vllm-ascend monkeypatch: Qwen3-ASR audio_tower NPU graph capture (masked forward).
#
# Runtime audio_tower = Qwen3OmniMoeAudioEncoder
# (vllm/model_executor/models/qwen3_omni_moe_thinker.py:321). We capture the
# encoder layers + head (ln_post/proj1/act/proj2) in an NPUGraph and replay it.
#
# KEY DESIGN (fixed-width grid + attention mask + crop), per the reference pattern:
# The model's attention is varlen cu_seqlens, which does cu_seqlens.cpu()->host
# ints (actual_seq_lengths) inside the graph -> (a) breaks fullgraph capture,
# (b) bakes cu_seqlens values -> one graph per distinct shape. We instead
# RE-IMPLEMENT the layer attention as batched FIA with a boolean attention_mask
# (NO cu_seqlens, NO host-sync):
#   * host-eager prelude: conv -> packed hidden_states (B, dim) [boolean gather,
#     data-dependent, stays eager]; place into a fixed-width grid
#     (NUM_WINDOWS, ws_grid, dim); build attn_mask (NUM_WINDOWS, ws_grid) bool:
#     real frames True, tail/padding False.
#   * graph body: 24x (qkv -> FIA(q,k,v,attn_mask=mask, BSND) -> out_proj -> LN
#     -> fc1/act/fc2) + ln_post/proj1/act/proj2. Static shapes; mask is a TENSOR
#     graph input -> values re-read each replay -> ONE graph per NUM_WINDOWS bin
#     covers arbitrary audio length (per-bin). Tail real frames attend only to
#     real frames (mask), padding frames are cropped out.
#   * crop output via real_mask gather.
#
# Grid width ws_grid: semantic window ws_real = window_aftercnn (104). ws_grid may
# equal ws_real (no extra padding, only tail windows <104 are masked within the
# row) or be larger (e.g. 128) to pad every window up to an Ascend-aligned width
# (the 24 padding frames per row are masked, FIA still computes the full
# ws_grid^2 attention matrix). 104-vs-128 is under same-card re-verification;
# default 128 restored for testing. Set VLLM_QWEN3_ASR_WS_GRID=104 for no pad.
#
# Conv stays host-eager; layers+head graphed. Installing this patch = graph
# capture always on (no toggle). Configurable via VLLM_QWEN3_ASR_BINS /
# VLLM_QWEN3_ASR_WS_GRID.
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu  # noqa: F401
from vllm.model_executor.models.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeAudioEncoder,
)

# NUM_WINDOWS bins = total number of attention windows across the WHOLE vllm
# batch (sum over audios of ceil(aftercnn_i / ws_real)). One graph per bin;
# runtime picks the next-larger bin and pads (supports any concurrency: 1,2,4,
# 8,16,32,64,...). ~4 windows/audio => C=64 -> ~256. Keep <=~68 graphs.
_BINS = [
    int(x)
    for x in os.environ.get("VLLM_QWEN3_ASR_BINS", "1,2,4,8,12,16,24,32,48,64,96,128,192,256").split(",")
    if x.strip()
]
# Grid window width for FIA (seq_len). ws_real = semantic window_aftercnn (104);
# ws_grid is the (fixed) grid width each window is placed into. Default 128
# pads every window up to an Ascend-aligned width (24 masked padding frames);
# set to 104 for no extra padding. See header note on the 104-vs-128 question.
_WS_GRID = int(os.environ.get("VLLM_QWEN3_ASR_WS_GRID", "128"))


_orig_forward = Qwen3OmniMoeAudioEncoder.forward


def _window_aftercnn(self):
    full = self._get_cnn_output_lengths(torch.tensor([self.n_window * 2], dtype=torch.long, device=self.device))
    return int(full.max().item()) * (self.n_window_infer // (self.n_window * 2))


class _LayersWrapper(nn.Module):
    """Encoder layers + head with batched masked FIA attention (no cu_seqlens).

    forward(hidden_states, attn_mask):
      hidden_states: (W, WS, dim)   fixed-shape grid (W = NUM_WINDOWS;
                                     WS = ws_grid, e.g. 128; semantic 104 padded)
      attn_mask:     (W, 1, WS, WS) bool  (FIA: True = masked)
    """

    def __init__(self, tower: Qwen3OmniMoeAudioEncoder):
        super().__init__()
        self.tower = tower
        lay = tower.layers[0].self_attn
        self._num_heads = lay.num_local_heads
        self._head_dim = lay.head_dim
        self._scale = lay.scaling
        self._embed_dim = lay.embed_dim

    def forward(self, hidden_states, attn_mask):
        # hidden_states: (W, WS, dim); attn_mask: (W, 1, WS, WS) bool.
        W = hidden_states.shape[0]
        WS = hidden_states.shape[1]
        H, D, dim = self._num_heads, self._head_dim, self._embed_dim

        def _linear_out(mod, x):
            r = mod(x)
            return r[0] if isinstance(r, (tuple, list)) else r

        h = hidden_states
        for layer in self.tower.layers:
            residual = h
            h = layer.self_attn_layer_norm(h)
            qkv = _linear_out(layer.self_attn.qkv, h)
            q, k, v = qkv.chunk(3, dim=-1)
            # chunk yields strided (non-contiguous) slices; FIA needs contiguous
            q = q.contiguous().view(W, WS, H, D)  # BSND
            k = k.contiguous().view(W, WS, H, D)
            v = v.contiguous().view(W, WS, H, D)
            ret = torch_npu.npu_fused_infer_attention_score(
                q,
                k,
                v,
                atten_mask=attn_mask,
                input_layout="BSND",
                actual_seq_lengths=[WS] * W,
                actual_seq_lengths_kv=[WS] * W,
                num_key_value_heads=H,
                num_heads=H,
                scale=self._scale,
                sparse_mode=0,
            )
            a = ret[0] if isinstance(ret, (tuple, list)) else ret
            a = a.view(W, WS, dim)
            h = _linear_out(layer.self_attn.out_proj, a)
            h = residual + h

            residual = h
            h = layer.final_layer_norm(h)
            h = _linear_out(layer.fc1, h)
            h = layer.activation_fn(h)
            h = _linear_out(layer.fc2, h)
            h = residual + h
        h = self.tower.ln_post(h)
        h = _linear_out(self.tower.proj1, h)
        h = self.tower.act(h)
        h = _linear_out(self.tower.proj2, h)
        return h


def _host_eager_meta(self, input_features, feature_lens, aftercnn_lens):
    """Conv -> packed hidden_states (B, dim). Data-dependent, host-eager."""
    chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()
    chunk_lengths = torch.tensor([self.n_window * 2] * chunk_num.sum(), dtype=torch.long, device=feature_lens.device)
    tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
    chunk_lengths[chunk_lengths == 0] = self.n_window * 2

    chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
    padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2).unsqueeze(1)
    feature_lens_after_cnn = self._get_cnn_output_lengths(chunk_lengths)
    max_len_after_cnn = feature_lens_after_cnn.max().item()
    indices = torch.arange(max_len_after_cnn, device=padded_feature.device)
    padded_mask_after_cnn = indices.unsqueeze(0) < feature_lens_after_cnn.unsqueeze(1)

    if padded_feature.size(0) <= self.conv_chunksize:
        e = F.gelu(self.conv2d1(padded_feature))
        e = F.gelu(self.conv2d2(e))
        e = F.gelu(self.conv2d3(e))
    else:
        es = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            x = F.gelu(self.conv2d1(chunk))
            x = F.gelu(self.conv2d2(x))
            x = F.gelu(self.conv2d3(x))
            es.append(x)
        e = torch.cat(es, dim=0)
    b, c, f, t = e.size()
    e = self.conv_out(e.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))
    pe = self.positional_embedding.positional_embedding[: e.shape[1], :].unsqueeze(0).to(e.dtype)
    e = e + pe
    hidden_states = e[padded_mask_after_cnn]  # (B, dim)
    return hidden_states


def _build_batched_grid_mask(self, hidden_states, aftercnn_lens, num_windows, ws_real, ws_grid, device):
    """Build ONE grid for all audios in the vllm batch (no per-audio split).

    hidden_states: (B_total, dim) packed real aftercnn frames, concat over
      audios (audio 0 frames, then audio 1, ...).
    aftercnn_lens: (N,) per-audio aftercnn count.
    ws_real: semantic window_aftercnn (=104); windowing unit.
    ws_grid: fixed grid width (>= ws_real; 128 pads each window, 104 = no pad).
    -> grid (num_windows, ws_grid, dim), attn_mask (num_windows,1,ws_grid,ws_grid)
       bool (FIA: True=masked), real_mask_flat (num_windows*ws_grid,) bool for crop.

    Each audio's frames are windowed into ws_real windows + a tail; each window
    is placed at the start of a ws_grid-wide grid row (rest zero-pad). Cross-audio
    isolation is automatic: each window is its own grid row, so attention never
    crosses audio boundaries. real_mask marks the real frames (used both to
    scatter input into the grid and to gather output back)."""
    B_total = hidden_states.shape[0]
    dim = hidden_states.shape[-1]

    aftercnn = aftercnn_lens.tolist()  # one D2H sync for the whole batch
    real_w_list = []
    for a in aftercnn:
        num_full = a // ws_real
        tail = a % ws_real
        real_w_list += [ws_real] * num_full
        if tail > 0:
            real_w_list += [tail]
    total_windows = len(real_w_list)
    real_w = torch.tensor(real_w_list, dtype=torch.long, device=device)
    if num_windows > total_windows:
        real_w = torch.cat([real_w, torch.zeros(num_windows - total_windows, dtype=torch.long, device=device)])

    # real_mask (num_windows, ws_grid): True where pos < real_w[wi] (real frame)
    real_mask = torch.arange(ws_grid, device=device).unsqueeze(0) < real_w.unsqueeze(1)  # (num_windows, ws_grid)

    # scatter packed real frames into the grid's real positions (order preserved)
    grid_flat = torch.zeros(num_windows * ws_grid, dim, dtype=hidden_states.dtype, device=device)
    grid_flat[real_mask.view(-1)] = hidden_states
    grid = grid_flat.view(num_windows, ws_grid, dim)

    # attn_mask (num_windows,1,ws_grid,ws_grid): True=masked. Real q (i<rw)
    # mask padding k (j>=rw); padding q attend all (avoid all-masked nan).
    i = torch.arange(ws_grid, device=device).view(1, ws_grid, 1)
    j = torch.arange(ws_grid, device=device).view(1, 1, ws_grid)
    rw = real_w.view(num_windows, 1, 1)
    mask = (i < rw) & (j >= rw)  # True=masked
    mask = mask.view(num_windows, 1, ws_grid, ws_grid)
    return grid, mask, real_mask.view(-1)


def _capture_one(self, num_windows, ws_grid, device, dt):
    """Capture _LayersWrapper for one NUM_WINDOWS bin.
    Returns (graphed, static_grid, static_mask)."""
    wrapper = _LayersWrapper(self)
    dim = wrapper._embed_dim
    dummy_grid = torch.randn(num_windows, ws_grid, dim, dtype=dt, device=device)
    # all-False = attend all (FIA: True=masked); valid warmup, no nan
    dummy_mask = torch.zeros(num_windows, 1, ws_grid, ws_grid, dtype=torch.bool, device=device)
    graphed = torch_npu.npu.make_graphed_callables(wrapper, (dummy_grid, dummy_mask), num_warmup_iters=3)
    return graphed, dummy_grid, dummy_mask


def _precompile_graphs(self):
    if getattr(self, "_q3a_precompiled", False):
        return
    self._q3a_precompiled = True
    ws_real = _window_aftercnn(self)
    ws_grid = _WS_GRID
    device = self.device
    dt = self.dtype
    cache = self.__dict__.setdefault("_q3a_graphs", {})
    for w in sorted(set(_BINS)):
        try:
            g, sg, sm = _capture_one(self, w, ws_grid, device, dt)
            cache[w] = (g, sg, sm)
        except Exception:
            pass  # skip this bin; runtime falls back to eager for it


def _forward_graph(self, input_features, feature_lens, aftercnn_lens):
    _precompile_graphs(self)  # one-time startup capture (no-op after first)
    ws_real = _window_aftercnn(self)
    ws_grid = _WS_GRID

    # batched conv (one pass over all audios' chunks) -> packed (B_total, dim)
    hidden_states = _host_eager_meta(self, input_features, feature_lens, aftercnn_lens)

    # total attention windows across all audios (bin key)
    aftercnn = aftercnn_lens.tolist()
    total_windows = sum((a + ws_real - 1) // ws_real for a in aftercnn)
    bins = sorted(set(_BINS))
    target_w = next((w for w in bins if w >= total_windows), None)
    cache = self.__dict__.get("_q3a_graphs", {})

    if target_w is None or target_w not in cache:
        # out of range or capture failed -> original eager forward
        return _orig_forward(self, input_features, feature_lens, aftercnn_lens)

    graphed, static_grid, static_mask = cache[target_w]
    grid, mask, real_mask_flat = _build_batched_grid_mask(
        self, hidden_states, aftercnn_lens, target_w, ws_real, ws_grid, hidden_states.device
    )
    static_grid.copy_(grid)
    static_mask.copy_(mask)
    with torch.inference_mode():
        out_padded = graphed(static_grid, static_mask).clone()
    # gather real frames back from per-window real positions -> (B_total, dim)
    out = out_padded.reshape(-1, out_padded.shape[-1])[real_mask_flat]
    return out


Qwen3OmniMoeAudioEncoder.forward = _forward_graph
