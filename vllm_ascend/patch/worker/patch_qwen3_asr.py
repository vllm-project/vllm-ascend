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
#
# vllm-ascend monkeypatch: Qwen3-ASR audio_tower NPU graph capture (masked forward).
#
# Runtime audio_tower = Qwen3OmniMoeAudioEncoder
# (vllm/model_executor/models/qwen3_omni_moe_thinker.py). We capture the
# encoder layers + head (ln_post/proj1/act/proj2) in an NPUGraph and replay it.
#
# KEY DESIGN (fixed-width grid + attention mask + crop):
# The model's attention is varlen cu_seqlens, which does cu_seqlens.cpu()->host
# ints (actual_seq_lengths) inside the graph -> (a) breaks fullgraph capture,
# (b) bakes cu_seqlens values -> one graph per distinct shape. We instead
# RE-IMPLEMENT the layer attention as batched FIA with a boolean attention_mask
# (NO cu_seqlens, NO host-sync):
#   * host-eager prelude: conv -> packed hidden_states (B, dim); place into a
#     fixed-width grid (NUM_WINDOWS, ws_grid, dim); build attn_mask
#     (NUM_WINDOWS, ws_grid) bool: real frames True, tail/padding False.
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
# (the padding frames per row are masked, FIA still computes the full
# ws_grid^2 attention matrix).
#
# Conv stays host-eager; layers+head graphed. Installing this patch = graph
# capture always on (no toggle).
#
# The prelude around the captured graph keeps every length, window and index
# computation on the host. Reading any of them back from the NPU (`.item()`,
# `.tolist()`, a boolean-mask gather or scatter, whose output size is only
# known on device) drains the queue, and the audio tower runs while other
# requests are decoding on the same engine, so each read stalls them too.
# Indices are built on the host and copied up with `async_tensor_h2d`.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu  # noqa: F401
from vllm.model_executor.models.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeAudioEncoder,
)
from vllm.utils.torch_utils import async_tensor_h2d

# NUM_WINDOWS bins = total number of attention windows across the WHOLE vllm
# batch (sum over audios of ceil(aftercnn_i / ws_real)). One graph per bin;
# runtime picks the next-larger bin and pads (supports any concurrency: 1,2,4,
# 8,16,32,64,...). ~4 windows/audio => C=64 -> ~256. Keep <=~68 graphs.
_BINS = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
# Grid window width for FIA (seq_len). ws_real = semantic window_aftercnn (104);
# ws_grid is the (fixed) grid width each window is placed into. 128 pads every
# window up to an Ascend-aligned width (24 masked padding frames); 104 would be
# no extra padding.
_WS_GRID = 128


_orig_forward = Qwen3OmniMoeAudioEncoder.forward


def _window_aftercnn(self):
    """Semantic window width after the CNN. Constant per tower, so cache it.

    The lengths are a function of the config, not of the request, so this runs
    on the host once instead of reading a fresh device tensor on every call.
    """
    ws = self.__dict__.get("_q3a_ws_real")
    if ws is None:
        full = self._get_cnn_output_lengths(torch.tensor([self.n_window * 2]))
        ws = int(full.max()) * (self.n_window_infer // (self.n_window * 2))
        self.__dict__["_q3a_ws_real"] = ws
    return ws


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
        # H is the LOCAL head count, so H * D is this rank's share of the
        # channels. out_proj is row-parallel and takes that, not embed_dim.
        H, D = self._num_heads, self._head_dim

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
            a = a.view(W, WS, H * D)
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
    """Conv -> packed hidden_states (B, dim); lengths and masks built on the host.

    `feature_lens` is expected on the host. The chunk lengths, the post-CNN
    lengths and the padding mask are all derived there, so the only transfer
    is the row index list for the final gather.
    """
    chunk_size = self.n_window * 2
    chunk_num = torch.ceil(feature_lens / chunk_size).long()
    chunk_lengths = torch.full((int(chunk_num.sum()),), chunk_size, dtype=torch.long)
    tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail_chunk_index] = feature_lens % chunk_size
    chunk_lengths[chunk_lengths == 0] = chunk_size

    chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
    padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2).unsqueeze(1)
    feature_lens_after_cnn = self._get_cnn_output_lengths(chunk_lengths)
    max_len_after_cnn = int(feature_lens_after_cnn.max())
    padded_mask_after_cnn = torch.arange(max_len_after_cnn).unsqueeze(0) < feature_lens_after_cnn.unsqueeze(1)

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
    # Gather the real frames with host-built row ids: a boolean-mask index
    # would have to read its output size back from the device.
    rows = async_tensor_h2d(
        padded_mask_after_cnn.flatten().nonzero().squeeze(1).tolist(),
        dtype=torch.long,
        device=e.device,
    )
    return e.flatten(0, 1).index_select(0, rows)  # (B, dim)


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
    # The eager fallback below is vLLM's own forward, which expects the lengths
    # where the caller put them, so keep the originals before moving to host.
    orig_feature_lens, orig_aftercnn_lens = feature_lens, aftercnn_lens
    feature_lens = feature_lens.cpu()
    aftercnn_lens = aftercnn_lens.cpu()

    # Window layout for the whole batch, on the host: one row per window, each
    # holding the real frame count of that window (tail windows are shorter).
    real_w_list = []
    for a in aftercnn_lens.tolist():
        real_w_list += [ws_real] * (a // ws_real)
        if a % ws_real:
            real_w_list.append(a % ws_real)
    total_windows = len(real_w_list)
    bins = sorted(set(_BINS))
    target_w = next((w for w in bins if w >= total_windows), None)
    cache = self.__dict__.get("_q3a_graphs", {})
    if target_w is None or target_w not in cache:
        # out of range or capture failed -> original eager forward
        return _orig_forward(self, input_features, orig_feature_lens, orig_aftercnn_lens)

    hidden_states = _host_eager_meta(self, input_features, feature_lens, aftercnn_lens)
    graphed, static_grid, static_mask = cache[target_w]
    device = hidden_states.device
    real_w_list += [0] * (target_w - total_windows)

    # Place the packed real frames at the start of each grid row. index_copy_
    # with host-built rows avoids the boolean-mask scatter, which would read
    # the number of selected rows back from the device.
    real_mask = torch.arange(ws_grid).unsqueeze(0) < torch.tensor(real_w_list).unsqueeze(1)
    rows = async_tensor_h2d(
        real_mask.view(-1).nonzero().squeeze(1).tolist(),
        dtype=torch.long,
        device=device,
    )
    static_grid.zero_()
    static_grid.view(-1, static_grid.shape[-1]).index_copy_(0, rows, hidden_states)

    # attn_mask (True = masked): real queries skip padding keys; padding
    # queries attend to everything (avoids all-masked NaN rows).
    rw = async_tensor_h2d(real_w_list, dtype=torch.long, device=device).view(target_w, 1, 1)
    ij = self.__dict__.get("_q3a_ij")
    if ij is None or ij[0].shape[1] != ws_grid:
        ar = torch.arange(ws_grid, device=device)
        ij = (ar.view(1, ws_grid, 1), ar.view(1, 1, ws_grid))
        self.__dict__["_q3a_ij"] = ij
    static_mask.copy_(((ij[0] < rw) & (ij[1] >= rw)).view(target_w, 1, ws_grid, ws_grid))

    with torch.inference_mode():
        out_padded = graphed(static_grid, static_mask)
    # Gather the real frames back. The result is a new tensor, so the graph's
    # output buffer can be reused by the next replay.
    return out_padded.reshape(-1, out_padded.shape[-1]).index_select(0, rows)


Qwen3OmniMoeAudioEncoder.forward = _forward_graph
