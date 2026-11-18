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

from types import SimpleNamespace

import torch
import torch.nn as nn

from vllm_ascend.patch.worker import patch_qwen3_omni_audio_encoder as patched


class _RecordingLayer(nn.Module):
    """Captures what the tower hands to the encoder layers."""

    def forward(self, hidden_states, cu_seqlens, max_seqlen):
        self.cu_seqlens = cu_seqlens
        self.max_seqlen = max_seqlen
        self.hidden_states = hidden_states
        return hidden_states


class _FakeTower(nn.Module):
    """The attributes `Qwen3OmniMoeAudioEncoder.forward` touches.

    Building the real tower needs a vLLM config and a process group; the
    patched function only uses these members, so a stand-in keeps the test to
    the bookkeeping it is meant to cover.
    """

    def __init__(self, num_mel_bins: int = 16, dim: int = 8):
        super().__init__()
        self.n_window = 10
        self.n_window_infer = 40
        self.conv_chunksize = 1024
        self.conv2d1 = nn.Conv2d(1, 4, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(4, 4, 3, 2, padding=1)
        self.conv2d3 = nn.Conv2d(4, 4, 3, 2, padding=1)
        conv_out_dim = 4 * ((((num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2)
        self.conv_out = nn.Linear(conv_out_dim, dim, bias=False)
        self.positional_embedding = SimpleNamespace(positional_embedding=torch.zeros(512, dim))
        self.layers = nn.ModuleList([_RecordingLayer()])
        self.ln_post = nn.LayerNorm(dim)
        self.proj1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(dim, dim)
        self.num_mel_bins = num_mel_bins

    def _get_cnn_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        lengths = input_lengths
        for _ in range(3):
            lengths = (lengths - 1) // 2 + 1
        return lengths

    def compute_attn_mask_seqlen(self, cu_seqlens: torch.Tensor) -> torch.Tensor | None:
        return None


def _without_pinned_memory(monkeypatch):
    """CPU-only torch has no pinned allocator, so take that branch out."""
    monkeypatch.setattr(patched, "PIN_MEMORY", False)


class _LengthsOnDevice:
    """Stands in for lengths that live on the NPU.

    Everything the patched code does with them happens after `.cpu()`, so the
    stand-in only has to record that the move was asked for. Without it the
    test would pass a host tensor and assert nothing, since `.cpu()` on a host
    tensor is a no-op.
    """

    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor
        self.moved_to_host = False

    def cpu(self) -> torch.Tensor:
        self.moved_to_host = True
        return self._tensor


def test_process_audio_input_moves_lengths_to_the_host():
    """The ASR entry point must hand the tower host lengths.

    `audio_output_lengths.tolist()` in the return runs on every request, so
    lengths left on the device cost a synchronization per request.
    """
    seen = {}

    class _AudioTower:
        dtype = torch.float32

        def __call__(self, features, feature_lens, aftercnn_lens):
            seen["feature_lens"] = feature_lens
            seen["aftercnn_lens"] = aftercnn_lens
            return torch.zeros(int(aftercnn_lens.sum()), 4)

    model = SimpleNamespace(audio_tower=_AudioTower())

    lengths = _LengthsOnDevice(torch.tensor([600, 424]))
    audio_input = {
        "input_features": torch.randn(16, 1024),
        "audio_feature_lengths": lengths,
    }

    out = patched._process_audio_input(model, audio_input)

    assert lengths.moved_to_host
    assert [chunk.shape[0] for chunk in out] == seen["aftercnn_lens"].tolist()


def test_audio_encoder_forward_gathers_only_the_real_frames(monkeypatch):
    """The tower emits one row per valid post-CNN frame, and no padding.

    The rows are gathered with host-built indices rather than a boolean mask,
    so a wrong index set shows up here as a wrong number of rows. The window
    lengths handed to the layers are checked too, since they are derived from
    the same host-side bookkeeping.
    """
    _without_pinned_memory(monkeypatch)
    torch.manual_seed(0)
    tower = _FakeTower()

    # Several chunks and windows, with a ragged tail in each item.
    feature_lens = _LengthsOnDevice(torch.tensor([95, 13]))
    aftercnn_lens = _LengthsOnDevice(tower._get_cnn_output_lengths(torch.tensor([95, 13])))
    features = torch.randn(tower.num_mel_bins, 95 + 13)

    with torch.no_grad():
        out = patched._audio_encoder_forward(tower, features, feature_lens, aftercnn_lens)

    assert feature_lens.moved_to_host
    assert aftercnn_lens.moved_to_host

    # 95 frames split into chunks of n_window*2=20 (20,20,20,20,15), then 13.
    chunk_lengths = torch.tensor([20, 20, 20, 20, 15, 13])
    expected_rows = int(tower._get_cnn_output_lengths(chunk_lengths).sum())
    assert out.shape[0] == expected_rows
    assert tower.layers[0].hidden_states.shape[0] == expected_rows

    # window_aftercnn is 3*2=6 here, so the 12-frame item fills two windows and
    # the 2-frame item is a single tail window.
    expected_cu = torch.tensor([0, 6, 6, 2], dtype=torch.int32).cumsum(-1, dtype=torch.int32)
    assert torch.equal(tower.layers[0].cu_seqlens, expected_cu)


def test_audio_encoder_forward_handles_an_exact_chunk_multiple(monkeypatch):
    """A length that divides evenly leaves no tail chunk.

    The tail index is written before the zero-length chunks are repaired, so
    the exact-multiple case is the one that breaks if that order is disturbed.
    """
    _without_pinned_memory(monkeypatch)
    torch.manual_seed(0)
    tower = _FakeTower()

    feature_lens = torch.tensor([40])
    aftercnn_lens = tower._get_cnn_output_lengths(feature_lens)
    features = torch.randn(tower.num_mel_bins, 40)

    with torch.no_grad():
        out = patched._audio_encoder_forward(tower, features, feature_lens, aftercnn_lens)

    assert out.shape[0] == int(tower._get_cnn_output_lengths(torch.tensor([20, 20])).sum())
