#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Format contract of the TurboQuant 4-bit latent slot.

These run on CPU against ``compress``, the reference implementation. The fused
device path (``compress_kernel`` plus the dequant inside the SFA operator) has to
produce and consume exactly this layout.
"""

import pytest
import torch

import vllm_ascend.ops.tq_latent_store as tq_latent_store
from vllm_ascend.ops.tq_latent_store import (
    base_slot_size,
    compress,
    fused_slot_size,
    had_fwd,
    had_inv,
    lutsq,
    packed_bytes,
)

HEAD_DIM = 512
ROPE_HEAD_DIM = 64


def _unpack_nibbles(slot: torch.Tensor, head_dim: int = HEAD_DIM) -> torch.Tensor:
    """Undo the int16 nibble packing the kernel reads back with a Cast."""
    packed = packed_bytes(head_dim)
    lo = slot[:, 0:packed:2].int()
    hi = slot[:, 1:packed:2].int()
    nibbles = torch.stack([lo & 0xF, (lo >> 4) & 0xF, hi & 0xF, (hi >> 4) & 0xF], dim=-1)
    return nibbles.reshape(slot.shape[0], head_dim)


def test_slot_sizes() -> None:
    # 256 nibble bytes + 2 B norm, padded to 64 B; the fused slot drops the
    # padding and carries the bf16 rope half instead.
    assert packed_bytes(HEAD_DIM) == 256
    assert base_slot_size(HEAD_DIM) == 320
    assert fused_slot_size(HEAD_DIM, ROPE_HEAD_DIM) == 386


def test_head_dim_must_be_a_power_of_two() -> None:
    with pytest.raises(ValueError):
        packed_bytes(384)


def test_compress_produces_the_expected_slot_layout() -> None:
    torch.manual_seed(0)
    latent = torch.randn(5, HEAD_DIM)
    slot = compress(latent)

    assert slot.shape == (5, base_slot_size(HEAD_DIM))
    assert slot.dtype == torch.uint8

    nibbles = _unpack_nibbles(slot)
    assert bool(((nibbles >= 0) & (nibbles < 16)).all())

    # Bytes [256, 258) hold the fp16 L2 norm the attention kernel rescales by.
    stored_norm = slot[:, 256:258].contiguous().view(torch.float16).float().flatten()
    expected_norm = latent.norm(dim=1)
    torch.testing.assert_close(stored_norm, expected_norm, rtol=2e-3, atol=0.0)

    # The tail is padding and must stay zeroed; the operator reads the full slot.
    assert bool((slot[:, 258:] == 0).all())


def test_lutsq_is_symmetric_under_nibble_swap() -> None:
    # lutsq[b] is the sum of the squared centroids of b's two nibbles, so swapping
    # them must not change the entry. A byte-order slip here would silently corrupt
    # every per-token scale.
    lut = lutsq(torch.device("cpu"))
    assert lut.shape == (256,)
    assert lut.dtype == torch.float32
    byte = torch.arange(256)
    swapped = ((byte & 0xF) << 4) | ((byte >> 4) & 0xF)
    torch.testing.assert_close(lut, lut[swapped])
    assert bool((lut > 0).all())


def test_centroids_match_the_cann_sparse_attention_codebook() -> None:
    tq_latent_store._build(torch.device("cpu"))
    expected = torch.tensor(
        [
            -0.12091285,
            -0.09111122,
            -0.07112455,
            -0.05513602,
            -0.04132067,
            -0.02874970,
            -0.01700489,
            -0.00568677,
            0.00547294,
            0.01680406,
            0.02857605,
            0.04108622,
            0.05492980,
            0.07101817,
            0.09115373,
            0.12037795,
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(tq_latent_store._CENT, expected, rtol=0.0, atol=0.0)


def test_hadamard_round_trip_is_the_identity() -> None:
    # The query is rotated before attention and the output rotated back, so the
    # forward and inverse transforms must compose to the identity.
    torch.manual_seed(0)
    x = torch.randn(4, HEAD_DIM)
    torch.testing.assert_close(had_inv(had_fwd(x)), x, rtol=0.0, atol=1e-4)
