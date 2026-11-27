# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from vllm.models.hy_v4_flash.reference import pooled_keys

from vllm_ascend.models.hy_v4_flash import attention, index_pool_fusions
from vllm_ascend.models.hy_v4_flash.index_pool_select import select_pool_candidates


def test_decode_pool_keys_keep_position_encoding(monkeypatch):
    monkeypatch.setattr(attention, "get_flash_attention_metadata", lambda: SimpleNamespace(prefill_reqs=(False,)))
    # Isolate the control flow from the device-only pooling kernel.
    monkeypatch.setattr(
        index_pool_fusions,
        "pool_norm",
        lambda raw, projection, ape, gamma, eps: pooled_keys(raw, projection[:128], projection[128:], ape, gamma, eps),
    )
    raw = torch.randn(8, 128, dtype=torch.bfloat16)
    weight = torch.randn(128, 128) * 0.05
    gamma, ape = torch.ones(128), torch.randn(4, 128)
    calls = []

    def rotate(value, positions, **kwargs):
        calls.append(positions.clone())
        phase = positions.float()[:, None] * 10000.0 ** (-torch.arange(0, 64, 2).float() / 64)
        nope, rotary = value.split((64, 64), dim=-1)
        even, odd = rotary.float()[..., ::2], rotary.float()[..., 1::2]
        return torch.cat(
            (
                nope,
                (even * phase.cos() - odd * phase.sin()).to(value.dtype),
                (odd * phase.cos() + even * phase.sin()).to(value.dtype),
            ),
            dim=-1,
        )

    spec = SimpleNamespace(
        ratio=4,
        key_value_projection=SimpleNamespace(weight=weight),
        key_gate_projection=SimpleNamespace(weight=weight * 0.7),
        intra_group_ape=ape,
        key_norm=SimpleNamespace(weight=gamma),
        eps=1e-5,
        rotate=rotate,
        _encode_fp8=lambda value, **kw: value,
    )
    positions = torch.arange(4096, 4104)
    pooled = pooled_keys(raw.reshape(2, 4, 128), weight, weight * 0.7, ape, gamma, spec.eps)
    expected = rotate(pooled, positions[::4])
    calls.clear()

    actual = attention.FlashIndexPool.encode_keys(spec, raw, positions, full_graph=True)

    assert len(calls) == 1
    assert torch.equal(calls[0], positions[::4])
    assert torch.equal(actual, expected)
    moved = attention.FlashIndexPool.encode_keys(spec, raw, positions + 97, full_graph=True)
    assert not torch.equal(actual, moved)


@pytest.mark.parametrize("batch_size", [2, 4, 8])
def test_mixed_short_long_and_padding_keep_candidate_sets(batch_size):
    visible = torch.tensor([-1, 0, 10, 2044, 2047, 2048] * batch_size)
    scores = torch.full((visible.numel(), 1024), -torch.inf)
    long_rows = visible // 4 > 511
    scores[long_rows, :512] = torch.arange(512).float()

    actual = select_pool_candidates(scores, visible, 1024, 4, 511)

    for position, row in zip(visible.tolist(), actual):
        if position < 2048:
            expected = torch.arange(2048).masked_fill(torch.arange(2048) > position, -1)
        else:
            # Scores select groups 1..511; group 512 contributes the tail.
            expected = torch.cat((torch.arange(4, 2048), torch.tensor([2048, -1, -1, -1])))
        assert torch.equal(row, expected)


@pytest.mark.parametrize("visible", [0, 10, 11, 12, 2043, 2044, 2047])
def test_batched_short_indexpool_keeps_all_visible_history(visible):
    scores = torch.full((6, 1024), -torch.inf)
    positions = torch.full((6,), visible, dtype=torch.int64)

    actual = select_pool_candidates(scores, positions, 1024, 4, 511)

    expected = torch.arange(2048).expand(6, -1).clone()
    expected.masked_fill_(expected > positions[:, None], -1)
    assert torch.equal(actual, expected)
