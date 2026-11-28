# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from unittest.mock import MagicMock

import pytest
import torch

from tests.ut.base import PytestBase
from vllm_ascend.ops.triton.qfa_kv_scatter import (
    scatter_k_descale,
    scatter_kv_payload,
)

MODULE = "vllm_ascend.ops.triton.qfa_kv_scatter"
HEADS, BLOCKS, DIM, BLOCK_SIZE = 2, 4, 256, 128


def _payload_cache(heads: int, blocks: int) -> torch.Tensor:
    frags = DIM // 32
    k_elems = heads * BLOCK_SIZE * DIM
    kd_bytes = 4 * heads * BLOCK_SIZE
    pitch = 2 * k_elems + kd_bytes
    shape = (blocks, heads, frags, BLOCK_SIZE, 32)
    tail, inner = [], 1
    for d in reversed(shape[1:]):
        tail.append(inner)
        inner *= d
    tail.reverse()
    raw = torch.empty(blocks * pitch, dtype=torch.int8)
    return torch.as_strided(raw, shape, (pitch, *tail))


def _descale_cache(heads: int, blocks: int) -> torch.Tensor:
    k_elems = heads * BLOCK_SIZE * DIM
    kd_bytes = 4 * heads * BLOCK_SIZE
    pitch = 2 * k_elems + kd_bytes
    raw = torch.empty(blocks * pitch, dtype=torch.int8)
    return torch.as_strided(
        raw.view(torch.float32),
        (blocks, heads, BLOCK_SIZE),
        (pitch // 4, BLOCK_SIZE, 1),
        storage_offset=k_elems // 4,
    )


def _slot_index(tokens: int):
    return (torch.arange(tokens), torch.arange(tokens) % BLOCK_SIZE)


class TestQfaScatterDispatch(PytestBase):
    def test_nonempty_payload_launches_with_token_head_grid(self, monkeypatch):
        kernel = MagicMock()
        monkeypatch.setattr(f"{MODULE}._scatter_kv_payload_kernel", kernel)
        tokens = 3
        payload = torch.ones(tokens, HEADS, DIM, dtype=torch.int8)
        cache = _payload_cache(HEADS, BLOCKS)
        scatter_kv_payload(payload, cache, _slot_index(tokens))
        kernel.__getitem__.assert_called_once_with((tokens * HEADS,))
        kernel.__getitem__.return_value.assert_called_once()
        launch_args = kernel.__getitem__.return_value.call_args.args
        assert torch.equal(launch_args[0], payload.view(torch.int8).view(-1))

    def test_nonempty_descale_launches_with_token_grid(self, monkeypatch):
        kernel = MagicMock()
        monkeypatch.setattr(f"{MODULE}._scatter_k_descale_kernel", kernel)
        tokens = 3
        source = torch.ones(tokens, HEADS, dtype=torch.float32)
        cache = _descale_cache(HEADS, BLOCKS)
        scatter_k_descale(source, cache, _slot_index(tokens))
        kernel.__getitem__.assert_called_once_with((tokens,))
        kernel.__getitem__.return_value.assert_called_once()
        launch_args = kernel.__getitem__.return_value.call_args.args
        assert torch.equal(launch_args[0], source)

    @pytest.mark.parametrize("wrapper", ["payload", "descale"])
    def test_empty_batch_returns_without_launch(self, wrapper, monkeypatch):
        kernel = MagicMock()
        monkeypatch.setattr(
            f"{MODULE}._scatter_kv_payload_kernel" if wrapper == "payload" else f"{MODULE}._scatter_k_descale_kernel",
            kernel,
        )
        cache = _payload_cache(HEADS, BLOCKS) if wrapper == "payload" else _descale_cache(HEADS, BLOCKS)
        source = (
            torch.empty(0, HEADS, DIM, dtype=torch.int8)
            if wrapper == "payload"
            else torch.empty(0, HEADS, dtype=torch.float32)
        )
        scatter = scatter_kv_payload if wrapper == "payload" else scatter_k_descale
        scatter(source, cache, _slot_index(0))
        kernel.__getitem__.assert_not_called()

    @pytest.mark.parametrize("wrapper", ["payload", "descale"])
    def test_mismatched_source_size_is_rejected_before_launch(self, wrapper, monkeypatch):
        kernel = MagicMock()
        monkeypatch.setattr(
            f"{MODULE}._scatter_kv_payload_kernel" if wrapper == "payload" else f"{MODULE}._scatter_k_descale_kernel",
            kernel,
        )
        tokens = 2
        if wrapper == "payload":
            source = torch.ones(tokens + 1, HEADS, DIM, dtype=torch.int8)
            scatter = scatter_kv_payload
            cache = _payload_cache(HEADS, BLOCKS)
            message = f"KV payload contains {source.numel()} elements; expected {tokens * HEADS * DIM}."
        else:
            source = torch.ones(tokens + 1, HEADS, dtype=torch.float32)
            scatter = scatter_k_descale
            cache = _descale_cache(HEADS, BLOCKS)
            message = f"K descale contains {source.numel()} elements; expected {tokens * HEADS}."
        with pytest.raises(ValueError, match=message):
            scatter(source, cache, _slot_index(tokens))
        kernel.__getitem__.assert_not_called()

    def test_non_pow2_heads_are_rejected(self, monkeypatch):
        kernel = MagicMock()
        monkeypatch.setattr(f"{MODULE}._scatter_kv_payload_kernel", kernel)
        payload = torch.ones(1, 3, DIM, dtype=torch.int8)
        cache = _payload_cache(3, BLOCKS)
        with pytest.raises(ValueError, match="power-of-two"):
            scatter_kv_payload(payload, cache, _slot_index(1))
        kernel.__getitem__.assert_not_called()
