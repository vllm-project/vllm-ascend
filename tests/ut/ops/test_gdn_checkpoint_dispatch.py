# SPDX-License-Identifier: Apache-2.0
"""Checkpoint integration tests using production functions and CPU tensors."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ops import gdn_checkpoint
from vllm_ascend.ops.triton.fla import chunk


def chunk_metadata():
    return SimpleNamespace(
        **{
            name: torch.zeros(1, dtype=torch.int32)
            for name in (
                "chunk_indices_chunk64",
                "chunk_offsets_chunk64",
                "chunk_indices_large_block",
                "block_indices_cumsum",
            )
        }
    )


@pytest.mark.parametrize(
    "missing",
    [None, "chunk_indices_chunk64", "chunk_offsets_chunk64", "chunk_indices_large_block", "block_indices_cumsum"],
)
def test_checkpoint_requires_device_metadata(missing):
    metadata = chunk_metadata() if missing else None
    if missing:
        setattr(metadata, missing, None)
    with pytest.raises(ValueError, match="prebuilt device chunk metadata"):
        chunk.chunk_gated_delta_rule_checkpoint(*(None for _ in range(7)), metadata, None, None)


def test_checkpoint_normalizes_inputs_preserving_write_buffer(monkeypatch):
    inputs = [torch.empty(1, 3, 4, 8).transpose(1, 2) for _ in range(6)]
    starts = torch.tensor([0, 0, 4, 0], dtype=torch.int32)[::2]
    offsets = torch.tensor([0, 9], dtype=torch.int32)[::2]
    workspace = torch.empty(1, 3, 8, 8)
    expected = (torch.empty(1), torch.empty(1))
    monkeypatch.setattr(chunk, "l2norm_fwd", lambda tensor: tensor)

    def forward(q, k, v, g, beta, scale, initial, final, **kwargs):
        assert all(
            t.is_contiguous() for t in (q, k, v, g, beta, initial, kwargs["cu_seqlens"], kwargs["checkpoint_offsets"])
        )
        assert kwargs["checkpoint_output"] is workspace
        assert kwargs["use_triton_recurrence"]
        return None, expected[0], None, expected[1], None, None, None

    monkeypatch.setattr(chunk, "chunk_gated_delta_rule_fwd", forward)
    actual = chunk.chunk_gated_delta_rule_checkpoint(*inputs, starts, chunk_metadata(), workspace, offsets)
    assert actual[0] is expected[0]
    assert actual[1] is expected[1]


@pytest.mark.parametrize("creates_checkpoint", [False, True])
def test_checkpoint_export_uses_metadata_and_cache_dtype(monkeypatch, creates_checkpoint):
    initial = torch.zeros(1, 2, 8, 4)
    final = torch.zeros(1, 2, 4, 8)
    output = torch.empty(1)
    workspace = torch.empty_like(final)
    checkpoint = SimpleNamespace(checkpoint_offsets=torch.tensor([64])) if creates_checkpoint else None
    metadata = SimpleNamespace(
        checkpoint=checkpoint,
        prefill_query_start_loc=torch.tensor([0, 128]),
        non_spec_prefill_metadata=SimpleNamespace(chunk=chunk_metadata()),
    )
    layer = SimpleNamespace(
        _ascend_checkpoint_workspace=workspace,
        kv_cache=(torch.empty(2, 3, 4), torch.empty(2, 2, 8, 4, dtype=torch.bfloat16)),
        _checkpoint_exporter=SimpleNamespace(export=Mock()),
    )
    backend = Mock(return_value=(output, final))
    monkeypatch.setattr(gdn_checkpoint, "chunk_gated_delta_rule_checkpoint", backend)
    raw = torch.empty(128, 4)
    result, state = gdn_checkpoint.run_gdn_checkpoint_prefill(
        layer, metadata, raw, *(torch.empty(1) for _ in range(5)), initial
    )
    assert result is output
    torch.testing.assert_close(state, final.transpose(-1, -2))
    args, kwargs = backend.call_args
    assert args[5].data_ptr() == initial.data_ptr()
    assert args[6] is metadata.prefill_query_start_loc
    assert kwargs["checkpoint_state_dtype"] == torch.bfloat16
    assert (args[8] is None) == (checkpoint is None)
    assert layer._checkpoint_exporter.export.call_count == int(creates_checkpoint)
    if creates_checkpoint:
        assert layer._checkpoint_exporter.export.call_args.kwargs["raw_qkv"] is raw


@pytest.mark.parametrize(
    "transfer,drop,lookahead,expected",
    [
        (False, None, 0, True),
        (False, False, 0, True),
        (False, True, 1, True),
        (False, True, 0, False),
        (True, None, 0, False),
        (True, True, 1, False),
    ],
)
def test_boundary_policy(transfer, drop, lookahead, expected):
    config = SimpleNamespace(
        kv_transfer_config=object() if transfer else None,
        speculative_config=None if drop is None else SimpleNamespace(use_eagle_block_drop=lambda: drop),
        num_hash_lookahead_tokens=lookahead,
    )
    assert gdn_checkpoint._supports_checkpoint_boundary_policy(config) is expected


def test_workspace_sharing_and_fallback():
    context: dict[str, SimpleNamespace] = {}
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        compilation_config=SimpleNamespace(static_forward_context=context),
    )

    def layer(heads=2, exports=True):
        return SimpleNamespace(
            _exports_checkpoints=exports,
            _ascend_checkpoint_workspace=None,
            vllm_config=config,
            num_v_heads=heads,
            tp_size=1,
            head_k_dim=4,
            head_v_dim=8,
        )

    first, second, different, fallback = layer(), layer(), layer(3), layer(exports=False)
    context.update(first=first, second=second, different=different, fallback=fallback)
    for candidate in context.values():
        gdn_checkpoint.ensure_gdn_checkpoint_workspace(candidate, torch.empty(1))
    assert first._ascend_checkpoint_workspace is second._ascend_checkpoint_workspace
    assert first._ascend_checkpoint_workspace is not different._ascend_checkpoint_workspace
    assert first._ascend_checkpoint_workspace.shape == (2, 2, 8, 4)
    assert first._ascend_checkpoint_workspace.dtype == torch.float32
    assert fallback._ascend_checkpoint_workspace is None


@pytest.mark.parametrize("suppress_level", [0, 3])
def test_chunk_forward_only_returns_converted_intermediates_when_requested(monkeypatch, suppress_level):
    q = torch.zeros(1, 4, 1, 8, dtype=torch.bfloat16)
    v = torch.zeros_like(q)
    g = torch.zeros(1, 4, 1)
    initial = torch.zeros(1, 1, 8, 8)
    h, v_new = torch.ones_like(q), torch.ones_like(v)
    monkeypatch.setattr(chunk, "get_forward_context", lambda: SimpleNamespace(attn_metadata=None))
    monkeypatch.setattr(chunk, "get_pcp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(chunk, "chunk_local_cumsum", lambda tensor, **kwargs: tensor)
    monkeypatch.setattr(chunk, "chunk_scaled_dot_kkt_fwd", lambda **kwargs: torch.zeros(1))
    monkeypatch.setattr(chunk, "solve_tril", lambda **kwargs: kwargs["A"])
    monkeypatch.setattr(chunk, "recompute_w_u_fwd", lambda **kwargs: (q, v))
    monkeypatch.setattr(chunk, "chunk_gated_delta_rule_fwd_h", lambda **kwargs: (h, v_new, initial))
    monkeypatch.setattr(chunk, "SUPPRESS_LEVEL", suppress_level)
    monkeypatch.setattr(
        torch.ops._C_ascend, "chunk_fwd_o_vllm", lambda *args, **kwargs: v.transpose(1, 2).contiguous(), raising=False
    )
    result = chunk.chunk_gated_delta_rule_fwd(q, q, v, g, g, 0.1, initial, True, use_triton_recurrence=True)
    torch.testing.assert_close(result[1], v)
    assert result[3] is initial
    if suppress_level == 0:
        assert result[4:] == (None, None, None)
    else:
        assert result[4] is q
        torch.testing.assert_close(result[5], h)
        torch.testing.assert_close(result[6], v_new)
