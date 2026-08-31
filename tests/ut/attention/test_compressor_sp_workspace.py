import pytest
import torch

from vllm_ascend.attention.context_parallel.compressor_sp import (
    GatherWorkspace,
    fused_gather_rows,
    gather_workspace_view,
)


def _fake_all_gather(per_rank_locals):
    """Simulate ``all_gather_into_tensor`` without a process group.

    ``per_rank_locals`` maps this rank's flat local buffer to the contribution of
    every rank, so the packing and unpacking layout can be checked on CPU.
    """
    calls = []

    def all_gather_into_tensor(gathered, local):
        calls.append(local.clone())
        blocks = per_rank_locals(local)
        assert gathered.numel() == sum(block.numel() for block in blocks)
        offset = 0
        for block in blocks:
            gathered[offset : offset + block.numel()] = block.reshape(-1)
            offset += block.numel()

    return all_gather_into_tensor, calls


def test_workspace_view_aliases_and_grows_in_place():
    workspace = GatherWorkspace()
    reference = torch.zeros(8, dtype=torch.float32)

    first = gather_workspace_view(workspace, reference, 16, field="send")
    assert first.shape == (16,)
    # The send side starts deterministic (zero-filled at allocation).
    assert torch.count_nonzero(first) == 0

    smaller = gather_workspace_view(workspace, reference, 8, field="send")
    assert smaller.data_ptr() == first.data_ptr(), "capacity reuse must not reallocate"
    assert smaller.numel() == 8

    bigger = gather_workspace_view(workspace, reference, 64, field="send")
    assert bigger.numel() == 64
    assert bigger.data_ptr() != first.data_ptr(), "growth must reallocate"

    gathered = gather_workspace_view(workspace, reference, 16, field="gathered")
    assert gathered.data_ptr() != workspace.send.data_ptr()
    assert workspace.gathered.numel() == 16


def test_workspace_view_respects_dtype_and_device_guards():
    workspace = GatherWorkspace()
    reference = torch.zeros(8, dtype=torch.float32)
    gather_workspace_view(workspace, reference, 16, field="send")

    other_dtype = torch.zeros(8, dtype=torch.float64)
    view = gather_workspace_view(workspace, other_dtype, 8, field="send")
    assert view.dtype == torch.float64, "dtype change must reallocate"
    assert workspace.send.dtype == torch.float64


def test_fused_gather_with_workspace_matches_fresh_buffers():
    tp_size = 3
    out_counts = (2, 1, 2)
    state_counts = (1, 1, 1)
    payloads = (
        (torch.tensor([[1.0, 2.0], [3.0, 4.0]]), out_counts),
        (torch.tensor([[10.0, 20.0, 30.0]]), state_counts),
    )
    per_rank = lambda local: [local + rank * 100.0 for rank in range(tp_size)]  # noqa: E731

    fresh_all_gather, _ = _fake_all_gather(per_rank)
    expected = fused_gather_rows(payloads, tp_size, fresh_all_gather)

    workspace = GatherWorkspace()
    ws_all_gather, calls = _fake_all_gather(per_rank)
    result = fused_gather_rows(payloads, tp_size, ws_all_gather, workspace=workspace)

    assert result is not None and expected is not None
    for got, want in zip(result, expected):
        assert torch.equal(got, want)
    # The send and receive sides both live in the persistent workspace now.
    assert workspace.send is not None and workspace.gathered is not None
    assert torch.equal(calls[0], workspace.send[: calls[0].numel()])


def test_workspace_reuse_keeps_stale_padding_out_of_results():
    """A larger call then a smaller call must not leak stale pad content."""
    tp_size = 2
    workspace = GatherWorkspace()

    big_payloads = ((torch.ones(4, 2), (4, 4)),)
    big_all_gather, _ = _fake_all_gather(lambda local: [local + rank * 10.0 for rank in range(tp_size)])
    fused_gather_rows(big_payloads, tp_size, big_all_gather, workspace=workspace)
    big_send_ptr = workspace.send.data_ptr()
    big_gathered_ptr = workspace.gathered.data_ptr()

    # Ragged smaller payload: rank rows 1 < max 2 leaves a pad row in the send
    # buffer that still holds the previous (much larger) call's data.
    small_payloads = ((torch.tensor([[5.0, 6.0]]), (1, 2)),)
    small_all_gather, calls = _fake_all_gather(lambda local: [local + rank * 10.0 for rank in range(tp_size)])
    result = fused_gather_rows(small_payloads, tp_size, small_all_gather, workspace=workspace)

    assert result is not None
    assert workspace.send.data_ptr() == big_send_ptr, "capacity suffices, no realloc"
    assert workspace.gathered.data_ptr() == big_gathered_ptr
    # Rank-major rebuild of the small payload: valid rows only, per rank order.
    assert result[0].shape == (4, 2)
    assert result[0][:1].tolist() == [[5.0, 6.0]]
    assert result[0][2:3].tolist() == [[15.0, 16.0]]
    # The pad rows (rows 1 and 3) carry the stale previous call's data; the
    # plan's compact selectors never select them, which the callers guarantee.
    small_gathered = workspace.gathered[: 4 * 2]
    assert torch.equal(result[0][1], small_gathered.view(4, 2)[1])


def test_indexer_debug_gate_env_reflection(monkeypatch):
    from vllm_ascend import envs

    monkeypatch.delenv("VLLM_ASCEND_COMPRESSOR_SP_DISABLE_INDEXER", raising=False)
    assert envs.VLLM_ASCEND_COMPRESSOR_SP_DISABLE_INDEXER is False

    monkeypatch.setenv("VLLM_ASCEND_COMPRESSOR_SP_DISABLE_INDEXER", "1")
    # envs caches nothing per attribute access order here, but the lazy getter
    # reads os.getenv on every access, so the flip is visible immediately.
    assert envs.VLLM_ASCEND_COMPRESSOR_SP_DISABLE_INDEXER is True


if __name__ == "__main__":
    pytest.main([__file__])
