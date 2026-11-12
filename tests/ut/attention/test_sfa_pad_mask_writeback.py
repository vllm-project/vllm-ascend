# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Contract tests for the NoPE padding-mask writeback in the SFA execution core."""

from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.attention.sfa_v1 as sfa_v1
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl


def _reference_writeback(source, destination, query_ends):
    """CPU stand-in for the Triton kernel: keep the valid rows, clear the rest."""
    valid = min(int(query_ends[-1]), source.shape[1])
    destination.zero_()
    destination[:, :valid].copy_(source[:, :valid])


class _StubDeviceOperator:
    """Record the operator call and return a kernel output the test owns."""

    def __init__(self, output: torch.Tensor) -> None:
        self.output = output
        self.calls: list[dict] = []

    def execute_sparse_flash_attention_process(
        self,
        impl,
        ql_nope,
        q_pe,
        kv_cache,
        topk_indices,
        attn_metadata,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table=None,
    ):
        self.calls.append(
            dict(
                impl=impl,
                ql_nope=ql_nope,
                q_pe=q_pe,
                kv_cache=kv_cache,
                topk_indices=topk_indices,
                attn_metadata=attn_metadata,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
                block_table=block_table,
            )
        )
        return self.output


def _build_impl(**overrides) -> "AscendSFAImpl":
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    # NoPE with the packed cache takes the floating NoPE operator, i.e. the
    # branch that has to repair the rows the kernel never writes.
    impl.qk_rope_head_dim = 0
    impl.enable_sparse_sfa_c8 = True
    impl.enable_sparse_sfa_turboquant = False
    impl.scale = 0.25
    for name, value in overrides.items():
        setattr(impl, name, value)
    return impl


def _run(monkeypatch, impl, stub, ql_nope, actual, written, block_size=1):
    cache = torch.zeros(4, block_size, 1, 8)
    metadata = SimpleNamespace(block_size=block_size)
    topk_indices = torch.zeros(ql_nope.shape[0], 1, dtype=torch.int32)
    query_ends = torch.tensor([actual], dtype=torch.int32)

    def write(source, destination, query_ends_arg):
        _reference_writeback(source, destination, query_ends_arg)
        written.append((source, destination, query_ends_arg))

    monkeypatch.setattr(sfa_v1, "DeviceOperator", stub)
    monkeypatch.setattr(sfa_v1, "write_recurrent_output", write)
    result = AscendSFAImpl._execute_sparse_flash_attention_process(
        impl,
        ql_nope,
        None,
        (cache,),
        topk_indices,
        metadata,
        query_ends,
        torch.tensor([1], dtype=torch.int32),
    )
    return result, cache, metadata, topk_indices, query_ends


@pytest.mark.parametrize("actual", [3, 7])
def test_nope_execution_clears_padding_with_one_in_place_writeback(monkeypatch, actual):
    capacity = 5
    impl = _build_impl()
    ql_nope = torch.randn(capacity, 1, 8)
    # The kernel writes the valid rows only; the rows above the scheduled token
    # count keep whatever the captured buffer held.
    output = torch.full((capacity, 1, 8), float("nan"))
    valid = min(actual, capacity)
    output[:valid] = torch.arange(1, valid + 1, dtype=torch.float32)[:, None, None]
    stub = _StubDeviceOperator(output)
    written: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    result, cache, metadata, topk_indices, query_ends = _run(monkeypatch, impl, stub, ql_nope, actual, written)

    # One launch on the same bytes replaces the
    # arange/cast/less/logical_not/masked_fill chain.
    assert len(written) == 1
    source, destination, query_ends_arg = written[0]
    assert query_ends_arg is query_ends
    assert source.shape == destination.shape == (1, capacity, 1, 8)
    assert source.data_ptr() == destination.data_ptr() == output.data_ptr()
    assert result is output
    torch.testing.assert_close(
        output[:valid], torch.arange(1, valid + 1, dtype=torch.float32)[:, None, None]
    )
    # Rows the kernel skipped are cleared by the writeback itself.
    assert torch.count_nonzero(output[valid:]) == 0

    call = stub.calls[0]
    assert call["impl"] is impl
    assert call["ql_nope"] is ql_nope
    assert call["q_pe"] is None
    assert call["kv_cache"][0].data_ptr() == cache.data_ptr()
    assert call["kv_cache"][0].shape[1] == metadata.block_size
    assert call["topk_indices"] is topk_indices
    assert call["attn_metadata"] is metadata
    assert call["actual_seq_lengths_query"] is query_ends


def test_nope_execution_falls_back_for_non_contiguous_output(monkeypatch):
    capacity, actual = 4, 2
    impl = _build_impl()
    ql_nope = torch.randn(capacity, 1, 8)
    backing = torch.full((2 * capacity, 1, 8), float("nan"))
    output = backing[::2]
    assert output.shape == (capacity, 1, 8)
    assert not output.is_contiguous()
    output[:actual] = torch.arange(1, actual + 1, dtype=torch.float32)[:, None, None]
    stub = _StubDeviceOperator(output)
    written: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    result, *_ = _run(monkeypatch, impl, stub, ql_nope, actual, written)

    # A strided result cannot host the in-place writeback, so the mask fallback
    # still produces the same values out of place.
    assert written == []
    assert result is not output
    torch.testing.assert_close(
        result[:actual], torch.arange(1, actual + 1, dtype=torch.float32)[:, None, None]
    )
    assert torch.count_nonzero(result[actual:]) == 0
    assert torch.isnan(output[actual:]).all()


def test_rope_heads_return_the_kernel_output_untouched(monkeypatch):
    capacity = 3
    impl = _build_impl(qk_rope_head_dim=64)
    ql_nope = torch.randn(capacity, 1, 8)
    output = torch.full((capacity, 1, 8), float("nan"))
    stub = _StubDeviceOperator(output)
    written: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    result, *_ = _run(monkeypatch, impl, stub, ql_nope, capacity, written)

    # With rope heads the kernel pads the output itself, so nothing is rewritten.
    assert written == []
    assert result is output
    assert torch.isnan(output).all()
