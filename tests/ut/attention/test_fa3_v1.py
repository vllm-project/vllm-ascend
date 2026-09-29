import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

import vllm_ascend.attention
from vllm_ascend.attention.attention_v1 import AscendMetadata


@pytest.fixture
def fa3_backend(monkeypatch):
    """Load the real backend with only its optional NPU kernel replaced."""
    kernel = Mock(side_effect=lambda query, *args, **kwargs: query + (20 if kwargs["causal"] else 10))
    fa3_stub = ModuleType("flash_attn_npu_v3")
    fa3_stub.flash_attn_with_kvcache = kernel
    monkeypatch.setitem(sys.modules, "flash_attn_npu_v3", fa3_stub)
    monkeypatch.setattr(torch.Tensor, "npu", lambda tensor: tensor, raising=False)

    # A private module instance prevents the stub from leaking into other tests
    # that import the optional FA3 backend or check whether its kernel is installed.
    source = Path(vllm_ascend.attention.__file__).with_name("fa3_v1.py")
    spec = importlib.util.spec_from_file_location("_fa3_v1_cpu_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AscendFAImpl, kernel


@pytest.mark.parametrize(
    "decode_lens,prefill_lens",
    [
        pytest.param([2, 2], [5], id="multi-token-decode-empty-prefill-table"),
        pytest.param([2], [5, 6], id="multi-token-decode-missing-prefill-row"),
        pytest.param([1, 1], [5], id="single-token-decode-mixed-batch"),
        pytest.param([2, 2], [], id="decode-only"),
        pytest.param([], [5, 6], id="prefill-only"),
    ],
)
def test_forward_impl_splits_block_tables_by_request(fa3_backend, decode_lens, prefill_lens):
    backend_cls, kernel = fa3_backend
    query_lens = decode_lens + prefill_lens
    num_requests = len(query_lens)
    num_tokens = sum(query_lens)
    num_decode_tokens = sum(decode_lens)
    num_heads, num_kv_heads, head_size, block_size = 2, 1, 128, 128
    padding_tokens = 2
    output_sentinel = -1.0
    query = torch.arange((num_tokens + padding_tokens) * num_heads * head_size, dtype=torch.float32).reshape(
        -1, num_heads, head_size
    )
    block_tables = torch.arange(num_requests * 2, dtype=torch.int32).reshape(num_requests, 2)
    query_start_loc = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    seq_lens = torch.tensor(query_lens, dtype=torch.int32) + 16
    metadata = AscendMetadata(
        num_actual_tokens=num_tokens,
        num_decodes=len(decode_lens),
        num_decode_tokens=num_decode_tokens,
        num_prefills=len(prefill_lens),
        actual_seq_lengths_q=query_start_loc[1:].tolist(),
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        block_tables=block_tables,
    )
    # forward_impl only needs the cache layout; construction normally requires
    # an engine configuration and is unrelated to request/token partitioning.
    impl = backend_cls.__new__(backend_cls)
    impl.num_kv_heads = num_kv_heads
    impl.head_size = head_size
    impl.key_cache = torch.zeros(num_requests * 2, block_size, num_kv_heads, head_size)
    impl.value_cache = torch.zeros_like(impl.key_cache)
    output = torch.full_like(query, output_sentinel)

    result = impl.forward_impl(query, query, query, (impl.key_cache, impl.value_cache), metadata, output)

    phases = []
    if decode_lens:
        phases.append((0, len(decode_lens), 0, num_decode_tokens, False))
    if prefill_lens:
        phases.append((len(decode_lens), num_requests, num_decode_tokens, num_tokens, True))
    assert kernel.call_count == len(phases)
    for call, (request_start, request_end, token_start, token_end, causal) in zip(kernel.call_args_list, phases):
        args, kwargs = call
        torch.testing.assert_close(kwargs["page_table"], block_tables[request_start:request_end])
        assert kwargs["page_table"].shape[0] == kwargs["cache_seqlens"].numel()
        torch.testing.assert_close(args[0], query[token_start:token_end])
        torch.testing.assert_close(kwargs["cache_seqlens"], seq_lens[request_start:request_end])
        torch.testing.assert_close(
            kwargs["cu_seqlens_q"].diff(), torch.tensor(query_lens[request_start:request_end], dtype=torch.int32)
        )
        assert kwargs["causal"] is causal
        torch.testing.assert_close(output[token_start:token_end], query[token_start:token_end] + (20 if causal else 10))
    assert result is output
    torch.testing.assert_close(output[num_tokens:], torch.full_like(output[num_tokens:], output_sentinel))
