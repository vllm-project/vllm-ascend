# SPDX-License-Identifier: Apache-2.0
"""Single-rank MLA cache regression; DCP performance needs separate A3 profiling."""

import math

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires an Ascend NPU"
)


@pytest.mark.parametrize("layout", ["BNSD", "BSND", "TND"])
@pytest.mark.parametrize("heads", [8, 128], ids=["local-heads", "dcp16-global-heads"])
@pytest.mark.parametrize("kv_lengths", [[127, 257], [0, 257]], ids=["nonempty", "empty-history-shard"])
def test_mla_dcp_component_cache_dim0_stride(layout, heads, kv_lengths):
    generator = torch.Generator().manual_seed(18256)
    batch, query_length = 2, 2
    blocks, block_size, head_dim, rope_dim = 6, 128, 512, 64
    dtype = torch.bfloat16
    scale = 1 / math.sqrt(head_dim + rope_dim)

    def random_tensor(*shape):
        return torch.randn(*shape, generator=generator).to(dtype)

    # Keep queries dense so only the cache inputs exercise the stride path.
    query_cpu = random_tensor(batch, query_length, heads, head_dim)
    query_rope_cpu = random_tensor(batch, query_length, heads, rope_dim)
    key_cpu = random_tensor(blocks, 1, block_size, head_dim)
    rope_cpu = random_tensor(blocks, 1, block_size, rope_dim)
    table_cpu = torch.tensor([[4, 0, 2], [3, 1, 5]], dtype=torch.int32)

    def format_query(tensor):
        if layout == "BNSD":
            return tensor.transpose(1, 2).contiguous()
        if layout == "TND":
            return tensor.flatten(0, 1).contiguous()
        return tensor.contiguous()

    query = format_query(query_cpu).npu()
    query_rope = format_query(query_rope_cpu).npu()
    assert query.is_contiguous() and query_rope.is_contiguous()

    # Each physical page contains a complete nope component followed by rope.
    # Both views share storage, with a prefix offset and poison between pages.
    prefix, gap = 256, 64
    key_elements = block_size * head_dim
    page_stride = block_size * (head_dim + rope_dim) + gap
    backing = torch.full((prefix + blocks * page_stride,), 37, dtype=dtype, device=query.device)
    key = torch.as_strided(backing, key_cpu.shape, (page_stride, key_elements, head_dim, 1), storage_offset=prefix)
    key_rope = torch.as_strided(
        backing,
        rope_cpu.shape,
        (page_stride, block_size * rope_dim, rope_dim, 1),
        storage_offset=prefix + key_elements,
    )
    key.copy_(key_cpu.npu())
    key_rope.copy_(rope_cpu.npu())
    before = backing.cpu().clone()
    assert not key.is_contiguous() and not key_rope.is_contiguous()
    assert key.storage_offset() > 0 and key_rope.storage_offset() > 0

    q_lengths = [query_length, batch * query_length] if layout == "TND" else [query_length] * batch
    kwargs = {
        "query_rope": query_rope,
        "num_query_heads": heads,
        "num_key_value_heads": 1,
        "input_layout": layout,
        "softmax_scale": scale,
        "sparse_mode": 0,
        "pre_tokens": 65535,
        "next_tokens": 65535,
        "block_table": table_cpu.npu(),
        "block_size": block_size,
        "actual_seq_qlen": q_lengths,
        "actual_seq_kvlen": kv_lengths,
        "return_softmax_lse": True,
    }

    def run(cache, rope):
        # MLA reuses the same latent cache for both K and V.
        output, lse = torch_npu.npu_fused_infer_attention_score_v2(query, cache, cache, key_rope=rope, **kwargs)
        assert lse.dtype == torch.float32
        return output.cpu(), lse.cpu()

    dense_output, dense_lse = run(key.contiguous(), key_rope.contiguous())
    output, lse = run(key, key_rope)
    torch.testing.assert_close(output, dense_output, rtol=0.01, atol=0.01)
    torch.testing.assert_close(lse, dense_lse, rtol=0.001, atol=0.005)
    candidates = [(output, lse), (dense_output, dense_lse)]
    if 0 in kv_lengths:
        # The migration must preserve empty-shard semantics used by DCP's
        # merge: zero output and positive-infinity LSE mark an invalid shard.
        legacy_kwargs = kwargs.copy()
        for old, new in (
            ("num_heads", "num_query_heads"),
            ("scale", "softmax_scale"),
            ("actual_seq_lengths", "actual_seq_qlen"),
            ("actual_seq_lengths_kv", "actual_seq_kvlen"),
            ("softmax_lse_flag", "return_softmax_lse"),
        ):
            legacy_kwargs[old] = legacy_kwargs.pop(new)
        legacy_output, legacy_lse = torch_npu.npu_fused_infer_attention_score(
            query, key, key, key_rope=key_rope, **legacy_kwargs
        )
        legacy_output, legacy_lse = legacy_output.cpu(), legacy_lse.cpu()
        assert legacy_lse.dtype == torch.float32
        torch.testing.assert_close(output, legacy_output, rtol=0.01, atol=0.01)
        torch.testing.assert_close(lse, legacy_lse, rtol=0.001, atol=0.005)
        candidates.append((legacy_output, legacy_lse))

    expected_outputs, expected_lse_rows = [], []
    for row, kv_length in enumerate(kv_lengths):
        if kv_length == 0:
            expected_outputs.append(torch.zeros(query_length, heads, head_dim))
            expected_lse_rows.append(torch.full((heads, query_length), torch.inf))
            for candidate_output, candidate_lse in candidates:
                if layout == "TND":
                    empty_output = candidate_output[row * query_length : (row + 1) * query_length]
                    empty_lse = candidate_lse[row * query_length : (row + 1) * query_length]
                else:
                    empty_output, empty_lse = candidate_output[row], candidate_lse[row]
                assert torch.count_nonzero(empty_output).item() == 0
                assert torch.isposinf(empty_lse).all().item()
            continue
        ids = table_cpu[row, : math.ceil(kv_length / block_size)].long()
        keys = key_cpu[ids].reshape(-1, head_dim)[:kv_length].float()
        ropes = rope_cpu[ids].reshape(-1, rope_dim)[:kv_length].float()
        q = query_cpu[row].float().transpose(0, 1)
        q_rope = query_rope_cpu[row].float().transpose(0, 1)
        scores = (q @ keys.T + q_rope @ ropes.T) * scale
        expected_outputs.append((scores.softmax(-1) @ keys).transpose(0, 1))
        expected_lse_rows.append(scores.logsumexp(-1))

    expected_output = format_query(torch.stack(expected_outputs))
    # FIA returns BNS1 LSE for both BNSD and BSND, and TN1 for TND.
    expected_lse = torch.stack(expected_lse_rows).unsqueeze(-1)
    if layout == "TND":
        expected_lse = expected_lse.transpose(1, 2).flatten(0, 1)
    for candidate_output, candidate_lse in candidates:
        torch.testing.assert_close(candidate_output.float(), expected_output, rtol=0.01, atol=0.01)
        torch.testing.assert_close(candidate_lse, expected_lse, rtol=0.001, atol=0.005)
    # Check valid cache elements, offsets, unused pages and all poison padding.
    torch.testing.assert_close(backing.cpu(), before, rtol=0, atol=0)


@pytest.mark.parametrize("heads", [8, 128])
def test_mla_dcp_current_tnd_causal_attention(heads):
    """Current-token attention uses dense TND KV and cumulative lengths."""
    generator = torch.Generator().manual_seed(18257)
    query_ends = [2, 5]
    tokens, head_dim, rope_dim = query_ends[-1], 512, 64
    scale = 1 / math.sqrt(head_dim + rope_dim)

    def random_tensor(*shape):
        return torch.randn(*shape, generator=generator).to(torch.bfloat16)

    query_cpu = random_tensor(tokens, heads, head_dim)
    query_rope_cpu = random_tensor(tokens, heads, rope_dim)
    key_cpu = random_tensor(tokens, 1, head_dim)
    key_rope_cpu = random_tensor(tokens, 1, rope_dim)
    query, query_rope = query_cpu.npu(), query_rope_cpu.npu()
    key, key_rope = key_cpu.npu(), key_rope_cpu.npu()
    # sparse_mode=3 consumes the compressed 2048x2048 right-down causal
    # template. Q and KV have equal per-request lengths in the current chunk.
    mask = torch.ones((2048, 2048), dtype=torch.bool).triu(1).npu()
    output, lse = torch_npu.npu_fused_infer_attention_score_v2(
        query,
        key,
        key,
        query_rope=query_rope,
        key_rope=key_rope,
        num_query_heads=heads,
        num_key_value_heads=1,
        input_layout="TND",
        atten_mask=mask,
        sparse_mode=3,
        softmax_scale=scale,
        actual_seq_qlen=query_ends,
        actual_seq_kvlen=query_ends,
        return_softmax_lse=True,
    )
    output, lse = output.cpu(), lse.cpu()
    assert output.shape == (tokens, heads, head_dim)
    assert lse.shape == (tokens, heads, 1) and lse.dtype == torch.float32

    expected_outputs, expected_lse = [], []
    for start, end in zip([0] + query_ends[:-1], query_ends):
        query_float = query_cpu[start:end].float().transpose(0, 1)
        query_rope_float = query_rope_cpu[start:end].float().transpose(0, 1)
        key_float = key_cpu[start:end, 0].float()
        key_rope_float = key_rope_cpu[start:end, 0].float()
        scores = (query_float @ key_float.T + query_rope_float @ key_rope_float.T) * scale
        causal_mask = torch.ones((end - start, end - start), dtype=torch.bool).triu(1)
        scores.masked_fill_(causal_mask, -torch.inf)
        expected_outputs.append((scores.softmax(-1) @ key_float).transpose(0, 1))
        expected_lse.append(scores.logsumexp(-1).transpose(0, 1).unsqueeze(-1))
    torch.testing.assert_close(output.float(), torch.cat(expected_outputs), rtol=0.01, atol=0.01)
    torch.testing.assert_close(lse, torch.cat(expected_lse), rtol=0.001, atol=0.005)
