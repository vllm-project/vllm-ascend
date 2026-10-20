import gc
from importlib import import_module

import pytest
import torch
import torch_npu

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.utils import enable_custom_op

torch_npu.npu.config.allow_internal_format = True
enable_custom_op()

BF16_ATOL = 6e-2
BF16_RTOL = 7.8125e-3


def _make_inputs():
    torch.manual_seed(1024)

    batch = 1
    query_seq = 1
    kv_seq = 4096
    actual_kv_seq = 4096
    query_heads = 64
    kv_heads = 1
    head_dim = 512
    rope_head_dim = 64
    tile_size = 128
    block_size = 256
    sparse_block_count = 2048
    block_num = kv_seq // block_size
    scale_value = (head_dim + rope_head_dim) ** -0.5

    query_nope = (
        torch.empty((batch, query_seq, query_heads, head_dim), dtype=torch.float32).uniform_(-10, 10).to(torch.bfloat16)
    )
    query_rope = (
        torch.empty((batch, query_seq, query_heads, rope_head_dim), dtype=torch.float32)
        .uniform_(-10, 10)
        .to(torch.bfloat16)
    )
    query = torch.cat((query_nope, query_rope), dim=-1).npu()

    key_nope = (
        torch.empty((block_num, block_size, kv_heads, head_dim), dtype=torch.float32).uniform_(-5, 10).to(torch.int8)
    )
    value = key_nope.clone()
    key_rope = (
        torch.empty((block_num, block_size, kv_heads, rope_head_dim), dtype=torch.float32)
        .uniform_(-10, 10)
        .to(torch.bfloat16)
    )
    dequant_scale = torch.empty((block_num, block_size, kv_heads, head_dim // tile_size), dtype=torch.float32).uniform_(
        0.1, 1.0
    )
    kv_cache = torch.cat(
        (
            key_nope,
            key_rope.contiguous().view(torch.int8),
            dequant_scale.contiguous().view(torch.int8),
        ),
        dim=-1,
    ).npu()

    sparse_indices = torch.randperm(actual_kv_seq, dtype=torch.int32)[:sparse_block_count].reshape(
        batch, query_seq, kv_heads, sparse_block_count
    )
    block_table = torch.arange(block_num, dtype=torch.int32).reshape(batch, block_num)
    actual_seq_lengths_query = torch.tensor([query_seq], dtype=torch.int32)
    actual_seq_lengths_kv = torch.tensor([actual_kv_seq], dtype=torch.int32)

    return {
        "query": query,
        "key": kv_cache,
        "value": value.npu(),
        "sparse_indices": sparse_indices.npu(),
        "block_table": block_table.npu(),
        "actual_seq_lengths_query": actual_seq_lengths_query.npu(),
        "actual_seq_lengths_kv": actual_seq_lengths_kv.npu(),
        "scale_value": scale_value,
        "sparse_block_size": 1,
        "layout_query": "BSND",
        "layout_kv": "PA_BSND",
        "sparse_mode": 3,
        "attention_mode": 2,
        "quant_scale_repo_mode": 1,
        "tile_size": tile_size,
        "rope_head_dim": rope_head_dim,
        "key_quant_mode": 2,
        "value_quant_mode": 2,
        "cpu": {
            "query_nope": query_nope,
            "query_rope": query_rope,
            "key_nope": key_nope,
            "value": value,
            "key_rope": key_rope,
            "dequant_scale": dequant_scale,
            "sparse_indices": sparse_indices,
            "block_table": block_table,
            "actual_seq_lengths_query": actual_seq_lengths_query,
            "actual_seq_lengths_kv": actual_seq_lengths_kv,
            "block_size": block_size,
        },
    }


def _sparse_token_indices(
    sparse_indices,
    sparse_block_size,
    sparse_block_count,
    query_idx,
    actual_seq_query,
    actual_seq_kv,
    sparse_mode,
):
    if sparse_mode == 0:
        threshold = actual_seq_kv
    elif sparse_mode == 3:
        threshold = actual_seq_kv - actual_seq_query + query_idx + 1
    else:
        raise AssertionError(f"unsupported sparse_mode in test: {sparse_mode}")

    valid_count = min(
        sparse_block_count,
        (threshold + sparse_block_size - 1) // sparse_block_size,
    )
    tokens: list[int] = []
    for sparse_id in sparse_indices[:valid_count].tolist():
        if sparse_id == -1:
            break
        begin = sparse_id * sparse_block_size
        end = min(begin + sparse_block_size, actual_seq_kv)
        if begin >= threshold:
            continue
        tokens.extend(range(begin, min(end, threshold)))
    return torch.tensor(tokens, dtype=torch.long)


def _reference_attention(inputs):
    cpu = inputs["cpu"]
    query = torch.cat((cpu["query_nope"], cpu["query_rope"]), dim=-1).float()

    dequant_scale = cpu["dequant_scale"].repeat_interleave(inputs["tile_size"], dim=-1)
    key_nope = (cpu["key_nope"].float() * dequant_scale).to(torch.bfloat16)
    value = (cpu["value"].float() * dequant_scale).to(torch.bfloat16).float()
    key = torch.cat((key_nope, cpu["key_rope"]), dim=-1).float()

    batch, query_seq, query_heads, _ = query.shape
    kv_heads = cpu["key_nope"].shape[2]
    group_size = query_heads // kv_heads
    head_dim = cpu["value"].shape[-1]
    output = torch.zeros((batch, query_seq, query_heads, head_dim), dtype=torch.float32)

    block_table = cpu["block_table"].long()
    block_size = cpu["block_size"]
    sparse_block_count = cpu["sparse_indices"].shape[-1]

    for batch_idx in range(batch):
        actual_seq_query = int(cpu["actual_seq_lengths_query"][batch_idx])
        actual_seq_kv = int(cpu["actual_seq_lengths_kv"][batch_idx])
        for kv_head_idx in range(kv_heads):
            head_begin = kv_head_idx * group_size
            head_end = head_begin + group_size
            for query_idx in range(actual_seq_query):
                token_indices = _sparse_token_indices(
                    cpu["sparse_indices"][batch_idx, query_idx, kv_head_idx],
                    inputs["sparse_block_size"],
                    sparse_block_count,
                    query_idx,
                    actual_seq_query,
                    actual_seq_kv,
                    inputs["sparse_mode"],
                )
                if token_indices.numel() == 0:
                    continue

                logical_blocks = token_indices // block_size
                block_offsets = token_indices % block_size
                physical_blocks = block_table[batch_idx, logical_blocks]
                k_sparse = key[physical_blocks, block_offsets, kv_head_idx]
                v_sparse = value[physical_blocks, block_offsets, kv_head_idx]
                q_current = query[batch_idx, query_idx, head_begin:head_end]

                scores = torch.matmul(q_current, k_sparse.T) * inputs["scale_value"]
                probs = torch.softmax(scores, dim=-1)
                output[batch_idx, query_idx, head_begin:head_end] = torch.matmul(
                    probs.to(torch.bfloat16).float(), v_sparse
                )
    return output


def _run_custom_op(inputs, return_softmax_lse=False):
    if get_ascend_device_type() == AscendDeviceType.A5:
        import_module("vllm_ascend.vllm_ascend_C")
    return torch.ops._C_ascend.npu_kv_quant_sparse_flash_attention_vllm(
        inputs["query"],
        inputs["key"],
        inputs["value"],
        inputs["sparse_indices"],
        inputs["scale_value"],
        key_quant_mode=inputs["key_quant_mode"],
        value_quant_mode=inputs["value_quant_mode"],
        block_table=inputs["block_table"],
        actual_seq_lengths_query=inputs["actual_seq_lengths_query"],
        actual_seq_lengths_kv=inputs["actual_seq_lengths_kv"],
        sparse_block_size=inputs["sparse_block_size"],
        layout_query=inputs["layout_query"],
        layout_kv=inputs["layout_kv"],
        sparse_mode=inputs["sparse_mode"],
        attention_mode=inputs["attention_mode"],
        quant_scale_repo_mode=inputs["quant_scale_repo_mode"],
        tile_size=inputs["tile_size"],
        rope_head_dim=inputs["rope_head_dim"],
        return_softmax_lse=return_softmax_lse,
    )


@torch.inference_mode()
def test_kv_quant_sparse_flash_attention_vllm():
    inputs = _make_inputs()
    reference = _reference_attention(inputs)

    output, softmax_max, softmax_sum = _run_custom_op(inputs)

    assert output.shape == (1, 1, 64, 512)
    assert output.dtype == torch.bfloat16
    assert softmax_max.numel() == 0
    assert softmax_sum.numel() == 0
    assert torch.isfinite(output.cpu()).all()
    torch.testing.assert_close(output.cpu().float(), reference, atol=BF16_ATOL, rtol=BF16_RTOL)

    output_lse, softmax_max_lse, softmax_sum_lse = _run_custom_op(inputs, return_softmax_lse=True)
    assert output_lse.shape == (1, 1, 64, 512)
    assert softmax_max_lse.shape == (1, 1, 1, 64)
    assert softmax_sum_lse.shape == (1, 1, 1, 64)
    assert output_lse.dtype == torch.bfloat16
    assert softmax_max_lse.dtype == torch.float32
    assert softmax_sum_lse.dtype == torch.float32
    assert torch.isfinite(output_lse.cpu()).all()
    assert torch.isfinite(softmax_max_lse.cpu()).all()
    assert torch.isfinite(softmax_sum_lse.cpu()).all()
    torch.testing.assert_close(output_lse.cpu().float(), reference, atol=BF16_ATOL, rtol=BF16_RTOL)
    torch.testing.assert_close(output_lse.cpu(), output.cpu(), atol=1e-2, rtol=1e-2)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A5, reason="Requires the A5 packed FP8 kernel.")
@pytest.mark.parametrize("length,count", [(4096, 2048), (1024, 512), (128, 32), (128, 0)])
@pytest.mark.parametrize("heads", [16, 32, 64])
def test_a5_c8_lse(length, count, heads):
    torch.manual_seed(16656)
    page_size, nope_dim, rope_dim, tile_size = 128, 512, 64, 128
    query = torch.randn(1, heads, nope_dim + rope_dim).to(torch.bfloat16)
    key_nope = (torch.randn(length, 1, nope_dim) * 2).to(torch.float8_e4m3fn)
    key_rope = torch.randn(length, 1, rope_dim).to(torch.bfloat16)
    scales = torch.rand(length, 1, nope_dim // tile_size) * 0.1 + 0.05
    cache = (
        torch.cat([key_nope.view(torch.uint8), key_rope.view(torch.uint8), scales.view(torch.uint8)], -1)
        .reshape(length // page_size, page_size, 1, -1)
        .view(torch.float8_e4m3fn)
    )
    indices = torch.full((1, 1, 2048), -1, dtype=torch.int32)
    indices[0, 0, :count] = torch.randperm(length)[:count].to(torch.int32)
    scale = (nope_dim + rope_dim) ** -0.5
    kwargs = dict(
        query=query.npu(),
        key=cache.npu(),
        value=cache.npu(),
        sparse_indices=indices.npu(),
        scale_value=scale,
        sparse_block_size=1,
        block_table=torch.arange(length // page_size, dtype=torch.int32).view(1, -1).npu(),
        actual_seq_lengths_query=torch.tensor([1], dtype=torch.int32).npu(),
        actual_seq_lengths_kv=torch.tensor([length], dtype=torch.int32).npu(),
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=tile_size,
        key_quant_mode=2,
        value_quant_mode=2,
        rope_head_dim=rope_dim,
    )
    output, max_value, exp_sum = _run_custom_op(kwargs, return_softmax_lse=True)
    output, max_value, exp_sum = output.cpu().float(), max_value.cpu().float(), exp_sum.cpu().float()
    assert torch.isfinite(output).all()
    if count == 0:
        assert torch.count_nonzero(output) == 0
        assert torch.count_nonzero(exp_sum) == 0
        assert torch.isneginf(max_value + exp_sum.log()).all()
        return
    selected = indices[0, 0, :count].long()
    dequant = (key_nope.float() * scales.repeat_interleave(tile_size, -1)).to(torch.bfloat16)
    key = torch.cat([dequant, key_rope], -1)[selected, 0].float()
    scores = query[0].float() @ key.T * scale
    reference = torch.softmax(scores, -1) @ dequant[selected, 0].float()
    torch.testing.assert_close(output[0], reference, atol=1e-3, rtol=1e-2)
    torch.testing.assert_close(
        (max_value + exp_sum.log()).reshape(-1), torch.logsumexp(scores, -1), atol=1e-5, rtol=1e-5
    )
    if count == 2048:
        builtin = torch_npu.npu_kv_quant_sparse_flash_attention(**kwargs).cpu().float()
        torch.testing.assert_close(output, builtin, atol=0, rtol=0)
