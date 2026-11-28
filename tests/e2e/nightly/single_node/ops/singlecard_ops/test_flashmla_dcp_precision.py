# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run real BF16/FP16 FlashMLA shards on one NPU and check a dense CPU reference.

This follows dcp.py's sequential-rank method. It checks the installed kernel,
not HCCL or multi-process serving. Run with the DCP-capable cann_ops_transformer
package after sourcing its CANN environment.
"""

import math
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")
pytest.importorskip("cann_ops_transformer")

from vllm_ascend.attention.flashmla import FlashMLAConfig, FlashMLAImpl  # noqa: E402
from vllm_ascend.device.device_op import DeviceOperator  # noqa: E402

BLOCK_SIZE, HEADS, QK_DIM, V_DIM = 128, 96, 576, 512


@pytest.mark.parametrize("block_size", [512, 768, 896])
@torch.inference_mode()
def test_native_page_current_kv_scatter(block_size):
    if not torch.npu.is_available():
        pytest.skip("requires an NPU")
    torch.npu.set_device(0)
    stride = block_size * QK_DIM + 32
    raw = torch.full((3 * stride + 64,), -17.0, dtype=torch.bfloat16, device="npu")
    cache = raw.as_strided((3, block_size, 1, QK_DIM), (stride, QK_DIM, QK_DIM, 1), 32)
    expected_raw = raw.cpu().clone()
    expected = expected_raw.as_strided(cache.shape, cache.stride(), cache.storage_offset())
    # Emulate PROLOG's 128-page workspace output and cross a reader boundary.
    workspace = torch.arange(130, dtype=torch.float32).view(-1, 1, 1).expand(-1, 1, QK_DIM).bfloat16()
    slots = torch.full((130,), -1, dtype=torch.int32)
    slots[0], slots[1], slots[-1] = block_size - 1, 2 * block_size, block_size
    for token, slot in enumerate(slots.tolist()):
        if slot >= 0:
            expected[slot // block_size, slot % block_size].copy_(workspace[token])
    DeviceOperator.reshape_and_cache(
        key=workspace[..., :V_DIM].contiguous().npu(),
        value=workspace[..., V_DIM:].contiguous().npu(),
        key_cache=cache[..., :V_DIM],
        value_cache=cache[..., V_DIM:],
        slot_mapping=slots.npu(),
    )
    torch.testing.assert_close(raw.cpu(), expected_raw, atol=0, rtol=0)


def _cache(sequences, *, world, rank, strided, dtype, device, block_size):
    local_keys = [sequence[(torch.arange(sequence.shape[0]) // block_size) % world == rank] for sequence in sequences]
    page_counts = [(keys.shape[0] + block_size - 1) // block_size for keys in local_keys]
    page_stride = block_size * QK_DIM + (32 if strided else 0)
    page_count = max(sum(page_counts), 1)
    backing = torch.full((page_count * page_stride + 64,), 17.0, dtype=dtype, device=device)
    cache = backing.as_strided((page_count, block_size, 1, QK_DIM), (page_stride, QK_DIM, QK_DIM, 1), 32)
    # One extra zero-used request owns the graph-style query padding.
    table = torch.zeros((len(sequences) + 1, max(max(page_counts), 1)), dtype=torch.int32)
    cursor = 0
    for request, keys in enumerate(local_keys):
        for page in range(page_counts[request]):
            table[request, page] = cursor
            rows = keys[page * block_size : (page + 1) * block_size]
            cache[cursor, : rows.shape[0], 0].copy_(rows.to(device))
            cursor += 1
    lengths = torch.tensor([keys.shape[0] for keys in local_keys] + [0], dtype=torch.int32, device=device)
    return cache, table.to(device), lengths


def _reference(query, sequences, query_lens, causal, scale):
    outputs, lses = [], []
    start = 0
    for keys, qlen in zip(sequences, query_lens):
        for offset in range(qlen):
            visible = keys.shape[0] - qlen + offset + 1 if causal else keys.shape[0]
            scores = query[start + offset].double() @ keys[:visible].double().T * scale
            outputs.append(scores.softmax(-1) @ keys[:visible, :V_DIM].double())
            lses.append(scores.logsumexp(-1))
        start += qlen
    return torch.stack(outputs), torch.stack(lses)


def _merge(partials):
    outputs = torch.stack([output for output, _ in partials])
    lses = torch.stack([lse for _, lse in partials])
    valid = torch.isfinite(lses)
    lses = torch.where(valid, lses, -torch.inf)
    maximum = lses.max(0).values
    assert torch.isfinite(maximum).all(), "every real query must have at least one visible shard"
    weights = torch.exp(lses - maximum)
    outputs = torch.where(valid.unsqueeze(-1), outputs, 0)
    denominator = weights.sum(0)
    return (outputs * weights.unsqueeze(-1)).sum(0) / denominator.unsqueeze(-1), maximum + denominator.log()


@pytest.mark.parametrize("block_size", [128, 512, 768, 896])
@pytest.mark.parametrize("world", [4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_custom_flashmla_dcp_matches_dense(world, dtype, causal, strided, block_size):
    if not torch.npu.is_available():
        pytest.skip("requires an NPU and the DCP-capable cann_ops_transformer package")
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    generator = torch.Generator().manual_seed(42)
    # Cross a full rank stripe and a single-block boundary; later ranks have
    # no KV for the second request. Padding must never contribute to either.
    query_lens = [4, 4]
    lengths = [world * block_size + 2, block_size + 2]
    sequences = [torch.randn(length, QK_DIM, generator=generator).to(dtype) * 0.2 for length in lengths]
    real_tokens = sum(query_lens)
    query = torch.randn(real_tokens + 2, HEADS, QK_DIM, generator=generator).to(dtype) * 0.2
    q = query.to(device)
    cu = torch.tensor([0, 4, 8, 10], dtype=torch.int32, device=device)
    used = torch.tensor([4, 4, 0], dtype=torch.int32, device=device)
    global_lens = torch.tensor(lengths + [0], dtype=torch.int32, device=device)
    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8, device=device), diagonal=1)
    scale = 1 / math.sqrt(QK_DIM)
    config = FlashMLAConfig(
        HEADS, scale, mask_mode=3 if causal else 0, return_softmax_lse=True, cache_block_size=block_size
    )
    adapter = FlashMLAImpl.load(config)

    def run(op, cache, table, local_lens, global_lengths=None):
        schedule = op.build_metadata(local_lens, cu, used, global_lengths)
        if op.config.enable_dcp:
            # Protocol asserted by the pinned 896/512 reduction example.
            header = schedule[:16].cpu().tolist()
            assert header[9] == 1 and header[10] == block_size, header
        output, lse = op.attention(
            q,
            cache,
            block_table=table,
            cache_seqlens=local_lens,
            cu_seqlens_q=cu,
            seqused_q=used,
            metadata=schedule,
            attn_mask=mask if op.config.mask_mode == 3 else None,
        )
        # The adapter retains the framework's NTD/NT contract around the
        # custom operator's TND/NT ABI. Compare real tokens only.
        return output[:, :real_tokens].transpose(0, 1).cpu().double(), lse[:, :real_tokens].T.cpu().double()

    full = run(
        adapter, *_cache(sequences, world=1, rank=0, strided=strided, dtype=dtype, device=device, block_size=block_size)
    )
    partials = []
    for rank in range(world):
        local_adapter = replace(
            adapter,
            config=replace(
                config,
                mask_mode=0,
                enable_dcp=causal,
                cp_world=world if causal else 1,
                cp_rank=rank if causal else 0,
                cp_kv_cache_interleave_size=block_size if causal else BLOCK_SIZE,
            ),
        )
        partials.append(
            run(
                local_adapter,
                *_cache(
                    sequences,
                    world=world,
                    rank=rank,
                    strided=strided,
                    dtype=dtype,
                    device=device,
                    block_size=block_size,
                ),
                global_lengths=global_lens if causal else None,
            )
        )
    merged = _merge(partials)
    expected = _reference(query, sequences, query_lens, causal, scale)
    for actual in (full, merged):
        torch.testing.assert_close(actual[0], expected[0], atol=0.003, rtol=0.03)
        torch.testing.assert_close(actual[1], expected[1], atol=0.003, rtol=0.001)
    torch.testing.assert_close(merged[0], full[0], atol=0.003, rtol=0.03)
    torch.testing.assert_close(merged[1], full[1], atol=0.003, rtol=0.001)
