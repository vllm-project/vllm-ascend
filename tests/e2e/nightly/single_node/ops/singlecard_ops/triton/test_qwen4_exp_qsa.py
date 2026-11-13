import math

import pytest
import torch
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: F401
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.models.qwen4_exp.lightning_indexer import qsa_select_paged_tokens_lightning_v2
from vllm_ascend.ops.triton.qwen4_exp.fia import QSAFIAMaskWorkspace, build_qsa_fia_mask
from vllm_ascend.ops.triton.qwen4_exp.gate import qsa_sigmoid_mul_
from vllm_ascend.ops.triton.qwen4_exp.gbsa import (
    pack_qsa_gbsa_indices,
    prepare_qsa_gbsa_inputs,
    prepare_qsa_gbsa_shared,
    prepare_qsa_lengths,
)
from vllm_ascend.ops.triton.qwen4_exp.qsa import (
    pack_qsa_group_indices_npu,
    qsa_store_key_position_rows,
    qsa_store_kv_rows,
)

pytestmark = [
    pytest.mark.skipif(
        get_ascend_device_type() not in (AscendDeviceType.A3, AscendDeviceType.A5),
        reason="Requires Ascend A3/A5",
    ),
    pytest.mark.skipif(not HAS_TRITON, reason="Triton is not available"),
]


@torch.inference_mode()
def test_qsa_pack_supports_non_power_of_two_output_width() -> None:
    groups, ratio = 512, 4
    block_indices = torch.arange(groups, dtype=torch.int32).repeat(2, 1).npu()
    query_positions = torch.tensor([2048, 4094], dtype=torch.int64).npu()
    sequence_lengths = torch.tensor([4096, 4096], dtype=torch.int32).npu()
    token_to_req = torch.tensor([0, 1], dtype=torch.int32).npu()

    actual = pack_qsa_group_indices_npu(
        block_indices,
        query_positions,
        sequence_lengths,
        token_to_req,
        token_topk=2048,
        compress_ratio=ratio,
    )

    expected = torch.full((2, groups + ratio - 1), -1, dtype=torch.int32)
    expected[0, :512] = torch.arange(512, dtype=torch.int32)
    expected[0, 512] = 2048
    expected[1, :512] = torch.arange(512, dtype=torch.int32)
    expected[1, 512:515] = torch.arange(4092, 4095, dtype=torch.int32)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@torch.inference_mode()
def test_qsa_fia_workspace_wide_to_narrow_reuse() -> None:
    workspace = QSAFIAMaskWorkspace(torch.device("npu:0"), max_rows=2, max_bytes=4096, max_update_width=520)
    compact = torch.full((2, 131), -1, dtype=torch.int32, device="npu")
    compact[0, :2] = torch.tensor([1, 5], dtype=torch.int32, device="npu")
    compact[1, :2] = torch.tensor([2, 9], dtype=torch.int32, device="npu")
    visible = torch.tensor([256, 256], dtype=torch.int32, device="npu")

    actual, selected = build_qsa_fia_mask(compact, visible, kv_length=256, mask_width=256, workspace=workspace)

    expected = torch.ones((2, 256), dtype=torch.bool)
    expected[0, 4:8] = False
    expected[0, 20:24] = False
    expected[1, 8:12] = False
    expected[1, 36:40] = False
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(selected.cpu(), torch.ones(2, dtype=torch.bool), rtol=0, atol=0)


@torch.inference_mode()
def test_qsa_gbsa_preparation_ragged_noncontiguous_table() -> None:
    compact = torch.tensor(
        [[0, 2, -1, 4, -1, -1, -1, -1, 99, 99, 99], [0, 1, 2, 3, 4, 5, 6, 7, 99, 99, 99]],
        dtype=torch.int32,
        device="npu",
    )
    positions = torch.tensor([18, 31], dtype=torch.int64, device="npu")
    lengths = torch.tensor([19, 32], dtype=torch.int32, device="npu")
    token_to_req = torch.tensor([0, 1], dtype=torch.int32, device="npu")
    table_storage = torch.arange(24, dtype=torch.int32, device="npu").reshape(2, 12)
    block_table = table_storage[:, ::2]

    sparse, counts, visible, cu_query, row_table = prepare_qsa_gbsa_inputs(
        compact, positions, lengths, token_to_req, block_table, num_kv_heads=2
    )

    sparse_cpu = sparse.cpu()
    for head in range(2):
        torch.testing.assert_close(sparse_cpu[head, 0, :3], torch.tensor([0, 2, 4], dtype=torch.int32), rtol=0, atol=0)
        torch.testing.assert_close(sparse_cpu[head, 1, :8], torch.arange(8, dtype=torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(counts.cpu(), torch.tensor([[3, 8], [3, 8]], dtype=torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(visible.cpu(), torch.tensor([19, 32], dtype=torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(cu_query.cpu(), torch.tensor([0, 1, 2], dtype=torch.int64), rtol=0, atol=0)
    torch.testing.assert_close(row_table.cpu(), block_table.cpu(), rtol=0, atol=0)


@torch.inference_mode()
def test_qsa_fused_key_position_store() -> None:
    key_cache = torch.full((2, 4, 1, 8), -7.0, dtype=torch.bfloat16, device="npu")
    position_cache = torch.full((2, 4, 1, 3), -7, dtype=torch.int64, device="npu")
    slots = torch.tensor([0, 3, 4, -1, 8], dtype=torch.int64, device="npu")
    key_rows = torch.arange(40, dtype=torch.bfloat16, device="npu").reshape(5, 1, 8)
    position_rows = torch.arange(15, dtype=torch.int64, device="npu").reshape(5, 1, 3)

    qsa_store_key_position_rows(key_cache, position_cache, slots, key_rows, position_rows)

    expected_keys = torch.full((2, 4, 1, 8), -7.0, dtype=torch.bfloat16)
    expected_positions = torch.full((2, 4, 1, 3), -7, dtype=torch.int64)
    for row, slot in enumerate((0, 3, 4)):
        expected_keys.view(8, 1, 8)[slot] = key_rows[row].cpu()
        expected_positions.view(8, 1, 3)[slot] = position_rows[row].cpu()
    torch.testing.assert_close(key_cache.cpu(), expected_keys, rtol=0, atol=0)
    torch.testing.assert_close(position_cache.cpu(), expected_positions, rtol=0, atol=0)


@torch.inference_mode()
def test_qsa_prepared_lengths_and_split_gbsa_match_monolithic() -> None:
    lengths = torch.tensor([0, 1, 3, 4, 19, 32], dtype=torch.int32, device="npu")
    compressed = torch.empty_like(lengths)
    residual = torch.empty_like(lengths)
    prepare_qsa_lengths(lengths, compressed, residual, lengths.numel())
    torch.testing.assert_close(compressed.cpu(), lengths.cpu() // 4, rtol=0, atol=0)
    torch.testing.assert_close(residual.cpu(), lengths.cpu() % 4, rtol=0, atol=0)

    compact = torch.tensor(
        [[0, 2, -1, 4, -1, -1, -1, -1, 99, 99, 99], [0, 1, 2, 3, 4, 5, 6, 7, 99, 99, 99]],
        dtype=torch.int32,
        device="npu",
    )
    positions = torch.tensor([18, 31], dtype=torch.int64, device="npu")
    seq_lens = torch.tensor([19, 32], dtype=torch.int32, device="npu")
    token_to_req = torch.tensor([0, 1], dtype=torch.int32, device="npu")
    table_storage = torch.arange(24, dtype=torch.int32, device="npu").reshape(2, 12)
    block_table = table_storage[:, ::2]
    expected_sparse, expected_counts, expected_visible, expected_cu, expected_table = prepare_qsa_gbsa_inputs(
        compact, positions, seq_lens, token_to_req, block_table, num_kv_heads=2
    )

    visible = torch.empty(2, dtype=torch.int32, device="npu")
    row_table = torch.empty((2, 6), dtype=torch.int32, device="npu")
    prepare_qsa_gbsa_shared(
        positions,
        seq_lens,
        token_to_req,
        block_table,
        visible,
        row_table,
        rows=2,
    )
    actual_sparse, actual_counts = pack_qsa_gbsa_indices(compact, visible, num_kv_heads=2)
    cu_query = torch.arange(3, dtype=torch.int64, device="npu")
    torch.testing.assert_close(actual_sparse.cpu(), expected_sparse.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(actual_counts.cpu(), expected_counts.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(visible.cpu(), expected_visible.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(cu_query.cpu(), expected_cu.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(row_table.cpu(), expected_table.cpu(), rtol=0, atol=0)


@torch.inference_mode()
def test_qsa_pack_and_main_kv_store_fusion() -> None:
    rows, groups, ratio, width = 2, 512, 4, 256
    block_indices = torch.arange(groups, dtype=torch.int32).repeat(rows, 1).npu()
    query_positions = torch.tensor([2048, 4094], dtype=torch.int64, device="npu")
    sequence_lengths = torch.tensor([4096, 4096], dtype=torch.int32, device="npu")
    token_to_req = torch.tensor([0, 1], dtype=torch.int32, device="npu")
    key = torch.arange(rows * width, dtype=torch.bfloat16, device="npu").reshape(rows, 1, width)
    value = (key + 7).to(torch.bfloat16)
    key_cache = torch.full((2, 4, 1, width), -1, dtype=torch.bfloat16, device="npu")
    value_cache = torch.full_like(key_cache, -1)
    slots = torch.tensor([0, 7], dtype=torch.int64, device="npu")

    actual = pack_qsa_group_indices_npu(
        block_indices,
        query_positions,
        sequence_lengths,
        token_to_req,
        compress_ratio=ratio,
        token_topk=2048,
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slots,
    )
    expected = pack_qsa_group_indices_npu(
        block_indices,
        query_positions,
        sequence_lengths,
        token_to_req,
        compress_ratio=ratio,
        token_topk=2048,
    )
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(key_cache.view(8, 1, width)[0].cpu(), key[0].cpu(), rtol=0, atol=0)
    torch.testing.assert_close(key_cache.view(8, 1, width)[7].cpu(), key[1].cpu(), rtol=0, atol=0)
    torch.testing.assert_close(value_cache.view(8, 1, width)[0].cpu(), value[0].cpu(), rtol=0, atol=0)
    torch.testing.assert_close(value_cache.view(8, 1, width)[7].cpu(), value[1].cpu(), rtol=0, atol=0)


@torch.inference_mode()
def test_qsa_combined_kv_store_and_sigmoid_mul() -> None:
    width = 256
    key_cache = torch.full((2, 4, 1, width), -1, dtype=torch.bfloat16, device="npu")
    value_cache = torch.full_like(key_cache, -1)
    slots = torch.tensor([1, 6, -1], dtype=torch.int64, device="npu")
    key = torch.arange(3 * width, dtype=torch.bfloat16, device="npu").reshape(3, 1, width)
    value = (key + 11).to(torch.bfloat16)
    qsa_store_kv_rows(key_cache, value_cache, slots, key, value)
    torch.testing.assert_close(key_cache.view(8, 1, width)[1].cpu(), key[0].cpu(), rtol=0, atol=0)
    torch.testing.assert_close(key_cache.view(8, 1, width)[6].cpu(), key[1].cpu(), rtol=0, atol=0)
    torch.testing.assert_close(value_cache.view(8, 1, width)[1].cpu(), value[0].cpu(), rtol=0, atol=0)
    torch.testing.assert_close(value_cache.view(8, 1, width)[6].cpu(), value[1].cpu(), rtol=0, atol=0)

    output = torch.randn(400, 4096, dtype=torch.bfloat16, device="npu")
    gate_storage = torch.randn(400, 8192, dtype=torch.bfloat16, device="npu")
    gate = gate_storage[:, :4096]
    expected = (output.float() * torch.sigmoid(gate.float())).to(torch.bfloat16)
    qsa_sigmoid_mul_(output, gate)
    torch.testing.assert_close(output.cpu(), expected.cpu(), rtol=1e-2, atol=1e-2)


@torch.inference_mode()
def test_qsa_lightning_prepared_inputs_and_fused_main_store() -> None:
    requests, rows, heads = 4, 16, 4
    seq_lens = torch.tensor([32, 33, 34, 35], dtype=torch.int32, device="npu")
    query_start = torch.arange(0, rows + 1, 4, dtype=torch.int32, device="npu")
    token_to_req = torch.arange(requests, dtype=torch.int32, device="npu").repeat_interleave(4)
    offsets = torch.arange(4, dtype=torch.int64, device="npu").repeat(requests)
    positions = seq_lens.to(torch.int64).repeat_interleave(4) - 4 + offsets
    query = torch.randn(rows, heads, 128, dtype=torch.bfloat16, device="npu")
    compressed_cache = torch.randn(requests, 16, 1, 128, dtype=torch.bfloat16, device="npu")
    block_table = torch.arange(requests, dtype=torch.int32, device="npu").view(requests, 1)

    baseline = qsa_select_paged_tokens_lightning_v2(
        query,
        compressed_cache,
        block_table,
        token_to_req,
        positions,
        seq_lens,
        query_start,
        token_topk=2048,
        compress_ratio=4,
        num_actual_requests=requests,
        num_actual_tokens=rows,
    )

    compressed = torch.empty(requests, dtype=torch.int32, device="npu")
    residual = torch.empty(requests, dtype=torch.int32, device="npu")
    prepare_qsa_lengths(seq_lens, compressed, residual, requests)
    score_weight = math.ldexp(round(math.ldexp(1.0 / math.sqrt(128), 11)), -11)
    weights = torch.full((rows, heads), score_weight, dtype=torch.float32, device="npu")
    out = torch.full((rows, 2051), -1, dtype=torch.int32, device="npu")
    key = torch.randn(rows, 1, 256, dtype=torch.bfloat16, device="npu")
    value = torch.randn_like(key)
    key_cache = torch.zeros(1, 128, 1, 256, dtype=torch.bfloat16, device="npu")
    value_cache = torch.zeros_like(key_cache)
    slots = torch.arange(rows, dtype=torch.int64, device="npu")
    actual = qsa_select_paged_tokens_lightning_v2(
        query,
        compressed_cache,
        block_table,
        token_to_req,
        positions,
        seq_lens,
        query_start,
        token_topk=2048,
        compress_ratio=4,
        out=out,
        num_actual_requests=requests,
        num_actual_tokens=rows,
        out_initialized=True,
        prepared_compressed_lengths=compressed,
        prepared_residual_lengths=residual,
        prepared_score_weights=weights,
        main_key=key,
        main_value=value,
        main_key_cache=key_cache,
        main_value_cache=value_cache,
        main_slot_mapping=slots,
    )
    torch.testing.assert_close(actual.cpu(), baseline.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(key_cache[0, :rows].cpu(), key.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(value_cache[0, :rows].cpu(), value.cpu(), rtol=0, atol=0)
