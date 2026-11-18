# SPDX-License-Identifier: Apache-2.0
"""Exercise production GBSA preparation and packing on CPU with a Triton interpreter."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.ut.models.qwen4_exp.test_gbsa_preparation import _Kernel, _Language, load_cpu_preparation

ROOT = Path(__file__).resolve().parents[4]


def load_cpu_packing():
    source = ROOT / "vllm_ascend/ops/triton/qwen4_exp/gbsa_packing.py"
    module = ast.parse(source.read_text())
    module.body = [node for node in module.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    language = _Language()
    namespace = {
        "torch": torch,
        "tl": language,
        "triton": SimpleNamespace(
            cdiv=lambda n, d: (n + d - 1) // d,
            next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
        ),
    }
    exec(compile(module, str(source), "exec"), namespace)
    namespace["_pack_selected_blocks"] = _Kernel(namespace["_pack_selected_blocks"], language)
    return namespace["pack_selected_kv"]


def load_cpu_adapter(enabled):
    source = ROOT / "vllm_ascend/models/qwen4_exp/gbsa.py"
    module = ast.parse(source.read_text())
    module.body = [node for node in module.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    namespace = {
        "torch": torch,
        "envs": SimpleNamespace(VLLM_BATCH_INVARIANT=enabled),
        "prepare_qsa_gbsa_inputs": load_cpu_preparation(),
        "pack_selected_kv": load_cpu_packing(),
    }
    exec(compile(module, str(source), "exec"), namespace)
    return namespace


def make_metadata():
    metadata = torch.arange(1024, dtype=torch.int32)
    metadata[:3] = torch.tensor([0x5341534D, 8, 272])
    return metadata


def test_non_split_schedule_preserves_payload_and_is_idempotent():
    select = load_cpu_adapter(True)["_use_batch_invariant_gbsa_schedule"]
    metadata = make_metadata()
    expected = metadata.clone()
    expected[5:11] = torch.tensor([0, 0, 0, 2, 0, 0])
    select(metadata)
    torch.testing.assert_close(metadata, expected, rtol=0, atol=0)
    select(metadata)
    torch.testing.assert_close(metadata, expected, rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["dtype", "size", "magic", "version", "header_size"])
def test_non_split_schedule_rejects_unknown_layout_before_mutation(invalid):
    metadata = make_metadata()
    if invalid == "dtype":
        metadata = metadata.long()
    elif invalid == "size":
        metadata = metadata[:-1]
    else:
        metadata[{"magic": 0, "version": 1, "header_size": 2}[invalid]] += 1
    before = metadata.clone()
    with pytest.raises(RuntimeError, match="metadata|scheduling"):
        load_cpu_adapter(True)["_use_batch_invariant_gbsa_schedule"](metadata)
    torch.testing.assert_close(metadata, before, rtol=0, atol=0)


def make_packing_inputs(length, groups, heads, dim, dtype):
    generator = torch.Generator().manual_seed(38198)
    rows = 4
    pages = (length + 127) // 128
    # Interleaved K/V pages and sliced metadata reproduce non-contiguous sources.
    backing = torch.randn(rows * pages, 2, 128, heads, dim, generator=generator).to(dtype)
    table = torch.zeros(rows, pages * 2, dtype=torch.int32)[:, ::2]
    table.copy_(torch.randperm(rows * pages, generator=generator).reshape(rows, pages))
    indices = torch.full((heads, rows, groups * 2), -1, dtype=torch.int32)[:, :, ::2]
    counts = torch.zeros(heads, rows * 2, dtype=torch.int32)[:, ::2]
    visible = torch.full((rows,), length, dtype=torch.int32)
    for row in range(rows):
        full_count = max(0, min(length // 4, groups - int(length % 4 != 0)) - row)
        selected = torch.randperm(length // 4, generator=generator)[:full_count].tolist()
        if length % 4:
            selected.append(length // 4)
        indices[:, row, : len(selected)] = torch.tensor(selected, dtype=torch.int32)
        counts[:, row] = len(selected)
    return backing, table, indices, counts, visible


def assert_packed_values(packed, backing, table, indices, counts, visible):
    keys, values, packed_indices, lengths, packed_table = packed
    rows = table.shape[0]
    capacity = keys.shape[0] // rows * 128
    torch.testing.assert_close(packed_table.cpu(), torch.arange(keys.shape[0], dtype=table.dtype).reshape(rows, -1))
    for row in range(rows):
        selected = indices[0, row, : counts[0, row]].tolist()
        tokens = [token for block in selected for token in range(block * 4, block * 4 + 4) if token < visible[row]]
        assert lengths[row].item() == len(tokens)
        for head in range(indices.shape[0]):
            expected_indices = torch.full((indices.shape[-1],), -1, dtype=indices.dtype)
            expected_indices[: len(selected)] = torch.arange(len(selected), dtype=indices.dtype)
            torch.testing.assert_close(packed_indices[head, row].cpu(), expected_indices)
        for component, actual in enumerate((keys, values)):
            expected = torch.zeros(capacity, backing.shape[-2], backing.shape[-1], dtype=backing.dtype)
            for slot, token in enumerate(tokens):
                expected[slot] = backing[table[row, token // 128], component, token % 128].cpu()
            torch.testing.assert_close(
                actual.reshape(rows, capacity, *actual.shape[2:])[row].cpu(), expected, rtol=0, atol=0
            )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("heads", [1, 2])
@pytest.mark.parametrize("length,groups", [(4, 8), (129, 33), (130, 33), (131, 33), (20481, 512)])
def test_packing_preserves_values_under_page_relocation_and_batch_permutation(dtype, heads, length, groups):
    backing, table, indices, counts, visible = make_packing_inputs(length, groups, heads, 7, dtype)
    original = backing.clone()
    pack = load_cpu_packing()
    packed = pack(backing[:, 0], backing[:, 1], table, indices, counts, visible)
    assert_packed_values(packed, backing, table, indices, counts, visible)

    page_order = torch.arange(backing.shape[0] - 1, -1, -1)
    relocated = torch.empty_like(backing)
    relocated[page_order] = backing
    relocated_table = page_order[table].to(table.dtype)
    order = [3, 0, 2, 1]
    mixed = pack(
        relocated[:, 0], relocated[:, 1], relocated_table[order], indices[:, order], counts[:, order], visible[order]
    )
    rows = len(order)
    for component in (0, 1):
        shape = (rows, -1, heads, 7)
        torch.testing.assert_close(
            mixed[component].reshape(shape), packed[component].reshape(shape)[order], rtol=0, atol=0
        )
    for row in range(rows):
        solo = pack(
            relocated[:, 0],
            relocated[:, 1],
            relocated_table[row : row + 1],
            indices[:, row : row + 1],
            counts[:, row : row + 1],
            visible[row : row + 1],
        )
        for component in (0, 1):
            torch.testing.assert_close(
                solo[component], packed[component].reshape(rows, *solo[component].shape)[row], rtol=0, atol=0
            )
    torch.testing.assert_close(backing, original, rtol=0, atol=0)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("rows", [1, 32, 33, 65])
def test_adapter_routes_all_rows_and_uses_matching_metadata(enabled, rows):
    namespace = load_cpu_adapter(enabled)
    calls = []
    metadata = make_metadata()
    namespace["generic_block_sparse_attention_metadata"] = lambda *args, **kwargs: metadata.clone()

    def attention(query, keys, values, indices, counts, block_shape, **kwargs):
        received = kwargs["metadata"]
        expected = metadata.clone()
        if enabled:
            expected[5:11] = torch.tensor([0, 0, 0, 2, 0, 0])
        torch.testing.assert_close(received, expected)
        assert kwargs["cu_seqlens_q"].tolist() == list(range(len(query) + 1))
        calls.append(len(query))
        table = kwargs["block_table"]
        # Read the first selected value through the exact page table passed to CANN.
        first_token = indices[0, :, 0].long() * 4
        pages = table[torch.arange(len(query)), first_token // 128].long()
        first_value = values[pages, first_token % 128, 0, 0]
        return query + first_value[:, None, None], torch.empty(0)

    namespace["generic_block_sparse_attention"] = attention
    if not enabled:

        def unexpected_packing(*args):
            pytest.fail("packing must only run in batch-invariant mode")

        namespace["pack_selected_kv"] = unexpected_packing
    query = torch.arange(rows, dtype=torch.bfloat16)[:, None, None].expand(rows, 2, 8).clone()
    backing = (torch.arange(rows * 2 * 128 * 8) % 251).to(torch.bfloat16).reshape(rows, 2, 128, 1, 8)
    table = torch.arange(rows - 1, -1, -1, dtype=torch.int32)[:, None]
    compact = torch.tensor([1, 0, -1, -1, -1], dtype=torch.int32).repeat(rows, 1)
    lengths = torch.full((rows,), 9, dtype=torch.int32)
    output = torch.full_like(query, float("nan"))
    result = namespace["qsa_gbsa_attention"](
        query, backing[:, 0], backing[:, 1], compact, table, torch.arange(rows), lengths - 1, lengths, output
    )
    assert result is output
    expected = query + backing[table[:, 0].long(), 1, 4, 0, 0][:, None, None]
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert calls == ([min(32, rows - start) for start in range(0, rows, 32)] if enabled else [rows])
