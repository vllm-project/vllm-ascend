# SPDX-License-Identifier: Apache-2.0

import ast
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

MODEL_DIR = Path(__file__).resolve().parents[4] / "vllm_ascend/models/qwen4_exp"


@pytest.fixture
def ops():
    scope = {"torch": torch, "dataclass": dataclass, "replace": replace, "__name__": __name__}
    for filename in ("gbsa.py", "attention_dispatch.py", "hisparse.py"):
        source = MODEL_DIR / filename
        module = ast.parse(source.read_text())
        module.body = [
            node
            for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.Assign))
            or (filename == "attention_dispatch.py" and isinstance(node, ast.ClassDef))
            or (isinstance(node, ast.ImportFrom) and node.module == "__future__")
        ]
        exec(compile(module, str(source), "exec"), scope)
    return scope


def test_bind_hisparse_uses_kernel_block_views_only_for_side_caches(ops):
    class CacheLayer:
        bind_kv_cache = None

    class Mamba(CacheLayer):
        pass

    class RawKeyCache(CacheLayer):
        pass

    class CompressedKeyCache(CacheLayer):
        pass

    ops["MambaBase"] = Mamba
    ops["qsa_indexer"] = SimpleNamespace(QSARawKeyCache=RawKeyCache, QSACompressedKeyCache=CompressedKeyCache)
    recurrent = Mock(spec=Mamba, kv_cache=torch.empty(3, 1, 1, 64, dtype=torch.int8))
    indexer = Mock(spec=RawKeyCache, kv_cache=torch.empty(3, 1, 32, 64))
    already_bound = Mock(spec=CompressedKeyCache, kv_cache=torch.empty(3, 32, 1, 64))
    token_cache = Mock(kv_cache=torch.empty(3, 128, 128))
    layers = {"gdn": recurrent, "raw": indexer, "compressed": already_bound, "token": token_cache}

    ops["bind_qsa_hisparse_kv_caches"](layers, [*layers, "token.hisparse_resident"])

    recurrent.bind_kv_cache.assert_called_once_with(recurrent.kv_cache)
    indexer.bind_kv_cache.assert_called_once_with(indexer.kv_cache)
    already_bound.bind_kv_cache.assert_not_called()
    token_cache.bind_kv_cache.assert_not_called()


@pytest.mark.parametrize("length_dtype", [torch.int32, torch.int64])
def test_resolver_keeps_partial_group_after_invalid_lanes(ops, length_dtype):
    compact = torch.full((4, 515), -1, dtype=torch.int32)
    compact[:, :512] = torch.arange(512)
    compact[1, 0] = -1
    visible = torch.tensor([2049, 2051, 2048, 0], dtype=length_dtype)
    logical = torch.empty((4, 2048), dtype=torch.int32)
    ops["prepare_resolver_indices"](compact, visible, logical)
    assert logical[0, :2044].tolist() == list(range(2044))
    assert logical[0, 2044:].tolist() == [2048, -1, -1, -1]
    assert logical[1].tolist() == [*range(4, 2051), -1]
    assert logical[2].tolist() == list(range(2048))
    assert (logical[3] == -1).all()


@pytest.mark.parametrize("kv_heads", [1, 2])
@pytest.mark.parametrize("block_stride", [4, 5], ids=["dense", "gapped"])
def test_physical_attention_reads_selected_rows_per_kv_head(ops, kv_heads, block_stride):
    dim = 8
    query = torch.zeros(2, 4, dim)
    storage = torch.arange(3 * block_stride * kv_heads * 2 * dim, dtype=query.dtype)
    packed = storage.view(3, block_stride, -1)[:, :4]
    packed[0, 0] = torch.nan
    physical = torch.tensor([[2 * block_stride + 1, -1, 2, block_stride + 2], [-1] * 4], dtype=torch.int32)
    expected_rows = packed[[2, 0, 1], [1, 2, 2]].view(3, kv_heads, 2 * dim)

    def reference_gbsa(q, key, value, compact, table, requests, positions, lengths, output):
        assert table.tolist() == [[0], [1]]
        assert positions.tolist() == [2, -1]
        assert lengths.tolist() == [3, 0]
        assert requests.tolist() == [0, 1]
        assert compact.is_contiguous()
        assert compact[:, 0].tolist() == [0, 0]
        torch.testing.assert_close(torch.cat((key, value), dim=-1)[0, :3], expected_rows)
        assert not key[1].any() and not value[1].any()
        output.fill_(1)

    ops["qsa_gbsa_attention"] = reference_gbsa
    actual = ops["attend_physical"](query, packed, physical, kv_heads, torch.empty_like(query))
    expected = torch.ones_like(query)
    expected[1].zero_()
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("num_decodes", "prefill_start", "all_resident"),
    [
        pytest.param(2, 0, True, id="resident-mixed"),
        pytest.param(2, 0, False, id="swap-in-mixed"),
        pytest.param(0, 24 * 1024, True, id="long-prefill"),
        pytest.param(2, 24 * 1024, True, id="long-prefill-with-decode"),
    ],
)
def test_hisparse_batch_routes_decode_and_staged_prefill(ops, num_decodes, prefill_start, all_resident):
    num_prefill_tokens = 3
    num_tokens = num_decodes + num_prefill_tokens
    prefill_length = prefill_start + num_prefill_tokens
    resident = torch.empty(3, 128, 16)
    staged_table = torch.tensor([[0]], dtype=torch.int32)
    group = Mock(logical_topk_indices=torch.empty(num_tokens, 8, dtype=torch.int32))
    group.physical_kv_cache.return_value = resident
    group.stage_prefill_rows.return_value = (
        resident[:1],
        staged_table,
        torch.zeros(num_prefill_tokens, dtype=torch.int32),
    )
    cache = Mock(
        all_context_pages_resident=all_resident,
        block_table=torch.tensor([[11], [21], [31]], dtype=torch.int32),
        source_block_table=torch.tensor([[10], [20], [30]], dtype=torch.int32),
        runtime=SimpleNamespace(
            request_state_indices=torch.tensor([2, 0, 1], dtype=torch.int32),
            host_cache=resident,
            row_width=16,
        ),
    )
    slots = torch.arange(num_tokens)
    cache.write_target.return_value = resident, slots, num_tokens
    layer = SimpleNamespace(
        index_group=group,
        hisparse_cache=cache,
        indexer=SimpleNamespace(token_topk=8, compress_ratio=4),
        num_kv_heads=1,
        head_dim=8,
    )
    plan = ops["build_qsa_request_plan"](
        torch.tensor([0, num_prefill_tokens]), torch.tensor([prefill_length]), num_prefill_tokens, torch.tensor([True])
    )
    requests = torch.tensor([*range(num_decodes), *([num_decodes] * num_prefill_tokens)], dtype=torch.int32)
    meta = SimpleNamespace(
        num_actual_tokens=num_tokens,
        num_decode_tokens=num_decodes,
        num_reqs=num_decodes + 1,
        num_decodes=num_decodes,
        slot_mapping=slots,
        block_size=128,
        req_id_per_token=requests,
        query_start_loc=torch.tensor([*range(num_decodes + 1), num_tokens], dtype=torch.int32),
        prefill_dispatch=plan,
    )
    side = SimpleNamespace(
        token_to_req=requests,
        logical_positions=torch.tensor([*([8191] * num_decodes), *range(prefill_start, prefill_length)]),
        seq_lens=torch.tensor([*([8192] * num_decodes), prefill_length], dtype=torch.int32),
    )
    for name in (
        "_store_qsa_cache_rows",
        "qsa_resident_gbsa_attention",
        "attend_physical",
        "qsa_fia_attention",
        "qsa_gbsa_attention",
        "qsa_union_gbsa_attention",
    ):
        ops[name] = Mock()
    qkv = torch.zeros(num_tokens, 1, 8)
    compact = torch.tensor([[0, 1, -1, -1, -1]], dtype=torch.int32).repeat(num_tokens, 1)
    ops["run_qsa_hisparse"](layer, meta, side, compact, qkv, qkv, qkv, qkv)

    group.stage_prefill_rows.assert_called_once()
    resident_attention = ops["qsa_resident_gbsa_attention"]
    if num_decodes and all_resident:
        fast = resident_attention.call_args
        assert fast.args[0].shape[0] == num_decodes and fast.args[3] is cache.block_table
        assert fast.kwargs["request_state_indices"] is cache.runtime.request_state_indices
        assert fast.args[4].tolist() == list(range(num_decodes))
    else:
        resident_attention.assert_not_called()
    if all_resident:
        cache.swap_in.assert_not_called()
        ops["attend_physical"].assert_not_called()
    else:
        cache.swap_in.assert_called_once()
        swap = cache.swap_in.call_args
        assert swap.args[0].tolist() == list(range(num_decodes))
        assert swap.kwargs["block_table"] is cache.source_block_table
        assert swap.kwargs["block_size"] == meta.block_size
        assert swap.kwargs["num_valid_rows"].tolist() == [num_tokens]
        assert swap.kwargs["logical_topk_indices"].tolist() == [list(range(8))] * num_decodes
        group.set_logical_topk_ready.assert_called_once_with(0)
        assert ops["attend_physical"].call_args.args[1:4] == (resident, cache.swap_in.return_value, 1)

    fia = ops["qsa_fia_attention"]
    paged = ops["qsa_gbsa_attention"]
    ops["qsa_union_gbsa_attention"].assert_not_called()
    if prefill_start:
        fia.assert_not_called()
        paged.assert_called_once()
        args = paged.call_args.args
        assert args[0].shape[0] == num_prefill_tokens and args[4] is staged_table
        assert args[5].tolist() == [0] * num_prefill_tokens
        assert args[6].tolist() == list(range(prefill_start, prefill_length))
        assert args[7].tolist() == [prefill_length]
    else:
        paged.assert_not_called()
        fia.assert_called_once()
        assert fia.call_args.args[0].shape[0] == num_prefill_tokens
