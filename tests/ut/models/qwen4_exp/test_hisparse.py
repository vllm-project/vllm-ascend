# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

MODEL_DIR = Path(__file__).resolve().parents[4] / "vllm_ascend/models/qwen4_exp"


@pytest.fixture
def ops():
    scope = {"torch": torch}
    for filename in ("gbsa.py", "hisparse.py"):
        source = MODEL_DIR / filename
        module = ast.parse(source.read_text())
        module.body = [
            node
            for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.Assign))
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
def test_physical_attention_reads_selected_rows_per_kv_head(ops, kv_heads):
    rng = torch.Generator().manual_seed(17)
    heads, dim = 4, 8
    query = torch.randn(2, heads, dim, generator=rng)
    packed = torch.randn(3, 4, kv_heads * 2 * dim, generator=rng)
    packed[0, 0] = torch.nan
    physical = torch.tensor([[9, -1, 2, 6], [-1, -1, -1, -1]], dtype=torch.int32)

    def reference_gbsa(q, key, value, compact, table, requests, positions, lengths, output):
        assert table.tolist() == [[0], [1]]
        assert positions.tolist() == [2, -1]
        assert lengths.tolist() == [3, 0]
        assert requests.tolist() == [0, 1]
        assert compact[:, 0].tolist() == [0, 0]
        for head in range(heads):
            kv_head = head * kv_heads // heads
            scores = (key[0, :3, kv_head] @ q[0, head]) * dim**-0.5
            output[0, head] = scores.softmax(0) @ value[0, :3, kv_head]
        output[1].fill_(torch.nan)

    ops["qsa_gbsa_attention"] = reference_gbsa
    actual = ops["attend_physical"](query, packed, physical, kv_heads, torch.empty_like(query))
    expected = torch.zeros_like(query)
    rows = packed.reshape(-1, kv_heads, 2 * dim)[[9, 2, 6]]
    for head in range(heads):
        key, value = rows[:, head * kv_heads // heads].split(dim, dim=-1)
        expected[0, head] = ((key @ query[0, head]) * dim**-0.5).softmax(0) @ value
    torch.testing.assert_close(actual, expected)


def test_mixed_resident_batch_uses_decode_fast_path_and_stages_prefill(ops):
    resident = torch.empty(3, 128, 16)
    group = Mock()
    group.stage_prefill_rows.return_value = (
        resident[:1],
        torch.tensor([[0]], dtype=torch.int32),
        torch.tensor([0, 0, 0], dtype=torch.int32),
    )
    cache = Mock(
        all_context_pages_resident=True,
        decode_batch=False,
        block_table=torch.tensor([[11], [21], [31]], dtype=torch.int32),
        runtime=SimpleNamespace(
            request_state_indices=torch.tensor([2, 0, 1], dtype=torch.int32),
            host_cache=resident,
            row_width=16,
        ),
    )
    slots = torch.arange(5)
    cache.write_target.return_value = resident, slots, 5
    layer = SimpleNamespace(
        index_group=group,
        hisparse_cache=cache,
        indexer=SimpleNamespace(token_topk=8, compress_ratio=4),
        num_kv_heads=1,
        head_dim=8,
    )
    meta = SimpleNamespace(
        num_actual_tokens=5,
        num_decode_tokens=2,
        num_reqs=3,
        num_decodes=2,
        slot_mapping=slots,
        block_size=128,
        prefill_dispatch=object(),
    )
    side = SimpleNamespace(
        token_to_req=torch.tensor([0, 1, 2, 2, 2]),
        logical_positions=torch.tensor([8192, 8193, 0, 1, 2]),
        seq_lens=torch.tensor([8193, 8194, 3]),
    )
    for name in ("_store_qsa_cache_rows", "qsa_resident_gbsa_attention", "dispatch_qsa_attention"):
        ops[name] = Mock()
    qkv = torch.zeros(5, 1, 8)
    ops["run_qsa_hisparse"](layer, meta, side, torch.zeros(5, 5, dtype=torch.int32), qkv, qkv, qkv, qkv)

    cache.swap_in.assert_not_called()
    group.stage_prefill_rows.assert_called_once()
    fast_call = ops["qsa_resident_gbsa_attention"].call_args
    fast_args = fast_call.args
    assert fast_args[0].shape[0] == 2
    assert fast_args[3] is cache.block_table
    assert fast_call.kwargs["request_state_indices"] is cache.runtime.request_state_indices
    assert fast_args[4].tolist() == [0, 1]
    prefill_args = ops["dispatch_qsa_attention"].call_args.args
    assert prefill_args[0].shape[0] == 3
    assert prefill_args[-1] is meta.prefill_dispatch
