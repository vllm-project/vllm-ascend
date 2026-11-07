# SPDX-License-Identifier: Apache-2.0
"""CPU checks of production SP preparation methods against the existing consumer."""

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]


def method(path, class_name, name, scope):
    tree = ast.parse((ROOT / path).read_text())
    container = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    function = next(node for node in container.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(ROOT / path), "exec"), scope)
    return scope[name]


def legacy_rows(codes, scales, ids, boundaries, tp):
    """Original head-owner lookup followed by head concatenation."""
    pieces = []
    heads = ids.shape[1] // tp
    for rank in range(tp):
        lower, upper = boundaries[rank * heads], boundaries[(rank + 1) * heads]
        selected = ids[:, rank * heads : (rank + 1) * heads]
        owned = (selected >= lower) & (selected < upper)
        values = codes[selected.clamp(0, len(codes) - 1)].float() * scales[
            selected.clamp(0, len(codes) - 1)
        ].repeat_interleave(32, dim=-1)
        pieces.append(values.bfloat16().masked_fill(~owned[..., None], 0))
    return torch.cat(pieces, dim=1)


@pytest.fixture
def embedding():
    boundaries = [0, 11, 24, 41, 60]
    codes = torch.arange(60 * 32).remainder(255).sub(127).to(torch.int8).reshape(60, 32)
    scales = torch.arange(60).remainder(4).add(1).float().mul(0.125).view(60, 1)
    calls = []

    def gather(c, s, ids, **launch):
        assert ids.is_contiguous()
        assert launch["head_start"] == 0 and launch["local_heads"] == 4
        calls.append(ids.clone())
        bounds = launch["head_bounds"]
        heads = launch["heads_per_shard"]
        for head in range(4):
            lower, upper = bounds[head // heads]
            selected = ids[:, head].long()
            valid = (selected >= lower) & (selected < upper)
            safe = selected.clamp(0, 59)
            value = (codes[safe].float() * scales[safe]).bfloat16().masked_fill(~valid[:, None], 0)
            launch["output"].view(-1, 4, 32)[:, head].copy_(value)

    lookup = method(
        "vllm_ascend/models/deepseek_v41/engram/embedding.py",
        "AscendParallelEngramEmbedding",
        "lookup_token_shard",
        dict(gather_dequantize_host_uva=gather, torch=torch),
    )
    return SimpleNamespace(boundaries=boundaries, codes=codes, scales=scales, calls=calls, lookup=lookup)


def table_for(embedding, tp):
    bounds = embedding.boundaries
    table = SimpleNamespace(
        dp_size=1,
        tp_size=tp,
        n_hash_cols=4,
        dim=32,
        num_embeddings=60,
        part_n_hash_cols=4 // tp,
        _codes_uva=SimpleNamespace(global_rows=True),
        _scales_uva=object(),
        _sp_head_bounds=torch.tensor([(bounds[i * 4 // tp], bounds[(i + 1) * 4 // tp]) for i in range(tp)]),
    )
    table.lookup_token_shard = MethodType(embedding.lookup, table)
    return table


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("tokens", [0, 1, 3, 7, 32])
@pytest.mark.parametrize("extra_padding", [0, 3])
def test_sp_lookup_matches_existing_sp_consumer_and_preserves_query_count(embedding, tp, tokens, extra_padding):
    bucket = tokens + extra_padding
    ids = torch.arange(tokens * 2 * 4).remainder(60).reshape(tokens, 2, 4)[:, 1]
    if tokens:
        ids[0, 0] = -1
        ids[0, 1] = 60
        ids[-1, 0] = 59  # belongs to another original head owner's range
    full = legacy_rows(embedding.codes, embedding.scales, ids, embedding.boundaries, tp).flatten(1)
    full = F.pad(full, (0, 0, 0, bucket - tokens))
    slot = (bucket + tp - 1) // tp
    full = F.pad(full, (0, 0, 0, slot * tp - bucket))
    table = table_for(embedding, tp)
    for rank in range(tp):
        target = torch.full((slot + 2, 4 * 32), 99, dtype=torch.bfloat16)
        result = table.lookup_token_shard(ids, target, bucket, rank)
        torch.testing.assert_close(result, full[rank * slot : (rank + 1) * slot], rtol=0, atol=0)
        assert torch.all(target[slot:] == 99), "wrote beyond active graph bucket"
    assert sum(query.numel() for query in embedding.calls) == tokens * 4


def test_model_preparation_keeps_full_history_and_never_gathers_sp_rows(embedding):
    tokens, bucket, tp, rank = 5, 7, 4, 2
    hashes = torch.arange(tokens * 2 * 4).remainder(60).reshape(tokens, 2, 4)
    history = Mock(return_value=hashes)
    history.ensure_cache.return_value = True
    history.use_slot_cache = False
    tables = [table_for(embedding, tp) for _ in range(2)]
    layers = [SimpleNamespace(engram=SimpleNamespace(embed_tokens=table)) for table in tables]
    gather = Mock(side_effect=AssertionError("SP direct preparation must not gather"))
    scope = dict(
        torch=torch,
        engram_dead_mask=lambda ids, *args: ids == 999,
        get_engram_dp_size=lambda: 0,
        gather_engram_hashes=gather,
    )
    prepare = method("vllm_ascend/models/deepseek_v41/model.py", "DeepseekV41Model", "prepare_engram", scope)
    model = SimpleNamespace(
        config=SimpleNamespace(engram_layer_ids=[0, 1], image_token_id=999),
        has_engram=True,
        engram_hash=history,
        engram_dp_shared_memory=True,
        _engram_sp_direct=True,
        _engram_tp_size=tp,
        _engram_tp_rank=rank,
        layers=layers,
    )
    slot = (bucket + tp - 1) // tp
    buffers = {layer: torch.full((slot, 128), 99, dtype=torch.bfloat16) for layer in (0, 1)}
    mask = torch.zeros(bucket, dtype=torch.bool)
    input_ids = torch.arange(tokens)
    result, _ = prepare(
        model,
        input_ids,
        input_ids,
        torch.full((1, 2), -1),
        torch.tensor([0, tokens]),
        output_buffers=buffers,
        output_tokens=bucket,
        mask_output_buffer=mask,
    )
    assert history.call_args.args[0].numel() == tokens
    assert mask.tolist() == [True] * tokens + [False] * (bucket - tokens)
    gather.assert_not_called()
    for layer in (0, 1):
        reference = legacy_rows(embedding.codes, embedding.scales, hashes[:, layer], embedding.boundaries, tp).flatten(
            1
        )
        reference = F.pad(reference, (0, 0, 0, slot * tp - tokens))
        torch.testing.assert_close(result[layer], reference[rank * slot : (rank + 1) * slot], rtol=0, atol=0)


def test_non_global_backing_fails_closed(embedding):
    table = table_for(embedding, 2)
    table._codes_uva.global_rows = False
    with pytest.raises(ValueError, match="global chunk"):
        table.lookup_token_shard(torch.zeros((1, 4), dtype=torch.int64), torch.zeros((1, 128)), 1, 0)


@pytest.mark.parametrize("enabled,expected_capacity", [(False, 8), (True, 2)])
def test_persistent_lookup_capacity_shrinks_but_mask_stays_global(enabled, expected_capacity):
    allocate = method(
        "vllm_ascend/models/deepseek_v41/model.py",
        "DeepseekV41Model",
        "_get_engram_input_buffers",
        dict(torch=torch, logger=Mock()),
    )
    model = SimpleNamespace(
        _engram_input_buffers=None,
        _engram_max_tokens=8,
        _engram_sp_direct=enabled,
        _engram_tp_size=4,
        engram_rotation=torch.eye(32),
        config=SimpleNamespace(engram_layer_ids=[0, 1]),
        layers=[
            SimpleNamespace(engram=SimpleNamespace(embed_tokens=SimpleNamespace(n_hash_cols=4, dim=32)))
            for _ in range(2)
        ],
    )
    buffers, mask = allocate(model)
    assert all(tuple(value.shape) == (expected_capacity, 128) for value in buffers.values())
    assert tuple(mask.shape) == (8,)
    again = allocate(model)
    assert again is model._engram_input_buffers
    assert sum(value.numel() * value.element_size() for value in buffers.values()) == expected_capacity * 128 * 2 * 2
