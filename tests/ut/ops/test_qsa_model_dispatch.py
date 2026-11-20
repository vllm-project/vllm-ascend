# SPDX-License-Identifier: Apache-2.0

"""Exercise real model dispatch on CPU; device kernels are boundary mocks."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


def _load_dispatch(class_name, methods, base, **dependencies):
    # Keep class compilation (including the __class__ cell for super()) and
    # production predicates intact, without importing Linux/NPU dependencies.
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/qwen4_exp/qsa.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    cls.bases = [ast.Name(id="Base", ctx=ast.Load())]
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_compress_qsa_groups"]
    module = ast.fix_missing_locations(
        ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *helpers, cls],
            type_ignores=[],
        )
    )
    namespace = {"torch": torch, "Base": base, **dependencies}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[class_name]()


def _rotary():
    return SimpleNamespace(
        mrope_section=[11, 11, 10],
        rotary_dim=64,
        is_neox_style=True,
        mrope_interleaved=True,
        _match_cos_sin_cache_dtype=Mock(return_value=torch.empty(16, 64, dtype=torch.bfloat16)),
    )


@pytest.mark.parametrize("case", ["eligible", "row_stride", "column_stride", "wrong_width", "mtp"])
def test_main_projection_dispatch_uses_real_eligibility(monkeypatch, case):
    fallback_result = tuple(object() for _ in range(4))
    fused_result = tuple(object() for _ in range(4))
    fallback = Mock(return_value=fallback_result)
    base = type("Base", (), {"_project_qkv_gate": fallback})
    layer = _load_dispatch("AscendQwen4ExpQSAAttention", {"_main_fused_norm_rope_eligible", "_project_qkv_gate"}, base)
    layer.layer_name = "mtp.layers.0" if case == "mtp" else "model.layers.0"
    layer.attn_output_gate = True
    layer.num_heads, layer.num_kv_heads, layer.head_dim = 2, 1, 256
    layer.q_size, layer.kv_size = 512, 256
    layer.q_norm = SimpleNamespace(weight=torch.empty(256, dtype=torch.bfloat16), variance_epsilon=1e-6)
    layer.k_norm = SimpleNamespace(weight=torch.empty(256, dtype=torch.bfloat16), variance_epsilon=1e-6)
    layer.rotary_emb = _rotary()
    fused = Mock(return_value=fused_result)
    monkeypatch.setattr(torch.ops.vllm, "triton_split_qkv_rmsnorm_mrope", fused, raising=False)
    if case == "row_stride":
        qkv = torch.empty(4, 1536, dtype=torch.bfloat16)[::2]
    elif case == "column_stride":
        qkv = torch.empty(2, 3072, dtype=torch.bfloat16)[:, ::2]
    else:
        qkv = torch.empty(2, 1535 if case == "wrong_width" else 1536, dtype=torch.bfloat16)
    positions = torch.arange(2)
    # MTP must fall back even though its tensors satisfy the fusion predicate.
    assert layer._main_fused_norm_rope_eligible(qkv, positions) == (case in ("eligible", "mtp"))
    result = layer._project_qkv_gate(qkv, positions)
    if case == "eligible":
        assert result is fused_result
        fallback.assert_not_called()
        fused.assert_called_once()
        kwargs = fused.call_args.kwargs
        assert kwargs["qkv"] is qkv
        assert kwargs["positions"] is positions
        assert kwargs["q_weight"] is layer.q_norm.weight
        assert kwargs["k_weight"] is layer.k_norm.weight
        assert kwargs["cos_sin"] is layer.rotary_emb._match_cos_sin_cache_dtype.return_value
        assert kwargs["has_gate"] is True
    else:
        assert result is fallback_result
        fallback.assert_called_once_with(qkv, positions)
        fused.assert_not_called()
        layer.rotary_emb._match_cos_sin_cache_dtype.assert_not_called()


@pytest.mark.parametrize("head_dim", [128, 64])
@pytest.mark.parametrize("with_position_ring", [False, True])
def test_compression_dispatch_and_ring_order(head_dim, with_position_ring):
    events = []
    pooled, first_positions, normalized = (object() for _ in range(3))

    def compress(*args, **kwargs):
        events.append("fused_compress" if kwargs else "pool")
        return None if kwargs else (pooled, first_positions)

    compression = Mock(side_effect=compress)
    stores = Mock()
    canonical = Mock(side_effect=lambda positions: positions.view(-1, 1, 1).expand(-1, 1, 3))
    layer = _load_dispatch(
        "AscendQSAIndexer",
        {"_fused_norm_rope_eligible", "_update_and_compress"},
        object,
        qsa_compress_groups_with_ratio_fused=compression,
        _store_qsa_cache_rows=stores,
        qsa_indexer=SimpleNamespace(canonical_qsa_rope_positions=canonical),
    )
    layer.compress_ratio, layer.index_head_dim = 4, head_dim
    layer.k_layernorm = SimpleNamespace(weight=torch.empty(head_dim, dtype=torch.bfloat16), variance_epsilon=1e-6)
    layer.rotary_emb = _rotary()
    raw_cache = torch.empty(1, 8, 1, head_dim, dtype=torch.bfloat16)
    position_ring = torch.empty(1, 8, 1, 3, dtype=torch.int64) if with_position_ring else None
    compressed_cache = torch.empty(1, 8, 1, head_dim, dtype=torch.bfloat16)
    layer.raw_key_cache = SimpleNamespace(key_cache=raw_cache, rope_position_cache=position_ring)
    layer.compressed_key_cache = SimpleNamespace(kv_cache=compressed_cache)

    def normalize(*args):
        events.append("normalize")
        assert args == (pooled, first_positions)
        return normalized

    layer.normalize_compressed_keys = Mock(side_effect=normalize)

    def store(cache, slots, values):
        if cache is compressed_cache:
            events.append("store_compressed")
            assert values is normalized
        elif cache is raw_cache:
            events.append("store_raw")
            torch.testing.assert_close(values, token_k[:2])
        else:
            assert cache is position_ring
            events.append("store_positions")

    stores.side_effect = store
    token_k = torch.zeros(3, 1, head_dim, dtype=torch.bfloat16)
    positions = torch.arange(3)
    raw_metadata = SimpleNamespace(
        num_actual_tokens=2,
        logical_positions=torch.tensor([6, 7]),
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        token_to_req=torch.zeros(2, dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2]),
        slot_mapping=torch.tensor([6, 7]),
    )
    compressed_metadata = SimpleNamespace(slot_mapping=torch.tensor([-1, 0]))
    layer._update_and_compress(token_k, positions, raw_metadata, compressed_metadata)
    compression.assert_called_once()
    args, kwargs = compression.call_args
    assert args[0].shape == (2, 1, head_dim)  # Exclude graph-padding rows.
    assert args[8] == 4
    assert args[9] is position_ring
    if head_dim == 128:
        assert kwargs["compressed_cache"] is compressed_cache
        assert kwargs["norm_weight"] is layer.k_layernorm.weight
        layer.normalize_compressed_keys.assert_not_called()
        expected = ["fused_compress", "store_raw"]
    else:
        assert kwargs == {}  # Same wrapper, but no norm/store fusion arguments.
        layer.normalize_compressed_keys.assert_called_once()
        expected = ["pool", "normalize", "store_compressed", "store_raw"]
    if with_position_ring:
        expected.append("store_positions")
    assert events == expected
