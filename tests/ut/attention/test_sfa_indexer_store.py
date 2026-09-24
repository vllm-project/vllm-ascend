# SPDX-License-Identifier: Apache-2.0
"""Exercise the real SFA forward's cache-write ordering and fallback routing."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def run_forward(rows, state="decode", ranks=8, cache_supported=True, has_indexer=True):
    """Run real SFA control flow with CPU stand-ins for unrelated NPU math."""
    source = ROOT / "vllm_ascend/attention/sfa_v1.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendSFAImpl")
    methods = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ("forward", "indexer_select_pre_process")
    ]
    events = []
    slots = torch.arange(rows, dtype=torch.int64) + 3
    keys = torch.arange(rows * 128, dtype=torch.int64).remainder(127).to(torch.int8).view(rows, 128)
    scales = torch.full((rows,), 1.0006, dtype=torch.float32)
    cache = (
        torch.zeros(32, 2),
        torch.zeros(32, 2),
        torch.zeros(32, 128, dtype=torch.int8),
        torch.zeros(32, 1, dtype=torch.float16),
    )
    if not has_indexer:
        cache = cache[:2]

    def scatter(destination, indices, values):
        assert "wait" in events
        events.append("scatter")
        destination.index_copy_(0, indices.flatten(), values)

    def store(quant, stats, indices, key_cache, scale_cache):
        assert "wait" in events and "publish" not in events
        assert stats.dtype == torch.float32
        events.append("fused_store")
        key_cache.index_copy_(0, indices, quant)
        scale_cache.index_copy_(0, indices, stats.to(torch.float16))

    namespace = {
        "torch": torch,
        "torch_npu": SimpleNamespace(npu_dynamic_quant=lambda *_, **__: (keys, scales), npu_scatter_nd_update_=scatter),
        "HAS_TRITON": True,
        "rope_forward_triton_siso": lambda x, *_, **__: x,
        "AscendSFAImpl": SimpleNamespace(k_hadamard=torch.eye(128, dtype=torch.bfloat16)),
        "AscendAttentionState": SimpleNamespace(DecodeOnly="decode", SpecDecoding="spec"),
        "PreprocessType": SimpleNamespace(NATIVE="native", MLAPO="mlapo", PROLOG_V3="prolog"),
        "MLAPO_MAX_SUPPORTED_TOKENS": 512,
        "can_fuse_store": lambda *args: cache_supported,
        "store_indexer_key_scale": store,
        "wait_for_kv_layer_from_connector": lambda *_: events.append("wait"),
        "notify_kv_cache_written": lambda *_: events.append("publish"),
        "record_attention_compute_start": lambda: events.append("read_fence"),
        "maybe_save_kv_layer_to_connector": lambda *_: None,
    }
    future = ast.parse("from __future__ import annotations").body
    exec(compile(ast.Module(body=future + methods, type_ignores=[]), str(source), "exec"), namespace)
    backend = SimpleNamespace(
        enable_dsa_cp=False,
        enable_dsa_cp_with_o_proj_tp=False,
        enable_sp=False,
        enable_sparse_li_c8=True,
        enable_sparse_sfa_c8=False,
        has_indexer=has_indexer,
        dcp_size=ranks,
        head_dim=128,
        qk_rope_head_dim=64,
        q_lora_rank=2,
        kv_lora_rank=2,
        kv_cache_indexer_k_idx=2,
        kv_cache_indexer_scale_idx=3,
        preprocess_type="native",
        skip_topk=not has_indexer,
        use_index_cache=False,
        is_rope_neox_style=False,
        c8_k_cache_dtype=torch.int8,
        c8_k_scale_cache_dtype=torch.float16,
        _compose_sfa_kv_cache=lambda value: value,
        _get_sfa_kv_slot_mapping=lambda metadata: metadata.slot_mapping,
        _use_li_c8_reshape_optim=lambda: False,
        fused_qkv_a_proj=lambda x: (torch.zeros(rows, 68), None),
        q_a_layernorm=lambda x: x,
        wk_weights_proj=lambda x: (torch.zeros(rows, 160, dtype=torch.bfloat16), None),
        k_norm=lambda x: x,
        exec_kv=lambda *args: (None, None, None),
        _maybe_gather_kv_for_dsacp=lambda *a: (a[3], a[4], None, []),
        _q_proj_and_k_up_proj=lambda x: (x, x),
        rope_single=lambda x, *args: x,
        _record_query_gather_context=lambda *args: None,
        _maybe_store_kvcache_for_c8_n_dsacp=lambda *a: (None, None, a[3], None, None),
        _get_full_kv=lambda value, *args: value,
        indexer_select_post_process=lambda **kwargs: torch.zeros(rows, 1, dtype=torch.int32),
        _get_indexcache_topk_indices=lambda *_: torch.zeros(rows, 1, dtype=torch.int32),
        _execute_sparse_flash_attention_process=lambda *args: torch.ones(rows, 4),
        _v_up_proj=lambda x: x,
        o_proj=lambda x: (x, None),
        layer_name="test_layer",
    )
    backend.indexer_select_pre_process = lambda **kwargs: namespace["indexer_select_pre_process"](backend, **kwargs)
    metadata = SimpleNamespace(
        cos=torch.ones(rows, 64),
        sin=torch.zeros(rows, 64),
        slot_mapping=slots,
        cum_query_lens=None,
        seq_lens=None,
        num_input_tokens=rows,
        attn_state=state,
        dsa_cp_context=None,
    )
    namespace["forward"](backend, "test_layer", torch.zeros(rows, 4), cache, metadata, output=torch.empty(rows, 4))
    assert events.index("wait") < events.index("publish") < events.index("read_fence")
    if not has_indexer:
        assert "fused_store" not in events and "scatter" not in events
        return events
    assert torch.equal(cache[2][slots], keys)
    assert torch.equal(cache[3][slots], scales.to(torch.float16).view(rows, 1))
    assert not cache[2][:3].any() and not cache[3][:3].any()
    return events


@pytest.mark.parametrize("rows,state", [(1, "decode"), (6, "spec"), (12, "spec")])
def test_direct_store_preserves_native_quant_bytes_and_publication_order(rows, state):
    events = run_forward(rows, state)
    assert events.count("fused_store") == 1 and "scatter" not in events


def test_layers_reusing_topk_do_not_access_missing_indexer_cache():
    run_forward(6, has_indexer=False)


@pytest.mark.parametrize("kwargs", [{"state": "prefill"}, {"ranks": 2}, {"cache_supported": False}])
def test_unsupported_routes_keep_two_native_scatters(kwargs):
    events = run_forward(6, **kwargs)
    assert events.count("scatter") == 2 and "fused_store" not in events


@pytest.mark.parametrize("rows,expected", [(0, False), (1, True), (6, True), (12, True), (13, False)])
def test_store_shape_gate_retains_large_batch_fallback(rows, expected):
    source = ROOT / "vllm_ascend/ops/triton/sfa_indexer_store.py"
    tree = ast.parse(source.read_text())
    method = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "can_fuse_store")
    namespace = {"torch": torch, "triton": object(), "_KEY_WIDTH": 128, "_MAX_DECODE_ROWS": 12}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    device = SimpleNamespace(type="npu")

    def metadata(value):
        return SimpleNamespace(
            device=device,
            dtype=value.dtype,
            ndim=value.ndim,
            shape=value.shape,
            numel=value.numel,
            is_contiguous=value.is_contiguous,
        )

    cache = metadata(torch.empty((2, 128, 1, 128), dtype=torch.int8))
    scales = metadata(torch.empty((2, 128, 1, 1), dtype=torch.float16))
    slots = metadata(torch.empty(rows, dtype=torch.int64))
    assert namespace["can_fuse_store"](cache, scales, slots, rows) is expected
