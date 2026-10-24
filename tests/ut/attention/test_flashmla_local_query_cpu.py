# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of retained local Q; projection, RoPE and collectives are emulated.

These tests execute production preprocessing methods loaded with AST, without
importing the NPU backend. They do not establish NPU operator correctness.
"""

import ast
import sys
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from typing import NamedTuple
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "vllm_ascend/attention/mla_v1.py"
NOPE_DIM = 512
PE_DIM = 64
LOCAL_HEADS = 2
TOKENS = 3


def production_scope(**dependencies):
    nodes = ast.parse(SOURCE.read_text(encoding="utf-8")).body
    result = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == "DecodeMLAPreprocessResult")
    owner = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == "AscendMLAImpl")
    names = {"_reorg_decode_q_with_current", "mla_preprocess_decode", "mla_preprocess_only_decode", "rope_single"}
    methods = [n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name in names]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    scope = {"torch": torch, "NamedTuple": NamedTuple, **dependencies}
    module = ast.fix_missing_locations(ast.Module(body=[future, result, *methods], type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), scope)
    return scope


def bind(impl, scope, *names):
    for name in names:
        setattr(impl, name, MethodType(scope[name], impl))
    return impl


def components(heads=LOCAL_HEADS, *, rank=0):
    q_nope = torch.arange(TOKENS * heads * NOPE_DIM, dtype=torch.float32).reshape(TOKENS, heads, NOPE_DIM)
    q_pe = torch.arange(TOKENS * heads * PE_DIM, dtype=torch.float32).reshape(TOKENS, heads, PE_DIM)
    return q_nope / 64 + rank * 100, q_pe / 32 + rank * 20 + 1


def rope_tables():
    # Position-dependent tables shared by every query head.
    angles = torch.arange(TOKENS * PE_DIM, dtype=torch.float32).reshape(TOKENS, 1, PE_DIM) / 200
    angles = angles.repeat_interleave(2, dim=-1)[..., :PE_DIM]
    return angles.cos(), angles.sin()


def reference_rope(query, cos, sin):
    result = torch.empty_like(query)
    result[..., 0::2] = query[..., 0::2] * cos[..., 0::2] - query[..., 1::2] * sin[..., 0::2]
    result[..., 1::2] = query[..., 1::2] * cos[..., 1::2] + query[..., 0::2] * sin[..., 1::2]
    return result


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("use_rope", [False, True])
@pytest.mark.parametrize("flash_mode", ["current", "history", "none"])
def test_normal_preprocess_retains_final_local_query_and_preserves_rotation_order(rank, use_rope, flash_mode):
    events = []
    all_raw = [components(rank=index) for index in range(2)]
    q_nope, q_pe = all_raw[rank]
    cos, sin = rope_tables()
    has_current = flash_mode == "current"
    flash = None if flash_mode == "none" else SimpleNamespace(current=object() if has_current else None)
    common_slots = torch.tensor([0, 1, -1])
    if flash is not None:
        flash.slots = torch.tensor([3, -1, 5])
    metadata = SimpleNamespace(
        num_decode_tokens=TOKENS,
        external_flashmla=flash,
        slot_mapping=common_slots,
        decode=SimpleNamespace(cos=cos[:, :, None, :], sin=sin[:, :, None, :]),
    )

    def rotary(query, passed_cos, passed_sin):
        events.append(("rotary_op", query.shape[1]))
        rotated = torch.stack((-query[..., 1::2], query[..., 0::2]), dim=-1).flatten(-2)
        return query * passed_cos + rotated * passed_sin

    scope = production_scope(torch_npu=SimpleNamespace(npu_interleave_rope=rotary))
    history = (torch.empty(1, 8, 1, NOPE_DIM), torch.empty(1, 8, 1, PE_DIM))
    current = (torch.full((TOKENS, 1, NOPE_DIM), 7.0), torch.full((TOKENS, 1, PE_DIM), 11.0))
    impl = bind(
        SimpleNamespace(
            num_heads=LOCAL_HEADS,
            use_mla_rope=use_rope,
            fa_quant_layer=False,
            support_fp8_attention=False,
            _q_proj_and_k_up_proj=Mock(return_value=(q_nope, q_pe)),
            _decode_requires_current_kv=lambda _: has_current,
        ),
        scope,
        "_reorg_decode_q_with_current",
        "mla_preprocess_decode",
    )
    actual_rope = MethodType(scope["rope_single"], impl)

    def rope(query, passed_cos, passed_sin):
        events.append(("rope", query.shape[1]))
        return actual_rope(query, passed_cos, passed_sin)

    def gather(nope, pe):
        events.append(("gather", pe.shape[1]))
        torch.testing.assert_close(nope, q_nope, rtol=0, atol=0)
        expected_local = reference_rope(q_pe, cos, sin) if has_current and use_rope else q_pe
        torch.testing.assert_close(pe, expected_local, rtol=0, atol=0)
        rank_pe = [reference_rope(raw_pe, cos, sin) if has_current and use_rope else raw_pe for _, raw_pe in all_raw]
        return torch.cat([item[0] for item in all_raw], dim=1), torch.cat(rank_pe, dim=1)

    def kv_decode(_kv, _cos, _sin, passed_cache, slots, *, return_current_kv):
        assert passed_cache is history
        assert slots is (flash.slots if flash is not None else common_slots) or torch.equal(slots, common_slots)
        assert return_current_kv is has_current
        return (history[1], history[0], current[1], current[0]) if has_current else (history[1], history[0])

    impl.rope_single = rope
    impl.reorg_decode_q = Mock(side_effect=gather)
    impl.exec_kv_decode = Mock(side_effect=kv_decode)
    result = impl.mla_preprocess_decode(torch.empty(TOKENS, 4), torch.empty(TOKENS, 576), history, metadata)

    # Independent old dataflow: gather raw heads first, rotate them, then
    # select this rank's head range, as the original code did for current Q.
    old_nope = torch.cat([item[0] for item in all_raw], dim=1)
    old_pe = torch.cat([item[1] for item in all_raw], dim=1)
    if use_rope:
        old_pe = reference_rope(old_pe, cos, sin)
    torch.testing.assert_close(result.ql_nope, old_nope, rtol=0, atol=0)
    torch.testing.assert_close(result.q_pe, old_pe, rtol=0, atol=0)
    owned = slice(rank * LOCAL_HEADS, (rank + 1) * LOCAL_HEADS)
    if has_current:
        assert result.current_q_nope is q_nope
        torch.testing.assert_close(result.current_q_nope, old_nope[:, owned], rtol=0, atol=0)
        torch.testing.assert_close(result.current_q_pe, old_pe[:, owned], rtol=0, atol=0)
        assert result.current_k_nope is current[0] and result.current_k_pe is current[1]
        if not use_rope:
            assert result.current_q_pe is q_pe
            assert torch.count_nonzero(result.current_q_pe) == result.current_q_pe.numel()
    else:
        assert result.current_q_nope is None and result.current_q_pe is None
        assert result.current_k_nope is None and result.current_k_pe is None
    calls = [(name, heads) for name, heads in events if name != "rotary_op"]
    assert calls == (
        [("rope", LOCAL_HEADS), ("gather", LOCAL_HEADS)]
        if has_current
        else [("gather", LOCAL_HEADS), ("rope", LOCAL_HEADS * 2)]
    )
    assert sum(name == "rotary_op" for name, _ in events) == int(use_rope)
    impl.reorg_decode_q.assert_called_once()


@pytest.mark.parametrize("has_current", [False, True])
def test_helper_preserves_strided_storage_and_only_trims_padded_heads_for_current(has_current):
    padded_heads = LOCAL_HEADS + 1
    width = (NOPE_DIM + PE_DIM) * 2
    offset = 17
    storage = torch.arange(TOKENS * padded_heads * width + offset, dtype=torch.float32)
    packed = storage[offset:].reshape(TOKENS, padded_heads, width)
    q_nope, q_pe = packed[..., : NOPE_DIM * 2 : 2], packed[..., NOPE_DIM * 2 :: 2]
    metadata = SimpleNamespace(external_flashmla=SimpleNamespace(current=object() if has_current else None))
    gather = Mock(side_effect=lambda nope, pe: (nope.repeat(1, 2, 1), pe.repeat(1, 2, 1)))
    impl = bind(
        SimpleNamespace(num_heads=LOCAL_HEADS, reorg_decode_q=gather),
        production_scope(),
        "_reorg_decode_q_with_current",
    )
    gathered_nope, gathered_pe, current_nope, current_pe = impl._reorg_decode_q_with_current(q_nope, q_pe, metadata)
    expected_heads = LOCAL_HEADS if has_current else padded_heads
    assert gathered_nope.shape == (TOKENS, expected_heads * 2, NOPE_DIM)
    assert gathered_pe.shape == (TOKENS, expected_heads * 2, PE_DIM)
    if has_current:
        for local, source in ((current_nope, q_nope), (current_pe, q_pe)):
            assert local.shape[1] == LOCAL_HEADS
            assert local.stride() == source.stride()
            assert local.data_ptr() == source.data_ptr()
            assert local.storage_offset() == source.storage_offset()
            torch.testing.assert_close(local, source[:, :LOCAL_HEADS], rtol=0, atol=0)
    else:
        assert current_nope is None and current_pe is None
        assert gather.call_args.args[0] is q_nope and gather.call_args.args[1] is q_pe


@pytest.mark.parametrize("use_rope", [False, True])
@pytest.mark.parametrize("has_current", [False, True])
def test_only_decode_prolog_retains_already_final_local_query(monkeypatch, use_rope, has_current):
    events = []
    heads = LOCAL_HEADS + int(has_current)
    q_nope, raw_pe = components(heads)
    cos, sin = rope_tables()
    q_pe = reference_rope(raw_pe, cos, sin) if use_rope else raw_pe
    history = (torch.empty(1, 8, 1, NOPE_DIM), torch.empty(1, 8, 1, PE_DIM))
    current = (torch.full_like(history[0], 7), torch.full_like(history[1], 11), torch.arange(8))
    slots = torch.tensor([0, -1, 2])
    flash = SimpleNamespace(current=object() if has_current else None)
    metadata = SimpleNamespace(
        num_decode_tokens=TOKENS,
        slot_mapping=slots,
        external_flashmla=flash,
        decode=SimpleNamespace(cos=cos[:, :, None, :], sin=sin[:, :, None, :]),
    )

    def prolog(**kwargs):
        events.append("prolog")
        torch.testing.assert_close(kwargs["cache_index"], torch.arange(TOKENS) if has_current else slots)
        assert kwargs["query_quant_mode"] == 0
        assert (kwargs["rope_cos"] is None) is (not use_rope)
        return q_nope, q_pe, None, None, None

    dependency = ModuleType("cann_ops_transformer")
    dependency.mla_prolog = Mock(side_effect=prolog)
    monkeypatch.setitem(sys.modules, "cann_ops_transformer", dependency)
    scatter = Mock()
    scope = production_scope(
        cdiv=lambda x, y: (x + y - 1) // y, DeviceOperator=SimpleNamespace(reshape_and_cache=scatter)
    )
    impl = bind(
        SimpleNamespace(
            num_heads=LOCAL_HEADS,
            mlapo_num_heads=heads,
            kv_lora_rank=NOPE_DIM,
            qk_rope_head_dim=PE_DIM,
            use_mla_rope=use_rope,
            support_fp8_attention=True,
            mlapo_weight_quant_mode=0,
            fa_quant_layer=False,
            enable_kv_nz=False,
            _dcp_current_kv_buffers=current,
            _decode_requires_current_kv=lambda _: has_current,
            weight_dq=object(),
            weight_uq_qr=object(),
            mlapo_W_UK_T=object(),
            weight_dkv_kr=object(),
            q_a_layernorm=SimpleNamespace(weight=SimpleNamespace(data=object())),
            kv_a_layernorm=SimpleNamespace(weight=SimpleNamespace(data=object())),
            rope_single=Mock(side_effect=AssertionError("PROLOG output must not be rotated again")),
        ),
        scope,
        "_reorg_decode_q_with_current",
        "mla_preprocess_only_decode",
    )

    def gather(nope, pe):
        events.append("gather")
        assert nope.shape[1] == pe.shape[1] == LOCAL_HEADS
        return nope.repeat(1, 2, 1), pe.repeat(1, 2, 1)

    impl.reorg_decode_q = Mock(side_effect=gather)
    result, prefill = impl.mla_preprocess_only_decode(torch.empty(TOKENS, 4), history, metadata)
    assert prefill is None and events == ["prolog", "gather"]
    expected_nope, expected_pe = q_nope[:, :LOCAL_HEADS], q_pe[:, :LOCAL_HEADS]
    torch.testing.assert_close(result.ql_nope, expected_nope.repeat(1, 2, 1), rtol=0, atol=0)
    torch.testing.assert_close(result.q_pe, expected_pe.repeat(1, 2, 1), rtol=0, atol=0)
    if has_current:
        for retained, source in ((result.current_q_nope, q_nope), (result.current_q_pe, q_pe)):
            torch.testing.assert_close(retained, source[:, :LOCAL_HEADS], rtol=0, atol=0)
            assert retained.stride() == source.stride() and retained.data_ptr() == source.data_ptr()
        scatter.assert_called_once()
        torch.testing.assert_close(scatter.call_args.kwargs["slot_mapping"], slots)
        assert scatter.call_args.kwargs["slot_mapping"].data_ptr() == slots.data_ptr()
        assert result.current_k_nope.untyped_storage().data_ptr() == current[0].untyped_storage().data_ptr()
        assert result.current_k_pe.untyped_storage().data_ptr() == current[1].untyped_storage().data_ptr()
    else:
        assert result.current_q_nope is None and result.current_q_pe is None
        scatter.assert_not_called()
    impl.rope_single.assert_not_called()
