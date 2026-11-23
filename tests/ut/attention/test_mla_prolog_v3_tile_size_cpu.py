# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check PROLOG call arguments without importing torch or executing NPU kernels."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _load_run_prolog_v3(torch, torch_npu):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/mla_v1.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    impl = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendMLAImpl")
    method = next(node for node in impl.body if isinstance(node, ast.FunctionDef) and node.name == "_run_prolog_v3")
    method.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[]))
    scope = {"torch": torch, "torch_npu": torch_npu}
    exec(compile(module, str(source), "exec"), scope)
    return scope[method.name]


@pytest.mark.parametrize("page_size", [128, 768])
@pytest.mark.parametrize("quant_mode", [0, 1])
def test_prolog_tile_size_is_independent_of_cache_pages(page_size, quant_mode):
    torch = SimpleNamespace(int64=object())
    outputs = object()
    operator = MagicMock(return_value=outputs)
    run = _load_run_prolog_v3(torch, SimpleNamespace(npu_mla_prolog_v3=operator))
    hidden_states = SimpleNamespace(shape=(3, 7168))
    cache = (
        SimpleNamespace(shape=(1, page_size, 1, 512)),
        SimpleNamespace(shape=(1, page_size, 1, 64)),
    )
    slots = MagicMock()
    slots.view.return_value = slots
    slots.to.return_value = slots
    slots.contiguous.return_value = slots
    norm = SimpleNamespace(weight=SimpleNamespace(data=object()), variance_epsilon=1e-6)
    weights = dict.fromkeys(["weight_dq", "weight_uq_qr", "weight_uk", "weight_dkv_kr"], object())
    impl = SimpleNamespace(
        q_a_layernorm=norm,
        kv_a_layernorm=norm,
        fa_quant_layer=bool(quant_mode),
        support_fp8_attention=False,
        quant_kscale=MagicMock(),
        _prepare_prolog_v3_token=MagicMock(return_value=(hidden_states, {}, weights)),
        _flatten_prolog_rope_table=lambda table, _: table,
        _prolog_v3_writer_workspace=lambda *_: (cache, slots, False),
    )

    assert run(impl, hidden_states, object(), object(), cache, slots, quantize_query=True) is outputs

    operator.assert_called_once()
    passed = operator.call_args.kwargs
    assert passed["tile_size"] == 128
    assert passed["kv_cache_quant_mode"] == quant_mode
    assert passed["query_quant_mode"] == quant_mode
    assert passed["cache_mode"] == "PA_BSND"
    assert passed["kv_cache"] is cache[0]
    assert passed["kr_cache"] is cache[1]
    assert cache[0].shape == (1, page_size, 1, 512)
    assert cache[1].shape == (1, page_size, 1, 64)
    assert passed["cache_index"] is slots
    slots.view.assert_called_once_with(-1)
    slots.to.assert_called_once_with(torch.int64)
    slots.contiguous.assert_called_once_with()
