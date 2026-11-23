# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests; no kernel or NPU correctness claim."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


def load_method(path, class_name, method, scope):
    root = Path(__file__).resolve().parents[3]
    tree = ast.parse((root / path).read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    node = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), scope)
    return scope[method]


def test_quant_dcp_rejects_unquantized_current_before_scatter():
    scatter = Mock()
    prepare = load_method(
        "vllm_ascend/attention/context_parallel/mla_cp.py",
        "AscendMlaDCPImpl",
        "_prepare_dcp_flashmla_current",
        dict(torch=torch, torch_npu=SimpleNamespace(npu_scatter_pa_kv_cache=scatter)),
    )
    impl = SimpleNamespace(use_quant_flashmla=True)
    flash = SimpleNamespace(current=SimpleNamespace(cache=torch.empty(1, dtype=torch.float8_e4m3fn)))
    with pytest.raises(RuntimeError, match="quantized FP8"):
        prepare(impl, flash, None, None, torch.ones(1, dtype=torch.bfloat16), torch.ones(1, dtype=torch.bfloat16))
    scatter.assert_not_called()


def test_metadata_capacity_bounds_do_not_change_empty_actual_lengths():
    lengths = torch.zeros(2, dtype=torch.int32)
    cuts = torch.tensor([0, 1, 3], dtype=torch.int32)
    used = torch.tensor([1, 2], dtype=torch.int32)
    operation = Mock(return_value=object())
    scope = dict(torch=torch, _check_lengths=Mock(), FLASHMLA_QK_DIM=576, FLASHMLA_V_DIM=512)
    build = load_method("vllm_ascend/attention/flashmla.py", "QuantFlashMLAImpl", "build_metadata", scope)
    impl = SimpleNamespace(
        metadata_op=operation, config=SimpleNamespace(num_heads=96, quant_mode=1, mask_mode=0, NUM_KV_HEADS=1)
    )
    build(impl, lengths, cuts, used, max_seqlen_q=3, max_seqlen_kv=2048)
    assert operation.call_args.args[0] is lengths
    assert operation.call_args.kwargs["seqused_q"] is used
    assert operation.call_args.kwargs["max_seqlen_q"] == 3
    assert operation.call_args.kwargs["max_seqlen_kv"] == 2048
    assert not torch.count_nonzero(lengths)
