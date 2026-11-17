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


@pytest.mark.parametrize("rank", [0, 1])
def test_quant_dcp_history_and_current_keep_head_descales(rank):
    tokens, local_heads, width = 3, 4, 576
    heads = local_heads * 2
    query = torch.zeros(tokens, heads, width, dtype=torch.float8_e4m3fn)
    scales = torch.arange(1, tokens * heads + 1, dtype=torch.float32).reshape(tokens, heads)
    k_scale = torch.tensor([0.125])
    history_query = torch.empty_like(query)
    current_query = torch.empty(tokens, local_heads, width, dtype=query.dtype)
    output = torch.zeros(heads, tokens, 512, dtype=torch.bfloat16)
    lse = torch.zeros(heads, tokens, dtype=torch.float32)
    history_adapter = SimpleNamespace(attention=Mock(return_value=(output, lse)))
    current_adapter = SimpleNamespace(attention=Mock(return_value=(output[:local_heads], lse[:local_heads])))
    current = SimpleNamespace(
        query=current_query,
        cache=torch.empty(1, 128, 1, width, dtype=query.dtype),
        adapter=current_adapter,
        schedule=object(),
        block_table=object(),
        attn_mask=object(),
    )
    flash = SimpleNamespace(
        current=current,
        query=history_query,
        adapter=history_adapter,
        schedule=object(),
        block_table=object(),
        cache_lens=object(),
        cu=object(),
        used_q=object(),
    )
    kind = SimpleNamespace(CURRENT=object(), HISTORY=object())
    scope = dict(torch=torch, CPKVScope=kind)
    run = load_method(
        "vllm_ascend/attention/context_parallel/mla_cp.py",
        "AscendMlaDCPImpl",
        "_run_dcp_flashmla_attention",
        scope,
    )
    local = load_method(
        "vllm_ascend/attention/context_parallel/common_cp.py",
        "DCPImplMixin",
        "_local_decode_query",
        dict(torch=torch),
    )
    impl = SimpleNamespace(
        use_quant_flashmla=True,
        kv_lora_rank=512,
        fak_descale_float=k_scale,
        dcp_rank=rank,
    )
    run(
        impl,
        flash,
        query[..., :512],
        query[..., 512:],
        current.cache,
        None,
        attention_kind=kind.HISTORY,
        q_descale=scales,
    )
    selected = local(impl, scales, num_heads=local_heads)[0]
    run(
        impl,
        flash,
        query[:, :local_heads, :512],
        query[:, :local_heads, 512:],
        None,
        None,
        attention_kind=kind.CURRENT,
        q_descale=selected,
    )
    history = history_adapter.attention.call_args.kwargs
    current_args = current_adapter.attention.call_args.kwargs
    torch.testing.assert_close(history["q_descale"], scales)
    torch.testing.assert_close(current_args["q_descale"], scales[:, rank * local_heads : (rank + 1) * local_heads])
    assert history["k_descale"].data_ptr() == current_args["k_descale"].data_ptr() == k_scale.data_ptr()
    assert history["q_descale"].shape == (tokens, heads)
    assert current_args["q_descale"].is_contiguous()


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
    build = load_method("vllm_ascend/attention/quant_flashmla.py", "QuantFlashMLAImpl", "build_metadata", scope)
    impl = SimpleNamespace(metadata_op=operation, config=SimpleNamespace(num_heads=96, quant_mode=1, mask_mode=0))
    build(impl, lengths, cuts, used, max_seqlen_q=3, max_seqlen_kv=2048)
    assert operation.call_args.args[0] is lengths
    assert operation.call_args.kwargs["seqused_q"] is used
    assert operation.call_args.kwargs["max_seqlen_q"] == 3
    assert operation.call_args.kwargs["max_seqlen_kv"] == 2048
    assert not torch.count_nonzero(lengths)
