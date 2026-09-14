# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

# Exercise metadata adaptation without importing the model/runtime or NPU backend.
_SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/kimi_kda.py"
_TREE = ast.parse(_SOURCE.read_text(encoding="utf-8"))
_METHOD = next(node for node in ast.walk(_TREE) if isinstance(node, ast.FunctionDef) and node.name == "_run_causal_conv1d")
_METHOD.decorator_list = []
_SCOPE = {"torch": torch, "envs": SimpleNamespace(VLLM_ASCEND_ENABLE_FLASH_MLA=True), "PAD_SLOT_ID": -1}
exec(compile(ast.Module(body=[_METHOD], type_ignores=[]), str(_SOURCE), "exec"), _SCOPE)
_RUN = _SCOPE["_run_causal_conv1d"]


def test_rank_two_cache_uses_first_column_and_keeps_state_alias():
    backing = torch.empty(2048, dtype=torch.bfloat16)
    state = backing.as_strided((4, 5, 16), (256, 32, 1), storage_offset=16)
    output = torch.empty(6, 16, dtype=torch.bfloat16)
    x = torch.empty_like(output)
    weight = torch.empty(4, 16, dtype=torch.bfloat16)
    qsl = torch.tensor([0, 3, 6], dtype=torch.int64)
    cache = torch.tensor([[2, 100, 101], [0, 200, 201]], dtype=torch.int64)
    accepted = torch.tensor([0, 2], dtype=torch.int64)
    with patch.object(torch.ops._C_ascend, "npu_causal_conv1d_custom", create=True) as native:
        native.return_value = output
        with (
            patch.object(torch.Tensor, "cpu", side_effect=AssertionError("device readback")),
            patch.object(torch.Tensor, "item", side_effect=AssertionError("device readback")),
            patch.object(torch.Tensor, "tolist", side_effect=AssertionError("device readback")),
        ):
            actual = _RUN(
                x,
                weight,
                state,
                qsl,
                cache,
                None,
                run_mode=1,
                max_query_len=3,
                num_accepted_tokens=accepted,
            )
        args = native.call_args.args
    assert actual is output
    assert args[3] is state
    assert state.stride() == (256, 32, 1)
    assert state.storage_offset() == 16
    assert args[5].dtype == args[6].dtype == args[8].dtype == torch.int32
    assert args[6].tolist() == [2, 0]
    assert args[8].tolist() == [0, 2]
    assert args[9:] == (1, -1, 1, 3)


def test_bool_initial_flags_and_int32_metadata_keep_existing_storage():
    tensor = torch.zeros(3, dtype=torch.int32)
    output = torch.empty(2, 16, dtype=torch.float16)
    initial = torch.tensor([True, False])
    with patch.object(torch.ops._C_ascend, "npu_causal_conv1d_custom", create=True) as native:
        _RUN(
            output.clone(),
            torch.empty(2, 16, dtype=torch.float16),
            torch.empty(2, 1, 16, dtype=torch.float16),
            tensor,
            torch.tensor([0, 1], dtype=torch.int32),
            initial,
            run_mode=0,
            max_query_len=2,
        )
        assert native.call_args.args[7] is initial
        assert native.call_args.args[5] is tensor
