# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Provider interfaces, active row bounds and model-owned constants; CPU only."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ops import gdn_recoverssm_backend as provider


def test_native_commit_slices_capacity_buffers_to_active_rows(monkeypatch):
    native = Mock()
    monkeypatch.setitem(sys.modules, "fla_npu.ops.ascendc", SimpleNamespace(gdn_recoverssm_commit=native))
    state, replay, table = object(), object(), object()
    indices = torch.tensor([2, 4], dtype=torch.int32)
    vectors = [torch.arange(7, dtype=torch.int32) for _ in range(4)]
    provider._ascendc_commit(state, replay, table, indices, *vectors, spec_query_len=4, align_mode=True)
    args, kwargs = native.call_args
    assert args[:4] == (state, replay, table, indices)
    assert all(tensor.shape == (2,) for tensor in args[4:])
    assert all(tensor.data_ptr() == original.data_ptr() for tensor, original in zip(args[4:], vectors))
    assert kwargs == {"spec_query_len": 4, "align_mode": True}


def test_native_compact_keeps_pooled_stride_tables(monkeypatch):
    native = Mock()
    monkeypatch.setitem(sys.modules, "fla_npu.ops.ascendc", SimpleNamespace(compact_conv_state=native))
    tables = [torch.tensor([123, 456], dtype=torch.int64) for _ in range(4)]
    indices = torch.tensor([1], dtype=torch.int32)
    plan = [torch.tensor([3, 7, 9], dtype=torch.int32) for _ in range(4)]
    conv = object()
    provider._ascendc_compact(conv, *tables, indices, *plan, conv_history_len=3, align_mode=False)
    args = native.call_args.args
    assert all(actual is expected for actual, expected in zip(args[1:5], tables))
    assert all(vector.shape == (1,) for vector in args[6:])


def test_constants_refresh_after_reload_without_global_tensor_cache():
    model = torch.nn.Sequential(torch.nn.Module())
    layer = model[0]
    layer._gdn_recoverssm_backend = "ascendc"
    layer.A_log = torch.nn.Parameter(torch.tensor([1.25], dtype=torch.float32))
    layer.dt_bias = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.bfloat16))
    provider.prepare_gdn_recoverssm_constants(model)
    assert layer._recoverssm_dt_bias_fp32.dtype == torch.float32
    assert "_recoverssm_dt_bias_fp32" not in model.state_dict()
    layer.dt_bias.data.fill_(2)
    provider.prepare_gdn_recoverssm_constants(model)
    torch.testing.assert_close(layer._recoverssm_dt_bias_fp32, torch.tensor([2.0]))


def test_invalid_provider_is_rejected_before_optional_import():
    with pytest.raises(ValueError, match="Unknown GDN"):
        provider.load_gdn_recoverssm_backend("missing")


def test_isolated_delta_does_not_execute_or_replace_base_controller(tmp_path):
    root = tmp_path / "fla_npu"
    root.mkdir()
    (root / "__init__.py").write_text('raise AssertionError("base controller must not run")')
    (root / "ops").mkdir()
    (root / "ops/__init__.py").write_text("")
    (root / "ops/ascendc.py").write_text("def gdn_recoverssm_verify(): return 7\n")
    library = root / "opp/vendors/fla_npu_transformer/op_api/lib/libcust_opapi.so"
    library.parent.mkdir(parents=True)
    library.touch()
    before = sys.modules.get("fla_npu")
    loaded = provider._load_native_package(str(root))
    assert loaded.gdn_recoverssm_verify() == 7
    assert sys.modules.get("fla_npu") is before
    assert provider._load_native_package(str(root)) is loaded


def test_incomplete_native_delta_fails_before_execution(tmp_path):
    with pytest.raises(ValueError, match="Incomplete GDN"):
        provider._load_native_package(str(tmp_path))
