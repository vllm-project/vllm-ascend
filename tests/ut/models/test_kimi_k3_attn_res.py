# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU numerical and contract tests; no vLLM installation or NPU required."""

import ast
import importlib.util
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
# Dynamo resolves function globals by module name. runpy's <run_path> is not
# importable in PyTorch 2.10; register a real module without importing the NPU plugin.
_backend_spec = importlib.util.spec_from_file_location(
    "_kimi_attn_res_test_backend", ROOT / "vllm_ascend/ops/kimi_attn_res.py"
)
backend = importlib.util.module_from_spec(_backend_spec)
sys.modules[_backend_spec.name] = backend
_backend_spec.loader.exec_module(backend)


def reference(rows, partial, query, epsilon):
    values = torch.cat((rows, partial.unsqueeze(1)), dim=1).double()
    logits = (values * query.double()).sum(-1) * torch.rsqrt(values.square().mean(-1) + epsilon)
    return (logits.softmax(-1).unsqueeze(-1) * values).sum(1).to(partial.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tokens,blocks", [(3, 0), (3, 4), (0, 2)])
@pytest.mark.parametrize("query_scale", [1.0, 100.0])
def test_online_softmax_matches_independent_reference(dtype, tokens, blocks, query_scale):
    torch.manual_seed(31)
    rows = torch.randn(tokens, blocks, 16).to(dtype)
    partial = torch.randn(tokens, 16).to(dtype)
    queries = torch.randn(5, 16) * query_scale
    stats = backend.prepare_attn_res_phase1(rows, queries, 1e-6)
    assert all(value.dtype == torch.float32 for value in stats)
    for idx, query in enumerate(queries):
        slot = backend.AttnResPhase1Stats(*(value[idx] for value in stats))
        output = backend.merge_attn_res_partial(partial, query, slot, 1e-6)
        torch.testing.assert_close(output, reference(rows, partial, query, 1e-6), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("mode", ["original", "two_phase"])
def test_backend_preserves_add_rounding_write_and_materialized_contract(mode):
    projection = SimpleNamespace(weight=torch.tensor([[0.3, -0.2, 0.1]]))
    norm = SimpleNamespace(weight=torch.tensor([1.0, 0.5, 2.0]), variance_epsilon=1e-6)
    plan = backend.AttnResQueryPlan([(projection, norm, 1)])
    op = backend.AttnResBackend(mode, plan)
    prefix = torch.tensor([[256.0, 1.0, -1.0]], dtype=torch.bfloat16)
    delta = torch.tensor([[1.0, 0.003, 0.003]], dtype=torch.bfloat16)
    bank = torch.tensor([[[1.0, -2.0, 3.0], [999.0, 999.0, 999.0]]], dtype=torch.bfloat16)
    before = bank.clone()
    output, raw, materialized = op(
        prefix,
        delta,
        bank,
        projection.weight,
        norm.weight,
        1e-6,
        1,
        output_norm_weight=norm.weight,
        output_norm_eps=1e-4,
        block_write_idx=1,
        return_materialized=True,
    )
    expected_raw = (prefix.float() + delta.float()).to(prefix.dtype)
    expected = reference(before[:, :1], expected_raw, projection.weight.squeeze(0) * norm.weight, 1e-6)
    normalized = expected.float() * torch.rsqrt(expected.float().square().mean(-1, keepdim=True) + 1e-4)
    torch.testing.assert_close(raw, expected_raw, atol=0, rtol=0)
    torch.testing.assert_close(materialized, expected)
    torch.testing.assert_close(output, (normalized * norm.weight).to(prefix.dtype))
    torch.testing.assert_close(bank[:, 1], raw)
    torch.testing.assert_close(bank[:, 0], before[:, 0])
    assert prefix[0, 0] == 256  # The caller's prefix is never updated in place.


def test_phase1_is_reused_and_never_cached_across_forwards(monkeypatch):
    projection = SimpleNamespace(weight=torch.ones(1, 4))
    norm = SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-6)
    plan = backend.AttnResQueryPlan([(projection, norm, 1)] * 3 + [(projection, norm, 2)])
    # The backend methods share their module's globals dictionary.
    prepare_globals = backend.AttnResBackend._materialize.__globals__
    prepare = MagicMock(wraps=prepare_globals["prepare_attn_res_phase1"])
    monkeypatch.setitem(prepare_globals, "prepare_attn_res_phase1", prepare)
    for _ in range(2):
        op = backend.AttnResBackend("two_phase", plan)
        bank = torch.randn(2, 3, 4)
        raw = torch.randn(2, 4)
        for valid in (1, 1, 1, 2):
            op(raw, None, bank, projection.weight, norm.weight, 1e-6, valid)
    assert prepare.call_count == 4


def test_query_plan_refresh_preserves_fp32_buffers():
    projection = SimpleNamespace(weight=torch.tensor([[0.3, 0.4]], dtype=torch.bfloat16))
    norm = SimpleNamespace(weight=torch.tensor([0.7, 0.8], dtype=torch.bfloat16), variance_epsilon=1e-6)
    specs = [(projection, norm, 1)]
    with torch.inference_mode():
        plan = backend.AttnResQueryPlan(specs)
    query = plan.queries[(1, 1e-6)]
    pointer = query.data_ptr()
    expected = projection.weight.float() * norm.weight.float()
    assert query.dtype == torch.float32
    torch.testing.assert_close(query, expected)
    projection.weight.mul_(2)
    plan.refresh(specs)
    assert plan.queries[(1, 1e-6)] is query
    assert query.data_ptr() == pointer
    assert any(buffer is query for buffer in plan.buffers())
    assert not plan.state_dict()
    torch.testing.assert_close(query, expected * 2)


@pytest.mark.parametrize("change", ["slots", "epsilon", "hidden_size"])
def test_query_plan_rejects_layout_changes_on_reload(change):
    projection = SimpleNamespace(weight=torch.ones(1, 4))
    norm = SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-6)
    specs = [(projection, norm, 1)]
    plan = backend.AttnResQueryPlan(specs)
    query = plan.queries[(1, 1e-6)]
    expected = query.clone()
    if change == "slots":
        specs.append((projection, norm, 1))
    elif change == "epsilon":
        norm.variance_epsilon = 1e-5
    else:
        projection.weight = torch.zeros(1, 8)
        norm.weight = torch.ones(8)
    with pytest.raises(ValueError, match="query layout"):
        plan.refresh(specs)
    torch.testing.assert_close(query, expected)


def test_modes_default_to_two_phase_and_reject_typos():
    assert backend.resolve_attn_res_mode(SimpleNamespace()) == "two_phase"
    assert backend.resolve_attn_res_mode(SimpleNamespace(), fused_available=True) == "fused"
    for mode in ("native", "two_phase", "original"):
        for available in (False, True):
            assert backend.resolve_attn_res_mode(SimpleNamespace(attn_res_mode=mode), available) == mode
    assert backend.resolve_attn_res_mode(SimpleNamespace(attn_res_mode="fused")) == "two_phase"
    with pytest.raises(ValueError, match="attn_res_mode"):
        backend.resolve_attn_res_mode(SimpleNamespace(attn_res_mode="typo"))


@pytest.mark.parametrize("missing", ["module", "prepare", "update", None])
@pytest.mark.parametrize("with_norm", [False, True])
def test_optional_fused_operator_imports(monkeypatch, missing, with_norm):
    ops = SimpleNamespace(block_attn_res_prepare=MagicMock(), block_attn_res_update=MagicMock())
    norm_ops = SimpleNamespace(block_attn_res_update_rms_norm_op=MagicMock())
    if missing in ("prepare", "update"):
        delattr(ops, f"block_attn_res_{missing}")

    def importer(name):
        if name == "cann_ops_transformer.ops" and missing != "module":
            return ops
        if name == "ops.block_attn_res_update_rms_norm" and with_norm:
            return norm_ops
        raise ImportError(name)

    monkeypatch.setitem(backend.load_attn_res_fused_ops.__globals__, "import_module", importer)
    result = backend.load_attn_res_fused_ops()
    if missing is not None:
        assert result is None
    else:
        assert result.prepare is ops.block_attn_res_prepare
        assert result.update is ops.block_attn_res_update
        assert result.update_rms_norm is (norm_ops.block_attn_res_update_rms_norm_op if with_norm else None)


def test_registered_prepare_is_used_for_graph_capture(monkeypatch):
    registered = MagicMock()
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "block_attn_res_prepare", registered, raising=False)
    ops = SimpleNamespace(block_attn_res_prepare=MagicMock(), block_attn_res_update=MagicMock())

    def importer(name):
        if name == "cann_ops_transformer.ops":
            return ops
        raise ImportError(name)

    monkeypatch.setitem(backend.load_attn_res_fused_ops.__globals__, "import_module", importer)
    assert backend.load_attn_res_fused_ops().prepare is registered


@pytest.fixture
def fused_ops():
    def prepare(values, count, queries, eps):
        assert values.dtype == torch.float32 and values.is_contiguous()
        assert count.dtype == torch.uint64 and count.tolist() == [values.shape[1]]
        return backend.prepare_attn_res_phase1(values, queries, eps)

    def update(partial, delta, query, numerator, maximum, exp_sum, eps):
        assert partial.dtype == torch.float32
        partial.add_(delta.float())
        return reference_output(partial, query, numerator, maximum, exp_sum, eps).to(delta.dtype)

    def reference_output(partial, query, numerator, maximum, exp_sum, eps):
        stats = backend.AttnResPhase1Stats(numerator, maximum, exp_sum)
        return backend.merge_attn_res_partial(partial, query, stats, eps)

    def update_norm(partial, delta, query, numerator, maximum, exp_sum, gamma, eps, norm_eps):
        value = update(partial, delta, query, numerator, maximum, exp_sum, eps).float()
        return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + norm_eps) * gamma).to(delta.dtype)

    return backend.AttnResFusedOps(
        MagicMock(side_effect=prepare), MagicMock(side_effect=update), MagicMock(side_effect=update_norm)
    )


@pytest.mark.parametrize("with_norm", [False, True])
@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("tuple_output", [False, True])
def test_fused_dispatch_preserves_partial_and_dspark_contract(fused_ops, with_norm, capture, tuple_output):
    projection = SimpleNamespace(weight=torch.ones(1, 4))
    norm = SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-6)
    specs = [(projection, norm, 1)] * 2
    if not with_norm:
        fused_ops = fused_ops._replace(update_rms_norm=None)
    if tuple_output:
        update = fused_ops.update
        fused_ops = fused_ops._replace(update=lambda *args, **kwargs: (update(*args, **kwargs), args[0]))
    op = backend.AttnResBackend("fused", backend.AttnResQueryPlan(specs), fused_ops)
    prefix = torch.full((2, 4), 256.0, dtype=torch.bfloat16)
    delta = torch.ones_like(prefix)
    bank = torch.zeros(2, 1, 4, dtype=torch.bfloat16)
    snapshots = []
    for _ in range(2):
        output, raw, materialized = op(
            prefix,
            delta,
            bank,
            projection.weight,
            norm.weight,
            1e-6,
            1,
            output_norm_weight=norm.weight,
            output_norm_eps=1e-4,
            return_materialized=capture,
        )
        snapshots.append(raw.clone())
        if capture:
            value = materialized.float()
            expected = (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-4)).to(output.dtype)
            torch.testing.assert_close(output, expected)
        else:
            assert materialized is output
    assert fused_ops.prepare.call_count == 1
    assert op.partial.dtype == torch.float32 and torch.equal(op.partial, prefix.float() + 2)
    torch.testing.assert_close(snapshots[0], prefix)
    torch.testing.assert_close(raw, (prefix.float() + 2).bfloat16())
    assert torch.equal(op.finish_partial(raw, delta), (prefix.float() + 3).bfloat16())
    if with_norm:
        assert fused_ops.update_rms_norm.call_count == (0 if capture else 2)


def test_fused_runtime_error_does_not_silently_change_backend(fused_ops):
    projection = SimpleNamespace(weight=torch.ones(1, 4))
    norm = SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-6)
    fused_ops.update.side_effect = RuntimeError("operator version mismatch")
    op = backend.AttnResBackend("fused", backend.AttnResQueryPlan([(projection, norm, 1)]), fused_ops)
    with pytest.raises(RuntimeError, match="operator version mismatch"):
        op(torch.ones(2, 4), None, torch.zeros(2, 1, 4), projection.weight, norm.weight, 1e-6, 1)


def test_fused_flag_defaults_to_enabled_and_can_disable(monkeypatch):
    env_namespace = runpy.run_path(str(ROOT / "vllm_ascend/envs.py"))
    name = "VLLM_ASCEND_KIMI_K3_ATTNRES_FUSED_ENABLED"
    monkeypatch.delenv(name, raising=False)
    assert env_namespace["env_variables"][name]() is True
    monkeypatch.setenv(name, "0")
    assert env_namespace["env_variables"][name]() is False


@pytest.mark.parametrize("compile_backend", ["eager", "aot_eager"])
def test_fused_dispatch_is_capturable_without_python_fallback(fused_ops, compile_backend):
    def prepare(values, count, queries, eps):
        return backend.prepare_attn_res_phase1(values, queries, eps)

    # Replace only the optional NPU kernels with traceable Torch functions;
    # all adapter state, FP32 accumulation and output normalization stay real.
    ops = backend.AttnResFusedOps(prepare, fused_ops.update.side_effect, fused_ops.update_rms_norm.side_effect)
    projection = SimpleNamespace(weight=torch.ones(1, 4))
    norm = SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-6)
    plan = backend.AttnResQueryPlan([(projection, norm, 1)] * 2)

    def forward(prefix, delta, bank):
        op = backend.AttnResBackend("fused", plan, ops)
        output, raw, _ = op(
            prefix,
            delta,
            bank,
            projection.weight,
            norm.weight,
            1e-6,
            1,
            output_norm_weight=norm.weight,
        )
        return op(
            raw,
            output,
            bank,
            projection.weight,
            norm.weight,
            1e-6,
            1,
            output_norm_weight=norm.weight,
        )[0]

    compiled = torch.compile(forward, backend=compile_backend, fullgraph=True)
    for _ in range(2):
        prefix, delta = torch.randn(2, 4).bfloat16(), torch.randn(2, 4).bfloat16()
        bank = torch.randn(2, 1, 4).bfloat16()
        torch.testing.assert_close(compiled(prefix, delta, bank), forward(prefix, delta, bank))


@pytest.mark.parametrize("mode", ["two_phase", "original"])
@pytest.mark.parametrize("compile_backend", ["eager", "aot_eager"])
def test_two_phase_forward_cache_supports_full_graph_capture(mode, compile_backend):
    projection = SimpleNamespace(weight=torch.ones(1, 4))
    norm = SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-6)
    plan = backend.AttnResQueryPlan([(projection, norm, 1)] * 2)

    def forward(partial, delta, bank):
        op = backend.AttnResBackend(mode, plan)
        output, raw, _ = op(partial, delta, bank, projection.weight, norm.weight, 1e-6, 1)
        return op(raw, output, bank, projection.weight, norm.weight, 1e-6, 1)[0]

    compiled = torch.compile(forward, backend=compile_backend, fullgraph=True)
    for _ in range(2):
        partial, delta = torch.randn(2, 4), torch.randn(2, 4)
        bank = torch.randn(2, 1, 4)
        torch.testing.assert_close(compiled(partial, delta, bank), forward(partial, delta, bank))


@pytest.mark.parametrize("group", ["tp", "dp", "mc2"])
def test_hccl_options_keep_runtime_communication_default(group):
    # PR a0403dcc2 removed forced CCU for prefill. v0.30 already delegates
    # communication to HCCL for every role; preserve buffer and group settings.
    source = ast.parse((ROOT / "vllm_ascend/utils.py").read_text())
    node = next(node for node in source.body if getattr(node, "name", None) == "create_hccl_pg_options")
    namespace = {
        "torch_npu": MagicMock(),
        "get_hccl_config_for_pg_options": lambda name: None if name == "mc2" else {"hccl_buffer_size": 200},
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "utils.py", "exec"), namespace)
    options = namespace["create_hccl_pg_options"](group)
    assert "hccl_op_expansion_mode" not in options.hccl_config
    assert options.hccl_config["group_name"] == group
    if group != "mc2":
        assert options.hccl_config["hccl_buffer_size"] == 200
