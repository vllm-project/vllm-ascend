# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test QSFA dispatch with CPU tensors: pytest --confcutdir=tests/ut/device.

Only operator and hardware boundaries are mocked; execute the production
adaptor methods without importing the NPU/vLLM runtime.
"""

import ast
import runpy
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
DEVICE_TYPES = runpy.run_path(str(ROOT / "vllm_ascend/device/hardware.py"))["AscendDeviceType"]


@pytest.fixture
def adaptor():
    source = ROOT / "vllm_ascend/device/device_op.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BaseDeviceAdaptor")
    methods = {
        "_load_cann_kv_quant_sparse_flash_attention",
        "_execute_kv_quant_sparse_flash_attention_vllm",
        "execute_sparse_flash_attention_process",
    }
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    operators = SimpleNamespace(cann=Mock(), custom=Mock(), public=Mock(), floating=Mock())
    cpu_torch = SimpleNamespace(
        Tensor=torch.Tensor,
        int8=torch.int8,
        float8_e4m3fn=torch.float8_e4m3fn,
        float8_e5m2=torch.float8_e5m2,
        float32=torch.float32,
        cat=torch.cat,
        full=Mock(wraps=torch.full),
        ops=SimpleNamespace(
            _C_ascend=SimpleNamespace(
                npu_kv_quant_sparse_flash_attention_vllm=operators.custom,
                npu_sparse_flash_attention=operators.floating,
            ),
        ),
    )
    importer = Mock(return_value=SimpleNamespace(kv_quant_sparse_flash_attention=operators.cann))
    device = Mock(return_value=DEVICE_TYPES.A3)
    scope = dict(
        torch=cpu_torch,
        torch_npu=SimpleNamespace(npu_kv_quant_sparse_flash_attention=operators.public),
        F=F,
        lru_cache=lru_cache,
        import_module=importer,
        get_ascend_device_type=device,
        AscendDeviceType=DEVICE_TYPES,
        _SFA_MAX_GENERIC_QUERY_GROUP_SIZE=64,
        _SFA_PADDED_QUERY_GROUP_SIZE=128,
    )
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[])), str(source), "exec"), scope
    )
    return SimpleNamespace(
        cls=scope["BaseDeviceAdaptor"], ops=operators, importer=importer, device=device, torch=cpu_torch
    )


def make_inputs(heads=12, rope_dim=64, *, kimi=True):
    query = torch.randn(heads, 3, 512 + rope_dim).transpose(0, 1)
    nope, rope = query.split((512, rope_dim), dim=-1)
    return SimpleNamespace(
        impl=SimpleNamespace(scale=0.125, qk_rope_head_dim=rope_dim, is_kimi_k31=kimi, sfa_qsfa_sinks=None),
        query=query,
        nope=nope,
        rope=rope,
        cache=torch.empty(2, 128, 1, 528 + 2 * rope_dim, dtype=torch.int8),
        indices=torch.full((3, 1, 2048), -1, dtype=torch.int32),
        metadata=SimpleNamespace(block_table=torch.tensor([[0, 1]], dtype=torch.int32)),
        query_lens=torch.tensor([3], dtype=torch.int32),
        kv_lens=torch.tensor([256], dtype=torch.int32),
    )


def execute(adaptor, inputs, *, packed=False, return_lse=False):
    return adaptor.cls.execute_sparse_flash_attention_process(
        inputs.impl,
        inputs.nope,
        inputs.rope,
        (inputs.cache,),
        inputs.indices,
        inputs.metadata,
        inputs.query_lens,
        inputs.kv_lens,
        sparse_mode=0 if packed else 3,
        return_lse=return_lse,
        packed_query=inputs.query if packed else None,
    )


@pytest.mark.parametrize("heads", [12, 24, 96])
@pytest.mark.parametrize("rope_dim", [0, 64])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("return_lse", [False, True])
def test_a3_k31_routes_native_heads_and_statistics_to_cann(adaptor, heads, rope_dim, packed, return_lse):
    inputs = make_inputs(heads, rope_dim)
    if packed:
        # DCP's gathered query has more heads than the local split fragments.
        inputs.nope = inputs.nope[:, :1]
        inputs.rope = inputs.rope[:, :1] if rope_dim else None
    result = (torch.randn(3, heads, 512), torch.randn(1, 3, heads), torch.ones(1, 3, heads))
    adaptor.ops.cann.return_value = result

    actual = execute(adaptor, inputs, packed=packed, return_lse=return_lse)

    assert actual is (result if return_lse else result[0])
    adaptor.importer.assert_called_once_with("cann_ops_transformer.ops")
    adaptor.ops.cann.assert_called_once()
    adaptor.ops.custom.assert_not_called()
    adaptor.ops.public.assert_not_called()
    kwargs = adaptor.ops.cann.call_args.kwargs
    torch.testing.assert_close(kwargs["query"], inputs.query)
    assert kwargs["query"].is_contiguous()
    assert kwargs["key"] is kwargs["value"] is inputs.cache
    assert kwargs["sparse_indices"] is inputs.indices
    assert kwargs["block_table"] is inputs.metadata.block_table
    assert kwargs["actual_seq_lengths_query"] is inputs.query_lens
    assert kwargs["actual_seq_lengths_kv"] is inputs.kv_lens
    assert kwargs["layout_query"] == "TND"
    assert kwargs["layout_kv"] == "PA_BSND"
    assert kwargs["sparse_mode"] == (0 if packed else 3)
    assert kwargs["scale_value"] == inputs.impl.scale
    assert kwargs["key_quant_mode"] == kwargs["value_quant_mode"] == 2
    assert kwargs["attention_mode"] == 2
    assert kwargs["quant_scale_repo_mode"] == 1
    assert kwargs["tile_size"] == 128
    assert kwargs["rope_head_dim"] == rope_dim
    assert kwargs["return_softmax_lse"] is True
    assert kwargs["sinks"].shape == (heads,)
    assert kwargs["sinks"].dtype == torch.float32
    assert torch.isneginf(kwargs["sinks"]).all()


@pytest.mark.parametrize(
    "device,kimi",
    [
        (DEVICE_TYPES.A2, False),
        (DEVICE_TYPES.A2, True),
        (DEVICE_TYPES.A3, False),
        (DEVICE_TYPES.A5, False),
        (DEVICE_TYPES.A5, True),
    ],
)
@pytest.mark.parametrize("packed,return_lse", [(False, False), (False, True), (True, True)])
def test_other_models_and_devices_keep_existing_operator_selection(adaptor, device, kimi, packed, return_lse):
    adaptor.device.return_value = device
    inputs = make_inputs(kimi=kimi)
    result = (torch.randn(3, 12, 512), torch.zeros(1, 3, 12), torch.ones(1, 3, 12))
    use_custom = kimi or packed or return_lse
    selected = adaptor.ops.custom if use_custom else adaptor.ops.public
    selected.return_value = result

    assert execute(adaptor, inputs, packed=packed, return_lse=return_lse) is (result if return_lse else result[0])

    selected.assert_called_once()
    adaptor.ops.cann.assert_not_called()
    adaptor.importer.assert_not_called()
    assert inputs.impl.sfa_qsfa_sinks is None


def test_a3_k31_floating_cache_does_not_load_cann(adaptor):
    inputs = make_inputs(rope_dim=0)
    inputs.cache = inputs.cache.to(torch.bfloat16)
    result = (torch.randn(3, 12, 512), torch.zeros(1, 3, 12), torch.ones(1, 3, 12))
    adaptor.ops.floating.return_value = result

    assert execute(adaptor, inputs) is result[0]
    adaptor.ops.floating.assert_called_once()
    adaptor.importer.assert_not_called()


def test_cann_no_sink_preserves_attention_and_dcp_lse(adaptor):
    inputs = make_inputs(heads=12, rope_dim=0)
    # A zero logit sink would halve this single-candidate attention output.
    inputs.query.zero_()
    values = torch.arange(512, dtype=torch.float32).expand(3, 12, -1)

    def reference_operator(**kwargs):
        logits = kwargs["query"].sum(-1)
        sinks = kwargs.get("sinks", torch.zeros(12)).expand_as(logits)
        maximum = torch.maximum(logits, sinks)
        total = (logits - maximum).exp() + (sinks - maximum).exp()
        output = values * ((logits - maximum).exp() / total).unsqueeze(-1)
        return output, maximum.unsqueeze(0), total.unsqueeze(0)

    adaptor.ops.cann.side_effect = reference_operator
    output, maximum, total = execute(adaptor, inputs, packed=True, return_lse=True)

    torch.testing.assert_close(output, values)
    torch.testing.assert_close(maximum + total.log(), torch.zeros(1, 3, 12))


def test_no_sink_buffer_and_operator_are_reused_after_warmup(adaptor):
    inputs = make_inputs()
    adaptor.ops.cann.return_value = (torch.empty(3, 12, 512), torch.empty(0), torch.empty(0))
    execute(adaptor, inputs)
    sinks = adaptor.ops.cann.call_args.kwargs["sinks"]
    execute(adaptor, inputs, packed=True)

    assert adaptor.ops.cann.call_args.kwargs["sinks"] is sinks
    assert inputs.impl.sfa_qsfa_sinks[(12, inputs.query.device)] is sinks
    adaptor.torch.full.assert_called_once()
    adaptor.importer.assert_called_once()

    gathered = make_inputs(heads=24)
    gathered.impl = inputs.impl
    execute(adaptor, gathered, packed=True)
    assert gathered.impl.sfa_qsfa_sinks[(24, gathered.query.device)].shape == (24,)
    execute(adaptor, inputs)
    assert adaptor.ops.cann.call_args.kwargs["sinks"] is sinks
    assert adaptor.torch.full.call_count == 2
    assert torch.isneginf(sinks).all()


@pytest.mark.parametrize("missing", ["package", "operator"])
def test_missing_cann_dependency_reports_requirement_without_fallback(adaptor, missing):
    if missing == "package":
        adaptor.importer.side_effect = ModuleNotFoundError("cann_ops_transformer")
    else:
        del adaptor.importer.return_value.kv_quant_sparse_flash_attention

    with pytest.raises(RuntimeError, match="Kimi K3.1 INT8 KV on Ascend A3 requires cann_ops_transformer"):
        execute(adaptor, make_inputs())

    adaptor.ops.custom.assert_not_called()
    adaptor.ops.public.assert_not_called()


def test_dcp_rejects_operator_without_softmax_statistics(adaptor):
    adaptor.ops.cann.return_value = torch.empty(3, 12, 512)
    with pytest.raises(RuntimeError, match="did not return softmax max/sum"):
        execute(adaptor, make_inputs(), packed=True, return_lse=True)
