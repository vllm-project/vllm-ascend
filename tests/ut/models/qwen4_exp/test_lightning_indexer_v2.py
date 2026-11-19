# SPDX-License-Identifier: Apache-2.0
"""CPU coverage for padded request bounds and the QSA/Liv2 buffer contract."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def selector(monkeypatch):
    # Load the real selector without importing vLLM or a Triton NPU runtime.
    # Only device kernels and the external stride utility are replaced.
    upstream = ModuleType("cann_ops_transformer")
    bindings = SimpleNamespace(
        lightning_indexer_v2_metadata=Mock(return_value=torch.empty(1024, dtype=torch.int32)),
        lightning_indexer_v2=Mock(
            side_effect=lambda q, k, w, topk, **kwargs: (
                torch.zeros(q.shape[0], 1, topk, dtype=torch.int32),
                torch.empty(0),
            )
        ),
    )
    for name, op in (
        ("lightning_indexer_metadata", bindings.lightning_indexer_v2_metadata),
        ("lightning_indexer", bindings.lightning_indexer_v2),
    ):
        monkeypatch.setattr(torch.ops.cann_ops_transformer, name, op, raising=False)

    def fake_pack(groups, positions, lengths, mapping, ratio, topk, out):
        # Check buffer wiring; real packing arithmetic is covered on NPU.
        out.fill_(-1)
        out[positions >= 0] = 7

    packing = ModuleType("vllm_ascend.ops.triton.qwen4_exp.qsa")
    packing.__dict__.update(pack_qsa_group_indices_npu=Mock(side_effect=fake_pack))
    utilities = ModuleType("vllm.utils.torch_utils")
    utilities.__dict__.update(canonicalize_singleton_dim_strides=Mock(side_effect=lambda tensor: tensor))
    preparation = ModuleType("vllm_ascend.ops.triton.qwen4_exp.lightning_v2_prepare")

    def reference_prepare(query, lengths, starts, ratio):
        # CPU oracle for the device boundary, not execution of the Triton kernel.
        weight = torch.tensor(1 / 128**0.5, dtype=torch.bfloat16).float()
        return (
            weight.expand(query.shape[:2]).clone(),
            starts[: lengths.numel() + 1].clamp_max(query.shape[0]),
            lengths.floor_divide(ratio),
            lengths.remainder(ratio),
        )

    preparation.__dict__.update(prepare_qsa_lightning_v2_inputs=Mock(side_effect=reference_prepare))
    for module in (upstream, packing, utilities, preparation):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    source = Path(__file__).resolve().parents[4] / "vllm_ascend/models/qwen4_exp/lightning_indexer.py"
    spec = importlib.util.spec_from_file_location("qsa_liv2_contract", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, bindings, packing, utilities


def _inputs():
    return dict(
        query=torch.empty(6, 4, 128, dtype=torch.bfloat16),
        compressed_key_cache=torch.empty(8, 16, 1, 128, dtype=torch.bfloat16)[::2],
        attn_metadata=SimpleNamespace(
            block_table=torch.arange(12, dtype=torch.int32).reshape(3, 4),
            token_to_req=torch.tensor([0, 0, 1, 1, 2, 2]),
            logical_positions=torch.tensor([31, 32, 32, 33, -1, -1]),
            seq_lens=torch.tensor([33, 34, 0], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 2, 4, 6], dtype=torch.int32),
            num_actual_tokens=6,
        ),
        token_topk=2048,
        compress_ratio=4,
        out=torch.full((6, 2051), 99, dtype=torch.int32),
    )


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize(
    "rows,starts,expected_starts",
    [
        (4, [0, 2, 4, 6], [0, 2, 4, 4]),
        (6, [0, 2, 4, 6], [0, 2, 4, 6]),
        (6, [0, 2, 4, 4], [0, 2, 4, 4]),
    ],
    ids=["trimmed-query", "dummy-query-zero-key", "repeated-terminal"],
)
def test_padded_bounds_preserve_output_and_metadata(selector, monkeypatch, rows, starts, expected_starts, prepared):
    module, bindings, packing, utilities = selector
    inputs = _inputs()
    inputs["query"] = inputs["query"][:rows]
    inputs["attn_metadata"].query_start_loc = torch.tensor(starts, dtype=torch.int32)
    if prepared:
        from tests.ut.models.qwen4_exp.test_input_adapter import (
            AscendQwen4ExpQSAAttention,
            _load_adapter,
            _Model,
        )

        adapter_cls, _ = _load_adapter(monkeypatch)
        monkeypatch.setattr(
            torch.ops.cann_ops_transformer, "lightning_indexer_metadata", bindings.lightning_indexer_v2_metadata
        )
        layer = AscendQwen4ExpQSAAttention()
        adapter = adapter_cls(_Model(layer))
        inputs["attn_metadata"].num_actual_tokens = rows
        adapter._prepare_lightning(layer, inputs["attn_metadata"], prepared_this_step={})
    result = module.qsa_select_paged_tokens_lightning_v2(**inputs)
    assert result is inputs["out"]
    assert torch.all(result[:4, :515] == 7)
    assert torch.all(result[:4, 515:] == -1)
    assert torch.all(result[4:] == -1)
    bindings.lightning_indexer_v2_metadata.assert_called_once()
    metadata = (
        inputs["attn_metadata"].qsa_lightning_v2_metadata
        if prepared
        else bindings.lightning_indexer_v2_metadata.return_value
    )
    metadata_kwargs = bindings.lightning_indexer_v2_metadata.call_args.kwargs
    assert metadata_kwargs["batch_size"] == 3
    bindings.lightning_indexer_v2.assert_called_once()
    args, kwargs = bindings.lightning_indexer_v2.call_args
    assert args[0] is inputs["query"]
    assert args[1] is inputs["compressed_key_cache"]
    assert not args[1].is_contiguous()
    assert args[2].dtype == torch.float32
    expected_weight = torch.tensor(1 / 128**0.5, dtype=torch.bfloat16).float()
    torch.testing.assert_close(args[2], expected_weight.expand(rows, 4))
    assert args[3] == 512
    assert kwargs["metadata"] is metadata
    assert kwargs["block_table"].shape == (3, 4)
    assert kwargs["block_table"].data_ptr() == inputs["attn_metadata"].block_table.data_ptr()
    for name in ("cu_seqlens_q", "seqused_k", "cmp_residual_k"):
        torch.testing.assert_close(metadata_kwargs[name], kwargs[name])
    torch.testing.assert_close(kwargs["cu_seqlens_q"], torch.tensor(expected_starts, dtype=torch.int32))
    torch.testing.assert_close(inputs["attn_metadata"].query_start_loc, torch.tensor(starts, dtype=torch.int32))
    torch.testing.assert_close(kwargs["seqused_k"], torch.tensor([8, 8, 0], dtype=torch.int32))
    torch.testing.assert_close(kwargs["cmp_residual_k"], torch.tensor([1, 2, 0], dtype=torch.int32))
    assert (kwargs["layout_q"], kwargs["layout_k"], kwargs["mask_mode"], kwargs["cmp_ratio"]) == (
        "TND",
        "PA_BBND",
        3,
        4,
    )
    utilities.canonicalize_singleton_dim_strides.assert_called_once()
    module.prepare_qsa_lightning_v2_inputs.assert_called_once()
    packing.pack_qsa_group_indices_npu.assert_called_once()


def test_empty_query_clears_output_without_launch(selector):
    module, bindings, packing, _ = selector
    inputs = _inputs()
    inputs["query"] = inputs["query"][:0]
    result = module.qsa_select_paged_tokens_lightning_v2(**inputs)
    assert result is inputs["out"]
    assert torch.all(result == -1)
    bindings.lightning_indexer_v2.assert_not_called()
    bindings.lightning_indexer_v2_metadata.assert_not_called()
    packing.pack_qsa_group_indices_npu.assert_not_called()
    module.prepare_qsa_lightning_v2_inputs.assert_not_called()


def test_further_trimmed_query_regenerates_prepared_schedule(selector):
    module, bindings, _, _ = selector
    inputs = _inputs()
    inputs["attn_metadata"].qsa_lightning_v2_metadata = torch.empty(1024, dtype=torch.int32)
    inputs["query"] = inputs["query"][:4]
    module.qsa_select_paged_tokens_lightning_v2(**inputs)
    bindings.lightning_indexer_v2_metadata.assert_called_once()
    assert (
        bindings.lightning_indexer_v2.call_args.kwargs["metadata"]
        is bindings.lightning_indexer_v2_metadata.return_value
    )
    torch.testing.assert_close(
        bindings.lightning_indexer_v2_metadata.call_args.kwargs["cu_seqlens_q"],
        torch.tensor([0, 2, 4, 4], dtype=torch.int32),
    )
