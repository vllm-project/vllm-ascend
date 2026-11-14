# SPDX-License-Identifier: Apache-2.0
"""CPU coverage for live-prefix slicing and the QSA/Liv2 buffer contract."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def selector(monkeypatch):
    # Load the real selector without importing vLLM or a Triton NPU runtime.
    # Only device kernels and the external stride utility are replaced.
    bindings = ModuleType("vllm_ascend.ops.lightning_indexer_v2")
    bindings.lightning_indexer_v2_metadata = Mock(return_value=torch.empty(2048, dtype=torch.int32))
    bindings.lightning_indexer_v2 = Mock(
        side_effect=lambda q, *args, **kwargs: (torch.zeros(q.shape[0], 1, 512, dtype=torch.int32), torch.empty(0))
    )
    packing = ModuleType("vllm_ascend.ops.triton.qwen4_exp.qsa")
    packing.pack_qsa_group_indices_npu = Mock(side_effect=lambda *args: args[-1].fill_(7))
    utilities = ModuleType("vllm.utils.torch_utils")
    utilities.canonicalize_singleton_dim_strides = Mock(side_effect=lambda tensor: tensor)
    for module in (bindings, packing, utilities):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    source = Path(__file__).resolve().parents[4] / "vllm_ascend/models/qwen4_exp/lightning_indexer.py"
    spec = importlib.util.spec_from_file_location("qsa_liv2_contract", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, bindings, packing, utilities


def _inputs():
    return dict(
        query=torch.empty(6, 4, 128, dtype=torch.bfloat16),
        compressed_key_cache=torch.empty(8, 16, 1, 128, dtype=torch.bfloat16)[::2],
        block_table=torch.arange(12, dtype=torch.int32).reshape(3, 4),
        token_to_req=torch.tensor([0, 0, 1, 1, 2, 2]),
        query_positions=torch.tensor([31, 32, 32, 33, 0, 0]),
        sequence_lengths=torch.tensor([33, 34, 0], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2, 4, 6], dtype=torch.int32),
        token_topk=2048,
        compress_ratio=4,
        out=torch.full((6, 2051), 99, dtype=torch.int32),
        num_actual_requests=2,
        num_actual_tokens=4,
    )


@pytest.mark.parametrize("prepared", [False, True])
def test_live_prefixes_preserve_output_and_metadata(selector, prepared):
    module, bindings, packing, utilities = selector
    inputs = _inputs()
    metadata = torch.empty(2048, dtype=torch.int32) if prepared else None
    result = module.qsa_select_paged_tokens_lightning_v2(**inputs, prepared_metadata=metadata)
    assert result is inputs["out"]
    assert torch.all(result[:4, :515] == 7)
    assert torch.all(result[:4, 515:] == -1)
    assert torch.all(result[4:] == -1)
    if prepared:
        bindings.lightning_indexer_v2_metadata.assert_not_called()
    else:
        bindings.lightning_indexer_v2_metadata.assert_called_once()
        metadata = bindings.lightning_indexer_v2_metadata.return_value
        assert bindings.lightning_indexer_v2_metadata.call_args.kwargs["batch_size"] == 2
    args, kwargs = bindings.lightning_indexer_v2.call_args
    assert args[0].shape == (4, 4, 128)
    assert args[0].data_ptr() == inputs["query"].data_ptr()
    assert args[1] is inputs["compressed_key_cache"]
    assert not args[1].is_contiguous()
    assert args[2].dtype == torch.float32
    expected_weight = torch.tensor(1 / 128**0.5, dtype=torch.bfloat16).float()
    torch.testing.assert_close(args[2], expected_weight.expand(4, 4))
    assert args[3] == 512
    assert kwargs["metadata"] is metadata
    assert kwargs["block_table"].shape == (2, 4)
    assert kwargs["block_table"].data_ptr() == inputs["block_table"].data_ptr()
    torch.testing.assert_close(kwargs["cu_seqlens_q"], torch.tensor([0, 2, 4], dtype=torch.int32))
    torch.testing.assert_close(kwargs["seqused_k"], torch.tensor([8, 8], dtype=torch.int32))
    torch.testing.assert_close(kwargs["cmp_residual_k"], torch.tensor([1, 2], dtype=torch.int32))
    assert (kwargs["layout_q"], kwargs["layout_k"], kwargs["mask_mode"], kwargs["cmp_ratio"]) == (
        "TND",
        "PA_BBND",
        3,
        4,
    )
    utilities.canonicalize_singleton_dim_strides.assert_called_once()
    packing.pack_qsa_group_indices_npu.assert_called_once()


def test_empty_live_prefix_clears_padding_without_launch(selector):
    module, bindings, packing, _ = selector
    inputs = _inputs()
    inputs.update(num_actual_requests=0, num_actual_tokens=0)
    result = module.qsa_select_paged_tokens_lightning_v2(**inputs)
    assert result is inputs["out"]
    assert torch.all(result == -1)
    bindings.lightning_indexer_v2.assert_not_called()
    bindings.lightning_indexer_v2_metadata.assert_not_called()
    packing.pack_qsa_group_indices_npu.assert_not_called()


@pytest.mark.parametrize(
    "override,message",
    [
        ({"num_actual_requests": 4}, "live counts"),
        ({"num_actual_tokens": 7}, "live counts"),
        ({"num_actual_tokens": -1}, "live counts"),
        ({"num_actual_requests": 0}, "nonempty queries"),
        ({"block_table": torch.empty(1, 4, dtype=torch.int32)}, "live requests"),
        ({"query_start_loc": torch.zeros(2, dtype=torch.int32)}, "live requests"),
        ({"token_topk": 1024}, "token_topk"),
        ({"compress_ratio": 2}, "compress_ratio"),
        ({"token_to_req": torch.zeros(5, dtype=torch.int64)}, "query rows"),
        ({"query_positions": torch.zeros(6, dtype=torch.float32)}, "integer tensors"),
        ({"out": torch.empty(6, 2051, dtype=torch.float32)}, "output must match"),
    ],
)
def test_invalid_capacity_and_packing_inputs_fail_before_dispatch(selector, override, message):
    module, bindings, packing, _ = selector
    inputs = _inputs()
    inputs.update(override)
    with pytest.raises(ValueError, match=message):
        module.qsa_select_paged_tokens_lightning_v2(**inputs)
    bindings.lightning_indexer_v2.assert_not_called()
    bindings.lightning_indexer_v2_metadata.assert_not_called()
    packing.pack_qsa_group_indices_npu.assert_not_called()
