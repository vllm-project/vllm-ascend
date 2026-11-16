# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the upstream LI V2 dispatcher boundary."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def binding(monkeypatch):
    # Isolate device execution and package import side effects, but exercise the
    # real wrapper and preserve all tensor identities across its API boundary.
    upstream = ModuleType("cann_ops_transformer")
    monkeypatch.setitem(sys.modules, "cann_ops_transformer", upstream)
    indexer = Mock(return_value=(object(), object()))
    metadata = Mock(return_value=torch.empty(2048, dtype=torch.int32))
    legacy_indexer = Mock(return_value=(object(), object()))
    legacy_metadata = Mock(return_value=torch.empty(2048, dtype=torch.int32))
    for namespace, name, value in (
        (torch.ops.cann_ops_transformer, "lightning_indexer", SimpleNamespace(default=indexer)),
        (torch.ops.cann_ops_transformer, "lightning_indexer_metadata", SimpleNamespace(default=metadata)),
        (torch.ops._C_ascend, "npu_lightning_indexer_v2", legacy_indexer),
        (torch.ops._C_ascend, "npu_lightning_indexer_v2_metadata", legacy_metadata),
    ):
        monkeypatch.setattr(namespace, name, value, raising=False)
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/lightning_indexer_v2.py"
    spec = importlib.util.spec_from_file_location("li_v2_contract", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, indexer, metadata, legacy_indexer, legacy_metadata


def test_metadata_uses_upstream_even_with_legacy_extension(binding):
    module, _, upstream, _, legacy = binding
    lengths = torch.tensor([8, 12], dtype=torch.int32)
    boundaries = torch.tensor([0, 1, 3], dtype=torch.int32)
    residual = torch.tensor([0, 2], dtype=torch.int32)
    result = module.lightning_indexer_v2_metadata(
        4,
        1,
        128,
        512,
        cu_seqlens_q=boundaries,
        seqused_k=lengths,
        cmp_residual_k=residual,
        batch_size=2,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=4,
    )
    upstream.assert_called_once()
    legacy.assert_not_called()
    assert result is upstream.return_value
    args, kwargs = upstream.call_args
    assert args == (4, 1, 128, 512)
    assert kwargs["cu_seqlens_q"] is boundaries
    assert kwargs["seqused_k"] is lengths
    assert kwargs["cmp_residual_k"] is residual
    assert (kwargs["batch_size"], kwargs["layout_q"], kwargs["layout_k"], kwargs["mask_mode"], kwargs["cmp_ratio"]) == (
        2,
        "TND",
        "PA_BBND",
        3,
        4,
    )


@pytest.mark.parametrize("return_value", [0, 1])
def test_indexer_preserves_prepared_metadata_and_strided_cache(binding, return_value):
    module, upstream, metadata_op, legacy, _ = binding
    query = torch.empty(3, 4, 128, dtype=torch.bfloat16)
    cache = torch.empty(8, 16, 1, 128, dtype=torch.bfloat16)[::2]
    weights = torch.empty(3, 4, dtype=torch.float32)
    metadata = torch.empty(2048, dtype=torch.int32)
    table = torch.tensor([[0, 1]], dtype=torch.int32)
    kwargs = dict(
        cu_seqlens_q=torch.tensor([0, 3], dtype=torch.int32),
        seqused_k=torch.tensor([32], dtype=torch.int32),
        cmp_residual_k=torch.tensor([2], dtype=torch.int32),
        block_table=table,
        metadata=metadata,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=4,
        return_value=return_value,
    )
    result = module.lightning_indexer_v2(query, cache, weights, 512, **kwargs)
    upstream.assert_called_once()
    legacy.assert_not_called()
    metadata_op.assert_not_called()
    assert result is upstream.return_value
    args, forwarded = upstream.call_args
    assert args[0] is query and args[1] is cache and args[2] is weights
    assert args[3] == 512
    for name, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            assert forwarded[name] is value
        else:
            assert forwarded[name] == value
    assert "candidate_topk_blocks" not in forwarded


def test_metadata_none_defaults_are_preserved(binding):
    module, _, upstream, _, _ = binding
    defaults = dict(
        batch_size=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        layout_q=None,
        layout_k=None,
        mask_mode=None,
        cmp_ratio=None,
    )
    module.lightning_indexer_v2_metadata(4, 1, 128, 16, seqused_k=torch.empty(1, dtype=torch.int32), **defaults)
    upstream.assert_called_once()
    kwargs = upstream.call_args.kwargs
    assert tuple(kwargs[key] for key in defaults) == (0, -1, -1, "BSND", "BSND", 0, 1)


def test_missing_length_tensors_fail_before_dispatch(binding):
    module, indexer, metadata, legacy_indexer, legacy_metadata = binding
    with pytest.raises(ValueError, match="at least one NPU length tensor"):
        module.lightning_indexer_v2_metadata(4, 1, 128, 16)
    for op in (indexer, metadata, legacy_indexer, legacy_metadata):
        op.assert_not_called()


def test_pure_meta_metadata_does_not_dispatch_to_npu(binding):
    module, _, upstream, _, legacy = binding
    result = module.lightning_indexer_v2_metadata(
        4, 1, 128, 16, seqused_k=torch.empty(2, dtype=torch.int32, device="meta")
    )
    assert result.device.type == "meta"
    assert result.dtype == torch.int32
    assert result.shape == (2048,)
    upstream.assert_not_called()
    legacy.assert_not_called()
