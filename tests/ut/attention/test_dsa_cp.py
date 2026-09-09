# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSACPImpl


class TestAscendDSACPLayerMetadata:
    def test_routes_by_cache_prefix(self):
        impl = AscendDSACPImpl.__new__(AscendDSACPImpl)
        impl.compress_ratio = 4
        impl.swa_cache_layer = SimpleNamespace(prefix="swa_cache")
        impl.compressor = SimpleNamespace(state_cache=SimpleNamespace(prefix="compressor.state_cache"))
        impl.indexer = SimpleNamespace(
            k_cache=SimpleNamespace(prefix="indexer.k_cache"),
            compressor=SimpleNamespace(state_cache=SimpleNamespace(prefix="indexer.compressor.state_cache")),
        )
        attention_metadata = object()
        compressor_state_metadata = object()
        indexer_cache_metadata = object()
        indexer_state_metadata = object()
        swa_metadata = object()

        metadata: Any = {
            "layer": attention_metadata,
            "compressor.state_cache": compressor_state_metadata,
            "indexer.k_cache": indexer_cache_metadata,
            "indexer.compressor.state_cache": indexer_state_metadata,
            "swa_cache": swa_metadata,
        }
        layer_metadata = impl._get_layer_metadata("layer", metadata)

        assert layer_metadata.swa is swa_metadata
        assert layer_metadata.compressor_cache is attention_metadata
        assert layer_metadata.compressor_state is compressor_state_metadata
        assert layer_metadata.indexer_cache is indexer_cache_metadata
        assert layer_metadata.indexer_state is indexer_state_metadata


def test_cp_indexer_postprocesses_formal_compressor_before_scatter():
    impl = AscendDSACPImpl.__new__(AscendDSACPImpl)
    impl.compressor_overlap = True
    impl.compress_ratio = 4
    impl.indexcom_wkv = SimpleNamespace(weight=torch.ones(256, 4096))
    impl.indexcom_wgate = SimpleNamespace(weight=torch.zeros(256, 4096))
    impl.indexcom_ape = torch.zeros(4, 256)
    raw = torch.ones(2, 128, dtype=torch.bfloat16)
    processed = torch.full((1, 128), 2.0, dtype=torch.bfloat16)
    postprocess = MagicMock(return_value=processed)
    impl.indexer = SimpleNamespace(compressor=SimpleNamespace(rotate=False, _postprocess=postprocess))
    cos = torch.ones(1, 1, 64)
    sin = torch.zeros_like(cos)
    slots = torch.tensor([1])
    impl._compute_compressor_metadata = MagicMock(return_value=(cos, sin, slots))
    table = torch.tensor([[1]], dtype=torch.int32)
    start_pos = torch.tensor([0], dtype=torch.int32)
    metadata = SimpleNamespace(
        indexer_state=SimpleNamespace(req_metadata=SimpleNamespace(block_table=table)),
        indexer_cache=SimpleNamespace(req_metadata=SimpleNamespace(start_pos=start_pos)),
    )
    state = torch.zeros(2, 128, 1, 512)
    key_cache = torch.empty(1)
    scale_cache = torch.empty(1)
    full_cache = torch.empty(1)
    lengths = torch.tensor([0, 4], dtype=torch.int32)
    operator_path = "vllm_ascend.attention.context_parallel.dsa_cp.DeviceOperator"
    with (
        patch(f"{operator_path}.unpack_dsa_indexer_kv_cache", return_value=(state, key_cache, scale_cache, full_cache)),
        patch(f"{operator_path}.indexer_quant_scatter_part1", return_value=(None, None)) as scatter,
        patch.object(torch.ops._C_ascend, "compressor", return_value=raw, create=True) as op,
    ):
        impl._update_indexer_cache(torch.ones(4, 4096), (), metadata, lengths)
    assert len(op.call_args.args) == 5
    assert op.call_args.kwargs["cu_seqlens"] is lengths
    assert op.call_args.kwargs["start_pos"] is start_pos
    assert "norm_eps" not in op.call_args.kwargs
    postprocess.assert_called_once()
    torch.testing.assert_close(postprocess.call_args.args[0], raw[:1])
    assert scatter.call_args.args[0] is processed
    assert scatter.call_args.args[3] is slots
