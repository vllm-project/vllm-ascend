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


class TestAscendDSACPKVAllGather:
    def test_starts_async_gather_in_tp_group(self):
        impl = AscendDSACPImpl.__new__(AscendDSACPImpl)
        impl.tp_group = object()
        local_hidden_states = torch.arange(8).view(2, 4)
        gathered_hidden_states = torch.arange(16).view(4, 4)
        handle = MagicMock()

        with patch(
            "vllm_ascend.attention.context_parallel.dsa_cp.all_gather_async",
            return_value=(gathered_hidden_states, handle),
        ) as gather:
            result, result_handle = impl._start_kv_hidden_states_all_gather(local_hidden_states, enabled=True)

        gather.assert_called_once_with(local_hidden_states, impl.tp_group, async_op=True)
        assert result is gathered_hidden_states
        assert result_handle is handle

    def test_skips_gather_when_sequence_parallel_is_disabled(self):
        impl = AscendDSACPImpl.__new__(AscendDSACPImpl)
        local_hidden_states = torch.arange(8).view(2, 4)

        with patch("vllm_ascend.attention.context_parallel.dsa_cp.all_gather_async") as gather:
            result, handle = impl._start_kv_hidden_states_all_gather(local_hidden_states, enabled=False)

        gather.assert_not_called()
        assert result is local_hidden_states
        assert handle is None

    def test_waits_before_slicing_gathered_hidden_states(self):
        gathered_hidden_states = torch.arange(24).view(6, 4)
        handle = MagicMock()

        result = AscendDSACPImpl._finish_kv_hidden_states_all_gather(
            gathered_hidden_states,
            handle,
            num_actual_tokens=4,
        )

        handle.wait.assert_called_once_with()
        assert torch.equal(result, gathered_hidden_states[:4])
