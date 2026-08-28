# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.models.deepseek_v4.compressor import (
    AscendCompressorMetadata,
    AscendCompressorStateCache,
    Compressor,
)


class TestCompressorMetadata:
    def test_compute_metadata_flattens_rotary_inputs(self):
        compressor = Compressor.__new__(Compressor)
        torch.nn.Module.__init__(compressor)
        compressor.compress_ratio = 4
        full_cos = torch.arange(8).view(2, 1, 1, 4)
        full_sin = -full_cos
        query_start_loc = torch.tensor([0, 2, 4], dtype=torch.int32)
        start_pos = torch.tensor([1, 3], dtype=torch.int32)
        block_table = torch.tensor([[0], [1]], dtype=torch.int32)
        metadata = SimpleNamespace(
            full_compress_cos=full_cos,
            full_compress_sin=full_sin,
            query_start_loc=query_start_loc,
            start_pos=start_pos,
            block_table=block_table,
            storage_block_size=128,
            num_compressed_tokens=3,
            num_actual_reqs=2,
        )
        result_cos = torch.ones((3, 4))
        result_sin = torch.zeros((3, 4))
        slot_mapping = torch.tensor([[0, 1]], dtype=torch.int32)

        with (
            patch.object(
                DeviceOperator,
                "get_dsa_compressor_slot_mapping_format",
                return_value=7,
            ) as get_slot_format,
            patch.object(
                torch.ops._C_ascend,
                "compressor_metadata",
                create=True,
                return_value=(result_cos, result_sin, slot_mapping),
            ) as metadata_op,
        ):
            actual = compressor._compute_metadata(metadata)

        assert actual[0] is result_cos
        assert actual[1] is result_sin
        assert actual[2] is slot_mapping
        get_slot_format.assert_called_once_with()
        args = metadata_op.call_args.args
        assert torch.equal(args[0], full_cos.view(2, 4))
        assert torch.equal(args[1], full_sin.view(2, 4))
        assert args[2] is query_start_loc
        assert args[3] is start_pos
        assert args[4] is block_table
        assert args[5:] == (128, 7, 4, 3, 2)


class TestCompressorForward:
    @pytest.mark.parametrize(
        ("compress_ratio", "overlap", "expected_coff"),
        [(4, True, 2), (128, False, 1)],
    )
    def test_routes_cache_and_state_metadata(
        self,
        compress_ratio: int,
        overlap: bool,
        expected_coff: int,
    ):
        compressor = Compressor.__new__(Compressor)
        torch.nn.Module.__init__(compressor)
        compressor.overlap = overlap
        compressor.compress_ratio = compress_ratio
        compressor.rope_head_dim = 2
        compressor.norm_eps = 1e-6
        compressor.ape = torch.ones((compress_ratio, 4))
        compressor.wkv = SimpleNamespace(weight=torch.ones((4, 4)))
        compressor.wgate = SimpleNamespace(weight=torch.ones((4, 4)))
        compressor.norm = SimpleNamespace(weight=torch.ones(4))
        cache_req_metadata = SimpleNamespace(
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            start_pos=torch.tensor([1], dtype=torch.int32),
        )
        state_req_metadata = SimpleNamespace(block_table=torch.tensor([[3]], dtype=torch.int32))
        metadata = AscendCompressorMetadata(
            cache=SimpleNamespace(req_metadata=cache_req_metadata),
            state=SimpleNamespace(req_metadata=state_req_metadata),
        )
        hidden_states = torch.ones((2, 4))
        state_cache = torch.ones((1, 2, 1, 4))
        compress_cos = torch.ones((1, 1, 2))
        compress_sin = torch.zeros((1, 1, 2))
        slot_mapping = torch.tensor([[0, 1]], dtype=torch.int32)
        compressed_kv = torch.ones((1, 1, 4))
        compute_metadata = MagicMock(return_value=(compress_cos, compress_sin, slot_mapping))
        compressor._compute_metadata = compute_metadata

        with patch.object(
            torch.ops._C_ascend,
            "compressor",
            create=True,
            return_value=compressed_kv,
        ) as compressor_op:
            actual_kv, actual_slot_mapping = compressor(
                hidden_states=hidden_states,
                state_cache=state_cache,
                metadata=metadata,
            )

        assert actual_kv is compressed_kv
        assert actual_slot_mapping is slot_mapping
        compute_metadata.assert_called_once_with(cache_req_metadata)
        call = compressor_op.call_args
        assert call.args[0] is hidden_states
        assert torch.equal(call.args[3], state_cache.squeeze(-2))
        assert call.kwargs["state_block_table"] is state_req_metadata.block_table
        assert call.kwargs["cu_seqlens"] is cache_req_metadata.query_start_loc
        assert call.kwargs["start_pos"] is cache_req_metadata.start_pos
        assert call.kwargs["cmp_ratio"] == compress_ratio
        assert call.kwargs["coff"] == expected_coff


class TestCompressorStateCache:
    @pytest.mark.parametrize(
        ("state_dim", "compress_ratio", "padding_index"),
        [(2 * 256, 4, 0), (2 * 1024, 4, 1), (2 * 512, 128, 1)],
    )
    def test_cache_spec_selects_expected_page_padding(
        self,
        state_dim: int,
        compress_ratio: int,
        padding_index: int,
    ):
        from vllm_ascend.models.layer.attention.layer import DSV4_BLOCK_SIZES

        cache = AscendCompressorStateCache.__new__(AscendCompressorStateCache)
        cache.state_dim = state_dim
        cache.compress_ratio = compress_ratio
        cache.block_size = 8
        cache.dtype = torch.float32
        cache.sliding_window = 64
        vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=128))

        spec = cache.get_kv_cache_spec(vllm_config)

        assert spec.block_size == 8
        assert spec.head_size == state_dim
        assert spec.sliding_window == 64
        assert spec.page_size_padded == DSV4_BLOCK_SIZES[128][1][padding_index]
