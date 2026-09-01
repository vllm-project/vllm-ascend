# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.attention import dsa_attn_kv_plan
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
        compressor.vllm_config = SimpleNamespace()
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
        plan = SimpleNamespace(get_dsa_compressor_slot_mapping_format=MagicMock(return_value=7))

        with (
            patch.object(
                dsa_attn_kv_plan,
                "get_dsa_attn_kv_plan",
                return_value=plan,
            ),
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
        plan.get_dsa_compressor_slot_mapping_format.assert_called_once_with()
        args = metadata_op.call_args.args
        assert torch.equal(args[0], full_cos.view(2, 4))
        assert torch.equal(args[1], full_sin.view(2, 4))
        assert args[2] is query_start_loc
        assert args[3] is start_pos
        assert args[4] is block_table
        assert args[5:] == (128, 7, 4, 3, 2)


class TestCompressorForward:
    def test_init_preallocates_packed_projection(self):
        config = SimpleNamespace(hidden_size=1024, qk_rope_head_dim=64, rms_norm_eps=1e-6)
        merged_projection = MagicMock()

        with (
            patch(
                "vllm_ascend.models.deepseek_v4.compressor.MergedColumnParallelLinear",
                return_value=merged_projection,
            ) as merged_linear,
            patch("vllm_ascend.models.deepseek_v4.compressor.RMSNorm"),
            patch("vllm_ascend.models.deepseek_v4.compressor.AscendCompressorStateCache"),
        ):
            compressor = Compressor(
                vllm_config=SimpleNamespace(),
                config=config,
                compress_ratio=4,
                head_dim=512,
                cache_config=SimpleNamespace(block_size=128),
                prefix="model.layers.0.compressor",
            )

        assert compressor.fused_wkv_wgate is merged_projection
        merged_linear.assert_called_once_with(
            1024,
            [1024, 1024],
            bias=False,
            quant_config=None,
            prefix="model.layers.0.compressor.fused_wkv_wgate",
            return_bias=False,
            disable_tp=True,
        )

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
        projection_dim = expected_coff * 4
        compressor.ape = torch.ones((compress_ratio, projection_dim))
        fused_weight = torch.arange(2 * projection_dim * 4, dtype=torch.float32).view(
            2 * projection_dim, 4
        )
        compressor.fused_wkv_wgate = SimpleNamespace(weight=fused_weight)
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

        with (
            patch(
                "vllm_ascend.models.deepseek_v4.compressor.envs_ascend.VLLM_ASCEND_DSA_COMPRESSOR_SPLIT",
                False,
            ),
            patch.object(
                torch.ops._C_ascend,
                "compressor",
                create=True,
                return_value=compressed_kv,
            ) as compressor_op,
        ):
            actual_kv, actual_slot_mapping = compressor(hidden_states, state_cache, metadata)

        assert actual_kv is compressed_kv
        assert actual_slot_mapping is slot_mapping
        compute_metadata.assert_called_once_with(cache_req_metadata)
        call = compressor_op.call_args
        assert call.args[0] is hidden_states
        assert torch.equal(call.args[1], fused_weight[:projection_dim])
        assert torch.equal(call.args[2], fused_weight[projection_dim:])
        assert torch.equal(call.args[3], state_cache.squeeze(-2))
        assert call.kwargs["state_block_table"] is state_req_metadata.block_table
        assert call.kwargs["cu_seqlens"] is cache_req_metadata.query_start_loc
        assert call.kwargs["start_pos"] is cache_req_metadata.start_pos
        assert call.kwargs["cmp_ratio"] == compress_ratio
        assert call.kwargs["coff"] == expected_coff

    def test_split_path_uses_one_linear_and_noncontiguous_views(self):
        compressor = Compressor.__new__(Compressor)
        torch.nn.Module.__init__(compressor)
        compressor.overlap = True
        compressor.compress_ratio = 4
        compressor.rope_head_dim = 2
        compressor.norm_eps = 1e-6
        compressor.ape = torch.ones((4, 8))
        fused_weight = torch.arange(64, dtype=torch.float32).view(16, 4)
        compressor.fused_wkv_wgate = SimpleNamespace(weight=fused_weight)
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
        state_cache = torch.ones((1, 2, 1, 16))
        compress_cos = torch.ones((1, 1, 2))
        compress_sin = torch.zeros((1, 1, 2))
        slot_mapping = torch.tensor([[0, 1]], dtype=torch.int32)
        compressed_kv = torch.ones((1, 1, 4))
        mm = torch.arange(32, dtype=torch.float32).view(2, 16)
        compressor._compute_metadata = MagicMock(return_value=(compress_cos, compress_sin, slot_mapping))

        with (
            patch(
                "vllm_ascend.models.deepseek_v4.compressor.envs_ascend.VLLM_ASCEND_DSA_COMPRESSOR_SPLIT",
                True,
            ),
            patch(
                "vllm_ascend.models.deepseek_v4.compressor.F.linear",
                return_value=mm,
            ) as linear,
            patch.object(
                torch.ops._C_ascend,
                "compress_norm_rope",
                create=True,
                return_value=compressed_kv,
            ) as compress_norm_rope,
        ):
            actual_kv, actual_slot_mapping = compressor(hidden_states, state_cache, metadata)

        assert actual_kv is compressed_kv
        assert actual_slot_mapping is slot_mapping
        linear.assert_called_once_with(hidden_states, fused_weight)
        call = compress_norm_rope.call_args
        mm_kv, mm_score = call.args[:2]
        assert torch.equal(mm_kv, mm[:, :8])
        assert torch.equal(mm_score, mm[:, 8:])
        assert mm_kv.stride() == (16, 1)
        assert mm_score.stride() == (16, 1)
        assert not mm_kv.is_contiguous()
        assert not mm_score.is_contiguous()
        assert call.kwargs["cmp_ratio"] == 4
        assert call.kwargs["coff"] == 2


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
        vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=128, cache_dtype="auto"),
        )

        spec = cache.get_kv_cache_spec(vllm_config)

        assert spec.block_size == 8
        assert spec.head_size == state_dim
        assert spec.sliding_window == 64
        assert spec.page_size_padded == DSV4_BLOCK_SIZES[128][1][padding_index]
