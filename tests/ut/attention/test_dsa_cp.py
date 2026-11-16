# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.attention.context_parallel import dsa_cp
from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSAPCPImpl, AscendDSAPCPMetadataBuilder
from vllm_ascend.worker.v2.pcp_manager import AscendPCPAttentionContext


class TestAscendDSAPCPMetadataBuilderWithoutContext:
    def test_builds_ordinary_metadata_without_pcp_context(self):
        """A replicated draft decode reaches the PCP builder without a PCP
        context; it must build the ordinary DSA metadata like PCP=1."""
        builder = AscendDSAPCPMetadataBuilder.__new__(AscendDSAPCPMetadataBuilder)
        builder._is_decode_sharded = True
        builder.use_pcp = True
        common = SimpleNamespace(
            pcp_context=None, pcp_cache_group_idx=None, slot_mapping=torch.zeros(1), num_input_tokens=1
        )
        sas_metadata: dict[Any, Any] = {}
        ordinary = object()
        with (
            patch.object(dsa_cp.dsa_v1.AscendDSAMetadataBuilder, "build", return_value=ordinary) as build,
            patch.object(builder, "_prepare_graph_pcp_context") as prepare,
        ):
            metadata = builder.build(
                0,
                common,
                num_actual_reqs=2,
                common_ratio_to_sas_metadata=sas_metadata,
                full_graph_mode=True,
            )
        assert metadata is ordinary
        prepare.assert_not_called()
        build.assert_called_once_with(
            0,
            common,
            False,
            num_actual_reqs=2,
            common_ratio_to_sas_metadata=sas_metadata,
            full_graph_mode=True,
        )


class TestAscendDSAPCPGraphPadding:
    @pytest.mark.parametrize(
        "full_graph_mode,expected_num_tokens",
        [
            (False, 6),
            (True, 8),
        ],
    )
    def test_restore_indices_match_graph_mode(self, full_graph_mode, expected_num_tokens):
        """FULL graphs replay the fixed padded batch, so restore indices cover
        every padded row; other modes restore only the actual token extent,
        keeping placeholder rows out of the cache updates entirely."""
        builder = AscendDSAPCPMetadataBuilder.__new__(AscendDSAPCPMetadataBuilder)
        builder._hidden_restore_idx_buffer = torch.full((16,), -1, dtype=torch.int64)

        global_batch = SimpleNamespace(num_tokens=6, num_tokens_after_padding=8, is_dummy=False)
        hidden_restore_idx = torch.arange(10, 16, dtype=torch.int64)
        pcp_context = AscendPCPAttentionContext(
            global_batch=global_batch,
            global_block_tables=(),
            global_slot_mappings=torch.zeros(0, dtype=torch.int64),
            hidden_restore_idx=hidden_restore_idx,
        )

        prepared = builder._prepare_graph_pcp_context(pcp_context, full_graph_mode=full_graph_mode)

        assert prepared.hidden_restore_idx.shape[0] == expected_num_tokens
        assert torch.equal(prepared.hidden_restore_idx[:6], hidden_restore_idx)
        if full_graph_mode:
            assert torch.equal(prepared.hidden_restore_idx[6:], torch.zeros(2, dtype=torch.int64))

    def test_swa_cache_update_consumes_metadata_as_built(self):
        """cos/sin/slot_mapping are built at the restored batch length (padded
        for FULL graphs, actual tokens otherwise), so the cache update passes
        them to the rotary/scatter kernels without any trimming."""
        impl = AscendDSAPCPImpl.__new__(AscendDSAPCPImpl)
        impl.nope_head_dim = 4
        impl.rope_head_dim = 2
        impl.head_dim = 6
        impl.vllm_config = SimpleNamespace()
        impl.wkv = MagicMock(side_effect=lambda x: x)
        impl.kv_norm = MagicMock(side_effect=lambda x: x)

        num_tokens = 3
        cos = torch.randn(num_tokens, 2)
        sin = torch.randn(num_tokens, 2)
        slot_mapping = torch.arange(num_tokens, dtype=torch.int64)
        req_metadata = SimpleNamespace(
            cos={"layer": cos},
            sin={"layer": sin},
            slot_mapping=slot_mapping,
        )
        rotary = MagicMock()
        scatter = MagicMock()
        with (
            patch(
                "vllm_ascend.attention.dsa_v1._require_req_metadata",
                return_value=req_metadata,
            ),
            # The op registers lazily at worker startup, so create the
            # attribute on the op namespace for this CPU-only UT.
            patch("torch.ops._C_ascend.inplace_partial_rotary_mul", rotary, create=True),
            patch(
                "vllm_ascend.attention.context_parallel.dsa_cp.get_dsa_attn_kv_plan",
                return_value=SimpleNamespace(dsa_kv_compress_scatter=scatter),
            ),
        ):
            impl._update_global_swa_cache(
                "layer",
                torch.randn(num_tokens, 6),
                torch.randn(1),
                MagicMock(),
            )

        rotary.assert_called_once()
        assert rotary.call_args.args[1] is cos
        assert rotary.call_args.args[2] is sin
        scatter.assert_called_once()
        assert scatter.call_args.args[2] is slot_mapping
