from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.config import CUDAGraphMode

from tests.ut.base import TestBase
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackendImpl,
    AscendAttentionState,
    AscendC8MXFPAttentionBackendImpl,
    C8MXFPGraphAttentionParams,
)


class TestC8MXFPFullGraph(TestBase):
    def test_piecewise_capture_keeps_regular_mxfp_path(self):
        impl = object.__new__(AscendC8MXFPAttentionBackendImpl)
        impl.enable_hamming_sparse = False

        query = torch.zeros(2, 2, 4)
        key = torch.zeros_like(query)
        value = torch.zeros_like(query)
        output = torch.zeros_like(query)
        query_scale = torch.zeros(2, 2, 1, 2)
        key_scale = torch.zeros_like(query_scale)
        kv_cache = tuple(torch.empty(1) for _ in range(4))
        metadata = SimpleNamespace(
            num_actual_tokens=2,
            attn_state=AscendAttentionState.DecodeOnly,
        )
        layer = SimpleNamespace(
            v_cache_scale_float_reciprocal=torch.ones(1),
            v_cache_scale=torch.ones(1),
        )

        with (
            patch(
                "vllm_ascend.attention.attention_v1._EXTRA_CTX",
                SimpleNamespace(capturing=True),
            ),
            patch(
                "vllm_ascend.attention.attention_v1.get_forward_context",
                return_value=SimpleNamespace(
                    cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE
                ),
            ),
            patch(
                "torch_npu.npu_dynamic_mx_quant",
                side_effect=((query, query_scale), (key, key_scale)),
            ),
            patch("torch_npu.npu_quantize", return_value=value),
            patch.object(impl, "reshape_and_cache"),
            patch.object(impl, "_transpose_kv_cache", return_value=kv_cache),
            patch.object(
                impl,
                "_forward_mxfp8_attention",
                return_value=output,
            ) as mock_regular_attention,
            patch.object(impl, "_full_graph_mxfp8_decode") as mock_full_attention,
        ):
            result = impl.forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                metadata,
                output,
            )

        self.assertIs(result, output)
        mock_regular_attention.assert_called_once()
        mock_full_attention.assert_not_called()

    def test_capture_registers_mxfp_fia_v2_task_params(self):
        impl = object.__new__(AscendC8MXFPAttentionBackendImpl)
        impl.num_heads = 8
        impl.num_kv_heads = 2
        impl.scale = 0.125

        query = torch.zeros(2, 8, 64)
        query_scale = torch.zeros(2, 8, 1, 2)
        key = torch.zeros(2, 2, 4, 64)
        value = torch.zeros_like(key)
        key_scale = torch.zeros(2, 2, 4, 1, 2)
        value_scale = torch.zeros(2, 2, 1, 64, 2)
        output = torch.zeros_like(query, dtype=torch.bfloat16)
        metadata = SimpleNamespace(
            num_decode_tokens=2,
            num_decodes=2,
            block_tables=torch.tensor([[0], [1]], dtype=torch.int32),
            actual_seq_lengths_q=[1, 2],
            seq_lens_list=[33, 45],
        )
        graph_params = SimpleNamespace(
            workspaces={2: None},
            events={2: []},
            attn_params={2: []},
            handles={2: []},
        )
        workspace = torch.empty(1)
        event = MagicMock()
        stream = MagicMock()
        handle = MagicMock()

        with (
            patch(
                "vllm_ascend.attention.attention_v1._EXTRA_CTX",
                SimpleNamespace(is_draft_model=False),
            ),
            patch(
                "vllm_ascend.attention.attention_v1.get_graph_params",
                return_value=graph_params,
            ),
            patch(
                "vllm_ascend.attention.attention_v1.update_graph_params_workspaces"
            ) as mock_update_workspace,
            patch(
                "vllm_ascend.attention.attention_v1.weak_ref_tensors",
                side_effect=lambda tensor: tensor,
            ),
            patch(
                "torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace",
                return_value=workspace,
            ),
            patch("torch_npu.npu.current_stream", return_value=stream),
            patch("torch.npu.ExternalEvent", return_value=event),
            patch("torch.npu.graph_task_group_begin") as mock_group_begin,
            patch("torch.npu.graph_task_group_end", return_value=handle),
            patch("torch_npu.npu_fused_infer_attention_score_v2.out") as mock_fia_out,
        ):
            result = impl._full_graph_mxfp8_decode(
                query,
                query_scale,
                (key, value, key_scale, value_scale),
                metadata,
                output,
            )

        self.assertIs(result, output)
        self.assertEqual(len(graph_params.attn_params[2]), 1)
        self.assertIsInstance(
            graph_params.attn_params[2][0], C8MXFPGraphAttentionParams
        )
        self.assertEqual(graph_params.attn_params[2][0].actual_seq_qlen, [1, 2])
        self.assertEqual(
            graph_params.attn_params[2][0].softmax_lse.dtype,
            torch.float32,
        )
        self.assertIs(graph_params.attn_params[2][0].query_scale, query_scale)
        self.assertEqual(
            graph_params.attn_params[2][0].query_scale.shape,
            (2, 8, 1, 2),
        )
        self.assertEqual(graph_params.events[2], [event])
        self.assertEqual(graph_params.handles[2], [handle])
        mock_update_workspace.assert_called_once_with(2, workspace)
        mock_group_begin.assert_called_once_with(stream)
        mock_fia_out.assert_called_once()
        self.assertEqual(mock_fia_out.call_args.kwargs["actual_seq_kvlen"], [33, 45])
        self.assertIs(
            mock_fia_out.call_args.kwargs["dequant_scale_query"],
            graph_params.attn_params[2][0].query_scale,
        )

    def test_replay_updates_mxfp_sequence_lengths(self):
        query = torch.zeros(2, 8, 64)
        query_scale = torch.zeros(2, 2, 4, 1, 2)
        key = torch.zeros(2, 2, 4, 64)
        value = torch.zeros_like(key)
        key_scale = torch.zeros(2, 2, 4, 1, 2)
        value_scale = torch.zeros(2, 2, 1, 64, 2)
        output = torch.zeros_like(query)
        softmax_lse = torch.empty(1)
        param = C8MXFPGraphAttentionParams(
            query=query,
            key=key,
            value=value,
            query_scale=query_scale,
            key_scale=key_scale,
            value_scale=value_scale,
            block_size=4,
            num_kv_heads=2,
            num_heads=8,
            scale=0.125,
            actual_seq_qlen=[1, 2],
            attn_output=output,
            softmax_lse=softmax_lse,
        )
        event = MagicMock()
        handle = MagicMock()
        workspace = torch.empty(1)
        graph_params = SimpleNamespace(
            attn_params={2: [param]},
            handles={2: [handle]},
            events={2: [event]},
            workspaces={2: workspace},
        )
        latest_block_tables = torch.tensor([[1], [0]], dtype=torch.int32)
        metadata = SimpleNamespace(
            block_tables=latest_block_tables,
            actual_seq_lengths_q=[1],
            seq_lens_list=[34],
        )
        forward_context = SimpleNamespace(attn_metadata={"layers.0.attn": metadata})
        update_stream = MagicMock()

        with (
            patch(
                "vllm_ascend.attention.attention_v1._EXTRA_CTX",
                SimpleNamespace(is_draft_model=False, sinks=False),
            ),
            patch(
                "vllm_ascend.attention.attention_v1.get_graph_params",
                return_value=graph_params,
            ),
            patch(
                "vllm_ascend.attention.attention_v1.using_paged_attention",
                return_value=False,
            ),
            patch("torch.npu.stream"),
            patch("torch.npu.graph_task_update_begin") as mock_update_begin,
            patch("torch.npu.graph_task_update_end") as mock_update_end,
            patch("torch_npu.npu_fused_infer_attention_score_v2.out") as mock_fia_out,
        ):
            AscendAttentionBackendImpl.update_graph_params(
                update_stream,
                forward_context,
                2,
                MagicMock(),
            )

        mock_update_begin.assert_called_once_with(update_stream, handle)
        mock_update_end.assert_called_once_with(update_stream)
        event.record.assert_called_once_with(update_stream)
        mock_fia_out.assert_called_once()
        kwargs = mock_fia_out.call_args.kwargs
        self.assertEqual(kwargs["actual_seq_qlen"], [1, 2])
        self.assertEqual(kwargs["actual_seq_kvlen"], [34, 0])
        self.assertIs(kwargs["block_table"], latest_block_tables)
        self.assertIs(kwargs["dequant_scale_query"], query_scale)
        self.assertIs(kwargs["dequant_scale_key"], key_scale)
        self.assertIs(kwargs["dequant_scale_value"], value_scale)
        self.assertIs(kwargs["workspace"], workspace)
