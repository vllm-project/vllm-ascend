import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.dsa_v1 import AscendDSAImpl, AscendDSAMetadataBuilder


class TestAscendDSAMetadataBuilder(unittest.TestCase):
    def test_build_for_graph_capture_supplies_default_cache_dicts(self):
        vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=128),
            scheduler_config=SimpleNamespace(
                max_num_batched_tokens=16,
                max_num_seqs=8,
            ),
            model_config=SimpleNamespace(
                max_model_len=4096,
                hf_text_config=SimpleNamespace(qk_rope_head_dim=64),
                hf_config=SimpleNamespace(model_type="unit_test_model"),
            ),
            speculative_config=None,
        )
        kv_cache_spec = SimpleNamespace(compress_ratio=4)
        builder = AscendDSAMetadataBuilder(
            kv_cache_spec=kv_cache_spec,
            layer_names=["layer_0"],
            vllm_config=vllm_config,
            device=torch.device("cpu"),
        )
        builder.build = MagicMock(return_value=SimpleNamespace(attn_state=None))

        common_attn_metadata = MagicMock()
        attn_metadata = builder.build_for_graph_capture(
            common_attn_metadata=common_attn_metadata,
            attn_state=AscendAttentionState.DecodeOnly,
        )

        self.assertEqual(attn_metadata.attn_state, AscendAttentionState.DecodeOnly)
        builder.build.assert_called_once()
        _, kwargs = builder.build.call_args
        self.assertEqual(kwargs["common_prefix_len"], 0)
        self.assertIs(kwargs["common_attn_metadata"], common_attn_metadata)
        self.assertEqual(kwargs["prefill_ratio_to_sas_metadata"], {})
        self.assertEqual(kwargs["decode_ratio_to_sas_metadata"], {})
        self.assertEqual(kwargs["common_ratio_to_sas_metadata"], {})
        self.assertEqual(kwargs["block_size"], 128)


class TestAscendDSAUpdateGraphParams(unittest.TestCase):

    @staticmethod
    def _metadata(
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
        slot_mapping: torch.Tensor,
        sas_metadata: torch.Tensor,
    ):
        return SimpleNamespace(
            decode=SimpleNamespace(
                block_table=block_table,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                slot_mapping=slot_mapping,
                sas_metadata=sas_metadata,
            ),
        )

    def test_copy_metadata_inplace_refreshes_dict_and_list_tensors(self):
        captured = SimpleNamespace(
            decode=SimpleNamespace(
                cos={"layer": torch.zeros(2, dtype=torch.float32)},
                sin={"layer": torch.zeros(2, dtype=torch.float32)},
                seq_lens_list=[0, 0],
                nested=[torch.zeros(2, dtype=torch.int32), 0],
            ),
        )
        runtime = SimpleNamespace(
            decode=SimpleNamespace(
                cos={"layer": torch.tensor([1.0, 2.0])},
                sin={"layer": torch.tensor([3.0, 4.0])},
                seq_lens_list=[5, 6],
                nested=[torch.tensor([7, 8], dtype=torch.int32), 9],
            ),
        )

        AscendDSAImpl._copy_metadata_inplace(captured, runtime)

        self.assertTrue(torch.equal(
            captured.decode.cos["layer"],
            torch.tensor([1.0, 2.0]),
        ))
        self.assertTrue(torch.equal(
            captured.decode.sin["layer"],
            torch.tensor([3.0, 4.0]),
        ))
        self.assertEqual(captured.decode.seq_lens_list, [5, 6])
        self.assertTrue(torch.equal(
            captured.decode.nested[0],
            torch.tensor([7, 8], dtype=torch.int32),
        ))
        self.assertEqual(captured.decode.nested[1], 9)

    @patch("vllm_ascend.attention.dsa_v1.torch.npu.stream")
    @patch("vllm_ascend.attention.dsa_v1.get_draft_graph_params")
    def test_update_graph_params_copies_draft_metadata_to_capture_buffers(
        self,
        mock_get_draft_graph_params,
        mock_stream,
    ):
        mock_stream.return_value = nullcontext()
        captured = self._metadata(
            block_table=torch.zeros((2, 2), dtype=torch.int32),
            seq_lens=torch.zeros(2, dtype=torch.int32),
            query_start_loc=torch.zeros(3, dtype=torch.int32),
            slot_mapping=torch.zeros((2, 2), dtype=torch.int32),
            sas_metadata=torch.zeros(4, dtype=torch.int32),
        )
        runtime = self._metadata(
            block_table=torch.tensor([[1, 2]], dtype=torch.int32),
            seq_lens=torch.tensor([7], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            slot_mapping=torch.tensor([[3, 4]], dtype=torch.int32),
            sas_metadata=torch.tensor([9, 8, 7, 6], dtype=torch.int32),
        )
        event = MagicMock()
        mock_get_draft_graph_params.return_value = SimpleNamespace(
            attn_params={4: [(captured,)]},
            events={4: [event]},
        )

        with patch(
            "vllm_ascend.attention.dsa_v1._EXTRA_CTX",
            SimpleNamespace(is_draft_model=True),
        ):
            AscendDSAImpl.update_graph_params(
                update_stream="update_stream",
                forward_context=SimpleNamespace(attn_metadata={}),
                num_tokens=4,
                draft_attn_metadatas=[{"mtp.layer": runtime}],
            )

        self.assertTrue(
            torch.equal(captured.decode.block_table,
                        torch.tensor([[1, 2], [0, 0]], dtype=torch.int32)))
        self.assertTrue(
            torch.equal(captured.decode.seq_lens,
                        torch.tensor([7, 0], dtype=torch.int32)))
        self.assertTrue(
            torch.equal(captured.decode.query_start_loc,
                        torch.tensor([0, 1, 1], dtype=torch.int32)))
        self.assertTrue(
            torch.equal(captured.decode.slot_mapping,
                        torch.tensor([[3, 4], [0, 0]], dtype=torch.int32)))
        self.assertTrue(
            torch.equal(captured.decode.sas_metadata,
                        torch.tensor([9, 8, 7, 6], dtype=torch.int32)))
        event.record.assert_called_once_with("update_stream")

    @patch("vllm_ascend.attention.dsa_v1.torch.npu.stream")
    @patch("vllm_ascend.attention.dsa_v1.get_draft_graph_params")
    def test_update_graph_params_copies_all_metadata_in_captured_tuple(
        self,
        mock_get_draft_graph_params,
        mock_stream,
    ):
        mock_stream.return_value = nullcontext()
        captured_0 = self._metadata(
            block_table=torch.zeros((1, 2), dtype=torch.int32),
            seq_lens=torch.zeros(1, dtype=torch.int32),
            query_start_loc=torch.zeros(2, dtype=torch.int32),
            slot_mapping=torch.zeros((1, 2), dtype=torch.int32),
            sas_metadata=torch.zeros(2, dtype=torch.int32),
        )
        captured_1 = self._metadata(
            block_table=torch.zeros((1, 2), dtype=torch.int32),
            seq_lens=torch.zeros(1, dtype=torch.int32),
            query_start_loc=torch.zeros(2, dtype=torch.int32),
            slot_mapping=torch.zeros((1, 2), dtype=torch.int32),
            sas_metadata=torch.zeros(2, dtype=torch.int32),
        )
        runtime_0 = self._metadata(
            block_table=torch.tensor([[1, 2]], dtype=torch.int32),
            seq_lens=torch.tensor([3], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            slot_mapping=torch.tensor([[4, 5]], dtype=torch.int32),
            sas_metadata=torch.tensor([6, 7], dtype=torch.int32),
        )
        runtime_1 = self._metadata(
            block_table=torch.tensor([[8, 9]], dtype=torch.int32),
            seq_lens=torch.tensor([10], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            slot_mapping=torch.tensor([[11, 12]], dtype=torch.int32),
            sas_metadata=torch.tensor([13, 14], dtype=torch.int32),
        )
        event = MagicMock()
        mock_get_draft_graph_params.return_value = SimpleNamespace(
            attn_params={4: [(captured_0, captured_1)]},
            events={4: [event]},
        )

        with patch(
            "vllm_ascend.attention.dsa_v1._EXTRA_CTX",
            SimpleNamespace(is_draft_model=True),
        ):
            AscendDSAImpl.update_graph_params(
                update_stream="update_stream",
                forward_context=SimpleNamespace(attn_metadata={}),
                num_tokens=4,
                draft_attn_metadatas=[{
                    "mtp.layer.attn": runtime_0,
                    "mtp.layer.swa_cache": runtime_1,
                }],
            )

        self.assertTrue(torch.equal(captured_0.decode.seq_lens,
                                    torch.tensor([3], dtype=torch.int32)))
        self.assertTrue(torch.equal(captured_1.decode.seq_lens,
                                    torch.tensor([10], dtype=torch.int32)))
        self.assertTrue(torch.equal(captured_1.decode.sas_metadata,
                                    torch.tensor([13, 14], dtype=torch.int32)))
        event.record.assert_called_once_with("update_stream")

    @patch("vllm_ascend.attention.dsa_v1.torch.npu.stream")
    @patch("vllm_ascend.attention.dsa_v1.get_draft_graph_params")
    def test_update_graph_params_handles_recursive_metadata(
        self,
        mock_get_draft_graph_params,
        mock_stream,
    ):
        mock_stream.return_value = nullcontext()
        captured = self._metadata(
            block_table=torch.zeros((1, 2), dtype=torch.int32),
            seq_lens=torch.zeros(1, dtype=torch.int32),
            query_start_loc=torch.zeros(2, dtype=torch.int32),
            slot_mapping=torch.zeros((1, 2), dtype=torch.int32),
            sas_metadata=torch.zeros(2, dtype=torch.int32),
        )
        runtime = self._metadata(
            block_table=torch.tensor([[1, 2]], dtype=torch.int32),
            seq_lens=torch.tensor([3], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            slot_mapping=torch.tensor([[4, 5]], dtype=torch.int32),
            sas_metadata=torch.tensor([6, 7], dtype=torch.int32),
        )
        captured.decode.self_ref = captured
        runtime.decode.self_ref = runtime

        event = MagicMock()
        mock_get_draft_graph_params.return_value = SimpleNamespace(
            attn_params={4: [(captured,)]},
            events={4: [event]},
        )

        with patch(
            "vllm_ascend.attention.dsa_v1._EXTRA_CTX",
            SimpleNamespace(is_draft_model=True),
        ):
            AscendDSAImpl.update_graph_params(
                update_stream="update_stream",
                forward_context=SimpleNamespace(attn_metadata={}),
                num_tokens=4,
                draft_attn_metadatas=[{"mtp.layer": runtime}],
            )

        self.assertTrue(torch.equal(captured.decode.seq_lens,
                                    torch.tensor([3], dtype=torch.int32)))
        event.record.assert_called_once_with("update_stream")

    @patch("vllm_ascend.attention.dsa_v1.torch.npu.stream")
    @patch("vllm_ascend.attention.dsa_v1.get_draft_graph_params")
    def test_update_graph_params_records_unmatched_events(
        self,
        mock_get_draft_graph_params,
        mock_stream,
    ):
        mock_stream.return_value = nullcontext()
        captured_0 = self._metadata(
            block_table=torch.zeros((1, 2), dtype=torch.int32),
            seq_lens=torch.zeros(1, dtype=torch.int32),
            query_start_loc=torch.zeros(2, dtype=torch.int32),
            slot_mapping=torch.zeros((1, 2), dtype=torch.int32),
            sas_metadata=torch.zeros(2, dtype=torch.int32),
        )
        captured_1 = self._metadata(
            block_table=torch.zeros((1, 2), dtype=torch.int32),
            seq_lens=torch.zeros(1, dtype=torch.int32),
            query_start_loc=torch.zeros(2, dtype=torch.int32),
            slot_mapping=torch.zeros((1, 2), dtype=torch.int32),
            sas_metadata=torch.zeros(2, dtype=torch.int32),
        )
        runtime = self._metadata(
            block_table=torch.tensor([[1, 2]], dtype=torch.int32),
            seq_lens=torch.tensor([3], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            slot_mapping=torch.tensor([[4, 5]], dtype=torch.int32),
            sas_metadata=torch.tensor([6, 7], dtype=torch.int32),
        )
        matched_event = MagicMock()
        unmatched_event = MagicMock()
        mock_get_draft_graph_params.return_value = SimpleNamespace(
            attn_params={4: [(captured_0,), (captured_1,)]},
            events={4: [matched_event, unmatched_event]},
        )

        with patch(
            "vllm_ascend.attention.dsa_v1._EXTRA_CTX",
            SimpleNamespace(is_draft_model=True),
        ):
            AscendDSAImpl.update_graph_params(
                update_stream="update_stream",
                forward_context=SimpleNamespace(attn_metadata={}),
                num_tokens=4,
                draft_attn_metadatas=[{"mtp.layer": runtime}],
            )

        self.assertTrue(torch.equal(captured_0.decode.seq_lens,
                                    torch.tensor([3], dtype=torch.int32)))
        self.assertTrue(torch.equal(captured_1.decode.seq_lens,
                                    torch.tensor([0], dtype=torch.int32)))
        matched_event.record.assert_called_once_with("update_stream")
        unmatched_event.record.assert_called_once_with("update_stream")
