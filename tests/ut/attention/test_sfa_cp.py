#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
import sys
from unittest.mock import MagicMock, patch

import torch
from vllm.distributed.parallel_state import GroupCoordinator

from tests.ut.attention.utils import patch_distributed_groups
from tests.ut.base import TestBase
from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState

if "torch_npu._inductor" not in sys.modules:
    sys.modules["torch_npu._inductor"] = MagicMock()

from vllm_ascend.attention.context_parallel.common_cp import AscendPCPMetadata
from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFACPImpl, AscendSFACPMetadataBuilder
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl, AscendSFAMetadata


def _make_indexer_mock():
    indexer = MagicMock()
    indexer.n_head = 64
    indexer.head_dim = 128
    indexer.wq_b = MagicMock()
    indexer.wk_weights_proj = MagicMock()
    indexer.k_norm = MagicMock()
    return indexer


def _make_impl_kwargs(extra=None):
    kv_a_layernorm = MagicMock()
    kv_a_layernorm.weight = torch.randn(96)
    kv_a_layernorm.variance_epsilon = 1e-6
    kwargs = {
        "kv_lora_rank": 32,
        "qk_nope_head_dim": 64,
        "qk_rope_head_dim": 32,
        "qk_head_dim": 96,
        "v_head_dim": 128,
        "q_lora_rank": 64,
        "q_proj": MagicMock(),
        "q_b_proj": MagicMock(),
        "kv_b_proj": MagicMock(),
        "o_proj": MagicMock(),
        "kv_a_proj_with_mqa": MagicMock(),
        "fused_qkv_a_proj": MagicMock(),
        "kv_a_layernorm": kv_a_layernorm,
        "q_a_layernorm": MagicMock(),
        "rotary_emb": MagicMock(),
        "indexer": _make_indexer_mock(),
        "layer_name": "layer_0",
    }
    if extra:
        kwargs.update(extra)
    return kwargs


class TestAscendSFACPMetadataBuilder(TestBase):
    """Tests for AscendSFACPMetadataBuilder."""

    @patch("vllm.distributed.parallel_state._TP", new_callable=lambda: MagicMock(spec=GroupCoordinator))
    def setUp(self, mock_tp):
        mock_tp.world_size = 2
        mock_tp.rank_in_group = MagicMock()
        mock_tp.device_group = MagicMock()

        self.mock_cfg = MagicMock()
        self.mock_cfg.parallel_config = MagicMock()
        self.mock_cfg.parallel_config.tensor_parallel_size = 1
        self.mock_cfg.parallel_config.prefill_context_parallel_size = 1
        self.mock_cfg.parallel_config.decode_context_parallel_size = 1

        self.mock_cfg.compilation_config = MagicMock()
        self.mock_cfg.compilation_config.pass_config = MagicMock()
        self.mock_cfg.compilation_config.pass_config.enable_sp = False

        self.mock_cfg.speculative_config.num_speculative_tokens = 0

        self.patcher = patch("vllm.config.get_current_vllm_config", return_value=self.mock_cfg)
        self.patcher.start()

        # Mock parent class __init__ to avoid complex initialization,
        # but still set the essential attributes that child class needs.
        def mock_parent_init(
            self, kv_cache_spec, layer_names, vllm_config, device, metadata_cls, supports_dcp_with_varlen
        ):
            self.metadata_cls = metadata_cls
            self.kv_cache_spec = kv_cache_spec
            self.model_config = vllm_config.model_config
            self.vllm_config = vllm_config
            self.device = device
            self.chunked_prefill_workspace_size = 128 * 1024
            self.chunked_prefill_workspace = torch.empty(
                (self.chunked_prefill_workspace_size, vllm_config.model_config.get_head_size()),
                dtype=vllm_config.model_config.dtype,
                device=device,
            )

        self.parent_init_patcher = patch(
            "vllm.model_executor.layers.attention.mla_attention.MLACommonMetadataBuilder.__init__", mock_parent_init
        )
        self.parent_init_patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.parent_init_patcher.stop()

    def _make_vllm_config(self):
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        vllm_config.model_config.hf_text_config = MagicMock(qk_rope_head_dim=64)
        vllm_config.model_config.hf_config.model_type = "deepseek_v3"
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 0
        vllm_config.speculative_config = speculative_config
        vllm_config.scheduler_config.max_num_seqs = 16
        vllm_config.scheduler_config.max_num_batched_tokens = 256
        vllm_config.parallel_config = MagicMock()
        vllm_config.parallel_config.cp_kv_cache_interleave_size = 1
        vllm_config.parallel_config.tensor_parallel_size = 1
        vllm_config.parallel_config.prefill_context_parallel_size = 2
        vllm_config.parallel_config.decode_context_parallel_size = 2
        vllm_config.kv_transfer_config = None
        return vllm_config

    def _build_builder(self, pcp_size=2, dcp_size=2):
        kv_cache_spec = MagicMock()
        layer_names = ["layer1", "layer2"]
        vllm_config = self._make_vllm_config()
        device = torch.device("cpu")
        builder = AscendSFACPMetadataBuilder(
            kv_cache_spec=kv_cache_spec,
            layer_names=layer_names,
            vllm_config=vllm_config,
            device=device,
        )
        # The parent mock above sets minimal attributes, set the rest by ourselves
        builder.block_size = 16
        builder.speculative_config = vllm_config.speculative_config
        builder.decode_threshold = 1
        builder.reorder_batch_threshold = 1
        return builder

    @patch("vllm_ascend.attention.context_parallel.sfa_cp.enabling_mlapo")
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_init_default(self, mock_enabling_mlapo):
        mock_enabling_mlapo.return_value = False
        builder = self._build_builder(pcp_size=2, dcp_size=2)
        self.assertEqual(builder.pcp_size, 2)
        self.assertEqual(builder.pcp_rank, 0)
        self.assertEqual(builder.dcp_size, 2)
        self.assertEqual(builder.dcp_rank, 0)
        self.assertFalse(builder.enable_mlapo)
        self.assertEqual(builder.cp_local_block_size, 1)
        # cp_virtual_block_size = 1 * 2 * 2 = 4
        self.assertEqual(builder.cp_virtual_block_size, 4)
        # block_size = lcm(16, 4) = 16
        self.assertEqual(builder.block_size, 16)
        self.assertIsNotNone(builder.slot_mapping_buf)
        self.assertEqual(builder.block_arange_buffer.shape[0], 4)

    @patch("vllm_ascend.attention.context_parallel.sfa_cp.enabling_mlapo")
    @patch_distributed_groups(dcp_size=1, pcp_size=1, needs_mocks=False)
    def test_init_no_cp(self, mock_enabling_mlapo):
        mock_enabling_mlapo.return_value = False
        builder = self._build_builder(pcp_size=1, dcp_size=1)
        self.assertEqual(builder.pcp_size, 1)
        self.assertEqual(builder.pcp_rank, 0)
        self.assertEqual(builder.dcp_size, 1)
        self.assertEqual(builder.dcp_rank, 0)
        self.assertIsNone(builder.pcp_group)
        self.assertIsNone(builder.dcp_group)

    @patch("vllm_ascend.attention.context_parallel.sfa_cp.enabling_mlapo")
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_init_with_mlapo_enabled(self, mock_enabling_mlapo):
        mock_enabling_mlapo.return_value = True
        builder = self._build_builder(pcp_size=2, dcp_size=2)
        self.assertTrue(builder.enable_mlapo)

    @patch("vllm_ascend.attention.context_parallel.sfa_cp.enabling_mlapo", return_value=False)
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_compact_varlen_decode_slot_mapping_basic(self, mock_enabling_mlapo):
        builder = self._build_builder()
        # pcp_size=2, total tokens=6 (3 per req with pcp gather)
        # decode_query_lens: [2, 1] (2 + 1 = 3 total valid)
        # slot_mapping with pcp expansion: each decode token has pcp_size=2 entries
        # Layout: [t0_p0, t0_p1, t1_p0, t1_p1, t2_p0, t2_p1] = 6 tokens
        # req0 spans 2 tokens => valid_in: [0, 2], req1 spans 1 token => valid_in: [4]
        decode_slot_mapping = torch.tensor([10, 20, 30, 40, 50, 60], dtype=torch.int32)
        decode_query_lens = torch.tensor([2, 1], dtype=torch.int64)
        builder._compact_varlen_decode_slot_mapping(decode_slot_mapping, decode_query_lens)
        # With pcp_size=2:
        # req_spans = [4, 2], req_starts = [0, 4]
        # token_offsets after rebase: [0, 1, 0]
        # valid_in_idx = [0, 2, 4] => slots [10, 30, 50]
        # valid_out_idx = [0, 1, 4]
        # Final: pos 0=10, 1=30, 4=50, others=-1
        self.assertEqual(decode_slot_mapping[0].item(), 10)
        self.assertEqual(decode_slot_mapping[1].item(), 30)
        self.assertEqual(decode_slot_mapping[2].item(), -1)
        self.assertEqual(decode_slot_mapping[3].item(), -1)
        self.assertEqual(decode_slot_mapping[4].item(), 50)
        self.assertEqual(decode_slot_mapping[5].item(), -1)

    @patch("vllm_ascend.attention.context_parallel.sfa_cp.enabling_mlapo", return_value=False)
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_compact_varlen_decode_slot_mapping_zero_tokens(self, mock_enabling_mlapo):
        builder = self._build_builder()
        decode_slot_mapping = torch.tensor([10, 20, 30], dtype=torch.int32)
        decode_query_lens = torch.tensor([0, 0], dtype=torch.int64)
        # Should return early without modification
        original = decode_slot_mapping.clone()
        builder._compact_varlen_decode_slot_mapping(decode_slot_mapping, decode_query_lens)
        self.assertTrue(torch.equal(decode_slot_mapping, original))

    @patch("vllm_ascend.attention.context_parallel.sfa_cp.enabling_mlapo", return_value=False)
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_build_prefill_compact_block_metadata(self, mock_enabling_mlapo):
        builder = self._build_builder()
        # Make a block_table with 3 reqs, 1 decode, 2 prefills, 4 blocks per req
        block_table = torch.tensor(
            [
                [0, 0, 0, 0],
                [1, 2, 3, 4],
                [5, 6, 1, 2],
            ],
            dtype=torch.int32,
        )
        valid_block_ids, block_table_cp = builder.build_prefill_compact_block_metadata(block_table, num_decodes=1)
        # prefill block_table covers reqs 1 and 2: blocks 1, 2, 3, 4, 5, 6, 1, 2 (8 entries)
        # unique: [1, 2, 3, 4, 5, 6]
        self.assertEqual(valid_block_ids.numel(), 6)
        # block_table_cp shape should be (num_prefill_reqs, num_blocks_per_req * pcp*dcp)
        self.assertEqual(block_table_cp.shape[0], 2)
        # 4 blocks per req * (pcp_size * dcp_size) = 4 * 4 = 16
        self.assertEqual(block_table_cp.shape[1], 16)

    @patch("vllm_ascend.attention.context_parallel.sfa_cp.enabling_mlapo", return_value=False)
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_build_cp_metadata(self, mock_enabling_mlapo):
        builder = self._build_builder()
        block_arange = builder.block_arange_buffer
        seq_lens = torch.tensor([8, 16], dtype=torch.int32)

        common_attn_metadata = MagicMock()
        long_seq_metadata = MagicMock()
        long_seq_metadata.q_head_idx_tensor = torch.tensor([0, 1])
        long_seq_metadata.q_tail_idx_tensor = torch.tensor([2, 3])
        long_seq_metadata.q_full_idx = torch.tensor([0, 1, 2, 3])
        long_seq_metadata.pcp_allgather_restore_idx = torch.tensor([0, 1, 2, 3])
        common_attn_metadata.prefill_context_parallel_metadata = long_seq_metadata
        common_attn_metadata.num_computed_tokens_cpu = torch.tensor([0, 0], dtype=torch.int32)

        result = builder.build_cp_metadata(block_arange, seq_lens, common_attn_metadata)
        self.assertIsInstance(result, AscendPCPMetadata)
        self.assertIs(result.q_head_idx, long_seq_metadata.q_head_idx_tensor)
        self.assertIs(result.q_tail_idx, long_seq_metadata.q_tail_idx_tensor)
        self.assertIsNotNone(result.head_attn_nomask_seqlens)
        self.assertIsNotNone(result.tail_attn_nomask_seqlens)

    @patch("vllm_ascend.attention.context_parallel.sfa_cp.enabling_mlapo", return_value=False)
    @patch("vllm_ascend.attention.context_parallel.sfa_cp.split_decodes_and_prefills")
    @patch_distributed_groups(dcp_size=1, pcp_size=1, needs_mocks=False)
    def test_build_decode_only_no_pcp(self, mock_split, mock_mlapo):
        # Build path with no prefills, no pcp, simplest case
        builder = self._build_builder(pcp_size=1, dcp_size=1)
        mock_split.return_value = (2, 0, 2, 0)  # decodes, prefills, decode_tokens, prefill_tokens

        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 2
        common_attn_metadata.num_actual_tokens = 2
        common_attn_metadata.num_input_tokens = 2

        long_seq_metadata = MagicMock()
        long_seq_metadata.q_head_idx_tensor = torch.tensor([0])
        long_seq_metadata.q_tail_idx_tensor = torch.tensor([1])
        long_seq_metadata.q_full_idx = torch.tensor([0, 1])
        long_seq_metadata.pcp_allgather_restore_idx = torch.tensor([0, 1])
        long_seq_metadata.num_actual_tokens_pcp_padded = 2
        common_attn_metadata.prefill_context_parallel_metadata = long_seq_metadata
        common_attn_metadata.num_computed_tokens_cpu = torch.tensor([0, 0], dtype=torch.int32)
        common_attn_metadata.slot_mapping = torch.arange(8, dtype=torch.int32)

        # Mock super().build()
        fake_metadata = AscendSFAMetadata(
            num_actual_tokens=2,
            slot_mapping=torch.zeros(2, dtype=torch.int32),
            seq_lens=torch.tensor([4, 4], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([4, 4], dtype=torch.int32),
            cum_query_lens=torch.tensor([1, 2], dtype=torch.int32),
            block_table=torch.zeros((2, 4), dtype=torch.int32),
            sin=torch.randn(2, 32),
            cos=torch.randn(2, 32),
            num_input_tokens=2,
            attn_state=AscendAttentionState.DecodeOnly,
        )
        with patch.object(
            AscendSFACPMetadataBuilder.__bases__[0],
            "build",
            return_value=fake_metadata,
        ):
            result = builder.build(common_prefix_len=0, common_attn_metadata=common_attn_metadata)
        self.assertIs(result, fake_metadata)
        self.assertEqual(result.num_decodes, 2)
        self.assertEqual(result.num_decode_tokens, 2)
        self.assertEqual(result.num_prefills, 0)
        # In pcp_size=1 path, sfa_cp_metadata should be set but block_table_cp/valid_block_ids None
        self.assertIsNotNone(result.sfa_cp_metadata)
        self.assertIsNone(result.sfa_cp_metadata.valid_block_ids)
        self.assertIsNone(result.sfa_cp_metadata.block_table_cp)

    @patch("vllm_ascend.attention.context_parallel.sfa_cp.enabling_mlapo", return_value=False)
    @patch("vllm_ascend.attention.context_parallel.sfa_cp.split_decodes_and_prefills")
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_build_with_prefills_and_decodes(self, mock_split, mock_mlapo):
        builder = self._build_builder(pcp_size=2, dcp_size=2)
        # 1 decode + 2 prefills, 1 decode token, 6 prefill tokens
        mock_split.return_value = (1, 2, 1, 6)

        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 3
        common_attn_metadata.num_actual_tokens = 7
        common_attn_metadata.num_input_tokens = 7

        long_seq_metadata = MagicMock()
        long_seq_metadata.q_head_idx_tensor = torch.tensor([0, 1])
        long_seq_metadata.q_tail_idx_tensor = torch.tensor([2, 3])
        long_seq_metadata.q_full_idx = torch.tensor([0, 1, 2, 3])
        long_seq_metadata.pcp_allgather_restore_idx = torch.tensor([0, 1, 2, 3])
        long_seq_metadata.num_actual_tokens_pcp_padded = 14
        long_seq_metadata.query_lens_pcp_full_cpu = torch.tensor([1, 3, 3], dtype=torch.int32)
        common_attn_metadata.prefill_context_parallel_metadata = long_seq_metadata
        common_attn_metadata.num_computed_tokens_cpu = torch.tensor([0, 0, 0], dtype=torch.int32)
        common_attn_metadata.slot_mapping = torch.arange(64, dtype=torch.int32)

        block_table = torch.tensor(
            [
                [0, 0, 0, 0],
                [1, 2, 3, 4],
                [5, 6, 7, 8],
            ],
            dtype=torch.int32,
        )
        fake_metadata = AscendSFAMetadata(
            num_actual_tokens=7,
            slot_mapping=torch.zeros(7, dtype=torch.int32),
            seq_lens=torch.tensor([4, 8, 8], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([4, 8, 8], dtype=torch.int32),
            cum_query_lens=torch.tensor([1, 4, 7], dtype=torch.int32),
            block_table=block_table,
            sin=torch.randn(7, 32),
            cos=torch.randn(7, 32),
            num_input_tokens=7,
            attn_state=AscendAttentionState.ChunkedPrefill,
        )
        with patch.object(
            AscendSFACPMetadataBuilder.__bases__[0],
            "build",
            return_value=fake_metadata,
        ):
            result = builder.build(common_prefix_len=0, common_attn_metadata=common_attn_metadata)
        self.assertEqual(result.num_decodes, 1)
        self.assertEqual(result.num_prefills, 2)
        self.assertEqual(result.num_decode_tokens, 1)
        self.assertIsNotNone(result.sfa_cp_metadata)
        self.assertIsNotNone(result.sfa_cp_metadata.valid_block_ids)
        self.assertIsNotNone(result.sfa_cp_metadata.block_table_cp)
        self.assertIsNotNone(result.sfa_cp_metadata.prefill_q_cum_seqlens)

    @patch("vllm_ascend.attention.context_parallel.sfa_cp.enabling_mlapo", return_value=False)
    @patch("vllm_ascend.attention.context_parallel.sfa_cp.split_decodes_and_prefills")
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_build_prefills_only(self, mock_split, mock_mlapo):
        # Verifies prefill_q_cum_seqlens equals actual_seq_lengths_query when no decodes
        builder = self._build_builder(pcp_size=2, dcp_size=2)
        mock_split.return_value = (0, 2, 0, 6)

        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 2
        common_attn_metadata.num_actual_tokens = 6
        common_attn_metadata.num_input_tokens = 6

        long_seq_metadata = MagicMock()
        long_seq_metadata.q_head_idx_tensor = torch.tensor([0, 1])
        long_seq_metadata.q_tail_idx_tensor = torch.tensor([2, 3])
        long_seq_metadata.q_full_idx = torch.tensor([0, 1, 2, 3])
        long_seq_metadata.pcp_allgather_restore_idx = torch.tensor([0, 1, 2, 3])
        long_seq_metadata.num_actual_tokens_pcp_padded = 12
        long_seq_metadata.query_lens_pcp_full_cpu = torch.tensor([3, 3], dtype=torch.int32)
        common_attn_metadata.prefill_context_parallel_metadata = long_seq_metadata
        common_attn_metadata.num_computed_tokens_cpu = torch.tensor([0, 0], dtype=torch.int32)
        common_attn_metadata.slot_mapping = torch.arange(64, dtype=torch.int32)

        block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        fake_metadata = AscendSFAMetadata(
            num_actual_tokens=6,
            slot_mapping=torch.zeros(6, dtype=torch.int32),
            seq_lens=torch.tensor([8, 8], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([8, 8], dtype=torch.int32),
            cum_query_lens=torch.tensor([3, 6], dtype=torch.int32),
            block_table=block_table,
            sin=torch.randn(6, 32),
            cos=torch.randn(6, 32),
            num_input_tokens=6,
            attn_state=AscendAttentionState.ChunkedPrefill,
        )
        with patch.object(
            AscendSFACPMetadataBuilder.__bases__[0],
            "build",
            return_value=fake_metadata,
        ):
            result = builder.build(common_prefix_len=0, common_attn_metadata=common_attn_metadata)
        self.assertEqual(result.num_decodes, 0)
        self.assertEqual(result.num_prefills, 2)
        self.assertIsNotNone(result.sfa_cp_metadata.prefill_q_cum_seqlens)

    @patch("vllm_ascend.attention.context_parallel.sfa_cp.enabling_mlapo", return_value=True)
    @patch("vllm_ascend.attention.context_parallel.sfa_cp.split_decodes_and_prefills")
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_build_with_mlapo_enabled(self, mock_split, mock_mlapo):
        # When mlapo is on: slot_mapping is compacted by pcp_size
        builder = self._build_builder(pcp_size=2, dcp_size=2)
        # 2 decodes, 0 prefills
        mock_split.return_value = (2, 0, 2, 0)

        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 2
        common_attn_metadata.num_actual_tokens = 2
        common_attn_metadata.num_input_tokens = 2

        long_seq_metadata = MagicMock()
        long_seq_metadata.q_head_idx_tensor = torch.tensor([0])
        long_seq_metadata.q_tail_idx_tensor = torch.tensor([1])
        long_seq_metadata.q_full_idx = torch.tensor([0, 1])
        long_seq_metadata.pcp_allgather_restore_idx = torch.tensor([0, 1])
        long_seq_metadata.num_actual_tokens_pcp_padded = 4
        common_attn_metadata.prefill_context_parallel_metadata = long_seq_metadata
        common_attn_metadata.num_computed_tokens_cpu = torch.tensor([0, 0], dtype=torch.int32)
        common_attn_metadata.slot_mapping = torch.tensor(
            [10, 20, 30, 40, 50, 60, 70, 80], dtype=torch.int32
        )  # 8 tokens, padded

        fake_metadata = AscendSFAMetadata(
            num_actual_tokens=2,
            slot_mapping=torch.zeros(2, dtype=torch.int32),
            seq_lens=torch.tensor([4, 4], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([4, 4], dtype=torch.int32),
            cum_query_lens=torch.tensor([1, 2], dtype=torch.int32),
            block_table=torch.zeros((2, 4), dtype=torch.int32),
            sin=torch.randn(2, 32),
            cos=torch.randn(2, 32),
            num_input_tokens=2,
            attn_state=AscendAttentionState.DecodeOnly,
        )
        with patch.object(
            AscendSFACPMetadataBuilder.__bases__[0],
            "build",
            return_value=fake_metadata,
        ):
            result = builder.build(common_prefix_len=0, common_attn_metadata=common_attn_metadata)
        # The first num_decode_tokens slot mappings are taken at every pcp_size stride
        self.assertEqual(result.num_decodes, 2)
        self.assertEqual(result.slot_mapping.shape[0], 4)

    @patch("vllm_ascend.attention.context_parallel.sfa_cp.enabling_mlapo", return_value=False)
    @patch("vllm_ascend.attention.context_parallel.sfa_cp.split_decodes_and_prefills")
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_build_with_speculative_and_pcp(self, mock_split, mock_mlapo):
        # Tests speculative_config branch (compact_varlen_decode_slot_mapping)
        builder = self._build_builder(pcp_size=2, dcp_size=2)
        builder.speculative_config = MagicMock()  # Truthy speculative_config
        # 2 decodes, 0 prefills, num_decode_tokens = 3 (varlen)
        mock_split.return_value = (2, 0, 3, 0)

        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 2
        common_attn_metadata.num_actual_tokens = 3
        common_attn_metadata.num_input_tokens = 3

        long_seq_metadata = MagicMock()
        long_seq_metadata.q_head_idx_tensor = torch.tensor([0])
        long_seq_metadata.q_tail_idx_tensor = torch.tensor([1])
        long_seq_metadata.q_full_idx = torch.tensor([0, 1])
        long_seq_metadata.pcp_allgather_restore_idx = torch.tensor([0, 1])
        long_seq_metadata.num_actual_tokens_pcp_padded = 6
        long_seq_metadata.query_lens_pcp_full_cpu = torch.tensor([2, 1], dtype=torch.int64)
        common_attn_metadata.prefill_context_parallel_metadata = long_seq_metadata
        common_attn_metadata.num_computed_tokens_cpu = torch.tensor([0, 0], dtype=torch.int32)
        common_attn_metadata.slot_mapping = torch.arange(20, dtype=torch.int32)

        fake_metadata = AscendSFAMetadata(
            num_actual_tokens=3,
            slot_mapping=torch.zeros(3, dtype=torch.int32),
            seq_lens=torch.tensor([4, 4], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([4, 4], dtype=torch.int32),
            cum_query_lens=torch.tensor([2, 3], dtype=torch.int32),
            block_table=torch.zeros((2, 4), dtype=torch.int32),
            sin=torch.randn(3, 32),
            cos=torch.randn(3, 32),
            num_input_tokens=3,
            attn_state=AscendAttentionState.SpecDecoding,
        )
        with patch.object(
            AscendSFACPMetadataBuilder.__bases__[0],
            "build",
            return_value=fake_metadata,
        ):
            result = builder.build(common_prefix_len=0, common_attn_metadata=common_attn_metadata)
        self.assertEqual(result.num_decodes, 2)
        self.assertIsNotNone(result.slot_mapping)


class TestAscendSFACPImpl(TestBase):
    """Tests for AscendSFACPImpl."""

    @patch("vllm_ascend.attention.context_parallel.sfa_cp.enabling_mlapo", return_value=False)
    @patch("vllm.distributed.parallel_state._TP", new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp_with_o_proj_tp", return_value=False)
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp_with_layer_shard", return_value=False)
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp", return_value=False)
    @patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config")
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def setUp(
        self,
        mock_get_current_vllm_config,
        _mock_enable_dsa_cp,
        _mock_enable_dsa_cp_with_layer_shard,
        _mock_enable_dsa_cp_with_o_proj_tp,
        mock_tp,
        _mock_enabling_mlapo,
    ):
        mock_tp.world_size = 2
        mock_tp.rank_in_group = MagicMock()
        mock_tp.device_group = MagicMock()

        vllm_config = MagicMock()
        speculative_config = MagicMock()
        model_config = MagicMock()
        parallel_config = MagicMock()
        parallel_config.prefill_context_parallel_size = 2
        parallel_config.decode_context_parallel_size = 2
        parallel_config.tensor_parallel_size = 2
        speculative_config.num_speculative_tokens = 0
        vllm_config.speculative_config = speculative_config
        model_config.dtype = torch.float16
        model_config.hf_config.model_type = "deepseek_v3"
        vllm_config.model_config = model_config
        vllm_config.kv_transfer_config = None
        vllm_config.additional_config = {"refresh": True}
        vllm_config.parallel_config = parallel_config
        mock_get_current_vllm_config.return_value = vllm_config
        init_ascend_config(vllm_config)

        self.kwargs = _make_impl_kwargs()
        self.impl = AscendSFACPImpl(
            num_heads=256,
            head_size=1024,
            scale=0.1,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type=None,
            kv_sharing_target_layer_name=None,
            **self.kwargs,
        )
        AscendSFAImpl.o_proj_full_pool = None
        AscendSFAImpl.q_hadamard = None
        AscendSFAImpl.k_hadamard = None

    def test_init_default(self):
        self.assertEqual(self.impl.pcp_size, 2)
        self.assertEqual(self.impl.dcp_size, 2)
        self.assertEqual(self.impl.pcp_rank, 0)
        self.assertEqual(self.impl.dcp_rank, 0)
        self.assertIsNotNone(self.impl.pcp_group)
        self.assertIsNotNone(self.impl.dcp_group)
        self.assertFalse(self.impl.enable_mlapo)

    @patch("vllm_ascend.attention.context_parallel.sfa_cp.enabling_mlapo", return_value=False)
    @patch("vllm.distributed.parallel_state._TP", new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp_with_o_proj_tp", return_value=False)
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp_with_layer_shard", return_value=False)
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp", return_value=False)
    @patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config")
    @patch_distributed_groups(dcp_size=1, pcp_size=1, needs_mocks=False)
    def test_init_no_cp(
        self,
        mock_get_current_vllm_config,
        _e_dsa,
        _e_layer_shard,
        _e_o_proj_tp,
        mock_tp,
        _e_mlapo,
    ):
        mock_tp.world_size = 1
        mock_tp.rank_in_group = MagicMock()
        vllm_config = MagicMock()
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 0
        vllm_config.speculative_config = speculative_config
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_config.model_type = "deepseek_v3"
        vllm_config.kv_transfer_config = None
        vllm_config.additional_config = {"refresh": True}
        parallel_config = MagicMock()
        parallel_config.prefill_context_parallel_size = 1
        parallel_config.decode_context_parallel_size = 1
        parallel_config.tensor_parallel_size = 1
        vllm_config.parallel_config = parallel_config
        mock_get_current_vllm_config.return_value = vllm_config
        init_ascend_config(vllm_config)

        impl = AscendSFACPImpl(
            num_heads=4,
            head_size=128,
            scale=0.1,
            num_kv_heads=2,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type=None,
            kv_sharing_target_layer_name=None,
            **_make_impl_kwargs(),
        )
        self.assertEqual(impl.pcp_size, 1)
        self.assertEqual(impl.dcp_size, 1)
        self.assertEqual(impl.pcp_rank, 0)
        self.assertEqual(impl.dcp_rank, 0)
        self.assertIsNone(impl.pcp_group)
        self.assertIsNone(impl.dcp_group)

    def test_align_to_graph_bucket_tokens_none_input(self):
        self.impl.pcp_size = 2
        result = self.impl._align_to_graph_bucket_tokens(None, MagicMock())
        self.assertIsNone(result)

    def test_align_to_graph_bucket_tokens_no_pcp(self):
        self.impl.pcp_size = 1
        attn_output = torch.randn(4, 8)
        result = self.impl._align_to_graph_bucket_tokens(attn_output, MagicMock())
        self.assertIs(result, attn_output)

    @patch("vllm_ascend.attention.context_parallel.sfa_cp.get_forward_context")
    def test_align_to_graph_bucket_tokens_already_aligned(self, mock_get_fc):
        self.impl.pcp_size = 2
        forward_context = MagicMock()
        forward_context.num_tokens = 8
        mock_get_fc.return_value = forward_context

        attn_metadata = MagicMock()
        attn_metadata.num_input_tokens = 8

        attn_output = torch.randn(8, 16)
        result = self.impl._align_to_graph_bucket_tokens(attn_output, attn_metadata)
        # Already aligned, returns same tensor
        self.assertIs(result, attn_output)

    @patch("vllm_ascend.attention.context_parallel.sfa_cp.get_forward_context")
    def test_align_to_graph_bucket_tokens_pad_smaller(self, mock_get_fc):
        self.impl.pcp_size = 2
        forward_context = MagicMock()
        forward_context.num_tokens = 16
        mock_get_fc.return_value = forward_context

        attn_metadata = MagicMock()
        attn_metadata.num_input_tokens = 8

        attn_output = torch.randn(8, 16)
        result = self.impl._align_to_graph_bucket_tokens(attn_output, attn_metadata)
        self.assertEqual(result.shape, (16, 16))
        # First 8 rows match input
        self.assertTrue(torch.equal(result[:8], attn_output))
        # Padded rows are zeros
        self.assertTrue(torch.all(result[8:] == 0))

    @patch("vllm_ascend.attention.context_parallel.sfa_cp.get_forward_context")
    def test_align_to_graph_bucket_tokens_truncate(self, mock_get_fc):
        # Edge: target is smaller than attn output (rare; valid_tokens = min)
        self.impl.pcp_size = 2
        forward_context = MagicMock()
        forward_context.num_tokens = 4
        mock_get_fc.return_value = forward_context

        attn_metadata = MagicMock()
        attn_metadata.num_input_tokens = 4

        attn_output = torch.randn(8, 16)
        result = self.impl._align_to_graph_bucket_tokens(attn_output, attn_metadata)
        self.assertEqual(result.shape, (4, 16))

    @patch("vllm_ascend.attention.context_parallel.sfa_cp.get_forward_context")
    def test_align_to_graph_bucket_tokens_no_forward_context(self, mock_get_fc):
        self.impl.pcp_size = 2
        mock_get_fc.return_value = None

        attn_metadata = MagicMock()
        attn_metadata.num_input_tokens = 16

        attn_output = torch.randn(8, 16)
        result = self.impl._align_to_graph_bucket_tokens(attn_output, attn_metadata)
        self.assertEqual(result.shape, (16, 16))

    def test_execute_sparse_flash_attention(self):
        ql_nope = torch.randn(2, 4, 32)
        q_pe = torch.randn(2, 4, 16)
        kv = torch.randn(2, 4, 1, 32)
        key_rope = torch.randn(2, 4, 1, 16)
        block_table = torch.tensor([[0]], dtype=torch.int32)
        topk_indices = torch.tensor([[0]], dtype=torch.int32)
        actual_seq_lengths_query = torch.tensor([1, 2], dtype=torch.int32)
        actual_seq_lengths_key = torch.tensor([1, 2], dtype=torch.int32)

        with patch.object(
            torch.ops._C_ascend,
            "npu_sparse_flash_attention",
            create=True,
            return_value=torch.randn(2, 4, 32),
        ) as mock_sfa:
            result = self.impl._execute_sparse_flash_attention(
                ql_nope, q_pe, kv, key_rope, block_table, topk_indices, actual_seq_lengths_query, actual_seq_lengths_key
            )
        self.assertIsNotNone(result)
        mock_sfa.assert_called_once()

    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_gather_kv_cross_cp(self):
        self.impl.pcp_size = 2
        self.impl.dcp_size = 2
        kv_cache = torch.randn(8, 4, 1, 16)
        block_tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)

        result, block_num = self.impl.gather_kv_cross_cp(kv_cache, block_tables)
        # block_num is num blocks selected before all_gather
        self.assertEqual(block_num, 4)
        # After both pcp and dcp all_gather, total blocks = 4 * 2 * 2 = 16
        self.assertEqual(result.shape[0], 16)

    @patch_distributed_groups(dcp_size=1, pcp_size=1, needs_mocks=False)
    def test_gather_kv_cross_cp_no_cp(self):
        self.impl.pcp_size = 1
        self.impl.dcp_size = 1
        kv_cache = torch.randn(8, 4, 1, 16)
        block_tables = torch.tensor([[0, 1]], dtype=torch.int32)

        result, block_num = self.impl.gather_kv_cross_cp(kv_cache, block_tables)
        self.assertEqual(block_num, 2)
        self.assertEqual(result.shape[0], 2)

    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_gather_kv_cross_cp_compact(self):
        self.impl.pcp_size = 2
        self.impl.dcp_size = 2
        kv_cache = torch.randn(8, 4, 1, 16)
        valid_block_ids = torch.tensor([0, 2, 4], dtype=torch.int64)

        result = self.impl.gather_kv_cross_cp_compact(kv_cache, valid_block_ids)
        # 3 blocks * 2 (dcp) * 2 (pcp) = 12
        self.assertEqual(result.shape[0], 12)

    @patch_distributed_groups(dcp_size=1, pcp_size=1, needs_mocks=False)
    def test_gather_kv_cross_cp_compact_no_cp(self):
        self.impl.pcp_size = 1
        self.impl.dcp_size = 1
        kv_cache = torch.randn(8, 4, 1, 16)
        valid_block_ids = torch.tensor([0, 2, 4], dtype=torch.int64)

        result = self.impl.gather_kv_cross_cp_compact(kv_cache, valid_block_ids)
        self.assertEqual(result.shape[0], 3)

    def test_gather_block_table(self):
        block_num = 4
        block_tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        block_arange = torch.arange(4, dtype=torch.int32)

        result = self.impl.gather_block_table(block_num, block_tables, block_arange)
        # Shape: (num_reqs, num_blocks_per_req * pcp*dcp) = (2, 2*4=8)
        self.assertEqual(result.shape, (2, 8))
        self.assertEqual(result.dtype, block_tables.dtype)

    def test_execute_indexer_select_torch_npu(self):
        self.impl.use_torch_npu_lightning_indexer = True
        q = torch.randn(2, 64, 128)
        key = torch.randn(2, 1, 1, 128)
        weights = torch.randn(2, 64)
        actual_seq_lengths_query = torch.tensor([1, 2])
        actual_seq_lengths_key = torch.tensor([1, 2])
        block_table = torch.tensor([[0]], dtype=torch.int32)

        with patch("vllm_ascend.attention.context_parallel.sfa_cp.torch_npu") as mock_torch_npu:
            mock_torch_npu.npu_lightning_indexer.return_value = (torch.tensor([[0]]), None)
            result = self.impl._execute_indexer_select(
                q, key, weights, actual_seq_lengths_query, actual_seq_lengths_key, block_table
            )
        self.assertIsNotNone(result)

    def test_execute_indexer_select_ascend_op(self):
        self.impl.use_torch_npu_lightning_indexer = False
        q = torch.randn(2, 64, 128)
        key = torch.randn(2, 1, 1, 128)
        weights = torch.randn(2, 64)
        actual_seq_lengths_query = torch.tensor([1, 2])
        actual_seq_lengths_key = torch.tensor([1, 2])
        block_table = torch.tensor([[0]], dtype=torch.int32)

        with patch.object(
            torch.ops._C_ascend,
            "npu_lightning_indexer",
            create=True,
            return_value=torch.tensor([[0]]),
        ) as mock_indexer:
            result = self.impl._execute_indexer_select(
                q, key, weights, actual_seq_lengths_query, actual_seq_lengths_key, block_table
            )
        self.assertIsNotNone(result)
        mock_indexer.assert_called_once()

    def test_get_full_kv_no_pcp(self):
        self.impl.pcp_size = 1
        k = torch.randn(4, 8, 16)
        result = self.impl._get_full_kv(k, MagicMock())
        self.assertIs(result, k)

    def test_get_full_kv_mlapo(self):
        self.impl.pcp_size = 2
        self.impl.enable_mlapo = True
        k = torch.randn(4, 8, 16)
        result = self.impl._get_full_kv(k, MagicMock())
        self.assertIs(result, k)

    @patch_distributed_groups(dcp_size=1, pcp_size=2, needs_mocks=False)
    def test_get_full_kv_with_pcp(self):
        self.impl.pcp_size = 2
        self.impl.enable_mlapo = False
        k = torch.randn(4, 8, 16)
        attn_metadata = MagicMock()
        sfa_cp_metadata = MagicMock()
        sfa_cp_metadata.pcp_allgather_restore_idx = torch.arange(8)
        attn_metadata.sfa_cp_metadata = sfa_cp_metadata

        result = self.impl._get_full_kv(k, attn_metadata)
        # After all_gather pcp_size=2 -> 8 entries, then index_select with 8 indices
        self.assertEqual(result.shape[0], 8)

    @patch("vllm_ascend.attention.context_parallel.sfa_cp.torch_npu")
    @patch_distributed_groups(dcp_size=1, pcp_size=1, needs_mocks=False)
    def test_exec_kv_no_pcp(self, mock_torch_npu):
        # When pcp_size==1, simply delegates to super().exec_kv
        self.impl.pcp_size = 1
        with patch.object(AscendSFAImpl, "exec_kv", return_value=("a", "b")) as mock_super:
            result = self.impl.exec_kv(
                kv_no_split=torch.randn(2, 64),
                cos=torch.randn(2, 32),
                sin=torch.randn(2, 32),
                kv_cache=(torch.randn(4, 1, 1, 32), torch.randn(4, 1, 1, 32)),
                slots=torch.tensor([0, 1], dtype=torch.int32),
                attn_metadata=MagicMock(),
            )
        mock_super.assert_called_once()
        self.assertEqual(result, ("a", "b"))

    @patch("vllm_ascend.attention.context_parallel.sfa_cp.torch_npu")
    @patch_distributed_groups(dcp_size=1, pcp_size=2, needs_mocks=False)
    def test_exec_kv_with_pcp(self, mock_torch_npu):
        self.impl.pcp_size = 2
        # Configure dimensions
        self.impl.kv_lora_rank = 32
        self.impl.qk_rope_head_dim = 16
        self.impl.num_kv_heads = 1

        kv_a_layernorm = MagicMock()
        kv_a_layernorm.side_effect = lambda x: x
        self.impl.kv_a_layernorm = kv_a_layernorm
        self.impl.rope_single = MagicMock(side_effect=lambda x, cos, sin: x)

        # 2 input tokens, [num_tokens, kv_lora_rank + qk_rope_head_dim]
        kv_no_split = torch.randn(2, 32 + 16)
        cos = torch.randn(2, 16)
        sin = torch.randn(2, 16)
        kv_cache = (torch.randn(4, 1, 1, 32), torch.randn(4, 1, 1, 16))
        slots = torch.tensor([0, 1, 2, 3], dtype=torch.int32)

        attn_metadata = MagicMock()
        sfa_cp_metadata = MagicMock()
        sfa_cp_metadata.pcp_allgather_restore_idx = torch.arange(4)
        attn_metadata.sfa_cp_metadata = sfa_cp_metadata
        attn_metadata.slot_mapping = slots

        result = self.impl.exec_kv(kv_no_split, cos, sin, kv_cache, slots, attn_metadata)
        self.assertEqual(result, (None, None))
        mock_torch_npu._npu_reshape_and_cache.assert_called_once()

    @patch_distributed_groups(dcp_size=1, pcp_size=1, needs_mocks=False)
    def test_execute_sparse_flash_attention_process_decode_only(self):
        # num_prefills < 1: returns aligned decode output
        self.impl.pcp_size = 1
        self.impl.dcp_size = 1
        ql_nope = torch.randn(2, 4, 32)
        q_pe = torch.randn(2, 4, 16)
        kv_cache = (
            torch.randn(4, 1, 1, 32),
            torch.randn(4, 1, 1, 16),
            torch.randn(4, 1, 1, 32),
        )
        topk_indices = torch.tensor([[0], [0]], dtype=torch.int32)
        attn_metadata = MagicMock()
        attn_metadata.num_decodes = 2
        attn_metadata.num_decode_tokens = 2
        attn_metadata.num_prefills = 0
        attn_metadata.block_table = torch.tensor([[0], [1]], dtype=torch.int32)
        sfa_cp_metadata = MagicMock()
        sfa_cp_metadata.block_arange = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        attn_metadata.sfa_cp_metadata = sfa_cp_metadata

        actual_seq_lengths_query = torch.tensor([1, 2], dtype=torch.int32)
        actual_seq_lengths_key = torch.tensor([1, 2], dtype=torch.int32)

        with patch.object(
            torch.ops._C_ascend,
            "npu_sparse_flash_attention",
            create=True,
            return_value=torch.randn(2, 4, 32),
        ):
            result = self.impl._execute_sparse_flash_attention_process(
                ql_nope, q_pe, kv_cache, topk_indices, attn_metadata, actual_seq_lengths_query, actual_seq_lengths_key
            )
        self.assertIsNotNone(result)

    @patch_distributed_groups(dcp_size=1, pcp_size=1, needs_mocks=False)
    def test_execute_sparse_flash_attention_process_prefill_only_no_pcp(self):
        # Case: only prefills, pcp_size==1
        self.impl.pcp_size = 1
        self.impl.dcp_size = 1
        ql_nope = torch.randn(4, 4, 32)
        q_pe = torch.randn(4, 4, 16)
        kv_cache = (
            torch.randn(4, 1, 1, 32),
            torch.randn(4, 1, 1, 16),
            torch.randn(4, 1, 1, 32),
        )
        topk_indices = torch.tensor([[0]] * 4, dtype=torch.int32)
        attn_metadata = MagicMock()
        attn_metadata.num_decodes = 0
        attn_metadata.num_decode_tokens = 0
        attn_metadata.num_prefills = 2
        attn_metadata.block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)

        sfa_cp_metadata = MagicMock()
        sfa_cp_metadata.valid_block_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        sfa_cp_metadata.block_table_cp = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        sfa_cp_metadata.prefill_q_cum_seqlens = torch.tensor([2, 4], dtype=torch.int32)
        sfa_cp_metadata.block_arange = torch.tensor([0], dtype=torch.int32)
        attn_metadata.sfa_cp_metadata = sfa_cp_metadata

        actual_seq_lengths_query = torch.tensor([2, 4], dtype=torch.int32)
        actual_seq_lengths_key = torch.tensor([4, 8], dtype=torch.int32)

        with patch.object(
            torch.ops._C_ascend,
            "npu_sparse_flash_attention",
            create=True,
            return_value=torch.randn(4, 4, 32),
        ):
            result = self.impl._execute_sparse_flash_attention_process(
                ql_nope, q_pe, kv_cache, topk_indices, attn_metadata, actual_seq_lengths_query, actual_seq_lengths_key
            )
        self.assertIsNotNone(result)
        self.assertEqual(result.shape[0], 4)

    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_execute_sparse_flash_attention_process_prefill_with_pcp(self):
        self.impl.pcp_size = 2
        self.impl.dcp_size = 2
        ql_nope = torch.randn(4, 4, 32)
        q_pe = torch.randn(4, 4, 16)
        kv_cache = (
            torch.randn(4, 1, 1, 32),
            torch.randn(4, 1, 1, 16),
            torch.randn(4, 1, 1, 32),
        )
        topk_indices = torch.tensor([[0]] * 4, dtype=torch.int32)
        attn_metadata = MagicMock()
        attn_metadata.num_decodes = 0
        attn_metadata.num_decode_tokens = 0
        attn_metadata.num_prefills = 2
        attn_metadata.num_input_tokens = 4
        attn_metadata.block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)

        sfa_cp_metadata = MagicMock()
        sfa_cp_metadata.valid_block_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        sfa_cp_metadata.block_table_cp = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        sfa_cp_metadata.prefill_q_cum_seqlens = torch.tensor([2, 4], dtype=torch.int32)
        sfa_cp_metadata.q_head_idx = torch.tensor([0, 1], dtype=torch.int64)
        sfa_cp_metadata.q_tail_idx = torch.tensor([2, 3], dtype=torch.int64)
        sfa_cp_metadata.q_full_idx = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        sfa_cp_metadata.head_attn_nomask_seqlens = torch.tensor([4, 4], dtype=torch.int32)
        sfa_cp_metadata.tail_attn_nomask_seqlens = torch.tensor([8, 8], dtype=torch.int32)
        sfa_cp_metadata.block_arange = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        attn_metadata.sfa_cp_metadata = sfa_cp_metadata

        actual_seq_lengths_query = torch.tensor([2, 4], dtype=torch.int32)
        actual_seq_lengths_key = torch.tensor([4, 8], dtype=torch.int32)

        with (
            patch.object(
                torch.ops._C_ascend,
                "npu_sparse_flash_attention",
                create=True,
                return_value=torch.randn(2, 4, 32),
            ),
            patch("vllm_ascend.attention.context_parallel.sfa_cp.get_forward_context") as mock_fc,
        ):
            mock_fc.return_value = MagicMock(num_tokens=4)
            result = self.impl._execute_sparse_flash_attention_process(
                ql_nope, q_pe, kv_cache, topk_indices, attn_metadata, actual_seq_lengths_query, actual_seq_lengths_key
            )
        self.assertIsNotNone(result)

    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_execute_sparse_flash_attention_process_decode_and_prefill_with_pcp(self):
        # Covers final torch.cat([decode_attn_out, attn_output]) (line 326)
        self.impl.pcp_size = 2
        self.impl.dcp_size = 2

        ql_nope = torch.randn(5, 4, 32)
        q_pe = torch.randn(5, 4, 16)
        kv_cache = (
            torch.randn(4, 1, 1, 32),
            torch.randn(4, 1, 1, 16),
            torch.randn(4, 1, 1, 32),
        )
        topk_indices = torch.tensor([[0]] * 5, dtype=torch.int32)
        attn_metadata = MagicMock()
        attn_metadata.num_decodes = 1
        attn_metadata.num_decode_tokens = 1
        attn_metadata.num_prefills = 2
        attn_metadata.num_input_tokens = 5
        attn_metadata.block_table = torch.tensor([[0], [1], [2]], dtype=torch.int32)

        sfa_cp_metadata = MagicMock()
        sfa_cp_metadata.valid_block_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        sfa_cp_metadata.block_table_cp = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        sfa_cp_metadata.prefill_q_cum_seqlens = torch.tensor([2, 4], dtype=torch.int32)
        sfa_cp_metadata.q_head_idx = torch.tensor([0, 1], dtype=torch.int64)
        sfa_cp_metadata.q_tail_idx = torch.tensor([2, 3], dtype=torch.int64)
        sfa_cp_metadata.q_full_idx = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        sfa_cp_metadata.head_attn_nomask_seqlens = torch.tensor([4, 4, 4], dtype=torch.int32)
        sfa_cp_metadata.tail_attn_nomask_seqlens = torch.tensor([8, 8, 8], dtype=torch.int32)
        sfa_cp_metadata.block_arange = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        attn_metadata.sfa_cp_metadata = sfa_cp_metadata

        actual_seq_lengths_query = torch.tensor([1, 3, 5], dtype=torch.int32)
        actual_seq_lengths_key = torch.tensor([4, 8, 8], dtype=torch.int32)

        def fake_sfa(query, **kwargs):
            return torch.randn(query.shape[0], query.shape[1], query.shape[2])

        with (
            patch.object(
                torch.ops._C_ascend,
                "npu_sparse_flash_attention",
                create=True,
                side_effect=fake_sfa,
            ),
            patch("vllm_ascend.attention.context_parallel.sfa_cp.get_forward_context") as mock_fc,
        ):
            mock_fc.return_value = MagicMock(num_tokens=5)
            result = self.impl._execute_sparse_flash_attention_process(
                ql_nope, q_pe, kv_cache, topk_indices, attn_metadata, actual_seq_lengths_query, actual_seq_lengths_key
            )
        self.assertIsNotNone(result)
        self.assertEqual(result.shape[0], 5)

    @patch_distributed_groups(dcp_size=1, pcp_size=1, needs_mocks=False)
    def test_execute_sparse_flash_attention_process_decode_and_prefill_no_pcp(self):
        self.impl.pcp_size = 1
        self.impl.dcp_size = 1
        ql_nope = torch.randn(3, 4, 32)
        q_pe = torch.randn(3, 4, 16)
        kv_cache = (
            torch.randn(4, 1, 1, 32),
            torch.randn(4, 1, 1, 16),
            torch.randn(4, 1, 1, 32),
        )
        topk_indices = torch.tensor([[0]] * 3, dtype=torch.int32)
        attn_metadata = MagicMock()
        attn_metadata.num_decodes = 1
        attn_metadata.num_decode_tokens = 1
        attn_metadata.num_prefills = 1
        attn_metadata.block_table = torch.tensor([[0], [1]], dtype=torch.int32)

        sfa_cp_metadata = MagicMock()
        sfa_cp_metadata.valid_block_ids = torch.tensor([0, 1], dtype=torch.int64)
        sfa_cp_metadata.block_table_cp = torch.tensor([[0, 1]], dtype=torch.int32)
        sfa_cp_metadata.prefill_q_cum_seqlens = torch.tensor([2], dtype=torch.int32)
        sfa_cp_metadata.block_arange = torch.tensor([0], dtype=torch.int32)
        attn_metadata.sfa_cp_metadata = sfa_cp_metadata

        actual_seq_lengths_query = torch.tensor([1, 3], dtype=torch.int32)
        actual_seq_lengths_key = torch.tensor([4, 8], dtype=torch.int32)

        # Use side_effect so each call returns a tensor with the q-shape
        def fake_sfa(query, **kwargs):
            return torch.randn(query.shape[0], query.shape[1], query.shape[2])

        with patch.object(
            torch.ops._C_ascend,
            "npu_sparse_flash_attention",
            create=True,
            side_effect=fake_sfa,
        ):
            result = self.impl._execute_sparse_flash_attention_process(
                ql_nope, q_pe, kv_cache, topk_indices, attn_metadata, actual_seq_lengths_query, actual_seq_lengths_key
            )
        self.assertIsNotNone(result)
        self.assertEqual(result.shape[0], 3)

    @patch("vllm_ascend.attention.context_parallel.sfa_cp.HAS_TRITON", True)
    @patch("vllm_ascend.attention.context_parallel.sfa_cp.rope_forward_triton_siso")
    @patch_distributed_groups(dcp_size=1, pcp_size=1, needs_mocks=False)
    def test_indexer_select_post_process_decode_only_simple(self, mock_rope):
        # Case: num_prefills==0, returns decode_topk_indices
        self.impl.pcp_size = 1
        self.impl.dcp_size = 1
        self.impl.use_torch_npu_lightning_indexer = False

        x = torch.randn(2, self.impl.qk_head_dim)
        q_c = torch.randn(2, self.impl.q_lora_rank)
        kv_cache = (
            torch.randn(4, 1, 1, 32),
            torch.randn(4, 1, 1, 16),
            torch.randn(4, 1, 1, self.impl.head_dim),
        )
        cos = torch.randn(2, self.impl.qk_rope_head_dim)
        sin = torch.randn(2, self.impl.qk_rope_head_dim)

        kw_out = torch.randn(2, self.impl.head_dim * 2)
        self.impl.wk_weights_proj.return_value = (kw_out, None)
        self.impl.wq_b.return_value = (
            torch.randn(2, self.impl.n_head * self.impl.head_dim),
            None,
        )
        mock_rope.return_value = torch.randn(2, self.impl.n_head, self.impl.head_dim)

        attn_metadata = MagicMock()
        attn_metadata.num_decodes = 2
        attn_metadata.num_decode_tokens = 2
        attn_metadata.num_prefills = 0
        attn_metadata.block_table = torch.tensor([[0], [1]], dtype=torch.int32)

        sfa_cp_metadata = MagicMock()
        sfa_cp_metadata.block_arange = torch.tensor([0], dtype=torch.int32)
        attn_metadata.sfa_cp_metadata = sfa_cp_metadata

        actual_seq_lengths_query = torch.tensor([1, 2], dtype=torch.int32)
        actual_seq_lengths_key = torch.tensor([4, 8], dtype=torch.int32)

        with patch.object(
            torch.ops._C_ascend,
            "npu_lightning_indexer",
            create=True,
            return_value=torch.tensor([[0]] * 2),
        ):
            result = self.impl.indexer_select_post_process(
                x, q_c, kv_cache, attn_metadata, cos, sin, actual_seq_lengths_query, actual_seq_lengths_key
            )
        self.assertIsNotNone(result)

    @patch("vllm_ascend.attention.context_parallel.sfa_cp.HAS_TRITON", False)
    @patch("vllm_ascend.attention.context_parallel.sfa_cp.torch_npu")
    @patch_distributed_groups(dcp_size=1, pcp_size=1, needs_mocks=False)
    def test_indexer_select_post_process_decode_only_no_triton(self, mock_torch_npu):
        # Test no-triton path
        self.impl.pcp_size = 1
        self.impl.dcp_size = 1
        self.impl.use_torch_npu_lightning_indexer = False
        self.impl.is_rope_neox_style = True

        x = torch.randn(2, self.impl.qk_head_dim)
        q_c = torch.randn(2, self.impl.q_lora_rank)
        kv_cache = (
            torch.randn(4, 1, 1, 32),
            torch.randn(4, 1, 1, 16),
            torch.randn(4, 1, 1, self.impl.head_dim),
        )
        cos = torch.randn(2, self.impl.qk_rope_head_dim)
        sin = torch.randn(2, self.impl.qk_rope_head_dim)

        kw_out = torch.randn(2, self.impl.head_dim * 2)
        self.impl.wk_weights_proj.return_value = (kw_out, None)
        self.impl.wq_b.return_value = (
            torch.randn(2, self.impl.n_head * self.impl.head_dim),
            None,
        )
        mock_torch_npu.npu_rotary_mul.return_value = torch.randn(2, self.impl.n_head, 1, self.impl.qk_rope_head_dim)

        attn_metadata = MagicMock()
        attn_metadata.num_decodes = 2
        attn_metadata.num_decode_tokens = 2
        attn_metadata.num_prefills = 0
        attn_metadata.block_table = torch.tensor([[0], [1]], dtype=torch.int32)

        sfa_cp_metadata = MagicMock()
        sfa_cp_metadata.block_arange = torch.tensor([0], dtype=torch.int32)
        attn_metadata.sfa_cp_metadata = sfa_cp_metadata

        actual_seq_lengths_query = torch.tensor([1, 2], dtype=torch.int32)
        actual_seq_lengths_key = torch.tensor([4, 8], dtype=torch.int32)

        with patch.object(
            torch.ops._C_ascend,
            "npu_lightning_indexer",
            create=True,
            return_value=torch.tensor([[0]] * 2),
        ):
            result = self.impl.indexer_select_post_process(
                x, q_c, kv_cache, attn_metadata, cos, sin, actual_seq_lengths_query, actual_seq_lengths_key
            )
        self.assertIsNotNone(result)

    @patch("vllm_ascend.attention.context_parallel.sfa_cp.HAS_TRITON", True)
    @patch("vllm_ascend.attention.context_parallel.sfa_cp.rope_forward_triton_siso")
    @patch_distributed_groups(dcp_size=1, pcp_size=1, needs_mocks=False)
    def test_indexer_select_post_process_prefill_only_no_pcp(self, mock_rope):
        # Case: only prefills, pcp_size==1
        self.impl.pcp_size = 1
        self.impl.dcp_size = 1
        self.impl.use_torch_npu_lightning_indexer = False

        x = torch.randn(2, self.impl.qk_head_dim)
        q_c = torch.randn(2, self.impl.q_lora_rank)
        kv_cache = (
            torch.randn(4, 1, 1, 32),
            torch.randn(4, 1, 1, 16),
            torch.randn(4, 1, 1, self.impl.head_dim),
        )
        cos = torch.randn(2, self.impl.qk_rope_head_dim)
        sin = torch.randn(2, self.impl.qk_rope_head_dim)

        kw_out = torch.randn(2, self.impl.head_dim * 2)
        self.impl.wk_weights_proj.return_value = (kw_out, None)
        self.impl.wq_b.return_value = (
            torch.randn(2, self.impl.n_head * self.impl.head_dim),
            None,
        )
        mock_rope.return_value = torch.randn(2, self.impl.n_head, self.impl.head_dim)

        attn_metadata = MagicMock()
        attn_metadata.num_decodes = 0
        attn_metadata.num_decode_tokens = 0
        attn_metadata.num_prefills = 1
        attn_metadata.block_table = torch.tensor([[0]], dtype=torch.int32)

        sfa_cp_metadata = MagicMock()
        sfa_cp_metadata.valid_block_ids = torch.tensor([0, 1], dtype=torch.int64)
        sfa_cp_metadata.block_table_cp = torch.tensor([[0, 1]], dtype=torch.int32)
        sfa_cp_metadata.prefill_q_cum_seqlens = torch.tensor([2], dtype=torch.int32)
        sfa_cp_metadata.block_arange = torch.tensor([0], dtype=torch.int32)
        attn_metadata.sfa_cp_metadata = sfa_cp_metadata

        actual_seq_lengths_query = torch.tensor([2], dtype=torch.int32)
        actual_seq_lengths_key = torch.tensor([4], dtype=torch.int32)

        with patch.object(
            torch.ops._C_ascend,
            "npu_lightning_indexer",
            create=True,
            return_value=torch.tensor([[0]] * 2),
        ):
            result = self.impl.indexer_select_post_process(
                x, q_c, kv_cache, attn_metadata, cos, sin, actual_seq_lengths_query, actual_seq_lengths_key
            )
        self.assertIsNotNone(result)

    @patch("vllm_ascend.attention.context_parallel.sfa_cp.HAS_TRITON", True)
    @patch("vllm_ascend.attention.context_parallel.sfa_cp.rope_forward_triton_siso")
    @patch_distributed_groups(dcp_size=1, pcp_size=1, needs_mocks=False)
    def test_indexer_select_post_process_decode_and_prefill_no_pcp(self, mock_rope):
        self.impl.pcp_size = 1
        self.impl.dcp_size = 1
        self.impl.use_torch_npu_lightning_indexer = False

        x = torch.randn(3, self.impl.qk_head_dim)
        q_c = torch.randn(3, self.impl.q_lora_rank)
        kv_cache = (
            torch.randn(4, 1, 1, 32),
            torch.randn(4, 1, 1, 16),
            torch.randn(4, 1, 1, self.impl.head_dim),
        )
        cos = torch.randn(3, self.impl.qk_rope_head_dim)
        sin = torch.randn(3, self.impl.qk_rope_head_dim)

        kw_out = torch.randn(3, self.impl.head_dim * 2)
        self.impl.wk_weights_proj.return_value = (kw_out, None)
        self.impl.wq_b.return_value = (
            torch.randn(3, self.impl.n_head * self.impl.head_dim),
            None,
        )
        mock_rope.return_value = torch.randn(3, self.impl.n_head, self.impl.head_dim)

        attn_metadata = MagicMock()
        attn_metadata.num_decodes = 1
        attn_metadata.num_decode_tokens = 1
        attn_metadata.num_prefills = 1
        attn_metadata.block_table = torch.tensor([[0], [1]], dtype=torch.int32)

        sfa_cp_metadata = MagicMock()
        sfa_cp_metadata.valid_block_ids = torch.tensor([0, 1], dtype=torch.int64)
        sfa_cp_metadata.block_table_cp = torch.tensor([[0, 1]], dtype=torch.int32)
        sfa_cp_metadata.prefill_q_cum_seqlens = torch.tensor([2], dtype=torch.int32)
        sfa_cp_metadata.block_arange = torch.tensor([0], dtype=torch.int32)
        attn_metadata.sfa_cp_metadata = sfa_cp_metadata

        actual_seq_lengths_query = torch.tensor([1, 3], dtype=torch.int32)
        actual_seq_lengths_key = torch.tensor([4, 8], dtype=torch.int32)

        # In each call, returned tensor has rows matching q
        call_counter = [0]

        def fake_indexer(query, **kwargs):
            call_counter[0] += 1
            return torch.tensor([[0]] * query.shape[0])

        with patch.object(
            torch.ops._C_ascend,
            "npu_lightning_indexer",
            create=True,
            side_effect=fake_indexer,
        ):
            result = self.impl.indexer_select_post_process(
                x, q_c, kv_cache, attn_metadata, cos, sin, actual_seq_lengths_query, actual_seq_lengths_key
            )
        self.assertIsNotNone(result)
        self.assertEqual(result.shape[0], 3)

    @patch("vllm_ascend.attention.context_parallel.sfa_cp.HAS_TRITON", True)
    @patch("vllm_ascend.attention.context_parallel.sfa_cp.rope_forward_triton_siso")
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_indexer_select_post_process_prefill_with_pcp(self, mock_rope):
        # Case: prefills + pcp head/tail processing
        self.impl.pcp_size = 2
        self.impl.dcp_size = 2
        self.impl.use_torch_npu_lightning_indexer = False

        # 4 prefill tokens
        x = torch.randn(4, self.impl.qk_head_dim)
        q_c = torch.randn(4, self.impl.q_lora_rank)
        kv_cache = (
            torch.randn(4, 1, 1, 32),
            torch.randn(4, 1, 1, 16),
            torch.randn(4, 1, 1, self.impl.head_dim),
        )
        cos = torch.randn(4, self.impl.qk_rope_head_dim)
        sin = torch.randn(4, self.impl.qk_rope_head_dim)

        kw_out = torch.randn(4, self.impl.head_dim * 2)
        self.impl.wk_weights_proj.return_value = (kw_out, None)
        self.impl.wq_b.return_value = (
            torch.randn(4, self.impl.n_head * self.impl.head_dim),
            None,
        )
        mock_rope.return_value = torch.randn(4, self.impl.n_head, self.impl.head_dim)

        attn_metadata = MagicMock()
        attn_metadata.num_decodes = 0
        attn_metadata.num_decode_tokens = 0
        attn_metadata.num_prefills = 2
        attn_metadata.block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)

        sfa_cp_metadata = MagicMock()
        sfa_cp_metadata.valid_block_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        sfa_cp_metadata.block_table_cp = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        sfa_cp_metadata.prefill_q_cum_seqlens = torch.tensor([2, 4], dtype=torch.int32)
        sfa_cp_metadata.q_head_idx = torch.tensor([0, 1], dtype=torch.int64)
        sfa_cp_metadata.q_tail_idx = torch.tensor([2, 3], dtype=torch.int64)
        sfa_cp_metadata.q_full_idx = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        sfa_cp_metadata.head_attn_nomask_seqlens = torch.tensor([4, 4], dtype=torch.int32)
        sfa_cp_metadata.tail_attn_nomask_seqlens = torch.tensor([8, 8], dtype=torch.int32)
        sfa_cp_metadata.block_arange = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        attn_metadata.sfa_cp_metadata = sfa_cp_metadata

        actual_seq_lengths_query = torch.tensor([2, 4], dtype=torch.int32)
        actual_seq_lengths_key = torch.tensor([4, 8], dtype=torch.int32)

        def fake_indexer(query, **kwargs):
            return torch.tensor([[0]] * query.shape[0])

        with patch.object(
            torch.ops._C_ascend,
            "npu_lightning_indexer",
            create=True,
            side_effect=fake_indexer,
        ):
            result = self.impl.indexer_select_post_process(
                x, q_c, kv_cache, attn_metadata, cos, sin, actual_seq_lengths_query, actual_seq_lengths_key
            )
        self.assertIsNotNone(result)
        self.assertEqual(result.shape[0], 4)

    @patch("vllm_ascend.attention.context_parallel.sfa_cp.HAS_TRITON", True)
    @patch("vllm_ascend.attention.context_parallel.sfa_cp.rope_forward_triton_siso")
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_indexer_select_post_process_decode_and_prefill_with_pcp(self, mock_rope):
        # Case: decodes + prefills + pcp; covers final torch.cat([decode, attn_output]).
        self.impl.pcp_size = 2
        self.impl.dcp_size = 2
        self.impl.use_torch_npu_lightning_indexer = False

        # 1 decode + 4 prefill = 5 total
        x = torch.randn(5, self.impl.qk_head_dim)
        q_c = torch.randn(5, self.impl.q_lora_rank)
        kv_cache = (
            torch.randn(4, 1, 1, 32),
            torch.randn(4, 1, 1, 16),
            torch.randn(4, 1, 1, self.impl.head_dim),
        )
        cos = torch.randn(5, self.impl.qk_rope_head_dim)
        sin = torch.randn(5, self.impl.qk_rope_head_dim)

        kw_out = torch.randn(5, self.impl.head_dim * 2)
        self.impl.wk_weights_proj.return_value = (kw_out, None)
        self.impl.wq_b.return_value = (
            torch.randn(5, self.impl.n_head * self.impl.head_dim),
            None,
        )
        mock_rope.return_value = torch.randn(5, self.impl.n_head, self.impl.head_dim)

        attn_metadata = MagicMock()
        attn_metadata.num_decodes = 1
        attn_metadata.num_decode_tokens = 1
        attn_metadata.num_prefills = 2
        attn_metadata.block_table = torch.tensor([[0], [1], [2]], dtype=torch.int32)

        sfa_cp_metadata = MagicMock()
        sfa_cp_metadata.valid_block_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        sfa_cp_metadata.block_table_cp = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        sfa_cp_metadata.prefill_q_cum_seqlens = torch.tensor([2, 4], dtype=torch.int32)
        sfa_cp_metadata.q_head_idx = torch.tensor([0, 1], dtype=torch.int64)
        sfa_cp_metadata.q_tail_idx = torch.tensor([2, 3], dtype=torch.int64)
        sfa_cp_metadata.q_full_idx = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        sfa_cp_metadata.head_attn_nomask_seqlens = torch.tensor([4, 4, 4], dtype=torch.int32)
        sfa_cp_metadata.tail_attn_nomask_seqlens = torch.tensor([8, 8, 8], dtype=torch.int32)
        sfa_cp_metadata.block_arange = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        attn_metadata.sfa_cp_metadata = sfa_cp_metadata

        actual_seq_lengths_query = torch.tensor([1, 3, 5], dtype=torch.int32)
        actual_seq_lengths_key = torch.tensor([4, 8, 8], dtype=torch.int32)

        def fake_indexer(query, **kwargs):
            return torch.tensor([[0]] * query.shape[0])

        with patch.object(
            torch.ops._C_ascend,
            "npu_lightning_indexer",
            create=True,
            side_effect=fake_indexer,
        ):
            result = self.impl.indexer_select_post_process(
                x, q_c, kv_cache, attn_metadata, cos, sin, actual_seq_lengths_query, actual_seq_lengths_key
            )
        self.assertIsNotNone(result)
        self.assertEqual(result.shape[0], 5)
