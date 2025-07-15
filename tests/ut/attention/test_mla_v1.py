from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm.model_executor.layers.linear import LinearBase

from tests.ut.base import TestBase
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.mla_v1 import (AscendMLABackend,
                                          AscendMLADecodeMetadata,
                                          AscendMLAImpl, AscendMLAMetadata,
                                          AscendMLAMetadataBuilder,
                                          AscendMLAPrefillMetadata)


class TestAscendMLABackend(TestBase):
    def test_get_name(self):
        self.assertEqual(AscendMLABackend.get_name(), "ASCEND")

    def test_get_impl_cls(self):
        self.assertEqual(AscendMLABackend.get_impl_cls(), AscendMLAImpl)

    def test_get_metadata_cls(self):
        self.assertEqual(AscendMLABackend.get_metadata_cls(),
                         AscendMLAMetadata)

    def test_get_builder_cls(self):
        self.assertEqual(AscendMLABackend.get_builder_cls(),
                         AscendMLAMetadataBuilder)

    def test_get_kv_cache_shape(self):
        result = AscendMLABackend.get_kv_cache_shape(2, 4, 8, 128)
        self.assertEqual(result, (2, 4, 8, 128))


class TestAscendMLAPrefillMetadata(TestBase):
    def test_ascend_mla_prefill_metadata_default(self):
        # 构造最小必要输入
        attn_mask = torch.tensor([[1, 0], [1, 1]], dtype=torch.bool)
        query_lens = [1, 2]
        seq_lens = [2, 2]
        context_lens = torch.tensor([1, 2])
        input_positions = torch.tensor([0, 1, 0, 1])
        query_start_loc = torch.tensor([0, 1, 3])
        block_table = torch.tensor([[0, 1], [2, 3]])
        max_query_len = 2
        max_seq_lens = 2

        metadata = AscendMLAPrefillMetadata(attn_mask=attn_mask,
                                            query_lens=query_lens,
                                            seq_lens=seq_lens,
                                            context_lens=context_lens,
                                            input_positions=input_positions,
                                            query_start_loc=query_start_loc,
                                            block_table=block_table,
                                            max_query_len=max_query_len,
                                            max_seq_lens=max_seq_lens)
        self.assertEqual(metadata.attn_mask, attn_mask)
        self.assertEqual(metadata.query_lens, query_lens)
        self.assertEqual(metadata.seq_lens, seq_lens)
        self.assertEqual(metadata.context_lens, context_lens)
        self.assertEqual(metadata.input_positions, input_positions)
        self.assertEqual(metadata.query_start_loc, query_start_loc)
        self.assertEqual(metadata.block_table, block_table)
        self.assertEqual(metadata.max_query_len, max_query_len)
        self.assertEqual(metadata.max_seq_lens, max_seq_lens)
        self.assertIsNone(metadata.chunked_context)

    def test_ascend_mla_prefill_metadata_with_chunked_context(self):
        # 构造嵌套的 ChunkedContextMetadata
        cu_seq_lens = torch.tensor([0, 2, 4])
        starts = torch.tensor([0, 2])
        seq_tot = [2, 2]
        max_seq_lens = [2, 2]
        workspace = torch.randn(2, 4)
        chunk_seq_lens = torch.tensor([2, 2])

        chunked_context = AscendMLAPrefillMetadata.ChunkedContextMetadata(
            cu_seq_lens=cu_seq_lens,
            starts=starts,
            seq_tot=seq_tot,
            max_seq_lens=max_seq_lens,
            workspace=workspace,
            chunk_seq_lens=chunk_seq_lens)

        # 构造主 metadata
        metadata = AscendMLAPrefillMetadata(
            attn_mask=torch.tensor([[1, 0], [1, 1]], dtype=torch.bool),
            query_lens=[1, 2],
            seq_lens=[2, 2],
            context_lens=torch.tensor([1, 2]),
            input_positions=torch.tensor([0, 1, 0, 1]),
            query_start_loc=torch.tensor([0, 1, 3]),
            block_table=torch.tensor([[0, 1], [2, 3]]),
            max_query_len=2,
            max_seq_lens=2,
            chunked_context=chunked_context)

        self.assertIsNotNone(metadata.chunked_context)
        self.assertEqual(metadata.chunked_context.cu_seq_lens, cu_seq_lens)
        self.assertEqual(metadata.chunked_context.starts, starts)
        self.assertEqual(metadata.chunked_context.seq_tot, seq_tot)
        self.assertEqual(metadata.chunked_context.max_seq_lens, max_seq_lens)
        self.assertEqual(metadata.chunked_context.workspace, workspace)
        self.assertEqual(metadata.chunked_context.chunk_seq_lens,
                         chunk_seq_lens)


class TestAscendMLADecodeMetadata(TestBase):
    def test_ascend_mla_decode_metadata_default(self):
        input_positions = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
        block_table = torch.tensor([[0, 3, 2, 1], [0, 2, 1, 3]])
        seq_lens = torch.tensor([[2], [3]])
        max_seq_lens = 4
        seq_lens_list = [2, 3]
        attn_mask = None

        metadata = AscendMLADecodeMetadata(input_positions, block_table,
                                           seq_lens, max_seq_lens,
                                           seq_lens_list, attn_mask)

        self.assertEqual(metadata.input_positions, input_positions)
        self.assertEqual(metadata.block_table, block_table)
        self.assertEqual(metadata.seq_lens, seq_lens)
        self.assertEqual(metadata.max_seq_lens, max_seq_lens)
        self.assertEqual(metadata.seq_lens_list, seq_lens_list)
        self.assertIsNone(attn_mask)


class TestAscendMLAMetadata(TestBase):
    def test_ascend_mla_metadata_default(self):
        num_actual_tokens = 100
        slot_mapping = torch.randn(100, 4, 1024)
        query_start_loc = torch.tensor([1, 2, 3, 4])
        seq_lens = [30, 50]
        block_tables = torch.randint(0, 100, (100, 4))

        num_decodes = 4
        num_decode_tokens = 8
        num_prefills = 8

        num_input_tokens = 2

        max_num_tokens_across_dp = 2
        with_prefill_across_dp = False
        query_lens = None
        head_dim = None
        attn_mask = None
        attn_state = AscendAttentionState.ChunkedPrefill

        decode = None
        prefill = None

        metadata = AscendMLAMetadata(
            num_actual_tokens, slot_mapping, query_start_loc, seq_lens,
            block_tables, num_decodes, num_decode_tokens, num_prefills,
            num_input_tokens, max_num_tokens_across_dp, with_prefill_across_dp,
            query_lens, head_dim, attn_mask, attn_state, decode, prefill)

        self.assertEqual(metadata.num_actual_tokens, num_actual_tokens)
        self.assertEqual(metadata.slot_mapping, slot_mapping)
        self.assertEqual(metadata.query_start_loc, query_start_loc)
        self.assertEqual(metadata.seq_lens, seq_lens)
        self.assertEqual(metadata.block_tables, block_tables)
        self.assertEqual(metadata.num_decodes, num_decodes)
        self.assertEqual(metadata.num_decode_tokens, num_decode_tokens)
        self.assertEqual(metadata.num_prefills, num_prefills)
        self.assertEqual(metadata.num_input_tokens, num_input_tokens)
        self.assertEqual(metadata.max_num_tokens_across_dp,
                         max_num_tokens_across_dp)
        self.assertEqual(metadata.with_prefill_across_dp,
                         with_prefill_across_dp)
        self.assertEqual(metadata.query_lens, query_lens)
        self.assertEqual(metadata.head_dim, head_dim)
        self.assertEqual(metadata.attn_mask, attn_mask)
        self.assertEqual(metadata.attn_state, attn_state)
        self.assertEqual(metadata.decode, decode)
        self.assertEqual(metadata.prefill, prefill)


class TestAscendMLAMetadataBuilder(TestBase):
    def test_ascend_mla_metadata_builder_default(self):
        runner = MagicMock()
        runner.scheduler_config = MagicMock()
        runner.model_config = MagicMock()
        runner.scheduler_config.max_num_seqs = 4
        runner.model_config.max_model_len = 1024
        runner.model_config.get_head_size.return_value = 64
        runner.model_config.dtype = torch.float16
        runner.chunked_prefill_enabled = False
        runner.device = "cpu"
        runner.block_size = 16

        ascend_config = MagicMock()
        ascend_config.torchair_graph_config = MagicMock()
        ascend_config.torchair_graph_config.enabled = True
        with patch("vllm_ascend.ascend_config.get_ascend_config",
                   return_value=ascend_config):
            builder = AscendMLAMetadataBuilder(runner)

            self.assertEqual(builder.runner, runner)
            self.assertEqual(builder.scheduler_config, runner.scheduler_config)
            self.assertEqual(builder.model_config, runner.model_config)
            self.assertEqual(builder.block_size, runner.block_size)
            self.assertEqual(builder.chunked_prefill_enabled,
                             runner.chunked_prefill_enabled)
            self.assertEqual(builder.torchair_graph_enabled, True)

    def test_reorder_batch_with_torchair_graph(self):
        ascend_config = MagicMock()
        runner = MagicMock()
        ascend_config.torchair_graph_config = MagicMock()
        ascend_config.torchair_graph_config.enabled = True  # 在使能torchair的情况下进行测试
        with patch("vllm_ascend.ascend_config.get_ascend_config",
                   return_value=ascend_config):
            builder = AscendMLAMetadataBuilder(runner)

        # 模拟 input_batch
        input_batch = MagicMock()
        input_batch.req_ids = [0, 1, 2, 3]

        # 模拟 scheduler_output
        scheduler_output = MagicMock()
        scheduler_output.num_scheduled_tokens = {0: 2, 1: 1, 2: 3, 3: 1}
        scheduler_output.scheduled_spec_decode_tokens = {
            0: [1],  # 2 - 1 = 1 → decode
            1: [],  # 1 - 0 = 1 → decode
            2: [1, 1],  # 3 - 2 = 1 → decode
            3: []  # 1 - 0 = 1 → decode
        }

        input_batch.swap_states = MagicMock()

        modified = builder.reorder_batch(input_batch, scheduler_output)

        # 所有请求都是 decode，不需要重排
        self.assertFalse(modified)
        self.assertEqual(builder._num_decodes, 4)
        self.assertEqual(builder._num_prefills, 0)
        self.assertEqual(builder._num_decode_tokens, 7)
        self.assertEqual(builder._num_prefill_tokens, 0)
        input_batch.swap_states.assert_not_called()

    def test_reorder_batch_without_torchair_graph(self):
        ascend_config = MagicMock()
        runner = MagicMock()
        ascend_config.torchair_graph_config = MagicMock()
        ascend_config.torchair_graph_config.enabled = False  # 在使能torchair的情况下进行测试
        with patch("vllm_ascend.ascend_config.get_ascend_config",
                   return_value=ascend_config):
            builder = AscendMLAMetadataBuilder(runner)

        input_batch = MagicMock()
        input_batch.req_ids = [0, 1, 2, 3]

        scheduler_output = MagicMock()
        scheduler_output.num_scheduled_tokens = {0: 1, 1: 3, 2: 1, 3: 2}
        scheduler_output.scheduled_spec_decode_tokens = {
            0: [],  # 1 → decode
            1: [1],  # 3 → prefill
            2: [],  # 1 → decode
            3: []  # 2 → prefill
        }

        input_batch.swap_states = MagicMock()

        modified = builder.reorder_batch(input_batch, scheduler_output)

        self.assertTrue(modified)
        self.assertEqual(builder._num_decodes, 2)
        self.assertEqual(builder._num_prefills, 2)
        self.assertEqual(builder._num_decode_tokens, 2)
        self.assertEqual(builder._num_prefill_tokens, 5)
        input_batch.swap_states.assert_called_once_with(1, 2)

    def test_get_graph_runner_block_tables_normal(self):
        runner = MagicMock()
        runner.graph_block_tables = torch.zeros(
            (8, 64), dtype=torch.int32)  # 用于构建builder的参数
        builder = AscendMLAMetadataBuilder(runner=runner)
        block_tables = torch.randint(0, 100, (3, 10), dtype=torch.int32)

        result = builder._get_graph_runner_block_tables(3, 10)
        self.assertEqual(result.shape[0], 3)
        self.assertEqual(result.shape[1], 64)
        self.assertEqual(result[:, :10], block_tables)

    def test_get_graph_runner_block_tables_truncated(self):
        runner = MagicMock()
        runner.graph_block_tables = torch.zeros(
            (8, 4), dtype=torch.int32)  # 用于构建builder的参数
        builder = AscendMLAMetadataBuilder(runner=runner)
        block_tables = torch.randint(0, 100, (3, 10), dtype=torch.int32)

        result = builder._get_graph_runner_block_tables(3, 10)
        self.assertEqual(result.shape[0], 3)
        self.assertEqual(result.shape[1], 4)
        self.assertEqual(result, block_tables[:, :4])

    def test_get_graph_runner_block_tables_from_numpy(self):
        runner = MagicMock()
        runner.graph_block_tables = np.zeros((8, 64), dtype=np.int32)
        builder = AscendMLAMetadataBuilder(runner=runner)

        block_tables = torch.randint(0, 100, (3, 10), dtype=torch.int32)

        result = builder._get_graph_runner_block_tables(3, block_tables)

        self.assertEqual(result.shape[0], 3)
        self.assertEqual(result.shape[1], 64)
        self.assertEqual(result[:, :10], block_tables)

    def test_build_dummy(self):
        # 需要首先构建一个builder 构建builder需要一个runner
        runner = MagicMock()
        runner.model_config = MagicMock()
        runner.device = "cpu"
        runner.graph_block_tables = torch.zeros((8, 64), dtype=torch.int32)
        runner.model_config.get_head_size.return_value = 64

        runner.attn_mask = torch.zeros((1, 1), dtype=torch.bool)
        runner.spec_attn_mask = torch.zeros((1, 1), dtype=torch.bool)

        builder = AscendMLAMetadataBuilder(
            runner=runner, metadata_cls=AscendMLAMetadata)  # 构建一个builder

        with patch.object(builder,
                          "_get_graph_runner_block_tables",
                          side_effect=lambda x, y: y):
            metadata = builder.build_dummy(3, 3)

        self.assertIsInstance(metadata, AscendMLAMetadata)
        self.assertEqual(metadata.num_input_tokens, 3)
        self.assertEqual(metadata.num_actual_tokens, 3)
        self.assertEqual(metadata.num_decodes, 1)
        self.assertEqual(metadata.num_decode_tokens, 1)
        self.assertEqual(metadata.num_prefills, 0)
        self.assertEqual(metadata.attn_state, AscendAttentionState.DecodeOnly)
        self.assertIsNone(metadata.prefill)
        self.assertIsInstance(metadata.decode, AscendMLADecodeMetadata)
        self.assertEqual(metadata.block_tables.shape[0], 3)
        self.assertEqual(metadata.block_tables.shape[1], 64)
        self.assertEqual(metadata.seq_lens.shape[0], 3)
        self.assertEqual(metadata.slot_mapping.shape[0], 3)
        self.assertEqual(metadata.query_start_loc.shape[0], 3)

    def test_build():  #! 后续完善
        # 模拟 runner
        runner = MagicMock()
        runner.device = "cpu"
        runner.attn_mask = torch.zeros((1, 1), dtype=torch.bool)

        # 创建 builder
        builder = AscendMLAMetadataBuilder(runner)

        # 模拟 common_attn_metadata
        common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=torch.tensor([0, 1, 2]))

        # 调用 build
        metadata = builder.build(
            num_reqs=3,
            num_actual_tokens=3,
            max_query_len=1,
            common_attn_metadata=common_attn_metadata,
        )

        # 验证返回类型
        assert isinstance(metadata, AscendMLAMetadata)

        # 验证字段
        assert metadata.num_actual_tokens == 3
        assert metadata.num_decodes == 2
        assert metadata.num_prefills == 1
        assert torch.equal(metadata.attn_mask, runner.attn_mask)
        assert metadata.prefill is None
        assert metadata.decode is None


class TestAscendMLAImpl(TestBase):
    def setUp(self):
        self.patcher1 = patch("vllm_ascend.ascend_config.get_ascend_config")
        self.patcher2 = patch(
            "vllm.distributed.get_tensor_model_parallel_world_size")
        self.patcher3 = patch("vllm.config.get_current_vllm_config")

        # 启动 patch 并保存 mock
        self.mock_ascend_config = self.patcher1.start()
        self.mock_tensor_parallel = self.patcher2.start()
        self.mock_vllm_config = self.patcher3.start()

        ascend_config = MagicMock()
        vllm_config = MagicMock()
        speculative_config = MagicMock()

        # 设置默认返回值
        self.mock_tensor_parallel.return_value = 2
        self.mock_ascend_config.return_value = ascend_config
        self.mock_vllm_config.return_value = vllm_config

        ascend_config.torchair_graph_config.enabled = True
        ascend_config.torchair_graph_config.enable_kv_nz = False
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config

        # 构建其他的参数
        num_heads = 256
        head_size = 1024
        scale = 0.1
        num_kv_heads = 8
        kv_cache_dtype = "auto"

        kv_a_layernorm = MagicMock()
        kv_a_layernorm.weight = torch.randn(96)
        kv_a_layernorm.variance_epsilon = 1e-6
        # 构建kwargs
        kwargs = {
            "q_lora_rank": 64,
            "kv_lora_rank": 32,
            "qk_nope_head_dim": 64,
            "qk_rope_head_dim": 32,
            "qk_head_dim": 96,
            "v_head_dim": 128,
            "rotary_emb": MagicMock(),
            "q_proj": MagicMock(),
            "kv_b_proj": MagicMock(),
            "o_proj": MagicMock(),
            "kv_a_proj_with_mqa": MagicMock(),
            "kv_a_layernorm": kv_a_layernorm,
        }

        self.impl = AscendMLAImpl(num_heads,
                                  head_size,
                                  scale,
                                  num_kv_heads,
                                  kv_cache_dtype=kv_cache_dtype,
                                  **kwargs)
        self.patcher1.stop()
        self.patcher2.stop()
        self.patcher3.stop()

    def test_init(self):
        self.assertEqual(self.impl.num_heads, 256)
        self.assertEqual(self.impl.head_size, 1024)
        self.assertEqual(self.impl.scale, 0.1)
        self.assertEqual(self.impl.num_kv_heads, 8)
        self.assertEqual(self.impl.kv_cache_dtype, "auto")
        self.assertEqual(self.impl.q_lora_rank, 64)
        self.assertEqual(self.impl.kv_lora_rank, 32)
        self.assertEqual(self.impl.qk_nope_head_dim, 64)
        self.assertEqual(self.impl.qk_rope_head_dim, 32)
        self.assertEqual(self.impl.qk_head_dim, 96)
        self.assertEqual(self.impl.v_head_dim, 128)
        self.assertIsNotNone(self.impl.rotary_emb)
        self.assertIsNotNone(self.impl.q_proj)
        self.assertIsNotNone(self.impl.kv_b_proj)
        self.assertIsNotNone(self.impl.o_proj)
        self.assertIsNotNone(self.impl.kv_a_proj_with_mqa)
        self.assertIsNotNone(self.impl.kv_a_layernorm)
        self.assertEqual(self.impl.num_queries_per_kv, 32)
        self.assertEqual(self.impl.tp_size, 2)
        self.assertEqual(self.impl.spec_token_num, 4)
        self.assertTrue(self.impl.torchair_graph_enabled)
        self.assertFalse(self.impl.enable_kv_nz)

    def test_v_up_proj_and_o_proj(self):
        batch_size = 4
        x = torch.randn(batch_size, self.impl.num_heads,
                        self.impl.kv_lora_rank)

        self.impl.o_proj.return_value = (torch.randn(
            batch_size, self.impl.num_heads * self.impl.v_head_dim), )
        if not hasattr(self.impl, 'W_UV') or self.impl.W_UV is None:
            self.impl.W_UV = torch.randn(self.impl.num_heads,
                                         self.impl.kv_lora_rank,
                                         self.impl.v_head_dim)

        result = self.impl._v_up_proj_and_o_proj(x)

        self.assertEqual(result.shape[0], batch_size)
        self.assertEqual(result.shape[1],
                         self.impl.kv_lora_rank * self.impl.v_head_dim)

    def test_q_proj_and_k_up_proj(self):
        batch_size = 4
        x = torch.randn(batch_size, self.impl.num_heads, self.impl.qk_head_dim)
        q_proj_output = torch.randn(batch_size, self.impl.num_heads,
                                    self.impl.qk_head_dim)
        self.impl.q_proj.return_value = (q_proj_output, )
        # q_nope, q_pe [bs num_heads nope_dim] [bs num_heads rope_dim]
        if not hasattr(self.impl, 'W_UK_T') or self.impl.W_UK_T is None:
            self.impl.W_UK_T = torch.randn(self.impl.num_heads,
                                           self.impl.qk_nope_head_dim,
                                           self.impl.kv_lora_rank)
        result = self.impl._q_proj_and_k_up_proj(x)
        ql_nope, q_pe = result
        self.assertEqual(ql_nope.shape[0], batch_size)
        self.assertEqual(ql_nope.shape[1], self.impl.num_heads)
        self.assertEqual(ql_nope.shape[2], self.impl.kv_lora_rank)
        self.assertEqual(q_pe.shape[0], batch_size)
        self.assertEqual(q_pe.shape[1], self.impl.num_heads)
        self.assertEqual(q_pe.shape[2], self.impl.qk_rope_head_dim)

    def test_process_weights_after_loading(self):
        weight = torch.randn(
            self.impl.kv_lora_rank,
            self.impl.num_heads *
            (self.impl.v_head_dim + self.impl.qk_nope_head_dim))
        layer = MagicMock(spec=LinearBase)
        layer.quant_method = type('FakeQuant', (), {})()  # 动态生成一个匿名类实例
        shape_0 = self.impl.num_heads * (self.impl.qk_nope_head_dim +
                                         self.impl.v_head_dim)
        shape_1 = self.impl.kv_lora_rank
        layer.weight = torch.randn(shape_0, shape_1)
        self.impl.process_weights_after_loading(torch.bfloat16)

        self.assertEqual(self.impl.W_UK_T.shape[0], self.impl.num_heads)
        self.assertEqual(self.impl.W_UK_T.shape[1], self.impl.qk_nope_head_dim)
        self.assertEqual(self.impl.W_UK_T.shape[2], self.impl.kv_lora_rank)

        self.assertEqual(self.impl.W_UV.shape[0], self.impl.num_heads)
        self.assertEqual(self.impl.W_UV.shape[1], self.impl.kv_lora_rank)
        self.assertEqual(self.impl.W_UV.shape[2], self.impl.v_head_dim)

    def test_compute_prefill_context_none(self):
        batch_size = 4
        kv_cache = torch.randn(1, 1, 1, 192)
        query = torch.randn(batch_size, self.impl.num_heads,
                            self.impl.qk_head_dim)
        metadata = MagicMock()
        metadata.prefill = None
        prefix_out = torch.randn(2, 16, 128)
        prefix_lse = torch.randn(2, 16, 8)
        out, lse = self.impl._compute_prefill_context(query, kv_cache, 32,
                                                      metadata, prefix_out,
                                                      prefix_lse)

        self.assertEqual(prefix_out, out)
        self.assertEqual(prefix_lse, lse)

    @patch("torch_npu.atb.npu_paged_cache_load")
    @patch("torch_npu.atb.npu_ring_mla")
    def test_compute_prefill_context(self, mock_ring, mock_load):
        B, N, D = 2, self.impl.num_heads, self.impl.qk_head_dim  # 这里的96是两个值 64 和 32的和 qk_head_dim
        query = torch.randn(B, N, D)  # [2 256 96]
        kv_cache = torch.randn(4, 8, N, D)  # [4 8 256 96]
        prefix_out = torch.randn(B, N, 128)  # [2 256 128]
        prefix_lse = torch.randn(B, N, 8)  # [2 256 8]

        self.impl.kv_b_proj.return_value = (torch.randn(8, N, 64 + 32),
                                            )  # [8, 256, 64 + 32]

        chunk_ctx = MagicMock()
        chunk_ctx.seq_tot = [8]
        chunk_ctx.chunk_seq_lens = [torch.tensor([8])]
        chunk_ctx.starts = [torch.tensor([0])]

        prefill_meta = MagicMock()
        prefill_meta.chunked_context = chunk_ctx
        prefill_meta.query_lens = [8]
        prefill_meta.block_table = torch.randint(0, 4, (B, 4))

        meta = MagicMock()
        meta.prefill = prefill_meta

        out, lse = self.impl._compute_prefill_context(query, kv_cache, 32,
                                                      meta, prefix_out,
                                                      prefix_lse)

        mock_load.assert_called_once()
        mock_ring.assert_called_once()

        self.assertEqual(out.shape, prefix_out.shape)
        self.assertEqual(lse.shape, prefix_lse.shape)

    @patch("torch_npu.npu_kv_rmsnorm_rope_cache")
    @patch("torch_npu.npu_stream_switch")
    def test_exec_kv(self, mock_stream, mock_kv_cache):
        batch_size = 2
        hidden = torch.randn(batch_size, 128)
        cos = torch.randn(batch_size, 32)
        sin = torch.randn(batch_size, 32)
        kv_cache = (torch.randn(
            4, 8, self.impl.kv_lora_rank + self.impl.qk_rope_head_dim),
                    torch.randn(
                        4, 8,
                        self.impl.kv_lora_rank + self.impl.qk_rope_head_dim))
        slots = torch.arange(batch_size, dtype=torch.long)

        # mock kv_a_proj_with_mqa
        proj_out = torch.randn(
            batch_size, self.impl.num_kv_heads, 1,
            self.impl.kv_lora_rank + self.impl.qk_rope_head_dim)
        self.impl.kv_a_proj_with_mqa.return_value = (proj_out, )

        # mock npu_kv_rmsnorm_rope_cache 返回值
        mock_kv_cache.return_value = (
            torch.randn(batch_size, self.impl.num_kv_heads, 1,
                        self.impl.qk_rope_head_dim),  # k_pe
            torch.randn(batch_size, self.impl.num_kv_heads, 1,
                        self.impl.kv_lora_rank),  # k_nope
            None,
            None)

        k_pe, k_nope = self.impl.exec_kv(hidden,
                                         cos,
                                         sin,
                                         kv_cache,
                                         slots,
                                         enable_multistream_mla=False)

        # 验证调用
        self.impl.kv_a_proj_with_mqa.assert_called_once_with(hidden)
        mock_kv_cache.assert_called_once()
        mock_stream.assert_called_once()

        # 验证形状
        self.assertEqual(k_pe.shape, (batch_size, self.impl.num_kv_heads, 1,
                                      self.impl.qk_rope_head_dim))
        self.assertEqual(
            k_nope.shape,
            (batch_size, self.impl.num_kv_heads, 1, self.impl.kv_lora_rank))

    @patch("torch_npu.npu_kv_rmsnorm_rope_cache")
    def test_exec_kv_prefill(self, mock_kv):
        B, S, H = 2, 16, 128
        hidden_states = torch.randn(B, S, H)
        cos = torch.randn(B, S, 32)
        sin = torch.randn(B, S, 32)
        kv_cache = (torch.randn(
            100, 8, self.impl.kv_lora_rank + self.impl.qk_rope_head_dim),
                    torch.randn(
                        100, 8,
                        self.impl.kv_lora_rank + self.impl.qk_rope_head_dim))

        slots = torch.arange(B * S, dtype=torch.long)

        # mock kv_a_proj_with_mqa
        proj_out = torch.randn(
            B, self.impl.num_kv_heads, S,
            self.impl.kv_lora_rank + self.impl.qk_rope_head_dim)
        self.impl.kv_a_proj_with_mqa.return_value = (proj_out, )

        # mock npu_kv_rmsnorm_rope_cache 返回值
        mock_kv.return_value = (
            None,
            None,
            torch.randn(B, self.impl.num_kv_heads, S,
                        self.impl.qk_rope_head_dim),  # k_pe
            torch.randn(B, self.impl.num_kv_heads, S,
                        self.impl.kv_lora_rank)  # k_nope
        )

        k_pe, k_nope = self.impl.exec_kv_prefill(hidden_states, cos, sin,
                                                 kv_cache, slots)

        # 断言调用
        self.impl.kv_a_proj_with_mqa.assert_called_once_with(hidden_states)
        mock_kv.assert_called_once()

        # 验证形状
        self.assertEqual(
            k_pe.shape,
            (B, self.impl.num_kv_heads, S, self.impl.qk_rope_head_dim))
        self.assertEqual(
            k_nope.shape,
            (B, self.impl.num_kv_heads, S, self.impl.kv_lora_rank))

    @patch("torch_npu.npu_interleave_rope")
    def test_rope_single(self, mock_rope):
        B, N, D = 2, 16, 1024
        x = torch.randn(B, N, D)
        cos = torch.randn(B, N, 1, D)
        sin = torch.randn(B, N, 1, D)
        mock_rope.return_value = x.view(B, N, 1, D)
        result = self.impl.rope_single(x, cos, sin)
        self.assertEqual(result.shape, (B, N, D))
        mock_rope.assert_called_once_with(x.view(B, N, 1, D), cos, sin)

    @patch("vllm_ascend.attention.mla_v1.AscendMLAImpl._v_up_proj_and_o_proj")
    @patch("torch_npu._npu_paged_attention_mla")
    @patch("vllm_ascend.multistream.context.get_multistream_comm_context")
    def test_forward_decode_without_graph(self, mock_stream_context,
                                          mock_page_attention_mla,
                                          mock_up_proj):
        self.impl.running_in_graph = False
        num_tokens = 100
        num_blocks = 256
        block_size = 4
        q_nope = torch.randn(num_tokens, self.impl.num_heads,
                             self.impl.qk_nope_head_dim)
        q_pe = torch.randn(num_tokens, self.impl.num_heads,
                           self.impl.qk_rope_head_dim)
        kv_c_and_k_pe_cache = torch.randn(num_blocks, block_size,
                                          self.impl.num_heads,
                                          self.impl.kv_lora_rank)
        metadata = MagicMock()
        metadata.decode = MagicMock()
        metadata.decode.block_table = MagicMock()
        metadata.decode.seq_lens = 10
        mock_stream_context.return_value = MagicMock()
        mock_page_attention_mla.return_value = torch.randn(
            num_tokens, self.impl.num_heads, self.impl.kv_lora_rank)
        mock_up_proj.return_value = torch.randn(num_tokens,
                                                self.impl.num_heads,
                                                self.impl.v_head_dim)
        result = self.impl._forward_decode(q_nope, q_pe, None, None,
                                           kv_c_and_k_pe_cache, metadata)
        self.assertEqual(result.shape[0], num_tokens)
        self.assertEqual(result.shape[1], self.impl.num_heads)
        self.assertEqual(result.shape[2], self.impl.v_head_dim)
        mock_up_proj.assert_called_once()
        mock_page_attention_mla.assert_called_once()
        mock_stream_context.assert_called_once()
