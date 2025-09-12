from unittest.mock import MagicMock, patch

import torch
from vllm.attention.backends.abstract import AttentionType
from vllm.distributed.parallel_state import GroupCoordinator

from tests.ut.base import TestBase
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.torchair.torchair_attention import \
    AscendAttentionTorchairBackendImpl


class TestAscendAttentionTorchairBackendImpl(TestBase):

    @patch("torch.zeros")
    @patch('vllm.distributed.parallel_state._TP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))  # TODO
    @patch("vllm.distributed.get_tensor_model_parallel_world_size",
           return_value=2)  # TODO
    @patch("vllm.config.get_current_vllm_config")  # TODO
    @patch("vllm_ascend.torchair.torchair_mla.get_ascend_config")  # TODO
    def setUp(self, ascend_config, vllm_config, mock_get_tp_size, mock_tp,
              mock_zeros):
        mock_tp.world_size = 2  # TODO
        ascend_config.torchair_graph_config.enabled = True  # TODO
        ascend_config.torchair_graph_config.enable_kv_nz = False  # TODO
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config

        num_heads = 32
        head_size = 128  # TODO
        scale = 0.1  # TODO
        num_kv_heads = 4
        kv_cache_dtype = "auto"
        attn_type = AttentionType.DECODER
        mock_zeros.return_value = torch.ones((),
                                             device='cpu',
                                             dtype=torch.int32)

        self.impl = AscendAttentionTorchairBackendImpl(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype=kv_cache_dtype,
            blocksparse_params=None,
            logits_soft_cap=None,
            attn_type=attn_type,
            kv_sharing_target_layer_name=None)

    @patch("torch_npu.npu_scatter_nd_update_")
    @patch("torch_npu.npu_fused_infer_attention_score")
    def test_forward_with_decode_only(self, mock_fused, _):
        layer = MagicMock()
        layer._k_scale_float = 1.0
        layer._v_scale_float = 1.0

        seq_len = 1
        num_tokens = 100
        num_blocks = 256
        block_size = 4

        query = torch.randn(num_tokens, seq_len,
                            self.impl.num_heads * self.impl.head_size)
        key = torch.randn(num_tokens, seq_len,
                          self.impl.num_kv_heads * self.impl.head_size)
        value = torch.randn(num_tokens, seq_len,
                            self.impl.num_kv_heads * self.impl.head_size)
        kv_cache = (torch.randn(num_blocks, block_size,
                                self.impl.num_heads * self.impl.head_size),
                    torch.randn(num_blocks, block_size,
                                self.impl.num_heads * self.impl.head_size))
        output = torch.randn(num_tokens, self.impl.num_heads,
                             self.impl.head_size)

        decode = MagicMock()  # TODO
        decode.seq_lens_list = [2] * num_tokens
        decode.block_table = torch.ones(num_tokens, 8, dtype=torch.int32)
        decode.attn_mask = None

        metadata = MagicMock()
        metadata.attn_state = AscendAttentionState.DecodeOnly
        metadata.slot_mapping = torch.arange(num_tokens, dtype=torch.int32)
        metadata.decode = decode

        mock_fused.return_value = (torch.ones(num_tokens, self.impl.num_heads,
                                              self.impl.head_size),
                                   torch.ones(1))

        result = self.impl.forward(layer, query, key, value, kv_cache,
                                   metadata, output, False)
        self.assertEqual(result.shape[0], num_tokens)
