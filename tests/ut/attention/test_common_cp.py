import unittest
from unittest.mock import MagicMock, patch
import torch

from vllm_ascend.attention.context_parallel.common_cp import (
    _npu_attention_update,
    _npu_attn_out_lse_update,
    _update_out_and_lse,
    _out_lse_reshape
)

class TestCommonCP(unittest.TestCase):

    @patch('vllm_ascend.attention.context_parallel.common_cp.get_pcp_group')
    @patch('vllm_ascend.attention.context_parallel.common_cp.get_decode_context_model_parallel_world_size')
    @patch('vllm_ascend.attention.context_parallel.common_cp.get_dcp_group')
    @patch('torch.distributed.all_to_all_single')
    def test_process_attn_out_lse_complex(self, 
                                        mock_all2all, 
                                        mock_get_dcp_group, 
                                        mock_get_dcp_size, 
                                        mock_get_pcp_group):
        from vllm_ascend.attention.context_parallel.common_cp import _process_attn_out_lse

        pcp_size = 2
        dcp_size = 2
        mock_get_pcp_group.return_value.world_size = pcp_size
        mock_get_dcp_size.return_value = dcp_size
        
        mock_group = MagicMock()
        mock_get_dcp_group.return_value.device_group = mock_group

        bs, num_heads, head_dim = 4, 8, 64
        attn_output = torch.randn(bs, num_heads, head_dim, dtype=torch.float16)
        softmax_lse = torch.randn(bs, num_heads, 1, dtype=torch.float16)

        def fake_all_gather(tensor, dim=0):
            return torch.cat([tensor, tensor], dim=dim)
        mock_get_pcp_group.return_value.all_gather = fake_all_gather

        output = _process_attn_out_lse(attn_output, softmax_lse)

        self.assertEqual(output.dtype, torch.float32)

        # [4, 8, 64] + [4, 8, 1] -> [4, 8, 65] (Cat)
        # DCP All2All -> [4, 8, 65]
        # PCP AllGather (dim=0, size=2) -> [8, 8, 65]
        expected_shape = (bs * pcp_size, num_heads, head_dim + 1)
        self.assertEqual(output.shape, expected_shape)

        mock_all2all.assert_called_once()
        called_args = mock_all2all.call_args
        self.assertEqual(called_args.kwargs['group'], mock_group)

    @patch('vllm_ascend.attention.context_parallel.common_cp.get_pcp_group')
    @patch('vllm_ascend.attention.context_parallel.common_cp.get_decode_context_model_parallel_world_size')
    def test_process_attn_out_lse_simple(self, mock_get_dcp_size, mock_get_pcp_group):
        from vllm_ascend.attention.context_parallel.common_cp import _process_attn_out_lse
        
        mock_get_pcp_group.return_value.world_size = 1
        mock_get_dcp_size.return_value = 1
        
        attn_output = torch.randn(2, 4, 16)
        softmax_lse = torch.randn(2, 4, 1)
        
        output = _process_attn_out_lse(attn_output, softmax_lse)
        
        # concat: [2, 4, 16+1]
        self.assertEqual(output.shape, (2, 4, 17))

    @patch('torch_npu.npu_attention_update')
    def test_npu_attn_out_lse_update(self, mock_npu_attention_update):
        # Mock input data
        attn_lse_mask = torch.randn(8, 128, 1)
        attn_lse_nomask = torch.randn(8, 128, 1)
        attn_out_mask = torch.randn(8, 128, 128)
        attn_out_nomask = torch.randn(8, 128, 128)

        mock_npu_attention_update.return_value = (
            torch.randn(8 * 128, 128), 
            None
        )

        # Call the method under test
        output = _npu_attn_out_lse_update(
            attn_lse_mask,
            attn_lse_nomask,
            attn_out_mask,
            attn_out_nomask
        )

        # Assertions
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (8, 128, 128))
        mock_npu_attention_update.assert_called_once()

    def test_update_out_and_lse(self):
        # Mock input data: [N, batch, heads, head_size/1]
        out_list = torch.randn(3, 2, 4, 8)
        lse_list = torch.randn(3, 2, 4, 1)

        # Call the method under test
        out_final, lse_final = _update_out_and_lse(out_list, lse_list)

        # Assert shapes
        self.assertEqual(out_final.shape, (2, 4, 8))
        self.assertEqual(lse_final.shape, (2, 4, 1))
        self.assertIsInstance(out_final, torch.Tensor)
        self.assertIsInstance(lse_final, torch.Tensor)

    @patch('vllm_ascend.attention.context_parallel.common_cp.get_pcp_group')
    @patch('vllm_ascend.attention.context_parallel.common_cp.get_decode_context_model_parallel_world_size')
    @patch('torch_npu.npu_attention_update')
    def test_npu_attention_update(self, mock_npu_update, mock_get_dcp, mock_get_pcp):
        pcp_size = 2
        dcp_size = 2
        mock_get_pcp.return_value.world_size = pcp_size
        mock_get_dcp.return_value = dcp_size
        
        head_size = 64
        S, H = 4, 8 # Sequence and Heads per segment
        
        attn_out_lse = torch.randn(pcp_size * S, dcp_size * H, head_size + 1)
        
        # mock NPU op return (N=PCP*DCP=4, S*H=32) -> [4*32, 64]
        mock_npu_update.return_value = (torch.randn(pcp_size * dcp_size * S * H, head_size), None)
        
        # test
        result = _npu_attention_update(head_size, attn_out_lse)
        
        self.assertEqual(result.shape, (pcp_size * dcp_size * S, H, head_size))
        mock_npu_update.assert_called_once()

    def test_out_lse_reshape(self):
        # Mock input data
        out_list = torch.randn(3, 2, 4,
                               8)  # [N, batch_size, num_heads, head_size]
        lse_list = torch.randn(3, 2, 4, 1)  # [N, batch_size, num_heads, 1]

        # Call the method under test
        out_final, lse_final = _update_out_and_lse(out_list, lse_list)

        # Assert the method call
        self.assertEqual(out_final.shape,
                         (2, 4, 8))  # [batch_size, num_heads, head_size]
        self.assertEqual(lse_final.shape,
                         (2, 4, 1))  # [batch_size, num_heads, 1]

        self.assertIsInstance(out_final, torch.Tensor)
        self.assertIsInstance(lse_final, torch.Tensor)