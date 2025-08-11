import math
from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.ops.rotary_embedding import (__set_cos_sin_cache,
                                              custom_rotary_embedding_enabled,
                                              native_rope_deepseek_forward,
                                              rope_forward_oot, rotate_half,
                                              yarn_find_correction_dim,
                                              yarn_get_mscale)


class TestCustomRotaryEmbeddingEnabled(TestBase):

    def setUp(self):
        # Common setup for tests
        self.positions = torch.tensor([1, 2, 3])
        self.query = torch.randn(3, 4, dtype=torch.float16)
        self.key = torch.randn(3, 4, dtype=torch.float16)
        self.head_size = 32
        self.cos_sin_cache = torch.randn(3, 4)

        # Mock self object for rope_forward_oot
        self.mock_self = MagicMock()
        self.mock_self.head_size = self.head_size
        self.mock_self.cos_sin_cache = self.cos_sin_cache
        self.mock_self.is_neox_style = True
        self.mock_self.forward_native.return_value = (self.query, self.key)

    def test_custom_rotary_embedding_enabled(self):
        # Test when all conditions are True
        with patch('vllm_ascend.ops.rotary_embedding.enable_custom_op',
                   return_value=True):
            result = custom_rotary_embedding_enabled(self.query, True,
                                                     self.head_size)
            self.assertTrue(result)

        # Test when dtype is not float16
        with patch('vllm_ascend.ops.rotary_embedding.enable_custom_op',
                   return_value=True):
            query = self.query.to(torch.float32)
            result = custom_rotary_embedding_enabled(query, True,
                                                     self.head_size)
            self.assertFalse(result)

        # Test when neox_style is False
        with patch('vllm_ascend.ops.rotary_embedding.enable_custom_op',
                   return_value=True):
            result = custom_rotary_embedding_enabled(self.query, False,
                                                     self.head_size)
            self.assertFalse(result)

        # Test when head_size is not divisible by 32
        with patch('vllm_ascend.ops.rotary_embedding.enable_custom_op',
                   return_value=True):
            result = custom_rotary_embedding_enabled(self.query, True,
                                                     self.head_size + 1)
            self.assertFalse(result)

        # Test when custom op is disabled
        with patch('vllm_ascend.ops.rotary_embedding.enable_custom_op',
                   return_value=False):
            result = custom_rotary_embedding_enabled(self.query, True,
                                                     self.head_size)
            self.assertFalse(result)


class TestRopeForwardOot(TestBase):

    def setUp(self):
        # Common setup for tests
        self.positions = torch.tensor([1, 2, 3])
        self.query = torch.randn(3, 4, dtype=torch.float16)
        self.key = torch.randn(3, 4, dtype=torch.float16)
        self.head_size = 32
        self.cos_sin_cache = torch.randn(3, 4)

        # Mock self object for rope_forward_oot
        self.mock_self = MagicMock()
        self.mock_self.head_size = self.head_size
        self.mock_self.cos_sin_cache = self.cos_sin_cache
        self.mock_self.is_neox_style = True
        self.mock_self.forward_native.return_value = (self.query, self.key)

    @patch('vllm_ascend.ops.rotary_embedding.get_ascend_config')
    def test_rope_forward_oot_torchair_enabled_base(self,
                                                    mock_get_ascend_config):
        # Setup mock for torchair enabled
        mock_config = MagicMock()
        mock_config.torchair_graph_config.enabled = True
        mock_get_ascend_config.return_value = mock_config

        result_q, result_k = rope_forward_oot(self.mock_self, self.positions,
                                              self.query, self.key)

        self.mock_self.forward_native.assert_called_once_with(
            self.positions, self.query, self.key, None)
        self.assertTrue(torch.equal(result_q, self.query))
        self.assertTrue(torch.equal(result_k, self.key))

    @patch('torch.ops._C')
    @patch('vllm_ascend.ops.rotary_embedding.get_ascend_config')
    @patch('vllm_ascend.ops.rotary_embedding.is_310p', return_value=False)
    @patch('vllm_ascend.ops.rotary_embedding.custom_rotary_embedding_enabled',
           return_value=True)
    @patch('torch.ops._npu_rotary_embedding')
    def test_rope_forward_oot_custom_kernel(self, mock_rotary_embedding,
                                            mock_custom_enabled, mock_is_310p,
                                            mock_get_ascend_config, mock__c):
        mock_config = MagicMock()
        mock_config.torchair_graph_config.enabled = False
        mock_get_ascend_config.return_value = mock_config

        # Setup mock for custom kernel path

        mock__c.rotary_embedding.return_value = self.query, self.key

        result_q, result_k = rope_forward_oot(self.mock_self, self.positions,
                                              self.query, self.key)

        self.assertEqual(result_q.shape, self.query.shape)
        self.assertEqual(result_k.shape, self.key.shape)

    @patch('vllm_ascend.ops.rotary_embedding.get_ascend_config')
    @patch('vllm_ascend.ops.rotary_embedding.custom_rotary_embedding_enabled',
           return_value=False)
    @patch('torch_npu._npu_rotary_embedding')
    def test_rope_forward_oot_contiguous(self, mock_npu_rotary,
                                         mock_custom_enabled,
                                         mock_get_ascend_config):
        mock_config = MagicMock()
        mock_config.torchair_graph_config.enabled = False
        mock_get_ascend_config.return_value = mock_config

        # Test contiguous path when custom is disabled
        non_contig_query = self.query.transpose(0, 1)
        non_contig_key = self.key.transpose(0, 1)

        result_q, result_k = rope_forward_oot(self.mock_self, self.positions,
                                              non_contig_query, non_contig_key)

        mock_npu_rotary.assert_called_once()
        self.assertEqual(result_q.shape, non_contig_query.shape)
        self.assertEqual(result_k.shape, non_contig_key.shape)

    @patch('vllm_ascend.ops.rotary_embedding.get_ascend_config')
    def test_rope_forward_oot_with_offsets(self, mock_get_ascend_config):
        mock_config = MagicMock()
        mock_config.torchair_graph_config.enabled = False
        mock_get_ascend_config.return_value = mock_config

        # Test that NotImplementedError is raised when offsets is provided
        offsets = torch.tensor([1, 2, 3])
        with self.assertRaises(NotImplementedError):
            rope_forward_oot(self.mock_self, self.positions, self.query,
                             self.key, offsets)

    @patch('vllm_ascend.ops.rotary_embedding.get_ascend_config')
    @patch('vllm_ascend.ops.rotary_embedding.custom_rotary_embedding_enabled',
           return_value=False)
    @patch('torch_npu._npu_rotary_embedding')
    def test_rope_forward_oot_neox_style_override(self, mock_npu_rotary,
                                                  mock_custom_enabled,
                                                  mock_get_ascend_config):
        mock_config = MagicMock()
        mock_config.torchair_graph_config.enabled = False
        mock_get_ascend_config.return_value = mock_config

        # Test neox_style override
        result_q, result_k = rope_forward_oot(self.mock_self,
                                              self.positions,
                                              self.query,
                                              self.key,
                                              is_neox_style_override=False)

        # Check that neox_style=False was passed to the NPU function
        args, kwargs = mock_npu_rotary.call_args
        self.assertFalse(args[-1])


class MockRopeModule:

    def __init__(self, max_seq_len=2048, is_neox_style=True):
        self.max_seq_len = max_seq_len
        self.is_neox_style = is_neox_style
        self.cos_cached = None
        self.sin_cached = None
        self.rotary_dim = 1
        self.base = 1


class TestNativeRopeDeepseekForward(TestBase):

    @patch('vllm_ascend.ops.rotary_embedding.rope_forward_oot')
    def test_native_rope_deepseek_forward_base(self, mock_rope_forward_oot):
        module = MockRopeModule()
        positions = torch.tensor([1, 2, 3])
        query = torch.randn(1, 8, 128)
        key = torch.randn(1, 8, 128)

        mock_rope_forward_oot.return_value = (query, key)

        q_pe, k_pe = native_rope_deepseek_forward(module, positions, query,
                                                  key)

        assert q_pe.shape == query.shape
        assert k_pe.shape == key.shape

    @patch('vllm_ascend.ops.rotary_embedding._set_cos_sin_cache')
    @patch('vllm_ascend.ops.rotary_embedding.rope_forward_oot')
    def test_native_rope_deepseek_forward_cache_handling(
            self, mock_rope_forward_oot, mock_set_cache):
        # Test cache situation is true
        module = MockRopeModule(max_seq_len=1024)
        positions = torch.tensor([1, 2, 3])
        query = torch.randn(1, 8, 128)
        key = torch.randn(1, 8, 128)

        mock_rope_forward_oot.return_value = (query, key)

        q_pe, k_pe = native_rope_deepseek_forward(module,
                                                  positions,
                                                  query,
                                                  key,
                                                  max_seq_len=2048)

        assert q_pe.shape == query.shape
        assert k_pe.shape == key.shape

    @patch('vllm_ascend.ops.rotary_embedding.rope_forward_oot')
    def test_native_rope_deepseek_forward_key_reshaping(
            self, mock_rope_forward_oot):
        module = MockRopeModule()
        positions = torch.tensor([1, 2, 3])
        query = torch.randn(1, 8, 128)
        key = torch.randn(1, 128)

        mock_rope_forward_oot.return_value = (query, key)

        q_pe, k_pe = native_rope_deepseek_forward(module, positions, query,
                                                  key)

        assert q_pe.shape == query.shape
        assert k_pe.shape == (1, 128)

    @patch('vllm_ascend.ops.rotary_embedding.rope_forward_oot')
    def test_native_rope_deepseek_forward_non_neox_style(
            self, mock_rope_forward_oot):
        module = MockRopeModule(is_neox_style=False)
        positions = torch.tensor([1, 2, 3])
        query = torch.randn(1, 8, 128)
        key = torch.randn(1, 8, 128)

        mock_rope_forward_oot.return_value = (query, key)

        q_pe, k_pe = native_rope_deepseek_forward(module, positions, query,
                                                  key)

        assert q_pe.shape == query.shape
        assert k_pe.shape == key.shape


class TestRotateHalf(TestBase):

    def test_rotate_half_even_dim(self):
        # Test with even dimension
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        expected = torch.tensor([-3.0, -4.0, 1.0, 2.0])
        result = rotate_half(x)
        self.assertTrue(torch.allclose(result, expected))


class TestYarnFindCorrectionDim(TestBase):

    def test_basic_case(self):
        # Test with standard values
        num_rotations = 100
        dim = 512
        base = 10000
        max_position_embeddings = 2048

        result = yarn_find_correction_dim(num_rotations, dim, base,
                                          max_position_embeddings)

        # Calculate expected value manually
        expected = (dim * torch.log(
            torch.tensor(max_position_embeddings) /
            (num_rotations * 2 * torch.pi))) / (2 *
                                                torch.log(torch.tensor(base)))

        self.assertTrue(torch.allclose(result, expected))


class TestYarnGetMscale(TestBase):

    def test_scale_less_than_or_equal_1(self):
        self.assertEqual(yarn_get_mscale(scale=0.5), 1.0)
        self.assertEqual(yarn_get_mscale(scale=1.0), 1.0)
        self.assertEqual(yarn_get_mscale(scale=0.999), 1.0)

    def test_scale_greater_than_1(self):
        test_cases = [(2.0, 1.0, 1.0 + 0.1 * math.log(2.0)),
                      (10.0, 1.0, 1.0 + 0.1 * math.log(10.0)),
                      (5.0, 2.0, 1.0 + 0.2 * math.log(5.0)),
                      (math.e, 1.0, 1.0 + 0.1)]

        for scale, mscale, expected in test_cases:
            result = yarn_get_mscale(scale, mscale)
            self.assertAlmostEqual(
                result,
                expected,
                places=6,
                msg=f"Failed for scale={scale}, mscale={mscale}")


class MockRotaryEmbedding(torch.nn.Module):

    def __init__(self, base, rotary_dim, max_position_embeddings):
        super().__init__()

        self.base = base

        self.rotary_dim = rotary_dim

        self.max_position_embeddings = max_position_embeddings

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        return __set_cos_sin_cache(self, seq_len, device, dtype)


class TestSetCosSinCache(TestBase):

    def test_set_cos_sin_cache(self):
        # prepare an instance with reasonable values
        base = 10000.0
        rotary_dim = 4
        max_pos = 10
        model = MockRotaryEmbedding(base, rotary_dim, max_pos)
        # mock out register_buffer
        model.register_buffer = MagicMock()
        # call the private method via name mangling
        model._set_cos_sin_cache(seq_len=8,
                                 device="cpu",
                                 dtype=torch.float32)
        # expect three calls: inv_freq, cos, sin
        assert model.register_buffer.call_count == 3
        names = [call.args[0] for call in model.register_buffer.call_args_list]
        assert set(names) == {"inv_freq", "cos", "sin"}
        # verify inv_freq shape
        inv_freq = model.register_buffer.call_args_list[0].args[1]
        assert isinstance(inv_freq, torch.Tensor)
        assert inv_freq.shape == (rotary_dim // 2, )
        # verify cos buffer
        cos = model.register_buffer.call_args_list[1].args[1]
        assert isinstance(cos, torch.Tensor)
        assert cos.shape == (max_pos, rotary_dim)
        assert cos.dtype == torch.float32
        # verify sin buffer
        sin = model.register_buffer.call_args_list[2].args[1]
        assert isinstance(sin, torch.Tensor)
        assert sin.shape == (max_pos, rotary_dim)
        assert sin.dtype == torch.float32
