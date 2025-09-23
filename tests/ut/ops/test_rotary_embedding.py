import math
import unittest
from unittest.mock import MagicMock, PropertyMock, patch

import torch
from transformers.configuration_utils import PretrainedConfig
from vllm.config import ModelConfig, VllmConfig
from vllm.model_executor.layers.rotary_embedding import (
    DeepseekScalingRotaryEmbedding, RotaryEmbedding)

from tests.ut.base import TestBase
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.ops.rotary_embedding import _custom_rotary_embedding_enabled

MODEL = "Qwen3-0.6B"
MAX_NUM_BATCHED_TOKEND = 10000


class TestCustomRotaryEmbeddingEnabled(unittest.TestCase):

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
            result = _custom_rotary_embedding_enabled(self.query, True,
                                                      self.head_size)
            self.assertTrue(result)

        # Test when dtype is not float16
        with patch('vllm_ascend.ops.rotary_embedding.enable_custom_op',
                   return_value=True):
            query = self.query.to(torch.float32)
            result = _custom_rotary_embedding_enabled(query, True,
                                                      self.head_size)
            self.assertFalse(result)

        # Test when neox_style is False
        with patch('vllm_ascend.ops.rotary_embedding.enable_custom_op',
                   return_value=True):
            result = _custom_rotary_embedding_enabled(self.query, False,
                                                      self.head_size)
            self.assertFalse(result)

        # Test when head_size is not divisible by 32
        with patch('vllm_ascend.ops.rotary_embedding.enable_custom_op',
                   return_value=True):
            result = _custom_rotary_embedding_enabled(self.query, True,
                                                      self.head_size + 1)
            self.assertFalse(result)

        # Test when custom op is disabled
        with patch('vllm_ascend.ops.rotary_embedding.enable_custom_op',
                   return_value=False):
            result = _custom_rotary_embedding_enabled(self.query, True,
                                                      self.head_size)
            self.assertFalse(result)


class TestAscendRotaryEmbedding(unittest.TestCase):

    def setUp(self):
        # Common setup for tests
        self.positions = torch.tensor([1, 2, 3])
        self.query = torch.randn(3, 1, 32, dtype=torch.float16)
        self.key = torch.randn(3, 1, 32, dtype=torch.float16)
        self.head_size = 32
        self.rotary_dim = self.head_size
        self.max_position = 16
        self.rope_theta = 10000
        self.is_neox_style = True
        self.cos_sin_cache = torch.randn(3, 1, 32)
        self.layer = RotaryEmbedding(self.head_size, self.rotary_dim,
                                     self.max_position, self.rope_theta,
                                     self.is_neox_style, torch.float16)

        # Mock self object for rope_forward_oot
        self.mock_self = MagicMock()
        self.mock_self.head_size = self.head_size
        self.mock_self.cos_sin_cache = self.cos_sin_cache
        self.mock_self.is_neox_style = self.is_neox_style

    @patch('torch.ops._C_ascend')
    @patch('vllm_ascend.ops.rotary_embedding.is_310p', return_value=False)
    @patch('vllm_ascend.ops.rotary_embedding._custom_rotary_embedding_enabled',
           return_value=True)
    @patch('torch.ops._npu_rotary_embedding')
    @patch('vllm.config.ModelConfig.__post_init__', MagicMock())
    @patch('vllm.config.VllmConfig.__post_init__', MagicMock())
    @patch('vllm.distributed.parallel_state._DP', MagicMock(world_size=1))
    @patch('vllm.distributed.parallel_state._TP', MagicMock(world_size=1))
    def test_rope_forward_oot_custom_kernel(self, mock_rotary_embedding,
                                            mock_custom_enabled, mock_is_310p,
                                            mock__c):
        mock_config = MagicMock()
        mock_config.torchair_graph_config.enabled = False

        # Setup mock for custom kernel path

        mock__c.rotary_embedding.return_value = self.query, self.key
        vllm_config = VllmConfig()
        model_config = ModelConfig(MODEL,
                                   tokenizer=MODEL,
                                   max_model_len=MAX_NUM_BATCHED_TOKEND)
        model_config.hf_config = PretrainedConfig()
        vllm_config.model_config = model_config
        with set_ascend_forward_context(None, vllm_config):
            result_q, result_k = self.layer.forward(self.positions, self.query,
                                                    self.key)

        mock__c.rotary_embedding.assert_called_once()
        self.assertEqual(result_q.shape, self.query.shape)
        self.assertEqual(result_k.shape, self.key.shape)

    @patch('vllm_ascend.ops.rotary_embedding._custom_rotary_embedding_enabled',
           return_value=False)
    @patch('torch_npu._npu_rotary_embedding')
    @patch('vllm.config.ModelConfig.__post_init__', MagicMock())
    @patch('vllm.config.VllmConfig.__post_init__', MagicMock())
    @patch('vllm.distributed.parallel_state._DP', MagicMock(world_size=1))
    @patch('vllm.distributed.parallel_state._TP', MagicMock(world_size=1))
    def test_rope_forward_oot_contiguous(self, mock_npu_rotary,
                                         mock_custom_enabled):
        mock_config = MagicMock()
        mock_config.torchair_graph_config.enabled = False

        # Test contiguous path when custom is disabled
        non_contig_query = self.query.transpose(0, 1)
        non_contig_key = self.key.transpose(0, 1)
        vllm_config = VllmConfig()
        model_config = ModelConfig(MODEL,
                                   tokenizer=MODEL,
                                   max_model_len=MAX_NUM_BATCHED_TOKEND)
        model_config.hf_config = PretrainedConfig()
        vllm_config.model_config = model_config
        with set_ascend_forward_context(None, vllm_config):
            result_q, result_k = self.layer.forward(self.positions,
                                                    non_contig_query,
                                                    non_contig_key)

        mock_npu_rotary.assert_called_once()
        self.assertEqual(result_q.shape, non_contig_query.shape)
        self.assertEqual(result_k.shape, non_contig_key.shape)

    @patch('vllm.config.ModelConfig.__post_init__', MagicMock())
    @patch('vllm.config.VllmConfig.__post_init__', MagicMock())
    @patch('vllm.distributed.parallel_state._DP', MagicMock(world_size=1))
    @patch('vllm.distributed.parallel_state._TP', MagicMock(world_size=1))
    def test_rope_forward_oot_with_offsets(self):
        mock_config = MagicMock()
        mock_config.torchair_graph_config.enabled = False

        # Test that NotImplementedError is raised when offsets is provided
        offsets = torch.tensor([1, 2, 3])
        with self.assertRaises(NotImplementedError):
            vllm_config = VllmConfig()
            model_config = ModelConfig(MODEL,
                                       tokenizer=MODEL,
                                       max_model_len=MAX_NUM_BATCHED_TOKEND)
            model_config.hf_config = PretrainedConfig()
            vllm_config.model_config = model_config
            with set_ascend_forward_context(None, vllm_config):
                self.layer.forward(self.positions, self.query, self.key,
                                   offsets)

    @patch('vllm_ascend.ops.rotary_embedding._custom_rotary_embedding_enabled',
           return_value=False)
    @patch('torch_npu._npu_rotary_embedding')
    @patch('vllm.config.ModelConfig.__post_init__', MagicMock())
    @patch('vllm.config.VllmConfig.__post_init__', MagicMock())
    @patch('vllm.distributed.parallel_state._DP', MagicMock(world_size=1))
    @patch('vllm.distributed.parallel_state._TP', MagicMock(world_size=1))
    def test_rope_forward_oot_neox_style_override(self, mock_npu_rotary,
                                                  mock_custom_enabled):
        mock_config = MagicMock()
        mock_config.torchair_graph_config.enabled = False

        # Test neox_style override
        vllm_config = VllmConfig()
        model_config = ModelConfig(MODEL,
                                   tokenizer=MODEL,
                                   max_model_len=MAX_NUM_BATCHED_TOKEND)
        model_config.hf_config = PretrainedConfig()
        vllm_config.model_config = model_config
        with set_ascend_forward_context(None, vllm_config):
            result_q, result_k = self.layer.forward(
                self.positions,
                self.query,
                self.key,
                is_neox_style_override=False)
        # Check that neox_style=False was passed to the NPU function
        args, kwargs = mock_npu_rotary.call_args
        self.assertFalse(args[-1])

    @patch('vllm_ascend.ops.rotary_embedding._custom_rotary_embedding_enabled',
           return_value=False)
    @patch('torch_npu._npu_rotary_embedding')
    @patch('vllm.config.ModelConfig.__post_init__', MagicMock())
    @patch('vllm.config.VllmConfig.__post_init__', MagicMock())
    @patch('vllm.distributed.parallel_state._DP', MagicMock(world_size=1))
    @patch('vllm.distributed.parallel_state._TP', MagicMock(world_size=1))
    def test_rope_forward_oot_rotary_dim_less_than_head_size(
            self, mock_npu_rotary, mock_custom_enabled):
        mock_config = MagicMock()
        mock_config.torchair_graph_config.enabled = False

        # test case when rotary_dim < head_size
        org_rotary_dim = self.layer.rotary_dim
        self.layer.rotary_dim = self.layer.head_size // 2

        vllm_config = VllmConfig()
        model_config = ModelConfig(MODEL,
                                   tokenizer=MODEL,
                                   max_model_len=MAX_NUM_BATCHED_TOKEND)
        model_config.hf_config = PretrainedConfig()
        vllm_config.model_config = model_config
        with set_ascend_forward_context(None, vllm_config):
            result_q, result_k = self.layer.forward(self.positions, self.query,
                                                    self.key)

        mock_npu_rotary.assert_called_once()
        self.assertEqual(result_q.shape, self.query.shape)
        self.assertEqual(result_k.shape, self.key.shape)

        # restore rotary_dim
        self.layer.rotary_dim = org_rotary_dim


class MockRopeModule:

    def __init__(self, max_seq_len=2048, is_neox_style=True):
        self.max_seq_len = max_seq_len
        self.is_neox_style = is_neox_style
        self.cos_cached = None
        self.sin_cached = None
        self.rotary_dim = 1
        self.base = 1


class TestAscendDeepseekScalingRotaryEmbedding(TestBase):

    def setUp(self):
        # Common setup for tests
        self.positions = torch.tensor([1, 2, 3])
        self.query = torch.randn(3, 1, 32, dtype=torch.float16)
        self.key = torch.randn(3, 1, 32, dtype=torch.float16)
        self.head_size = 32
        self.rotary_dim = self.head_size
        self.max_position = 16
        self.rope_theta = 10000
        self.is_neox_style = True
        self.scaling_factor = 1
        self.layer = None

    def _create_layer(self):
        self.layer = DeepseekScalingRotaryEmbedding(
            self.head_size, self.rotary_dim, self.max_position,
            self.rope_theta, self.is_neox_style, self.scaling_factor,
            torch.float16)
        return self.layer

    @patch("vllm.platforms.current_platform.device_type",
           new=torch.device("cpu"))
    @patch("vllm_ascend.ops.rotary_embedding.NPUPlatform",
           new_callable=PropertyMock)
    def test_native_rope_deepseek_forward_base(self, mock_npuplatform):
        mock_npuplatform.device_type = torch.device("cpu")
        self.layer = self._create_layer()
        with patch("vllm_ascend.ops.rotary_embedding._rope_forward_oot",
                   return_value=(self.query,
                                 self.key)) as mock_rope_forward_oot:
            q_pe, k_pe = self.layer.forward(self.positions, self.query,
                                            self.key)
        mock_rope_forward_oot.assert_called_once()
        assert q_pe.shape == self.query.shape
        assert k_pe.shape == self.key.shape

    @patch('vllm_ascend.ops.rotary_embedding._rope_forward_oot')
    @patch("vllm.platforms.current_platform.device_type",
           new=torch.device("cpu"))
    @patch("vllm_ascend.ops.rotary_embedding.NPUPlatform",
           new_callable=PropertyMock)
    def test_native_rope_deepseek_forward_key_reshaping(
            self, mock_npuplatform, mock_rope_forward_oot):
        mock_npuplatform.device_type = torch.device("cpu")
        self.layer = self._create_layer()

        key = torch.randn(1, 32)

        mock_rope_forward_oot.return_value = (self.query, key)

        q_pe, k_pe = self.layer.forward(self.positions, self.query, key)
        mock_rope_forward_oot.assert_called_once()
        assert q_pe.shape == self.query.shape
        assert k_pe.shape == key.shape

    @patch('vllm_ascend.ops.rotary_embedding._rope_forward_oot')
    @patch("vllm.platforms.current_platform.device_type",
           new=torch.device("cpu"))
    @patch("vllm_ascend.ops.rotary_embedding.NPUPlatform",
           new_callable=PropertyMock)
    def test_native_rope_deepseek_forward_non_neox_style(
            self, mock_npuplatform, mock_rope_forward_oot):
        mock_npuplatform.device_type = torch.device("cpu")
        self.layer = self._create_layer()

        mock_rope_forward_oot.return_value = (self.query, self.key)

        q_pe, k_pe = self.layer.forward(self.positions, self.query, self.key)

        mock_rope_forward_oot.assert_called_once()
        assert q_pe.shape == self.query.shape
        assert k_pe.shape == self.key.shape

    @patch("vllm.platforms.current_platform.device_type",
           new=torch.device("cpu"))
    @patch("vllm_ascend.ops.rotary_embedding.NPUPlatform",
           new_callable=PropertyMock)
    def test_basic_case(self, mock_npuplatform):
        # Test with standard values
        mock_npuplatform.device_type = torch.device("cpu")
        self.layer = self._create_layer()
        num_rotations = 100
        dim = 512
        base = 10000
        max_position_embeddings = 2048

        result = self.layer._yarn_find_correction_dim(num_rotations, dim, base,
                                                      max_position_embeddings)

        # Calculate expected value manually
        expected = (dim * torch.log(
            torch.tensor(max_position_embeddings) /
            (num_rotations * 2 * torch.pi))) / (2 *
                                                torch.log(torch.tensor(base)))

        self.assertTrue(torch.allclose(result, expected))

    @patch("vllm.platforms.current_platform.device_type",
           new=torch.device("cpu"))
    @patch("vllm_ascend.ops.rotary_embedding.NPUPlatform",
           new_callable=PropertyMock)
    def test_yarn_get_mscale(self, mock_npuplatform):
        mock_npuplatform.device_type = torch.device("cpu")
        self.layer = self._create_layer()

        # test_scale_less_than_or_equal_1
        self.assertEqual(self.layer._yarn_get_mscale(scale=0.5), 1.0)
        self.assertEqual(self.layer._yarn_get_mscale(scale=1.0), 1.0)
        self.assertEqual(self.layer._yarn_get_mscale(scale=0.999), 1.0)

        # test_scale_greater_than_1:
        test_cases = [(2.0, 1.0, 1.0 + 0.1 * math.log(2.0)),
                      (10.0, 1.0, 1.0 + 0.1 * math.log(10.0)),
                      (5.0, 2.0, 1.0 + 0.2 * math.log(5.0)),
                      (math.e, 1.0, 1.0 + 0.1)]

        for scale, mscale, expected in test_cases:
            result = self.layer._yarn_get_mscale(scale, mscale)
            self.assertAlmostEqual(
                result,
                expected,
                places=6,
                msg=f"Failed for scale={scale}, mscale={mscale}")
