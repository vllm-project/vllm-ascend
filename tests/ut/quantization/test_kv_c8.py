import unittest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch


class TestWeightLoader(unittest.TestCase):
    """Test cases for weight_loader function in kv_c8.py"""

    def setUp(self):
        """Set up test environment before each test"""
        # Import the module under test
        from vllm_ascend.quantization.methods.kv_c8 import weight_loader
        self.weight_loader = weight_loader

        # Mock distributed functions
        self.tp_rank_patch = patch(
            "vllm_ascend.quantization.methods.kv_c8.get_tensor_model_parallel_rank"
        )
        self.tp_size_patch = patch(
            "vllm_ascend.quantization.methods.kv_c8.get_tensor_model_parallel_world_size"
        )
        self.mock_tp_rank = self.tp_rank_patch.start()
        self.mock_tp_size = self.tp_size_patch.start()

    def tearDown(self):
        """Clean up after each test"""
        self.tp_rank_patch.stop()
        self.tp_size_patch.stop()

    def test_weight_loader_single_element(self):
        """Test weight_loader when both tensors contain a single element"""
        # Create tensors with single element
        param = torch.tensor([0.0])
        loaded_weight = torch.tensor([5.0])

        # Call weight_loader
        self.weight_loader(param, loaded_weight)

        # Verify the value was filled correctly
        self.assertEqual(param.item(), 5.0)
        self.assertEqual(param.dtype, torch.float32)

    def test_weight_loader_single_element_int(self):
        """Test weight_loader with integer tensors"""
        param = torch.tensor([0], dtype=torch.int32)
        loaded_weight = torch.tensor([10], dtype=torch.int32)

        self.weight_loader(param, loaded_weight)

        self.assertEqual(param.item(), 10)

    def test_weight_loader_tp_sharding_first_rank(self):
        """Test weight_loader with tensor parallelism sharding for first rank"""
        # Configure mocks for rank 0 of 4
        self.mock_tp_rank.return_value = 0
        self.mock_tp_size.return_value = 4

        # Create test tensors
        param = torch.zeros(2, 5)  # Target param shape (2,5)
        loaded_weight = torch.ones(8, 5)  # Full weight (8,5)

        # Mock narrow to track the call
        with patch.object(loaded_weight, 'narrow', wraps=loaded_weight.narrow) as mock_narrow:
            self.weight_loader(param, loaded_weight)

            # Verify narrow was called correctly: narrow(dim=0, start=0, length=2)
            mock_narrow.assert_called_once_with(0, 0, 2)

            # Verify data was copied
            self.assertTrue(torch.all(param == 1))

    def test_weight_loader_tp_sharding_middle_rank(self):
        """Test weight_loader with tensor parallelism sharding for middle rank"""
        # Configure mocks for rank 2 of 4
        self.mock_tp_rank.return_value = 2
        self.mock_tp_size.return_value = 4

        param = torch.zeros(2, 5)
        loaded_weight = torch.ones(8, 5)

        with patch.object(loaded_weight, 'narrow', wraps=loaded_weight.narrow) as mock_narrow:
            self.weight_loader(param, loaded_weight)

            # Verify narrow was called correctly: start = shard_size * rank = 2 * 2 = 4
            mock_narrow.assert_called_once_with(0, 4, 2)

            self.assertTrue(torch.all(param == 1))

    def test_weight_loader_tp_sharding_last_rank(self):
        """Test weight_loader with tensor parallelism sharding for last rank"""
        # Configure mocks for rank 3 of 4
        self.mock_tp_rank.return_value = 3
        self.mock_tp_size.return_value = 4

        param = torch.zeros(2, 5)
        loaded_weight = torch.ones(8, 5)

        with patch.object(loaded_weight, 'narrow', wraps=loaded_weight.narrow) as mock_narrow:
            self.weight_loader(param, loaded_weight)

            # Verify narrow was called correctly: start = 2 * 3 = 6
            mock_narrow.assert_called_once_with(0, 6, 2)

            self.assertTrue(torch.all(param == 1))

    def test_weight_loader_shape_mismatch(self):
        """Test weight_loader raises assertion error on shape mismatch"""
        self.mock_tp_rank.return_value = 0
        self.mock_tp_size.return_value = 2

        param = torch.zeros(2, 3)
        loaded_weight = torch.ones(4, 4)  # Different shape after sharding

        # Mock narrow to return tensor with wrong shape
        with patch.object(loaded_weight, 'narrow', return_value=torch.ones(2, 4)):
            with self.assertRaises(AssertionError) as context:
                self.weight_loader(param, loaded_weight)

            # Verify error message contains expected information
            self.assertIn("Attempted to load weight", str(context.exception))
            self.assertIn("into parameter", str(context.exception))

    def test_weight_loader_with_different_dtypes(self):
        """Test weight_loader handles different dtypes correctly"""
        self.mock_tp_rank.return_value = 0
        self.mock_tp_size.return_value = 1  # No sharding

        param = torch.zeros(2, 3, dtype=torch.float32)
        loaded_weight = torch.ones(2, 3, dtype=torch.float16)

        self.weight_loader(param, loaded_weight)

        # Verify data was copied and converted
        self.assertTrue(torch.all(param == 1))
        self.assertEqual(param.dtype, torch.float32)


class TestAscendFAQuantAttentionMethodInit(unittest.TestCase):
    """Test cases for AscendFAQuantAttentionMethod initialization"""

    def setUp(self):
        """Set up test environment"""
        # Mock vllm_config
        self.config_patch = patch("vllm_ascend.quantization.methods.kv_c8.get_current_vllm_config")
        self.mock_get_config = self.config_patch.start()

        # Create mock config with attributes
        self.mock_config = Mock()
        self.mock_hf_config = Mock()
        self.mock_hf_config.kv_lora_rank = 128
        self.mock_hf_config.qk_rope_head_dim = 64
        self.mock_config.model_config.hf_config = self.mock_hf_config
        self.mock_get_config.return_value = self.mock_config

        # Import the class after patching
        from vllm_ascend.quantization.methods.kv_c8 import AscendFAQuantAttentionMethod
        self.method_class = AscendFAQuantAttentionMethod

    def tearDown(self):
        """Clean up after each test"""
        self.config_patch.stop()

    def test_init_with_full_config(self):
        """Test initialization when config has all attributes"""
        method = self.method_class()

        self.assertTrue(method.transpose_weight)
        self.assertFalse(method.printFlag)
        self.assertEqual(method.kv_lora_rank, 128)
        self.assertEqual(method.qk_rope_head_dim, 64)

    def test_init_without_kv_lora_rank(self):
        """Test initialization when config lacks kv_lora_rank"""
        delattr(self.mock_hf_config, "kv_lora_rank")

        method = self.method_class()

        self.assertEqual(method.kv_lora_rank, 0)
        self.assertEqual(method.qk_rope_head_dim, 64)

    def test_init_without_qk_rope_head_dim(self):
        """Test initialization when config lacks qk_rope_head_dim"""
        delattr(self.mock_hf_config, "qk_rope_head_dim")

        method = self.method_class()

        self.assertEqual(method.kv_lora_rank, 128)
        self.assertEqual(method.qk_rope_head_dim, 0)

    def test_init_without_both_attributes(self):
        """Test initialization when config lacks both attributes"""
        delattr(self.mock_hf_config, "kv_lora_rank")
        delattr(self.mock_hf_config, "qk_rope_head_dim")

        method = self.method_class()

        self.assertEqual(method.kv_lora_rank, 0)
        self.assertEqual(method.qk_rope_head_dim, 0)


class TestAscendFAQuantAttentionMethodCreateWeights(unittest.TestCase):
    """Test cases for create_weights method"""

    def setUp(self):
        """Set up test environment"""
        # Mock vllm_config
        self.config_patch = patch("vllm_ascend.quantization.methods.kv_c8.get_current_vllm_config")
        self.mock_get_config = self.config_patch.start()

        self.mock_config = Mock()
        self.mock_hf_config = Mock()
        self.mock_hf_config.kv_lora_rank = 128
        self.mock_hf_config.qk_rope_head_dim = 64
        self.mock_config.model_config.hf_config = self.mock_hf_config
        self.mock_get_config.return_value = self.mock_config

        # Import the class
        from vllm_ascend.quantization.methods.kv_c8 import AscendFAQuantAttentionMethod
        self.method_class = AscendFAQuantAttentionMethod

        # Mock torch functions
        self.default_dtype_patch = patch("torch.get_default_dtype", return_value=torch.float32)
        self.mock_default_dtype = self.default_dtype_patch.start()

        # Create a real nn.Module for testing
        self.layer = nn.Module()
        self.layer.num_heads = 32
        self.layer.num_kv_heads = 8

    def tearDown(self):
        """Clean up after each test"""
        self.config_patch.stop()
        self.default_dtype_patch.stop()

    def test_create_weights_adds_submodules(self):
        """Test that create_weights adds fa_q, fa_k, fa_v submodules"""
        method = self.method_class()

        with patch("torch.empty") as mock_empty:
            mock_empty.return_value = torch.zeros(1, 1)

            method.create_weights(self.layer)

            # Verify submodules were added
            self.assertTrue(hasattr(self.layer, "fa_q"))
            self.assertTrue(hasattr(self.layer, "fa_k"))
            self.assertTrue(hasattr(self.layer, "fa_v"))

            # Verify they are instances of nn.Module
            self.assertIsInstance(self.layer.fa_q, nn.Module)
            self.assertIsInstance(self.layer.fa_k, nn.Module)
            self.assertIsInstance(self.layer.fa_v, nn.Module)

    def test_create_weights_sets_num_kv_heads_to_one(self):
        """Test that num_kv_heads is set to 1"""
        method = self.method_class()

        with patch("torch.empty") as mock_empty:
            mock_empty.return_value = torch.zeros(1, 1)

            method.create_weights(self.layer)

            self.assertEqual(self.layer.num_kv_heads, 1)

    def test_create_weights_creates_correct_tensors(self):
        """Test that create_weights creates tensors with correct shapes and dtypes"""
        method = self.method_class()

        # Track torch.empty calls
        empty_calls = []

        def mock_empty(size, dtype=None):
            empty_calls.append((size, dtype))
            return torch.zeros(size, dtype=dtype if dtype else torch.float32)

        with patch("torch.empty", side_effect=mock_empty):
            method.create_weights(self.layer)

            # Verify tensor creations
            expected_calls = [
                ((32, 1), torch.float32),  # fa_q.scale
                ((1, 1), torch.float32),  # fa_k.scale
                ((1, 1), torch.float32),  # fa_v.scale
                ((32, 1), torch.int8),  # fa_q.offset
                ((1, 1), torch.int8),  # fa_k.offset
                ((1, 1), torch.int8),  # fa_v.offset
            ]

            # Compare without considering order
            self.assertEqual(len(empty_calls), len(expected_calls))
            for call in expected_calls:
                self.assertIn(call, empty_calls)

    def test_create_weights_registers_parameters(self):
        """Test that create_weights registers parameters with correct attributes"""
        method = self.method_class()

        # Create real tensors for testing
        def create_tensor(*args, **kwargs):
            size = args[0] if args else kwargs.get('size', (1,))
            dtype = kwargs.get('dtype', torch.float32)
            return torch.zeros(*size, dtype=dtype)

        with patch("torch.empty", side_effect=create_tensor):
            method.create_weights(self.layer)

            # Import weight_loader for comparison
            from vllm_ascend.quantization.methods.kv_c8 import weight_loader

            # Verify each parameter exists and has weight_loader
            self.assertTrue(hasattr(self.layer.fa_q, "scale"))
            self.assertTrue(hasattr(self.layer.fa_q.scale, "weight_loader"))
            self.assertEqual(self.layer.fa_q.scale.weight_loader, weight_loader)
            self.assertFalse(self.layer.fa_q.scale.requires_grad)

            self.assertTrue(hasattr(self.layer.fa_k, "scale"))
            self.assertTrue(hasattr(self.layer.fa_k.scale, "weight_loader"))

            self.assertTrue(hasattr(self.layer.fa_v, "scale"))
            self.assertTrue(hasattr(self.layer.fa_v.scale, "weight_loader"))

            self.assertTrue(hasattr(self.layer.fa_q, "offset"))
            self.assertTrue(hasattr(self.layer.fa_q.offset, "weight_loader"))
            self.assertEqual(self.layer.fa_q.offset.dtype, torch.int8)


class TestAscendFAQuantAttentionMethodProcessWeights(unittest.TestCase):
    """Test cases for process_weights_after_loading method"""

    def setUp(self):
        """Set up test environment"""
        # Mock vllm_config
        self.config_patch = patch("vllm_ascend.quantization.methods.kv_c8.get_current_vllm_config")
        self.mock_get_config = self.config_patch.start()

        self.mock_config = Mock()
        self.mock_hf_config = Mock()
        self.mock_hf_config.kv_lora_rank = 64
        self.mock_hf_config.qk_rope_head_dim = 32
        self.mock_config.model_config.hf_config = self.mock_hf_config
        self.mock_get_config.return_value = self.mock_config

        # Import the class
        from vllm_ascend.quantization.methods.kv_c8 import AscendFAQuantAttentionMethod
        self.method_class = AscendFAQuantAttentionMethod

        # Create method instance with real layer
        self.method = self.method_class()

        # Create a real nn.Module for testing
        self.layer = nn.Module()

        # Create real tensors for fa_k
        self.fa_k_scale = torch.tensor([[2.0, 3.0, 4.0]], dtype=torch.float16)  # Shape (1,3)
        self.fa_k_offset = torch.tensor([[1, 2, 3]], dtype=torch.int8)  # Shape (1,3)

        # Create fa_k module with parameters
        self.layer.fa_k = nn.Module()
        self.layer.fa_k.scale = nn.Parameter(self.fa_k_scale, requires_grad=False)
        self.layer.fa_k.offset = nn.Parameter(self.fa_k_offset, requires_grad=False)

    def tearDown(self):
        """Clean up after each test"""
        self.config_patch.stop()

    def test_process_weights_with_single_value_scale(self):
        """Test process_weights with single value scale"""
        # Create new layer with single value scale
        layer = nn.Module()
        layer.fa_k = nn.Module()
        layer.fa_k.scale = nn.Parameter(torch.tensor([[2.0]], dtype=torch.float16), requires_grad=False)
        layer.fa_k.offset = nn.Parameter(torch.tensor([[1]], dtype=torch.int8), requires_grad=False)

        self.method.kv_lora_rank = 4
        self.method.process_weights_after_loading(layer)

        self.assertEqual(layer.quant_kscale.shape, (1, 4))
        self.assertEqual(layer.quant_kscale.dtype, torch.float32)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete kv_c8 functionality"""

    def setUp(self):
        """Set up test environment"""
        # Mock vllm_config
        self.config_patch = patch("vllm_ascend.quantization.methods.kv_c8.get_current_vllm_config")
        self.mock_get_config = self.config_patch.start()

        self.mock_config = Mock()
        self.mock_hf_config = Mock()
        self.mock_hf_config.kv_lora_rank = 64
        self.mock_hf_config.qk_rope_head_dim = 32
        self.mock_config.model_config.hf_config = self.mock_hf_config
        self.mock_get_config.return_value = self.mock_config

        # Mock distributed functions
        self.tp_rank_patch = patch(
            "vllm_ascend.quantization.methods.kv_c8.get_tensor_model_parallel_rank"
        )
        self.tp_size_patch = patch(
            "vllm_ascend.quantization.methods.kv_c8.get_tensor_model_parallel_world_size"
        )
        self.mock_tp_rank = self.tp_rank_patch.start()
        self.mock_tp_size = self.tp_size_patch.start()

    def tearDown(self):
        """Clean up after each test"""
        self.config_patch.stop()
        self.tp_rank_patch.stop()
        self.tp_size_patch.stop()

    def test_complete_workflow(self):
        """Test complete workflow from weight creation to processing"""
        from vllm_ascend.quantization.methods.kv_c8 import AscendFAQuantAttentionMethod

        # Create method instance
        method = AscendFAQuantAttentionMethod()

        # Create real layer
        layer = nn.Module()
        layer.num_heads = 32
        layer.num_kv_heads = 8

        # Step 1: Create weights
        method.create_weights(layer)

        # Verify weights were created with correct structure
        self.assertTrue(hasattr(layer, "fa_q"))
        self.assertTrue(hasattr(layer, "fa_k"))
        self.assertTrue(hasattr(layer, "fa_v"))

        self.assertTrue(hasattr(layer.fa_q, "scale"))
        self.assertTrue(hasattr(layer.fa_q, "offset"))
        self.assertTrue(hasattr(layer.fa_k, "scale"))
        self.assertTrue(hasattr(layer.fa_k, "offset"))
        self.assertTrue(hasattr(layer.fa_v, "scale"))
        self.assertTrue(hasattr(layer.fa_v, "offset"))

        # Step 2: Simulate weight loading
        self.mock_tp_rank.return_value = 0
        self.mock_tp_size.return_value = 1

        # Create dummy weights
        q_scale = torch.randn(32, 1)
        k_scale = torch.randn(1, 1)
        v_scale = torch.randn(1, 1)
        q_offset = torch.randint(-128, 127, (32, 1), dtype=torch.int8)
        k_offset = torch.randint(-128, 127, (1, 1), dtype=torch.int8)
        v_offset = torch.randint(-128, 127, (1, 1), dtype=torch.int8)

        # Load weights using weight_loader
        from vllm_ascend.quantization.methods.kv_c8 import weight_loader

        with torch.no_grad():
            weight_loader(layer.fa_q.scale, q_scale)
            weight_loader(layer.fa_k.scale, k_scale)
            weight_loader(layer.fa_v.scale, v_scale)
            weight_loader(layer.fa_q.offset, q_offset)
            weight_loader(layer.fa_k.offset, k_offset)
            weight_loader(layer.fa_v.offset, v_offset)

        # Verify weights were loaded correctly
        self.assertTrue(torch.all(layer.fa_q.scale == q_scale))
        self.assertTrue(torch.all(layer.fa_k.scale == k_scale))
        self.assertTrue(torch.all(layer.fa_v.scale == v_scale))

        # Step 3: Process after loading
        method.process_weights_after_loading(layer)

        # Verify processed parameters
        self.assertTrue(hasattr(layer, "fak_descale"))
        self.assertTrue(hasattr(layer, "fak_offset"))
        self.assertTrue(hasattr(layer, "quant_kscale"))


if __name__ == "__main__":
    unittest.main(verbosity=2)