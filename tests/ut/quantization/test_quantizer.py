from unittest.mock import MagicMock, patch

from tests.ut.base import TestBase
from vllm_ascend.quantization.quant_config import AscendQuantConfig
from vllm_ascend.quantization.quantizer import (VLLMAscendQuantizer,
                                                W4A8DYNAMICQuantizer,
                                                W8A8Quantizer)

SUPPORT_ASCEND_QUANTIZER_TYPE = {"test": "1"}


class TestGetQuantizer(TestBase):

    def setUp(self):
        # Setup common test fixtures
        self.supported_types = {
            'INT8': MagicMock(_instance=None),
            'FP16': MagicMock(_instance=None),
            'C8': MagicMock(_instance=None)
        }
        self.original_supported_types = SUPPORT_ASCEND_QUANTIZER_TYPE.copy()
        SUPPORT_ASCEND_QUANTIZER_TYPE.update(self.supported_types)
        self.mock_quant_config = MagicMock(spec=AscendQuantConfig)
        self.mock_quant_config.quant_description = {"some_config": "value"}

    def tearDown(self):
        # Restore original supported types
        SUPPORT_ASCEND_QUANTIZER_TYPE.clear()
        SUPPORT_ASCEND_QUANTIZER_TYPE.update(self.original_supported_types)

    def test_get_quantizer_fa(self):
        """Test successful quantizer retrieval for different cases."""
        # Setup
        quant_description = {'fa_quant_type': 'C8'}
        prefix = '.attn'
        expected_type = 'C8'
        with patch.dict(
                'vllm_ascend.quantization.quantizer.SUPPORT_ASCEND_QUANTIZER_TYPE',
                SUPPORT_ASCEND_QUANTIZER_TYPE):

            result = VLLMAscendQuantizer.get_quantizer(
                quant_description,
                prefix,
                packed_modules_mapping={"some": "mapping"})

            # Verify
            self.assertIsNotNone(result)
            self.assertEqual(result,
                             self.supported_types[expected_type]._instance)
            self.supported_types[expected_type].assert_called_once_with(
                quant_description)

    def test_get_quantizer_kv(self):
        """Test successful quantizer retrieval for different cases."""
        # Setup
        quant_description = {'kv_quant_type': 'C8'}
        prefix = '.attn'
        expected_type = 'C8'
        with patch.dict(
                'vllm_ascend.quantization.quantizer.SUPPORT_ASCEND_QUANTIZER_TYPE',
                SUPPORT_ASCEND_QUANTIZER_TYPE):

            result = VLLMAscendQuantizer.get_quantizer(
                quant_description,
                prefix,
                packed_modules_mapping={"some": "mapping"})

            # Verify
            self.assertIsNotNone(result)
            self.assertEqual(result,
                             self.supported_types[expected_type]._instance)
            self.supported_types[expected_type].assert_called_once_with(
                quant_description)

    def test_get_quantizer_linear(self):
        """Test successful quantizer retrieval for different cases."""
        # Setup
        quant_description = {'linear_type': 'INT8'}
        prefix = 'nothing'
        expected_type = 'INT8'
        with patch('vllm_ascend.quantization.quantizer.VLLMAscendQuantizer.get_linear_quant_type',
                            return_value=expected_type), \
            patch.dict('vllm_ascend.quantization.quantizer.SUPPORT_ASCEND_QUANTIZER_TYPE', SUPPORT_ASCEND_QUANTIZER_TYPE):

            result = VLLMAscendQuantizer.get_quantizer(
                quant_description,
                prefix,
                packed_modules_mapping={"some": "mapping"})

            # Verify
            self.assertIsNotNone(result)
            self.assertEqual(result,
                             self.supported_types[expected_type]._instance)
            self.supported_types[expected_type].assert_called_once_with(
                quant_description)


class TestW8A8Quantizer(TestBase):

    def setUp(self):
        self.quantizer = W8A8Quantizer(quant_description={})

    def test_build_linear_method(self):
        with patch('vllm_ascend.quantization.quantizer.AscendW8A8LinearMethod',
                   return_value=MagicMock()) as mock_linear:
            result = self.quantizer.build_linear_method()
            mock_linear.assert_called_once_with()
            self.assertIsInstance(result, MagicMock)

    def test_build_moe_method(self):
        with patch(
                'vllm_ascend.quantization.quantizer.AscendW8A8FusedMoEMethod',
                return_value=MagicMock()) as mock_linear:
            result = self.quantizer.build_moe_method()
            mock_linear.assert_called_once_with()
            self.assertIsInstance(result, MagicMock)

    def test_build_attention_method(self):
        with patch('vllm_ascend.quantization.quantizer.AscendC8KVCacheMethod',
                   return_value=MagicMock()) as mock_linear:
            result = self.quantizer.build_attention_method()
            mock_linear.assert_called_once_with()
            self.assertIsInstance(result, MagicMock)


class TestW4A8DYNAMICQuantizer(TestBase):

    def setUp(self):
        self.quantizer = W4A8DYNAMICQuantizer(quant_description={})

    def test_build_linear_method(self):
        with patch(
                'vllm_ascend.quantization.quantizer.AscendW4A8DynamicLinearMethod',
                return_value=MagicMock()) as mock_linear:
            result = self.quantizer.build_linear_method()
            mock_linear.assert_called_once_with()
            self.assertIsInstance(result, MagicMock)

    def test_build_moe_method(self):
        with patch(
                'vllm_ascend.quantization.quantizer.AscendW4A8DynamicFusedMoEMethod',
                return_value=MagicMock()) as mock_fused_moe:
            result = self.quantizer.build_moe_method()
            mock_fused_moe.assert_called_once_with()
            self.assertIsInstance(result, MagicMock)
