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

from unittest.mock import MagicMock, patch

from vllm.config import VllmConfig

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import clear_ascend_config, get_ascend_config, init_ascend_config
from vllm_ascend.utils import AscendDeviceType


class TestAscendConfig(TestBase):
    @staticmethod
    def _clean_up_ascend_config(func):
        def wrapper(*args, **kwargs):
            clear_ascend_config()
            func(*args, **kwargs)
            clear_ascend_config()

        return wrapper

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def test_init_ascend_config_without_additional_config(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        # No additional config given, check the default value here.
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertFalse(ascend_config.multistream_overlap_shared_expert)
        self.assertFalse(ascend_config.enable_kv_nz)

        ascend_compilation_config = ascend_config.ascend_compilation_config
        self.assertTrue(ascend_compilation_config.fuse_norm_quant)

        ascend_fusion_config = ascend_config.ascend_fusion_config
        self.assertTrue(ascend_fusion_config.fusion_ops_gmmswigluquant)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def test_init_ascend_config_with_additional_config(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {
            "ascend_compilation_config": {
                "fuse_norm_quant": False,
            },
            "ascend_fusion_config": {
                "fusion_ops_gmmswigluquant": False,
            },
            "multistream_overlap_shared_expert": True,
            "eplb_config": {"num_redundant_experts": 2},
            "refresh": True,
            "enable_kv_nz": False,
        }
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertEqual(ascend_config.eplb_config.num_redundant_experts, 2)
        self.assertTrue(ascend_config.multistream_overlap_shared_expert)

        ascend_compilation_config = ascend_config.ascend_compilation_config
        self.assertFalse(ascend_compilation_config.fuse_norm_quant)
        self.assertFalse(ascend_config.enable_kv_nz)
        self.assertTrue(ascend_compilation_config.enable_npugraph_ex)
        self.assertFalse(ascend_compilation_config.enable_static_kernel)

        ascend_fusion_config = ascend_config.ascend_fusion_config
        self.assertFalse(ascend_fusion_config.fusion_ops_gmmswigluquant)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def test_init_ascend_config_enable_npugraph_ex(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {
            "ascend_compilation_config": {
                "enable_npugraph_ex": True,
                "enable_static_kernel": True
            },
            "refresh": True
        }
        ascend_compilation_config = init_ascend_config(
            test_vllm_config).ascend_compilation_config
        self.assertTrue(ascend_compilation_config.enable_npugraph_ex)
        self.assertTrue(ascend_compilation_config.enable_static_kernel)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def test_get_ascend_config(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertEqual(get_ascend_config(), ascend_config)

    @_clean_up_ascend_config
    def test_get_ascend_config_without_init(self):
        with self.assertRaises(RuntimeError):
            get_ascend_config()

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def test_clear_ascend_config(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertEqual(get_ascend_config(), ascend_config)
        clear_ascend_config()
        with self.assertRaises(RuntimeError):
            get_ascend_config()

    @_clean_up_ascend_config
    @patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A3)
    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def test_sparse_c8_without_layer_filter(self, mock_fix_incompatible_config, mock_get_ascend_device_type):
        test_vllm_config = VllmConfig()
        test_vllm_config.model_config = MagicMock()
        test_vllm_config.model_config.hf_text_config = MagicMock(index_topk=2048)
        test_vllm_config.quant_config = MagicMock(quant_description={"indexer_quant_type": "INT8_DYNAMIC"})
        test_vllm_config.additional_config = {
            "enable_sparse_c8": True,
        }

        ascend_config = init_ascend_config(test_vllm_config)

        self.assertTrue(ascend_config.enable_sparse_c8)
        self.assertTrue(ascend_config.is_sparse_c8_layer("model.layers.0.attn"))
        self.assertTrue(ascend_config.is_sparse_c8_layer("model.layers.17.attn"))

    @_clean_up_ascend_config
    @patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A3)
    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def test_sparse_c8_with_quant_description_filter(self, mock_fix_incompatible_config, mock_get_ascend_device_type):
        test_vllm_config = VllmConfig()
        test_vllm_config.model_config = MagicMock()
        test_vllm_config.model_config.hf_text_config = MagicMock(index_topk=2048)
        test_vllm_config.quant_config = MagicMock(
            quant_description={
                "model.layers.1.self_attn.indexer.quant_type": "INT8_DYNAMIC",
                "model.layers.3.self_attn.indexer.quant_type": "INT8_DYNAMIC",
                "model.layers.5.self_attn.indexer.quant_type": "FLOAT",
            }
        )
        test_vllm_config.additional_config = {
            "enable_sparse_c8": True,
        }

        ascend_config = init_ascend_config(test_vllm_config)

        self.assertTrue(ascend_config.is_sparse_c8_layer("model.layers.1.attn"))
        self.assertTrue(ascend_config.is_sparse_c8_layer("model.layers.3.attn"))
        self.assertFalse(ascend_config.is_sparse_c8_layer("model.layers.2.attn"))
        self.assertFalse(ascend_config.is_sparse_c8_layer("model.layers.5.attn"))

    @_clean_up_ascend_config
    @patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A3)
    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def test_sparse_c8_without_any_int8_dynamic_layer_falls_back_to_non_c8(
        self, mock_fix_incompatible_config, mock_get_ascend_device_type
    ):
        test_vllm_config = VllmConfig()
        test_vllm_config.model_config = MagicMock()
        test_vllm_config.model_config.hf_text_config = MagicMock(index_topk=2048)
        test_vllm_config.quant_config = MagicMock(
            quant_description={
                "model.layers.1.self_attn.indexer.quant_type": "FLOAT",
                "model.layers.3.self_attn.indexer.quant_type": "FLOAT",
            }
        )
        test_vllm_config.additional_config = {
            "enable_sparse_c8": True,
        }

        ascend_config = init_ascend_config(test_vllm_config)

        self.assertFalse(ascend_config.is_sparse_c8_layer("model.layers.1.attn"))
        self.assertFalse(ascend_config.is_sparse_c8_layer("model.layers.3.attn"))
