from unittest.mock import patch

from vllm.config import VllmConfig

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import clear_ascend_config, init_ascend_config
from vllm_ascend.compilation.graph_fusion_pass_manager import GraphFusionPassManager


class TestGraphFusionPassManagerConfig(TestBase):
    def tearDown(self):
        clear_ascend_config()

    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_configure_consumes_validated_ascend_compilation_config(self, mock_platform):
        vllm_config = VllmConfig()
        vllm_config.additional_config = {
            "ascend_compilation_config": {
                "fuse_norm_quant": "false",
                "fuse_qknorm_rope": "false",
                "fuse_muls_add": "false",
            }
        }
        init_ascend_config(vllm_config)

        manager = GraphFusionPassManager()
        manager.configure(vllm_config)

        self.assertFalse(manager.ascend_compilation_config.fuse_norm_quant)
        self.assertFalse(manager.ascend_compilation_config.fuse_qknorm_rope)
        self.assertFalse(manager.ascend_compilation_config.fuse_muls_add)
        self.assertEqual(manager.passes, [])
