from unittest.mock import MagicMock, patch

import torch
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

    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_fuse_gemm_comms_registers_pattern_matcher_pass(self, mock_platform):
        vllm_config = VllmConfig()
        vllm_config.model_config = MagicMock(dtype=torch.bfloat16)
        vllm_config.compilation_config.pass_config.fuse_gemm_comms = True
        vllm_config.additional_config = {
            "ascend_compilation_config": {
                "fuse_norm_quant": "false",
                "fuse_qknorm_rope": "false",
                "fuse_muls_add": "false",
            }
        }
        init_ascend_config(vllm_config)

        profile = MagicMock()
        profile.supports.side_effect = lambda capability: capability.name == "GRAPH_MM_REDUCE_SCATTER_FUSION"
        with (
            patch(
                "vllm_ascend.compilation.graph_fusion_pass_manager.get_current_hardware_profile",
                return_value=profile,
            ),
            patch(
                "vllm_ascend.compilation.passes.mm_reduce_scatter_fusion_pass.get_tensor_model_parallel_world_size",
                return_value=2,
            ),
            patch(
                "vllm_ascend.compilation.passes.mm_reduce_scatter_fusion_pass.get_tp_group",
                return_value=MagicMock(unique_name="tp:0"),
            ),
        ):
            manager = GraphFusionPassManager()
            manager.configure(vllm_config)

        self.assertEqual(len(manager.passes), 1)
