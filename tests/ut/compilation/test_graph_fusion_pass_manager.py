from unittest.mock import MagicMock, patch

from vllm.config import VllmConfig

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import clear_ascend_config, init_ascend_config
from vllm_ascend.compilation.graph_fusion_pass_manager import GraphFusionPassManager
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile


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
        self.assertEqual(manager.graph_passes, [])

    def _configure_with_gemm_comms(self, fuse_gemm_comms: bool) -> GraphFusionPassManager:
        with patch("vllm_ascend.platform.NPUPlatform.check_and_update_config"):
            vllm_config = VllmConfig()
        vllm_config.compilation_config.pass_config.fuse_gemm_comms = fuse_gemm_comms
        vllm_config.additional_config = {
            "ascend_compilation_config": {
                "fuse_norm_quant": "false",
                "fuse_qknorm_rope": "false",
                "fuse_muls_add": "false",
            }
        }
        init_ascend_config(vllm_config)

        manager = GraphFusionPassManager()
        with patch(
            "vllm_ascend.compilation.passes.mm_reduce_scatter_fusion_pass.MatmulReduceScatterFusionPass",
            return_value=MagicMock(),
        ):
            manager.configure(vllm_config)
        return manager

    def test_fuse_gemm_comms_registers_the_mm_reduce_scatter_pass(self):
        if not get_current_hardware_profile().supports(HardwareCapability.GRAPH_MM_REDUCE_SCATTER_FUSION):
            self.skipTest("device does not support matmul reduce-scatter fusion")

        manager = self._configure_with_gemm_comms(True)

        self.assertEqual(len(manager.graph_passes), 1)

    def test_mm_reduce_scatter_pass_is_off_by_default(self):
        manager = self._configure_with_gemm_comms(False)

        self.assertEqual(manager.graph_passes, [])
