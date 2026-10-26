from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import VllmConfig

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import clear_ascend_config, init_ascend_config
from vllm_ascend.compilation.graph_fusion_pass_manager import GraphFusionPassManager
from vllm_ascend.utils import vllm_version_is


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
        if vllm_version_is("0.31.0"):
            # v0.31.0 gates MMRS on the upstream PassConfig flag.
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
            # vLLM #60517: on main the MMRS pass is gated on Ascend SP.
            patch(
                "vllm_ascend.compilation.graph_fusion_pass_manager.enable_sp",
                return_value=True,
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


@pytest.mark.parametrize("method", [None, "dspark", "mtp"])
def test_qknorm_rope_fusion_skips_dspark(method):
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(use_dspark=lambda: method == "dspark") if method else None
    )
    ascend_config = SimpleNamespace(
        ascend_compilation_config=SimpleNamespace(fuse_norm_quant=False, fuse_qknorm_rope=True, fuse_muls_add=False)
    )
    with (
        patch("vllm_ascend.ascend_config.get_ascend_config", return_value=ascend_config),
        patch("vllm_ascend.compilation.passes.qknorm_rope_fusion_pass.QKNormRopeFusionPass") as fusion_pass,
    ):
        manager = GraphFusionPassManager()
        manager.configure(config)
        if method == "dspark":
            fusion_pass.assert_not_called()
            assert manager.passes == []
        else:
            fusion_pass.assert_called_once_with(config)
            assert manager.passes == [fusion_pass.return_value]
