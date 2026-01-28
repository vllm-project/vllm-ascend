import copy
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import torch
import torchair
from torchair._acl_concrete_graph import graph_pass
from vllm.distributed import tensor_model_parallel_all_reduce


def create_fusion_pass_wrapper(assert_func):
    from torchair._acl_concrete_graph.graph_pass import _apply_fusion_passes as original_func

    def wrapper(gm, *args, **kwargs):
        graph_before = copy.deepcopy(gm)
        ret = original_func(gm, *args, **kwargs)
        graph_after = copy.deepcopy(gm)
        assert_func(graph_before, graph_after)
        return ret

    return wrapper


class TestMatmulAllReduceRMSNormFusion(unittest.TestCase):
    _patterns_registered = False
    vllm_config = None
    _distributed_patches: list[Any] = []

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._setup_forward_context_mock()
        cls._setup_distributed_mock()
        if not cls._patterns_registered:
            cls._setup_vllm_config()
            cls._register_patterns()
            cls._patterns_registered = True

    @classmethod
    def _setup_vllm_config(cls):
        try:
            from vllm.config import ModelConfig, VllmConfig

            cls.vllm_config = VllmConfig(model_config=ModelConfig(dtype=torch.bfloat16))
        except ImportError:
            from types import SimpleNamespace

            cls.vllm_config = SimpleNamespace()
            cls.vllm_config.model_config = SimpleNamespace()
            cls.vllm_config.model_config.dtype = torch.bfloat16

    @classmethod
    def _setup_forward_context_mock(cls):
        try:
            import vllm.forward_context as fc_module

            class MockForwardContext:
                sp_enabled = False
                tp_size = 2
                ep_size = 1
                virtual_engine = 0
                attn_metadata = None
                valid_runtime_modes = set()

            fc_module._forward_context = MockForwardContext()
        except Exception:
            print("Failed to setup forward context mock")

    @classmethod
    def _setup_distributed_mock(cls):
        cls._distributed_patches = []

        mock_group = MagicMock()
        mock_backend = MagicMock()
        mock_backend.get_hccl_comm_name.return_value = "mock_hccl_comm_0"
        mock_group.device_group._get_backend.return_value = mock_backend

        def mock_all_reduce(tensor):
            return tensor

        mock_group.all_reduce = mock_all_reduce

        patch_targes = [
            ("vllm.distributed.parallel_state._TP", mock_group),
            ("vllm.distributed.parallel_state.get_tp_group", mock_group),
            ("vllm.distributed.communication_op.get_tp_group", mock_group),
            ("vllm.distributed.get_tp_group", mock_group),
            ("vllm.distributed.parallel_state.get_tensor_model_parallel_world_size", 2),
            ("vllm.distributed.get_tensor_model_parallel_world_size", 2),
            ("vllm.distributed.parallel_state.gen_tensor_model_parallel_rank", 0),
            ("torch.distributed.get_rank", 0),
            ("torch.distributed.get_world_size", 2),
            ("torch.distributed.is_initialized", True),
        ]

        for target, return_val in patch_targes:
            try:
                p = patch(target, return_val=return_val)
                p.start()
                cls._distributed_patches.append(p)
            except Exception:
                print("Failed to patch {target}")

    @classmethod
    def _register_patterns(cls):
        from vllm_ascend.compilation.npugraph_ex_passes.graphex_allreduce_rmsnorm_fusion_pass import (
            GraphEXMatmulAllReduceAddRMSNormPass,
        )

        GraphEXMatmulAllReduceAddRMSNormPass(cls.vllm_config)

    @classmethod
    def tearDownClass(cls):
        for p in getattr(cls, "_distributed_patches", []):
            try:
                p.stop()
            except Exception:
                print("Failed to stop patches")
        super().tearDownClass()

    def setUp(self):
        self.dtype = torch.bfloat16
        self.device = "npu"

    def _create_backend_with_wrapper(self, assert_func):
        try:
            from torchair._acl_concrete_graph.graph_pass import _apply_fusion_passes

            self.original_pass = _apply_fusion_passes
            graph_pass._apply_fusion_passes = create_fusion_pass_wrapper(assert_func)
        except (ImportError, AttributeError):
            self._original_pass = None

        config = torchair.CompilerConfig()
        config.mode = "reduce-overhead"

        return torchair.get_npu_backend(compiler_config=config)

    def tearDown(self):
        if hasattr(self, "_original_pass") and self._original_pass:
            graph_pass._apply_fusion_passes = self._original_pass

    def _get_input_tensor(self):
        batch, seq_len = 4, 8
        hidden_dim = 64

        return [
            # x: [batch * seq_len, hidden_dim]
            torch.randn(batch * seq_len, hidden_dim, dtype=self.dtype, device=self.device),
            # weight: [hidden_dim, hidden_dim]
            torch.randn(hidden_dim, hidden_dim, dtype=self.dtype, device=self.device),
            # residual: [batch * seq_len, hidden_dim]
            torch.randn(batch * seq_len, hidden_dim, dtype=self.dtype, device=self.device),
            # rms_norm_weight: [hidden_dim]
            torch.randn(hidden_dim, dtype=self.dtype, device=self.device),
        ]

    def test_matmul_allreduce_rmsnorm(self):
        def f(x, weight, residual, rms_norm_weight):
            mm = torch.ops.vllm.unquantized_gemm(x, weight, None)
            all_reduce_ = tensor_model_parallel_all_reduce(mm)
            output = torch.ops.npu.npu_add_rms_norm(all_reduce_, residual, rms_norm_weight, 1e-6)
            return output[0], output[2]

        def assert_fusion(graph_before, graph_after):
            has_unquantized_gemm_before = any(
                "unquantized_gemm" in str(n.target) for n in graph_before.graph.nodes if n.op == "call_function"
            )
            has_allreduce_before = any(
                "all_reduce" in str(n.target) or "allreduce" in str(n.target)
                for n in graph_before.graph.nodes
                if n.op == "call_function"
            )
            has_rms_norm_before = any(
                "npu_add_rms_norm" in str(n.target) for n in graph_before.graph.nodes if n.op == "call_function"
            )
            has_fused_op = any(
                "matmul_allreduce_add_rmsnorm" in str(n.target)
                for n in graph_after.graph.nodes
                if n.op == "call_function"
            )

            if has_unquantized_gemm_before and has_allreduce_before and has_rms_norm_before:
                self.assertTrue(
                    has_fused_op,
                    "unquantized_gemm + all_reduce + npu_add_rms_norm should be fused to matmul_addreduce_add_rmsnorm",
                )

        backend = self._create_backend_with_wrapper(assert_fusion)

        try:
            model = torch.compile(f, backend=backend, dynamic=True)
            inputs = self._get_input_tensor()
            model(*inputs)
        finally:
            self.tearDown()
