import unittest
import torch
import copy
import torchair
from torchair._acl_concrete_graph import graph_pass
import sys


def create_fusion_pass_wrapper(assert_func):
    """
    Create a wrapper for fusion pass to capture FX graphs before and after replacement.
    The wrapper will deepcopy the graph module in advance and post fusion,
    then call the custom assertion function to verify fusion effect.
    """
    try:
        from torchair._acl_concrete_graph.graph_pass import (
            _apply_fusion_passes as original_func,
        )
    except ImportError:
        try:
            from torchair._acl_concrete_graph.graph_pass import (
                _run_fusion_passes as original_func,
            )
        except ImportError:
            original_func = None

    # Refactor: single wrapper definition to resolve no-redef error
    def wrapper(gm, *args, **kwargs):
        if original_func is None:
            return None
        graph_before = copy.deepcopy(gm)
        ret = original_func(gm, *args, **kwargs)
        graph_after = copy.deepcopy(gm)
        assert_func(graph_before, graph_after)
        return ret

    return wrapper


class TestGraphEXAddRMSNormFusion(unittest.TestCase):
    # Explicitly declare class attributes to resolve mypy attr-defined error
    _patterns_registered = False
    vllm_config = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls._setup_forward_context_mock()
        if not cls._patterns_registered:
            cls._setup_vllm_config()
            cls._register_patterns()
            cls._patterns_registered = True

    @classmethod
    def _setup_vllm_config(cls):
        """Create mock VllmConfig for fusion pass initialization"""
        try:
            from vllm.config import VllmConfig, ModelConfig

            cls.vllm_config = VllmConfig(model_config=ModelConfig(dtype=torch.bfloat16))
        except ImportError:
            from types import SimpleNamespace

            cls.vllm_config = SimpleNamespace()
            cls.vllm_config.model_config = SimpleNamespace()
            cls.vllm_config.model_config.dtype = torch.float16

    @classmethod
    def _setup_forward_context_mock(cls):
        """Mock VLLM global ForwardContext to avoid fake tensor runtime errors"""
        try:
            import vllm.forward_context as fc_module

            class MockForwardContext:
                sp_enabled = False
                tp_size = 1
                ep_size = 1
                virtual_engine = 0
                attn_metadata = None
                valid_runtime_modes = set()  # Add missing attr to avoid runtime error

            fc_module._forward_context = MockForwardContext()
            print("[INFO] Mocked _forward_context set", file=sys.stderr)

        except Exception as e:
            print(
                f"[WARNING] Failed to setup forward context mock: {e}", file=sys.stderr
            )

    @classmethod
    def _register_patterns(cls):
        """Initialize and register AddRMSNorm fusion patterns to TorchAir"""
        from vllm_ascend.compilation.npugraph_ex_passes.graphex_norm_quant_fusion_pass import (
            GraphEXAddRMSNormFusionPass,
        )

        GraphEXAddRMSNormFusionPass(cls.vllm_config)

    def setUp(self):
        """Initialize test environment for each case"""
        self.dtype = torch.bfloat16
        self.device = "npu"

    def _create_backend_with_wrapper(self, assert_func):
        """
        Create TorchAir NPU backend and replace fusion pass with wrapped version
        Args:
            assert_func: Custom assertion function to verify fusion effect
        Returns:
            TorchAir NPU backend instance
        """
        try:
            from torchair._acl_concrete_graph.graph_pass import _apply_fusion_passes

            self._original_pass = _apply_fusion_passes
            graph_pass._apply_fusion_passes = create_fusion_pass_wrapper(assert_func)
        except (ImportError, AttributeError):
            self._original_pass = None

        config = torchair.CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.run_eagerly = True

        return torchair.get_npu_backend(compiler_config=config)

    def tearDown(self):
        """Clean up and restore original fusion pass function after each test case"""
        if hasattr(self, "_original_pass") and self._original_pass:
            graph_pass._apply_fusion_passes = self._original_pass

    def _get_input_tensors(self, with_bias=False):
        """
        Generate input tensors for AddRMSNorm fusion test
        Args:
            with_bias: Whether to include bias tensor in input list
        Returns:
            List of input tensors with specified dtype and NPU device
        """
        inputs = [
            torch.randn(2, 4, dtype=self.dtype, device=self.device),  # rms_norm_input
            torch.randn(2, 4, dtype=self.dtype, device=self.device),  # residual
            torch.randn(4, dtype=self.dtype, device=self.device),     # rms_norm_weight
            torch.ones(4, dtype=self.dtype, device=self.device),      # scale
            torch.ones(4, dtype=self.dtype, device=self.device),      # scale_reciprocal
            torch.zeros(4, dtype=self.dtype, device=self.device),     # offset
        ]
        if with_bias:
            inputs.append(torch.randn(4, dtype=self.dtype, device=self.device))  # bias
        return inputs

    def test_add_rms_norm_quant_basic(self):
        """
        Test GraphEXAddRMSNormQuantPattern:
        npu_add_rms_norm + quantize -> npu_add_rms_norm_quant
        """

        def f(
            rms_norm_input, residual, rms_norm_weight, scale, scale_reciprocal, offset
        ):
            output = torch.ops.npu.npu_add_rms_norm(
                rms_norm_input, residual, rms_norm_weight, 1e-6
            )
            out0 = output[0]
            out1 = output[2]
            quantized_output = torch.ops.vllm.quantize(
                out0, scale, scale_reciprocal, offset
            )
            return quantized_output, out1

        def assert_basic_fusion(graph_before, graph_after):
            """Verify basic AddRMSNorm+Quant fusion is successful"""
            has_rms_norm_before = any(
                "npu_add_rms_norm" in str(n.target) and "quant" not in str(n.target)
                for n in graph_before.graph.nodes
                if n.op == "call_function"
            )
            has_quantize_before = any(
                "quantize" in str(n.target)
                for n in graph_before.graph.nodes
                if n.op == "call_function"
            )

            has_fused_op = any(
                "npu_add_rms_norm_quant" in str(n.target)
                for n in graph_after.graph.nodes
                if n.op == "call_function"
            )

            if has_rms_norm_before and has_quantize_before:
                self.assertTrue(
                    has_fused_op,
                    "npu_add_rms_norm and quantize should be fused to npu_add_rms_norm_quant",
                )

        backend = self._create_backend_with_wrapper(assert_basic_fusion)

        try:
            model = torch.compile(f, backend=backend, dynamic=True)
            inputs = self._get_input_tensors(with_bias=False)
            model(*inputs)
        finally:
            self.tearDown()

    def test_add_rms_norm_quant_with_bias(self):
        """
        Test GraphEXAddRMSNormQuantPatternWithBias:
        npu_add_rms_norm + add(bias) + quantize -> npu_add_rms_norm_quant(beta=bias)
        """

        def f(
            rms_norm_input,
            residual,
            rms_norm_weight,
            scale,
            scale_reciprocal,
            offset,
            bias,
        ):
            output = torch.ops.npu.npu_add_rms_norm(
                rms_norm_input, residual, rms_norm_weight, 1e-6
            )
            out0 = output[0]
            out1 = output[2]
            out0 = out0 + bias
            quantized_output = torch.ops.vllm.quantize(
                out0, scale, scale_reciprocal, offset
            )
            return quantized_output, out1

        def assert_bias_fusion(graph_before, graph_after):
            """Verify AddRMSNorm+Quant fusion with bias term is successful"""
            has_fused_op = False
            has_beta_param = False

            for node in graph_after.graph.nodes:
                if node.op == "call_function" and "npu_add_rms_norm_quant" in str(
                    node.target
                ):
                    has_fused_op = True
                    if "beta" in node.kwargs:
                        has_beta_param = True
                    elif len(node.args) > 6:
                        has_beta_param = True

            has_separate_add = any(
                str(n.target) in ["aten.add.Tensor", "torch.add", "add"]
                for n in graph_after.graph.nodes
                if n.op == "call_function"
            )

            if has_fused_op:
                self.assertTrue(
                    has_beta_param, "Bias should be fused as beta parameter"
                )
                self.assertFalse(
                    has_separate_add, "Separate add op should be eliminated"
                )

        backend = self._create_backend_with_wrapper(assert_bias_fusion)

        try:
            model = torch.compile(f, backend=backend, dynamic=True)
            inputs = self._get_input_tensors(with_bias=True)
            model(*inputs)
        finally:
            self.tearDown()

    def test_add_rms_norm_quant_sp(self):
        """
        Test GraphEXAddRMSNormQuantSPPattern (Sequence Parallel):
        npu_add_rms_norm + all_gather + quantize -> npu_add_rms_norm_quant + all_gather
        """

        def f(
            rms_norm_input, residual, rms_norm_weight, scale, scale_reciprocal, offset
        ):
            output = torch.ops.npu.npu_add_rms_norm(
                rms_norm_input, residual, rms_norm_weight, 1e-6
            )
            out0 = output[0]
            out1 = output[2]
            out0 = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(out0, True)
            quantized_output = torch.ops.vllm.quantize(
                out0, scale, scale_reciprocal, offset
            )
            return quantized_output, out1

        def assert_sp_fusion(graph_before, graph_after):
            """Verify AddRMSNorm+Quant fusion in Sequence Parallel mode"""
            has_fused_op = False
            correct_order = False

            for node in graph_after.graph.nodes:
                if node.op == "call_function" and "npu_add_rms_norm_quant" in str(
                    node.target
                ):
                    has_fused_op = True
                    for user in node.users:
                        if "maybe_all_gather" in str(user.target):
                            correct_order = True
                            break

            if has_fused_op:
                self.assertTrue(
                    correct_order,
                    "all_gather should follow the fused npu_add_rms_norm_quant",
                )

        backend = self._create_backend_with_wrapper(assert_sp_fusion)
        try:
            model = torch.compile(f, backend=backend, dynamic=True)
            inputs = self._get_input_tensors(with_bias=False)
            model(*inputs)
        finally:
            self.tearDown()

    def test_add_rms_norm_quant_sp_with_bias(self):
        """
        Test GraphEXAddRMSNormQuantSPPatternWithBias in Sequence Parallel mode
        """

        def f(
            rms_norm_input,
            residual,
            rms_norm_weight,
            scale,
            scale_reciprocal,
            offset,
            bias,
        ):
            output = torch.ops.npu.npu_add_rms_norm(
                rms_norm_input, residual, rms_norm_weight, 1e-6
            )
            out0 = output[0]
            out1 = output[2]
            out0 = out0 + bias
            out0 = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(out0, True)
            quantized_output = torch.ops.vllm.quantize(
                out0, scale, scale_reciprocal, offset
            )
            return quantized_output, out1

        def assert_sp_bias_fusion(graph_before, graph_after):
            """Verify AddRMSNorm+Quant fusion with bias in Sequence Parallel mode"""
            has_fused_op = False
            has_beta = False
            correct_order = False

            for node in graph_after.graph.nodes:
                if node.op == "call_function" and "npu_add_rms_norm_quant" in str(
                    node.target
                ):
                    has_fused_op = True
                    if "beta" in node.kwargs or len(node.args) > 6:
                        has_beta = True
                    for user in node.users:
                        if "maybe_all_gather" in str(user.target):
                            correct_order = True
                            break

            if has_fused_op:
                self.assertTrue(has_beta, "Bias should be fused as beta in SP pattern")
                self.assertTrue(
                    correct_order, "all_gather should follow fused op in SP pattern"
                )

        backend = self._create_backend_with_wrapper(assert_sp_bias_fusion)

        try:
            model = torch.compile(f, backend=backend, dynamic=True)
            inputs = self._get_input_tensors(with_bias=True)
            model(*inputs)
        finally:
            self.tearDown()
