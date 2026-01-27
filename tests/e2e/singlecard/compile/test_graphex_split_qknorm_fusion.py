import sys
import unittest
import torch
import copy
import torchair
from torchair._acl_concrete_graph import graph_pass
from torchair.configs.compiler_config import CompilerConfig

from torchair._acl_concrete_graph.graph_pass import (
    _apply_fusion_passes as original_func,
)


def create_fusion_pass_wrapper(assert_func):
    if original_func is None:

        def wrapper(*args, **kwargs):
            return None

        return wrapper

    def wrapper(gm, *args, **kwargs):
        graph_before = copy.deepcopy(gm)
        ret = original_func(gm, *args, **kwargs)
        graph_after = copy.deepcopy(gm)
        assert_func(graph_before, graph_after)
        return ret

    return wrapper


class TestGraphEXQKNormRopeFusion(unittest.TestCase):
    _patterns_registered = False

    HEAD_DIM = 128
    NUM_HEADS = 32
    NUM_KV_HEADS = 8
    Q_SIZE = NUM_HEADS * HEAD_DIM
    KV_SIZE = NUM_KV_HEADS * HEAD_DIM
    SEQ_LEN = 5
    EPS = 1e-6

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
        def mock_get_layers(*args, **kwargs):
            return {"mock_attn": MockAttentionLayer()}

        try:
            from vllm.config import VllmConfig, ModelConfig

            cls.vllm_config = VllmConfig(model_config=ModelConfig(dtype=torch.bfloat16))

            class MockAttentionLayer:
                head_size = cls.HEAD_DIM
                num_heads = cls.NUM_HEADS
                num_kv_heads = cls.NUM_KV_HEADS

            from vllm.config import get_layers_from_vllm_config

            cls.original_get_layers = get_layers_from_vllm_config
            get_layers_from_vllm_config = mock_get_layers
        except ImportError:
            from types import SimpleNamespace

            cls.vllm_config = SimpleNamespace()
            cls.vllm_config.model_config = SimpleNamespace()
            cls.vllm_config.model_config.dtype = torch.float16
            cls.vllm_config.device_config = SimpleNamespace(device="npu")

    @classmethod
    def _setup_forward_context_mock(cls):
        try:
            import vllm.forward_context as fc_module

            class MockForwardContext:
                sp_enabled = False
                tp_size = 1
                ep_size = 1
                virtual_engine = 0
                attn_metadata = None
                valid_runtime_modes = set()

            fc_module._forward_context = MockForwardContext()
            print(
                "[INFO] Mocked _forward_context set for QKNormRope fusion",
                file=sys.stderr,
            )

        except Exception as e:
            print(
                f"[WARNING] Failed to setup forward context mock: {e}", file=sys.stderr
            )

    @classmethod
    def _register_patterns(cls):
        """Initialize and register QKNormRope fusion patterns"""
        from vllm_ascend.compilation.npugraph_ex_passes.graphex_qknorm_rope_fusion_pass import (
            GraphEXQKNormRopeFusionPass,
        )

        GraphEXQKNormRopeFusionPass(cls.vllm_config)

    def setUp(self):
        self.dtype = torch.bfloat16
        self.device = "npu"

    def _create_backend_with_wrapper(self, assert_func):
        """Helper to create backend with fusion pass wrapper"""
        try:
            from torchair._acl_concrete_graph.graph_pass import _apply_fusion_passes

            self._original_pass = _apply_fusion_passes
            graph_pass._apply_fusion_passes = create_fusion_pass_wrapper(assert_func)
        except (ImportError, AttributeError):
            self._original_pass = None

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        config.debug.run_eagerly = True

        return torchair.get_npu_backend(compiler_config=config)

    def tearDown(self):
        """Cleanup: restore original function"""
        if hasattr(self, "_original_pass") and self._original_pass:
            graph_pass._apply_fusion_passes = self._original_pass

    def _get_input_tensors(self, with_bias=False):
        total_qkv_dim = self.Q_SIZE + 2 * self.KV_SIZE
        inputs = [
            torch.randn(
                self.SEQ_LEN, total_qkv_dim, dtype=self.dtype, device=self.device
            ),  # qkv
            torch.randn(
                self.HEAD_DIM, dtype=self.dtype, device=self.device
            ),  # q_weight
            torch.randn(
                self.HEAD_DIM, dtype=self.dtype, device=self.device
            ),  # k_weight
        ]
        if with_bias:
            inputs.extend(
                [
                    torch.randn(
                        self.HEAD_DIM, dtype=self.dtype, device=self.device
                    ),  # q_bias
                    torch.randn(
                        self.HEAD_DIM, dtype=self.dtype, device=self.device
                    ),  # k_bias
                ]
            )

        inputs.extend(
            [
                torch.randn(
                    1,
                    self.SEQ_LEN,
                    1,
                    self.HEAD_DIM,
                    dtype=self.dtype,
                    device=self.device,
                ),  # cos
                torch.randn(
                    1,
                    self.SEQ_LEN,
                    1,
                    self.HEAD_DIM,
                    dtype=self.dtype,
                    device=self.device,
                ),  # sin
            ]
        )
        return inputs

    @unittest.skipIf(torch.__version__ < "2.2", "torch.compile requires torch >= 2.2")
    def test_qk_norm_rope_fusion_basic(self):
        """
        Test GraphEXQKNormRopeFusionPattern:
        qkv.split + npu_rms_norm(q/k) + npu_apply_rotary_pos_emb -> vllm.qkv_rmsnorm_rope
        """

        def f(qkv, q_weight, k_weight, cos, sin):
            q, k, v = qkv.split([self.Q_SIZE, self.KV_SIZE, self.KV_SIZE], dim=-1)

            q_by_head = q.view(
                *q.shape[:-1], q.shape[-1] // self.HEAD_DIM, self.HEAD_DIM
            )
            q_norm_out, _ = torch.ops.npu.npu_rms_norm(q_by_head, q_weight, self.EPS)

            k_by_head = k.view(
                *q.shape[:-1], k.shape[-1] // self.HEAD_DIM, self.HEAD_DIM
            )
            k_norm_out, _ = torch.ops.npu.npu_rms_norm(k_by_head, k_weight, self.EPS)

            q_flat = q_norm_out.view(q.shape)
            q_reshape = q_flat.contiguous().view(1, q_flat.shape[0], -1, self.HEAD_DIM)

            k_flat = k_norm_out.view(k.shape)
            k_reshape = k_flat.contiguous().view(1, k_flat.shape[0], -1, self.HEAD_DIM)

            q_rope, k_rope = torch.ops.npu.npu_apply_rotary_pos_emb(
                q_reshape, k_reshape, cos, sin
            )

            return q_rope, k_rope, v

        def assert_basic_fusion(graph_before, graph_after):
            """Verify fusion happened"""

            has_split = any(
                "split" in str(n.target)
                for n in graph_before.graph.nodes
                if n.op == "call_function"
            )
            has_rms_norm = any(
                "npu_rms_norm" in str(n.target)
                for n in graph_before.graph.nodes
                if n.op == "call_function"
            )
            has_rope = any(
                "npu_apply_rotary_pos_emb" in str(n.target)
                for n in graph_before.graph.nodes
                if n.op == "call_function"
            )
            has_contiguous = any(
                "contiguous" in str(n.target)
                for n in graph_before.graph.nodes
                if n.op == "call_function"
            )

            has_fused_op = any(
                "qkv_rmsnorm_rope" in str(n.target)
                for n in graph_after.graph.nodes
                if n.op == "call_function"
            )

            if has_split and has_rms_norm and has_rope and has_contiguous:
                self.assertTrue(
                    has_fused_op,
                    "qkv.split + npu_rms_norm + rope should be fused to vllm.qkv_rmsnorm_rope",
                )

        backend = self._create_backend_with_wrapper(assert_basic_fusion)

        try:
            model = torch.compile(f, backend=backend, dynamic=True)
            inputs = self._get_input_tensors(with_bias=False)
            model(*inputs)
        finally:
            self.tearDown()

    @unittest.skipIf(torch.__version__ < "2.2", "torch.compile requires torch >= 2.2")
    def test_qk_norm_rope_fusion_with_bias(self):
        """
        Test GraphEXQKNormRopeFusionPatternWithBias:
        qkv.split + npu_rms_norm(q/k) + add(bias) + npu_apply_rotary_pos_emb -> vllm.qkv_rmsnorm_rope(with bias)
        """

        def f(qkv, q_weight, k_weight, q_bias, k_bias, cos, sin):
            q, k, v = qkv.split([self.Q_SIZE, self.KV_SIZE, self.KV_SIZE], dim=-1)

            q_by_head = q.view(
                *q.shape[:-1], q.shape[-1] // self.HEAD_DIM, self.HEAD_DIM
            )
            q_norm_out, _ = torch.ops.npu.npu_rms_norm(q_by_head, q_weight, self.EPS)
            q_normed = q_norm_out + q_bias

            k_by_head = k.view(
                *q.shape[:-1], k.shape[-1] // self.HEAD_DIM, self.HEAD_DIM
            )
            k_norm_out, _ = torch.ops.npu.npu_rms_norm(k_by_head, k_weight, self.EPS)
            k_normed = k_norm_out + k_bias

            q_flat = q_normed.view(q.shape)
            q_reshape = q_flat.contiguous().view(1, q_flat.shape[0], -1, self.HEAD_DIM)

            k_flat = k_normed.view(k.shape)
            k_reshape = k_flat.contiguous().view(1, k_flat.shape[0], -1, self.HEAD_DIM)

            q_rope, k_rope = torch.ops.npu.npu_apply_rotary_pos_emb(
                q_reshape, k_reshape, cos, sin
            )

            return q_rope, k_rope, v

        def assert_bias_fusion(graph_before, graph_after):
            """Verify fusion happened with bias"""

            has_split = any(
                "split" in str(n.target)
                for n in graph_before.graph.nodes
                if n.op == "call_function"
            )
            has_rms_norm = any(
                "npu_rms_norm" in str(n.target)
                for n in graph_before.graph.nodes
                if n.op == "call_function"
            )
            has_rope = any(
                "npu_apply_rotary_pos_emb" in str(n.target)
                for n in graph_before.graph.nodes
                if n.op == "call_function"
            )
            has_bias_add = any(
                str(n.target) in ["aten.add.Tensor", "torch.add"]
                for n in graph_before.graph.nodes
                if n.op == "call_function"
            )

            has_fused_op = any(
                "qkv_rmsnorm_rope" in str(n.target)
                for n in graph_after.graph.nodes
                if n.op == "call_function"
            )

            if has_split and has_rms_norm and has_rope and has_bias_add:
                self.assertTrue(
                    has_fused_op,
                    "qkv.split + npu_rms_norm + add(bias) + rope should be fused to vllm.qkv_rmsnorm_rope",
                )

        backend = self._create_backend_with_wrapper(assert_bias_fusion)

        try:
            model = torch.compile(f, backend=backend, dynamic=True)
            inputs = self._get_input_tensors(with_bias=True)
            model(*inputs)
        finally:
            self.tearDown()
