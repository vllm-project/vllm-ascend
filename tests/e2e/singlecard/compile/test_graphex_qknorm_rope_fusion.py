import copy

import pytest
import torch
import torch.nn as nn
import torchair
import vllm.config
from vllm.config import ModelConfig, VllmConfig
from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
from vllm.utils.system_utils import update_environment_variables

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.compilation.passes.qknorm_rope_fusion_pass import (
    QKNormRopeFusionPattern,
    QKNormRopeFusionPatternWithBias,
)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def find_op(gm, op_default):
    return any(node.op == "call_function" and node.target == op_default for node in gm.graph.nodes)


def create_pattern_wrapper(assert_func):
    original_func = torchair.npu_fx_compiler._optimize_fx

    def wrapper(gm, example_inputs=None, config=None):
        ret = original_func(gm, example_inputs, config)
        graph_after = copy.deepcopy(gm)
        assert_func(graph_after)
        return ret

    return wrapper


@pytest.fixture(scope="module", autouse=True)
def init_triton():
    init_device_properties_triton()


class ModelQKNormRopeWithoutBias(nn.Module):
    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        dtype: torch.dtype = torch.bfloat16,
        eps: float = 1e-6,
        device="npu",
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.eps = eps

        # RMSNorm weight per head (shared across heads of same type)
        self.q_weight = nn.Parameter(torch.randn(head_dim, dtype=dtype, device=device))
        self.k_weight = nn.Parameter(torch.randn(head_dim, dtype=dtype, device=device))

    def forward(self, qkv, cos_sin_cache, positions):
        """
        Args:
            qkv: [T, q_size + 2*kv_size]
            cos: [1, T, 1, head_dim]
            sin: [1, T, 1, head_dim]
        Returns:
            q_rope, k_rope, v
        """
        # Split QKV
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Q RMSNorm (per-head)
        q_by_head = q.view(*q.shape[:-1], self.num_heads, self.head_dim)
        q_norm_out, _ = torch.ops.npu.npu_rms_norm(q_by_head, self.q_weight, self.eps)

        # K RMSNorm (per-head)
        k_by_head = k.view(*k.shape[:-1], self.num_kv_heads, self.head_dim)
        k_norm_out, _ = torch.ops.npu.npu_rms_norm(k_by_head, self.k_weight, self.eps)

        # Reshape for RoPE: [T, num_heads, head_dim] -> [1, T, num_heads, head_dim]
        q_flat = q_norm_out.view(q.shape)
        k_flat = k_norm_out.view(k.shape)

        # Apply RoPE
        q_rope, k_rope = torch.ops.vllm.npu_rotary_embedding(
            positions, q_flat, k_flat, cos_sin_cache, self.head_dim, self.head_dim, True
        )

        return q_rope, k_rope, v


class ModelQKNormRopeWithBias(nn.Module):
    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        dtype: torch.dtype = torch.bfloat16,
        eps: float = 1e-6,
        device="npu",
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.eps = eps

        self.q_weight = nn.Parameter(torch.randn(head_dim, dtype=dtype, device=device))
        self.k_weight = nn.Parameter(torch.randn(head_dim, dtype=dtype, device=device))
        self.q_bias = nn.Parameter(torch.randn(head_dim, dtype=dtype, device=device))
        self.k_bias = nn.Parameter(torch.randn(head_dim, dtype=dtype, device=device))

    def forward(self, qkv, cos_sin_cache, positions):
        # Split QKV
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Q RMSNorm + Bias
        q_by_head = q.view(*q.shape[:-1], self.num_heads, self.head_dim)
        q_norm_out, _ = torch.ops.npu.npu_rms_norm(q_by_head, self.q_weight, self.eps)
        q_normed = q_norm_out + self.q_bias

        # K RMSNorm + Bias
        k_by_head = k.view(*k.shape[:-1], self.num_kv_heads, self.head_dim)
        k_norm_out, _ = torch.ops.npu.npu_rms_norm(k_by_head, self.k_weight, self.eps)
        k_normed = k_norm_out + self.k_bias

        # Reshape for RoPE
        q_flat = q_normed.view(q.shape)
        k_flat = k_normed.view(k.shape)

        # Apply RoPE
        q_rope, k_rope = torch.ops.vllm.npu_rotary_embedding(
            positions, q_flat, k_flat, cos_sin_cache, self.head_dim, self.head_dim, True
        )

        return q_rope, k_rope, v


def assert_qknorm_rope_fusion(after_gm, expect_fused=True, use_bias=False):
    check_rules = [
        (torch.ops.vllm.qkv_rmsnorm_rope.default, expect_fused),
        (torch.ops.npu.npu_rms_norm.default, not expect_fused),
        (torch.ops.vllm.npu_rotary_embedding.default, not expect_fused),
    ]
    if use_bias:
        check_rules.append((torch.ops.aten.add.Tensor, not expect_fused))
    for torch_op, expect_exist in check_rules:
        found = find_op(after_gm, torch_op)
        if expect_exist:
            assert found, f"Expected operator '{torch_op}' but not find"
        else:
            assert not found, f"Not expected operator '{torch_op}' but find"


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [64])
@pytest.mark.parametrize("num_tokens", [257])
@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("use_bias", [False, True])
def test_rmsnorm_quant_fusion(
    dtype: torch.dtype,
    hidden_size: int,
    num_tokens: int,
    eps: float,
    use_bias: bool,
):
    vllm_config = VllmConfig(model_config=ModelConfig(dtype=dtype))
    with vllm.config.set_current_vllm_config(vllm_config):
        update_environment_variables(
            {
                "RANK": "0",
                "LOCAL_RANK": "0",
                "WORLD_SIZE": "1",
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "12345",
            }
        )
        init_distributed_environment()
        ensure_model_parallel_initialized(1, 1)
    num_heads = 16
    num_kv_heads = 8
    head_dim = 128
    with vllm.config.set_current_vllm_config(vllm_config), set_ascend_forward_context(None, vllm_config):
        fusion_pattern = None
        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim
        qkv_size = q_size + 2 * kv_size
        if use_bias:
            model = ModelQKNormRopeWithBias(head_dim, num_heads, num_kv_heads, dtype, eps, device="npu")
            fusion_pattern = QKNormRopeFusionPatternWithBias(
                vllm_config=vllm_config, head_dim=head_dim, num_heads=num_heads, num_kv_heads=num_kv_heads, eps=eps
            )
        else:
            model = ModelQKNormRopeWithoutBias(head_dim, num_heads, num_kv_heads, dtype, eps, device="npu")
            fusion_pattern = QKNormRopeFusionPattern(
                vllm_config=vllm_config, head_dim=head_dim, num_heads=num_heads, num_kv_heads=num_kv_heads, eps=eps
            )
        from torch._inductor.pattern_matcher import PatternMatcherPass
        pm_pass = PatternMatcherPass()
        fusion_pattern.register(pm_pass)
        model = model.to("npu")
        seq_len = 5
        qkv = torch.randn(seq_len, qkv_size, device="npu", dtype=dtype)
        cos = torch.randn(1, seq_len, 1, head_dim, device="npu", dtype=dtype)
        sin = torch.randn(1, seq_len, 1, head_dim, device="npu", dtype=dtype)

        with torch.no_grad():
            original_optimize = torchair.npu_fx_compiler._optimize_fx
            torchair.npu_fx_compiler._optimize_fx = create_pattern_wrapper(
                lambda gm: assert_qknorm_rope_fusion(gm, expect_fused=True, use_bias=use_bias)
            )

            compiled_model = torch.compile(model, backend="npugraph_ex", fullgraph=True, dynamic=True)

            compiled_model(qkv, cos, sin)

            torchair.npu_fx_compiler._optimize_fx = original_optimize
