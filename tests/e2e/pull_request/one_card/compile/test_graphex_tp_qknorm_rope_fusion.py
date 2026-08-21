import copy

import npugraph_ex as nge
import numpy as np
import pytest
import torch
import torch.nn as nn
import vllm.config
from vllm.config import ModelConfig, VllmConfig
from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
from vllm.utils.system_utils import update_environment_variables

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.compilation.passes.tp_qknorm_rope_fusion_pass import TPQKNormRopeFusionPattern
from vllm_ascend.ops import rotary_embedding as rope_module
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

MAX_POSITION_EMBEDDING = 8192


def find_op(gm, op_default):
    return any(node.op == "call_function" and node.target == op_default for node in gm.graph.nodes)


def create_pattern_wrapper(assert_func):
    original_func = nge.npu_fx_compiler._optimize_fx

    def wrapper(gm, example_inputs=None, config=None):
        ret = original_func(gm, example_inputs, config)
        graph_after = copy.deepcopy(gm)
        assert_func(graph_after)
        return ret

    return wrapper


class ModelTPQKNormRope(nn.Module):
    """Eager MiniMax-style TP qk-norm + rope sequence the pass replaces.

    Norms run over the full q/k channel dim (weights are [q_size] / [kv_size]),
    exactly like ``MiniMaxText01RMSNormTP.forward_qkv`` under TP > 1. The test
    runs single-card with tp_world=1 baked into the op constants, so the fused
    kernel skips the TP all-reduce while exercising the same match/replace
    path (the tp_world > 1 all-reduce path is covered by multi-card runs).
    """

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        eps: float = 1e-6,
        device="npu",
    ):
        super().__init__()
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.eps = eps
        self.tp_rank = 0
        self.tp_world = 1

        self.q_weight = nn.Parameter(torch.randn(self.q_size, dtype=dtype, device=device))
        self.k_weight = nn.Parameter(torch.randn(self.kv_size, dtype=dtype, device=device))

    def forward(self, qkv, cos_sin_cache, positions):
        q, k = torch.ops.vllm.minimax_qk_norm_fusion(
            qkv,
            self.q_weight,
            self.k_weight,
            self.q_size,
            self.kv_size,
            self.tp_rank,
            self.tp_world,
            self.eps,
            None,
        )
        _, _, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_rope, k_rope = torch.ops.vllm.npu_rotary_embedding(
            positions, q, k, cos_sin_cache, self.head_dim, self.rope_dim, True
        )
        return q_rope, k_rope, v


def assert_tp_qknorm_rope_fusion(after_gm, expect_fused=True):
    check_rules = [
        (torch.ops.vllm.qkv_tp_rmsnorm_rope.default, expect_fused),
        (torch.ops.vllm.minimax_qk_norm_fusion.default, not expect_fused),
        (torch.ops.vllm.npu_rotary_embedding.default, not expect_fused),
    ]
    for torch_op, expect_exist in check_rules:
        found = find_op(after_gm, torch_op)
        if expect_exist:
            assert found, f"Expected operator '{torch_op}' but not find"
        else:
            assert not found, f"Not expected operator '{torch_op}' but find"


@pytest.fixture(scope="module", autouse=True)
def init_triton():
    init_device_properties_triton()


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [257])
@pytest.mark.parametrize("eps", [1e-6])
def test_tp_qknorm_rope_fusion(dtype: torch.dtype, num_tokens: int, eps: float, tmp_path):
    # A local minimal MiniMax-M2 config keeps ModelConfig hermetic (no HF
    # network access) and makes get_rope_dim() resolve rotary_dim=64.
    import json

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "architectures": ["MiniMaxM2ForCausalLM"],
                "model_type": "minimax_m2",
                "hidden_size": 512,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 128,
                "rotary_dim": 64,
                "rms_norm_eps": eps,
                "vocab_size": 1024,
                "intermediate_size": 1024,
            }
        )
    )
    vllm_config = VllmConfig(model_config=ModelConfig(model=str(tmp_path), dtype=dtype))
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

    num_heads = 4
    num_kv_heads = 2
    head_dim = 128
    rope_dim = 64
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    qkv_size = q_size + 2 * kv_size

    with vllm.config.set_current_vllm_config(vllm_config), set_ascend_forward_context(None, vllm_config):
        model = ModelTPQKNormRope(head_dim, num_heads, num_kv_heads, rope_dim, dtype, eps, device="npu").to("npu")
        fusion_pattern = TPQKNormRopeFusionPattern(
            vllm_config=vllm_config,
            head_dim=head_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            tp_rank=0,
            tp_world=1,
            eps=eps,
        )
        from torch._inductor.pattern_matcher import PatternMatcherPass

        pm_pass = PatternMatcherPass()
        fusion_pattern.register(pm_pass)

        qkv = torch.randn(num_tokens, qkv_size, device="npu", dtype=dtype)
        cos_sin_cache = torch.from_numpy(np.random.uniform(0, 1, [MAX_POSITION_EMBEDDING, rope_dim])).to(dtype).npu()
        positions = torch.randint(
            low=0, high=MAX_POSITION_EMBEDDING, size=(num_tokens,), dtype=torch.int64, device="npu"
        )

        # Populate the runner-maintained per-step cos/sin slices that the fused
        # op reads at execution time; reset them afterwards so the module
        # globals stay pristine for other tests in the session.
        saved_state = (rope_module._cos, rope_module._sin, rope_module._cos_sin_cache)
        try:
            rope_module._record_cos_sin_cache(cos_sin_cache)
            rope_module._cos = torch.empty(1, num_tokens, 1, rope_dim, dtype=dtype, device="npu")
            rope_module._sin = torch.empty(1, num_tokens, 1, rope_dim, dtype=dtype, device="npu")
            rope_module.update_cos_sin(positions)

            # Warm up the fused op eagerly first: its split kernel is
            # @triton.autotune'd, and the autotune benchmark synchronizes the
            # device, which must not happen inside npugraph_ex capture.
            with torch.no_grad():
                torch.ops.vllm.qkv_tp_rmsnorm_rope(
                    qkv,
                    model.q_weight,
                    model.k_weight,
                    q_size,
                    kv_size,
                    head_dim,
                    rope_dim,
                    eps,
                    1,
                )

            with torch.no_grad():
                original_optimize = nge.npu_fx_compiler._optimize_fx
                nge.npu_fx_compiler._optimize_fx = create_pattern_wrapper(
                    lambda gm: assert_tp_qknorm_rope_fusion(gm, expect_fused=True)
                )

                compiled_model = torch.compile(model, backend="npugraph_ex", fullgraph=True, dynamic=True)

                compiled_model(qkv, cos_sin_cache, positions)

                nge.npu_fx_compiler._optimize_fx = original_optimize
        finally:
            rope_module._cos, rope_module._sin, rope_module._cos_sin_cache = saved_state
            rope_module._cos_slice = None
            rope_module._sin_slice = None
