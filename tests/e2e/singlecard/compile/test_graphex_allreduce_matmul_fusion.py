import copy

import pytest
import torch
import torch.nn as nn
import torchair
import vllm.config
from vllm.config import ModelConfig, VllmConfig
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
    tensor_model_parallel_all_reduce,
)
from vllm.utils.system_utils import update_environment_variables

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.compilation.npugraph_ex_passes.graphex_allreduce_rmsnorm_fusion_pass import (
    GraphEXLastLayerMatmulAllReduceAddRMSNormPattern,
    GraphEXMiddleLayerMatmulAllReduceAddRMSNormPattern,
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


class ModelMiddleLayerMatmulAllReduceRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, dtype: torch.dtype = torch.bfloat16, eps: float = 1e-6, device="npu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

        self.weight = nn.Parameter(torch.randn(hidden_size, hidden_size, dtype=dtype, device=device))
        self.rms_norm_weight = nn.Parameter(torch.randn(hidden_size, dtype=dtype, device=device))

    def forward(self, x, residual):
        """
        Args:
            x: [batch, seq, hidden]
            residual: [batch, seq, hidden]
        Returns:
            norm_output, updated_residual
        """
        mm = torch.ops.vllm.unquantized_gemm(x, self.weight, None)

        all_reduce_out = tensor_model_parallel_all_reduce(mm)

        output = torch.ops._C_ascend.npu_add_rms_norm_bias(all_reduce_out, residual, self.rms_norm_weight, None)

        return output[0], output[2]


class ModelLastLayerMatmulAllReduceRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, dtype: torch.dtype = torch.bfloat16, eps: float = 1e-6, device="npu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

        self.weight = nn.Parameter(torch.randn(hidden_size, hidden_size, dtype=dtype, device=device))
        self.rms_norm_weight = nn.Parameter(torch.randn(hidden_size, dtype=dtype, device=device))

    def forward(self, x, residual):
        """
        Args:
            x: [batch, seq, hidden]
            residual: [batch, seq, hidden]
        Returns:
            norm_output
        """
        # MatMul
        mm = torch.ops.vllm.unquantized_gemm(x, self.weight, None)

        all_reduce_out = tensor_model_parallel_all_reduce(mm)

        output = torch.ops._C_ascend.npu_add_rms_norm_bias(all_reduce_out, residual, self.rms_norm_weight, None)
        return output[0]


def assert_allreduce_matmul_fusion(after_gm, expect_fused=True, is_middle_layer=True):
    check_rules = [
        (torch.ops.vllm.unquantized_gemm.default, not expect_fused),
        (torch.ops._C_ascend.npu_add_rms_norm_bias.default, expect_fused),
    ]
    for torch_op, expect_exist in check_rules:
        found = find_op(after_gm, torch_op)
        if expect_exist:
            assert found, f"Expected operator '{torch_op}' but not find"
        else:
            assert not found, f"Not expected operator '{torch_op}' but find"


_registered_patterns = set()


def register_pattern_safe(pattern_class, vllm_config, eps, pattern_key):
    global _registered_patterns
    if pattern_key in _registered_patterns:
        print(f"Pattern {pattern_key} already registered, skipping...")
        return None

    pattern = pattern_class(vllm_config=vllm_config, eps=eps)
    try:
        pattern.register()
        _registered_patterns.add(pattern_key)
        print(f"Successfully registered pattern: {pattern_key}")
    except RuntimeError as e:
        if "Duplicate pattern" in str(e):
            print(f"Pattern {pattern_key} already exists (caught from RuntimeError), skipping...")
            _registered_patterns.add(pattern_key)
        else:
            raise e
    return pattern


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [64])
@pytest.mark.parametrize("num_tokens", [257])
@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("is_middle_layer", [True, False])
def test_rmsnorm_quant_fusion(
    dtype: torch.dtype,
    hidden_size: int,
    num_tokens: int,
    eps: float,
    is_middle_layer: bool,
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
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="hccl", rank=0, world_size=1)
        init_distributed_environment(backend="hccl")
        ensure_model_parallel_initialized(1, 1)

    with vllm.config.set_current_vllm_config(vllm_config), set_ascend_forward_context(None, vllm_config):
        if is_middle_layer:
            model = ModelMiddleLayerMatmulAllReduceRMSNorm(hidden_size, dtype, eps, device="npu")
            register_pattern_safe(
                GraphEXMiddleLayerMatmulAllReduceAddRMSNormPattern,
                vllm_config,
                eps,
                "GraphEXMiddleLayerMatmulAllReduceAddRMSNormPattern",
            )
        else:
            model = ModelLastLayerMatmulAllReduceRMSNorm(hidden_size, dtype, eps, device="npu")
            register_pattern_safe(
                GraphEXLastLayerMatmulAllReduceAddRMSNormPattern,
                vllm_config,
                eps,
                "GraphEXLastLayerMatmulAllReduceAddRMSNormPattern",
            )
        batch_size = 2
        seq_len = 5
        model = model.to("npu")
        x = torch.randn(batch_size, seq_len, hidden_size, device="npu", dtype=dtype)
        residual = torch.randn(batch_size, seq_len, hidden_size, device="npu", dtype=dtype)

        with torch.no_grad():
            original_optimize = torchair.npu_fx_compiler._optimize_fx
            torchair.npu_fx_compiler._optimize_fx = create_pattern_wrapper(
                lambda gm: assert_allreduce_matmul_fusion(gm, expect_fused=True, is_middle_layer=is_middle_layer)
            )

            compiled_model = torch.compile(model, backend="npugraph_ex", fullgraph=True, dynamic=True)

            compiled_model(x, residual)

            torchair.npu_fx_compiler._optimize_fx = original_optimize
