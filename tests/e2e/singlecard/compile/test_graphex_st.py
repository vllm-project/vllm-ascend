import re

import pytest
import torch
import torch.fx as fx
import torch.nn as nn
import torchair
import vllm
from torch._dynamo.backends.common import aot_autograd
from vllm.config import ModelConfig, VllmConfig
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.utils.system_utils import update_environment_variables

from vllm_ascend.ascend_forward_context import set_ascend_forward_context

# ---------------------------------------------------------------------------
# System-test backend
# ---------------------------------------------------------------------------


class TestBackendST:
    """
    System-test backend for GraphEX fusion.

    Principle:
    - Let torch.compile go through full AOT + torchair
    - Observe which NPU ops torchair attempts to lower
    - Do NOT inspect FX graph
    """

    def __init__(self):
        self.lowered_ops: set[str] = set()

    def _extract_npu_ops(self, msg: str):
        """
        Extract npu op names from torchair / AscendIR error message.
        Example:
            Failed to converter: npu_add_rms_norm_quant to AscendIR
        """
        for m in re.findall(r"npu_[a-zA-Z0-9_]+", msg):
            self.lowered_ops.add(m)

    def __call__(self, gm: fx.GraphModule, example_inputs):
        assert example_inputs is not None

        def fw_compiler(graph_module: fx.GraphModule, example_inputs_inner):
            config = torchair.CompilerConfig()
            npu_backend = torchair.get_npu_backend(compiler_config=config)

            try:
                return npu_backend(graph_module, example_inputs_inner)

            except RuntimeError as e:
                msg = str(e)

                # Expected in ST: AscendIR conversion may fail
                if "Failed to converter" in msg and "AscendIR" in msg:
                    self._extract_npu_ops(msg)
                    # Fallback to original graph_module for testing
                    return graph_module
                else:
                    raise

        return aot_autograd(
            fw_compiler=fw_compiler,
        )(gm, example_inputs)


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------


class TestModelWithoutBias(nn.Module):
    def __init__(self, hidden_size, dtype, eps, device="npu"):
        super().__init__()
        self.eps = eps
        self.rms_norm_weight = nn.Parameter(torch.randn(hidden_size, device=device))
        self.scale = torch.ones(hidden_size, dtype=dtype, device=device)
        self.scale_reciprocal = torch.ones(hidden_size, dtype=dtype, device=device)
        self.offset = torch.zeros(hidden_size, dtype=dtype, device=device)

    def forward(self, x):
        residual = torch.zeros_like(x)
        norm, _, new_residual = torch.ops.npu.npu_add_rms_norm(x, residual, self.rms_norm_weight, self.eps)
        q = torch.ops.vllm.quantize(norm, self.scale, self.scale_reciprocal, self.offset)
        return q, new_residual


class TestModelWithBias(nn.Module):
    def __init__(self, hidden_size, dtype, eps, device="npu"):
        super().__init__()
        self.eps = eps
        self.rms_norm_weight = nn.Parameter(torch.randn(hidden_size, device=device))
        self.bias = nn.Parameter(torch.randn(hidden_size, device=device))
        self.scale = torch.ones(hidden_size, dtype=dtype, device=device)
        self.scale_reciprocal = torch.ones(hidden_size, dtype=dtype, device=device)
        self.offset = torch.zeros(hidden_size, dtype=dtype, device=device)

    def forward(self, x):
        residual = torch.zeros_like(x)
        norm, _, new_residual = torch.ops.npu.npu_add_rms_norm(x, residual, self.rms_norm_weight, self.eps)
        norm = norm + self.bias
        q = torch.ops.vllm.quantize(norm, self.scale, self.scale_reciprocal, self.offset)
        return q, new_residual


class TestModelSPWithoutBias(nn.Module):
    def __init__(self, hidden_size, dtype, eps, device="npu"):
        super().__init__()
        self.eps = eps
        self.rms_norm_weight = nn.Parameter(torch.randn(hidden_size, device=device))
        self.scale = torch.ones(hidden_size, dtype=dtype, device=device)
        self.scale_reciprocal = torch.ones(hidden_size, dtype=dtype, device=device)
        self.offset = torch.zeros(hidden_size, dtype=dtype, device=device)

    def forward(self, x):
        residual = torch.zeros_like(x)
        norm, _, new_residual = torch.ops.npu.npu_add_rms_norm(x, residual, self.rms_norm_weight, self.eps)
        norm = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(norm, True)
        q = torch.ops.vllm.quantize(norm, self.scale, self.scale_reciprocal, self.offset)
        return q, new_residual


class TestModelSPWithBias(nn.Module):
    def __init__(self, hidden_size, dtype, eps, device="npu"):
        super().__init__()
        self.eps = eps
        self.rms_norm_weight = nn.Parameter(torch.randn(hidden_size, device=device))
        self.bias = nn.Parameter(torch.randn(hidden_size, device=device))
        self.scale = torch.ones(hidden_size, dtype=dtype, device=device)
        self.scale_reciprocal = torch.ones(hidden_size, dtype=dtype, device=device)
        self.offset = torch.zeros(hidden_size, dtype=dtype, device=device)

    def forward(self, x):
        residual = torch.zeros_like(x)
        norm, _, new_residual = torch.ops.npu.npu_add_rms_norm(x, residual, self.rms_norm_weight, self.eps)
        norm = norm + self.bias
        norm = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(norm, True)
        q = torch.ops.vllm.quantize(norm, self.scale, self.scale_reciprocal, self.offset)
        return q, new_residual


# ---------------------------------------------------------------------------
# PyTest
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [64])
@pytest.mark.parametrize("num_tokens", [257])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("sp_enable", [False, True])
def test_graphex_addrmsnorm_quant_st(
    dtype,
    hidden_size,
    num_tokens,
    eps,
    use_bias,
    sp_enable,
):
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)

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

    with vllm.config.set_current_vllm_config(vllm_config), set_ascend_forward_context(None, vllm_config):
        backend = TestBackendST()

        if use_bias:
            model = (TestModelSPWithBias if sp_enable else TestModelWithBias)(hidden_size, dtype, eps)
        else:
            model = (TestModelSPWithoutBias if sp_enable else TestModelWithoutBias)(hidden_size, dtype, eps)

        # model = model.to("npu")

        x = torch.rand(
            num_tokens,
            hidden_size,
            device="npu",
            dtype=dtype,
            requires_grad=False,
        )

        compiled = torch.compile(model, backend=backend)
        compiled(x)

        # ---------------- ST ASSERT ----------------
        assert "npu_add_rms_norm_quant" in backend.lowered_ops, (
            f"GraphEX AddRMSNorm+Quant fusion not observed.\nLowered ops: {backend.lowered_ops}"
        )
