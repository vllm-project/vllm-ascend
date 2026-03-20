import random
import math

import numpy as np
import pytest
import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def rms_norm(x, eps):
    """RMSNorm: x / sqrt(mean(x^2) + eps)"""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x / torch.sqrt(variance + eps)


def gelu_tanh(x):
    """GELU with tanh approximation"""
    sqrt_2_pi = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + torch.tanh(sqrt_2_pi * (x + 0.044715 * x.pow(3))))


def sum_lstm_golden(states_4d, z4_4d, prev_cell,
                    w_cell=None, b_cell=None,
                    w_state=None, b_state=None,
                    alpha=1.0, eps_cell=1e-6, eps_state=1e-6,
                    use_fast_gelu=True):
    """
    Golden implementation of SumLstm.

    states_4d: shape (..., 4*D) - contains [s0, s1, s2, s3] for gates
    z4_4d: shape (..., 4*D) - contains [z0, z1, z2, z3] for gates
    prev_cell: shape (..., D) - previous cell state

    Returns:
        out_state: shape (..., D)
        out_cell: shape (..., D)
    """
    # Convert to float32 for computation
    states_4d = states_4d.float()
    z4_4d = z4_4d.float()
    prev_cell = prev_cell.float()

    hidden_dim = prev_cell.shape[-1]

    # Separate four gate inputs
    s0 = states_4d[..., 0:hidden_dim]           # forget gate
    s1 = states_4d[..., hidden_dim:2*hidden_dim]  # input gate
    s2 = states_4d[..., 2*hidden_dim:3*hidden_dim]  # output gate
    s3 = states_4d[..., 3*hidden_dim:4*hidden_dim]  # cell gate

    z0 = z4_4d[..., 0:hidden_dim]
    z1 = z4_4d[..., hidden_dim:2*hidden_dim]
    z2 = z4_4d[..., 2*hidden_dim:3*hidden_dim]
    z3 = z4_4d[..., 3*hidden_dim:4*hidden_dim]

    # Step 1: Compute pre-activation values
    pre_f = s0 + alpha * z0  # forget gate
    pre_i = s1 + alpha * z1  # input gate
    cpre = s3 + alpha * z3   # cell pre-activation (note: s3, z3)

    # Step 2: Sigmoid on preF, preI
    f = torch.sigmoid(pre_f)
    i = torch.sigmoid(pre_i)

    # Step 3: RMSNorm on cpre
    out_cell_f = rms_norm(cpre, eps_cell)

    # Step 4: Optional weights w_cell, b_cell
    if w_cell is not None:
        out_cell_f = out_cell_f * w_cell.float()
    if b_cell is not None:
        out_cell_f = out_cell_f + b_cell.float()

    # Step 5: GELU
    if use_fast_gelu:
        cact = gelu_tanh(out_cell_f)
    else:
        cact = torch.nn.functional.gelu(out_cell_f)

    # Step 6: out_cell = prev_cell * f + cact * i
    out_cell = prev_cell * f + cact * i

    # Step 7: Delayed compute output gate
    pre_o = s2 + alpha * z2
    o = torch.sigmoid(pre_o)

    # Step 8: RMSNorm on out_cell
    out_state_f = rms_norm(out_cell, eps_state)

    # Step 9: Optional weights w_state, b_state
    if w_state is not None:
        out_state_f = out_state_f * w_state.float()
    if b_state is not None:
        out_state_f = out_state_f + b_state.float()

    # Step 10: GELU
    if use_fast_gelu:
        sact = gelu_tanh(out_state_f)
    else:
        sact = torch.nn.functional.gelu(out_state_f)

    # Step 11: out_state = sact * o
    out_state = sact * o

    return out_state, out_cell


@pytest.mark.parametrize(
    'batch_size',
    [1, 4, 16, 32, 64],
)
@pytest.mark.parametrize(
    'hidden_dim',
    [64, 128, 256, 512, 1024],
)
@pytest.mark.parametrize(
    'use_optional_weights',
    [False, True],
)
def test_sum_lstm_basic(batch_size: int, hidden_dim: int, use_optional_weights: bool):
    """Test SumLstm with basic configurations."""
    dtype = torch.bfloat16
    atol = 0.01
    rtol = 0.01

    # Create input data
    states_4d = torch.randn(batch_size, 4 * hidden_dim, dtype=dtype)
    z4_4d = torch.randn(batch_size, 4 * hidden_dim, dtype=dtype)
    prev_cell = torch.randn(batch_size, hidden_dim, dtype=dtype)

    # Optional weights
    w_cell = torch.randn(hidden_dim, dtype=dtype) if use_optional_weights else None
    b_cell = torch.randn(hidden_dim, dtype=dtype) if use_optional_weights else None
    w_state = torch.randn(hidden_dim, dtype=dtype) if use_optional_weights else None
    b_state = torch.randn(hidden_dim, dtype=dtype) if use_optional_weights else None

    # NPU computation
    out_state, out_cell = torch.ops._C_ascend.npu_sum_lstm(
        states_4d.npu(),
        z4_4d.npu(),
        prev_cell.npu(),
        w_cell.npu() if w_cell is not None else None,
        b_cell.npu() if b_cell is not None else None,
        w_state.npu() if w_state is not None else None,
        b_state.npu() if b_state is not None else None,
        1.0,   # alpha
        1e-6,  # eps_cell
        1e-6,  # eps_state
        True   # use_fast_gelu
    )
    out_state = out_state.cpu()
    out_cell = out_cell.cpu()

    # Golden computation
    out_state_golden, out_cell_golden = sum_lstm_golden(
        states_4d, z4_4d, prev_cell,
        w_cell, b_cell, w_state, b_state,
        alpha=1.0, eps_cell=1e-6, eps_state=1e-6,
        use_fast_gelu=True
    )
    out_state_golden = out_state_golden.to(dtype)
    out_cell_golden = out_cell_golden.to(dtype)

    # Verify results
    torch.testing.assert_close(out_state, out_state_golden, atol=atol, rtol=rtol)
    torch.testing.assert_close(out_cell, out_cell_golden, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    'alpha',
    [0.5, 1.0, 2.0],
)
@pytest.mark.parametrize(
    'eps_cell, eps_state',
    [(1e-6, 1e-6), (1e-5, 1e-5), (1e-8, 1e-8)],
)
def test_sum_lstm_params(alpha: float, eps_cell: float, eps_state: float):
    """Test SumLstm with different alpha and epsilon parameters."""
    batch_size = 16
    hidden_dim = 256
    dtype = torch.bfloat16
    atol = 0.01
    rtol = 0.01

    # Create input data
    states_4d = torch.randn(batch_size, 4 * hidden_dim, dtype=dtype)
    z4_4d = torch.randn(batch_size, 4 * hidden_dim, dtype=dtype)
    prev_cell = torch.randn(batch_size, hidden_dim, dtype=dtype)

    # NPU computation
    out_state, out_cell = torch.ops._C_ascend.npu_sum_lstm(
        states_4d.npu(),
        z4_4d.npu(),
        prev_cell.npu(),
        None, None, None, None,
        alpha,
        eps_cell,
        eps_state,
        True
    )
    out_state = out_state.cpu()
    out_cell = out_cell.cpu()

    # Golden computation
    out_state_golden, out_cell_golden = sum_lstm_golden(
        states_4d, z4_4d, prev_cell,
        alpha=alpha, eps_cell=eps_cell, eps_state=eps_state,
        use_fast_gelu=True
    )
    out_state_golden = out_state_golden.to(dtype)
    out_cell_golden = out_cell_golden.to(dtype)

    # Verify results
    torch.testing.assert_close(out_state, out_state_golden, atol=atol, rtol=rtol)
    torch.testing.assert_close(out_cell, out_cell_golden, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    'shape',
    [
        (2, 8, 128),      # 3D: (batch, seq, 4*hidden)
        (2, 4, 8, 256),   # 4D: (batch, heads, seq, 4*hidden)
    ],
)
def test_sum_lstm_multidim(shape):
    """Test SumLstm with multi-dimensional inputs."""
    dtype = torch.bfloat16
    atol = 0.01
    rtol = 0.01

    hidden_dim = shape[-1] // 4
    cell_shape = shape[:-1] + (hidden_dim,)

    # Create input data
    states_4d = torch.randn(*shape, dtype=dtype)
    z4_4d = torch.randn(*shape, dtype=dtype)
    prev_cell = torch.randn(*cell_shape, dtype=dtype)

    # Flatten for NPU (keep last dimension)
    flat_shape = (-1, shape[-1])
    flat_cell_shape = (-1, hidden_dim)

    states_4d_flat = states_4d.reshape(flat_shape)
    z4_4d_flat = z4_4d.reshape(flat_shape)
    prev_cell_flat = prev_cell.reshape(flat_cell_shape)

    # NPU computation
    out_state_flat, out_cell_flat = torch.ops._C_ascend.npu_sum_lstm(
        states_4d_flat.npu(),
        z4_4d_flat.npu(),
        prev_cell_flat.npu(),
        None, None, None, None,
        1.0, 1e-6, 1e-6, True
    )
    out_state = out_state_flat.cpu().reshape(cell_shape)
    out_cell = out_cell_flat.cpu().reshape(cell_shape)

    # Golden computation
    out_state_golden, out_cell_golden = sum_lstm_golden(
        states_4d, z4_4d, prev_cell,
        alpha=1.0, eps_cell=1e-6, eps_state=1e-6,
        use_fast_gelu=True
    )
    out_state_golden = out_state_golden.to(dtype)
    out_cell_golden = out_cell_golden.to(dtype)

    # Verify results
    torch.testing.assert_close(out_state, out_state_golden, atol=atol, rtol=rtol)
    torch.testing.assert_close(out_cell, out_cell_golden, atol=atol, rtol=rtol)
