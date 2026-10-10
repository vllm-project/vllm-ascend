# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-kernel precision tests for the core KDA Triton kernels.

Kernel-to-test mapping (each test directly launches exactly one production
kernel and never calls a production wrapper):

* ``layer_norm_gated_fwd_kernel`` ->
  ``test_layer_norm_gated_fwd_kernel``
* ``layer_norm_gated_fwd_kernel1`` ->
  ``test_layer_norm_gated_fwd_kernel1``
* ``chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter`` ->
  ``test_chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter``
* ``chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra`` ->
  ``test_chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra``
* ``recompute_w_u_fwd_kernel`` -> ``test_recompute_w_u_fwd_kernel``
* ``chunk_gla_fwd_kernel_o`` -> ``test_chunk_gla_fwd_kernel_o``
* ``kda_gate_fwd_kernel`` -> ``test_kda_gate_fwd_kernel``

All references use independent PyTorch operations with FP32 accumulation.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.kda.kda import (
    chunk_gla_fwd_kernel_o,
    chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter,
    chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra,
    kda_gate_fwd_kernel,
    layer_norm_gated_fwd_kernel,
    layer_norm_gated_fwd_kernel1,
    recompute_w_u_fwd_kernel,
)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

DEVICE = "npu"
CHUNK_SIZE = 64
SUB_CHUNK_SIZE = 16
EPS = 1e-5


@dataclass(frozen=True)
class SequenceLayout:
    batch_size: int
    sequence_width: int
    segments: tuple[tuple[int, int, int], ...]
    grid_chunks: int
    total_chunks: int
    cu_seqlens: torch.Tensor | None
    chunk_indices: torch.Tensor | None

    @property
    def is_varlen(self) -> bool:
        return self.cu_seqlens is not None


@pytest.fixture(scope="module", autouse=True)
def init_triton_device_properties():
    init_device_properties_triton()


def _ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _randn(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    *,
    scale: float = 1.0,
) -> torch.Tensor:
    return (torch.randn(shape, dtype=torch.float32) * scale).to(dtype).contiguous()


def _make_layout(is_varlen: bool) -> SequenceLayout:
    if not is_varlen:
        batch_size, sequence_width = 2, 75
        grid_chunks = _ceil_div(sequence_width, CHUNK_SIZE)
        return SequenceLayout(
            batch_size=batch_size,
            sequence_width=sequence_width,
            segments=tuple((batch, 0, sequence_width) for batch in range(batch_size)),
            grid_chunks=grid_chunks,
            total_chunks=batch_size * grid_chunks,
            cu_seqlens=None,
            chunk_indices=None,
        )

    lengths = (37, 70)
    cumulative = [0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)
    indices = [
        (sequence, chunk) for sequence, length in enumerate(lengths) for chunk in range(_ceil_div(length, CHUNK_SIZE))
    ]
    return SequenceLayout(
        batch_size=1,
        sequence_width=cumulative[-1],
        segments=tuple((0, cumulative[index], length) for index, length in enumerate(lengths)),
        grid_chunks=len(indices),
        total_chunks=len(indices),
        cu_seqlens=torch.tensor(cumulative, dtype=torch.int64),
        chunk_indices=torch.tensor(indices, dtype=torch.int64),
    )


def _iter_chunks(layout: SequenceLayout) -> Iterator[tuple[int, int, int, int]]:
    global_chunk = 0
    for batch, sequence_start, sequence_length in layout.segments:
        for chunk in range(_ceil_div(sequence_length, CHUNK_SIZE)):
            token_start = sequence_start + chunk * CHUNK_SIZE
            chunk_length = min(CHUNK_SIZE, sequence_length - chunk * CHUNK_SIZE)
            state_chunk = global_chunk if layout.is_varlen else batch * layout.grid_chunks + chunk
            yield batch, token_start, chunk_length, state_chunk
            global_chunk += 1


def _npu_metadata(layout: SequenceLayout) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    cu_seqlens = None if layout.cu_seqlens is None else layout.cu_seqlens.to(DEVICE)
    chunk_indices = None if layout.chunk_indices is None else layout.chunk_indices.to(DEVICE)
    return cu_seqlens, chunk_indices


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.bfloat16:
        return 4e-2, 4e-2
    if dtype == torch.float16:
        return 1.5e-2, 1.5e-2
    return 5e-4, 5e-4


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    dtype: torch.dtype,
    *,
    rtol: float | None = None,
    atol: float | None = None,
) -> None:
    default_rtol, default_atol = _tolerances(dtype)
    torch.testing.assert_close(
        actual.detach().cpu().float(),
        expected.detach().cpu().float(),
        rtol=default_rtol if rtol is None else rtol,
        atol=default_atol if atol is None else atol,
    )


def _layer_norm_gated_reference(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
    activation: str,
    is_rms_norm: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    residual_out = x.float()
    if residual is not None:
        residual_out = residual_out + residual.float()

    mean = None if is_rms_norm else residual_out.mean(dim=-1)
    centered = residual_out if mean is None else residual_out - mean[:, None]
    variance = centered.square().mean(dim=-1)
    rstd = torch.rsqrt(variance + EPS)
    normalized = residual_out * rstd[:, None] if is_rms_norm else centered * rstd[:, None]
    if weight is not None:
        normalized = normalized * weight.float()[None, :]
    if bias is not None:
        normalized = normalized + bias.float()[None, :]

    gate_fp32 = gate.float()
    if activation in ("silu", "swish"):
        normalized = normalized * F.silu(gate_fp32)
    elif activation == "sigmoid":
        normalized = normalized * torch.sigmoid(gate_fp32)
    return normalized.to(x.dtype), mean, rstd, residual_out.to(x.dtype)


def _scaled_dot_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    layout: SequenceLayout,
    scale: float,
    *,
    inter_sub_chunk: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, heads, _ = q.shape
    a_ref = torch.zeros(
        layout.batch_size,
        layout.sequence_width,
        heads,
        CHUNK_SIZE,
        dtype=torch.float32,
    )
    aqk_ref = torch.zeros_like(a_ref)

    for batch, token_start, chunk_length, _ in _iter_chunks(layout):
        for head in range(heads):
            q_chunk = q[batch, token_start : token_start + chunk_length, head].float()
            k_chunk = k[batch, token_start : token_start + chunk_length, head].float()
            gate_chunk = gate[batch, token_start : token_start + chunk_length, head].float()
            gated_k = k_chunk * torch.exp2(gate_chunk)
            inverse_gated_k = k_chunk * torch.exp2(-gate_chunk)
            kkt = gated_k @ inverse_gated_k.transpose(0, 1)
            qkt = (q_chunk * torch.exp2(gate_chunk)) @ inverse_gated_k.transpose(0, 1)

            for row in range(chunk_length):
                sub_chunk = row // SUB_CHUNK_SIZE
                if inter_sub_chunk:
                    column_start = 0
                    a_column_end = sub_chunk * SUB_CHUNK_SIZE
                    aqk_column_end = a_column_end
                else:
                    column_start = sub_chunk * SUB_CHUNK_SIZE
                    a_column_end = row
                    aqk_column_end = row + 1

                output_row = token_start + row
                if a_column_end > column_start:
                    a_ref[batch, output_row, head, column_start:a_column_end] = (
                        kkt[row, column_start:a_column_end] * beta[batch, output_row, head].float()
                    )
                if aqk_column_end > column_start:
                    aqk_ref[batch, output_row, head, column_start:aqk_column_end] = (
                        qkt[row, column_start:aqk_column_end] * scale
                    )
    return a_ref, aqk_ref


def _recompute_reference(
    q: torch.Tensor | None,
    k: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    coefficients: torch.Tensor,
    gate: torch.Tensor,
    layout: SequenceLayout,
    *,
    store_qg: bool,
    store_kg: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    dtype = k.dtype
    _, _, heads, _ = k.shape
    w_ref = torch.zeros_like(k)
    u_ref = torch.zeros_like(value)
    qg_ref = torch.zeros_like(q) if store_qg and q is not None else None
    kg_ref = torch.zeros_like(k) if store_kg else None

    for batch, token_start, chunk_length, _ in _iter_chunks(layout):
        for head in range(heads):
            token_slice = slice(token_start, token_start + chunk_length)
            a_chunk = coefficients[batch, token_slice, head, :chunk_length].float()
            beta_chunk = beta[batch, token_slice, head].float()
            gate_chunk = gate[batch, token_slice, head].float()

            weighted_value = (value[batch, token_slice, head].float() * beta_chunk[:, None]).to(dtype).float()
            weighted_key = (k[batch, token_slice, head].float() * beta_chunk[:, None]).to(dtype).float()
            weighted_key = (weighted_key * torch.exp2(gate_chunk)).to(dtype).float()
            u_ref[batch, token_slice, head] = (a_chunk @ weighted_value).to(dtype)
            w_ref[batch, token_slice, head] = (a_chunk @ weighted_key).to(dtype)

            if qg_ref is not None and q is not None:
                qg_ref[batch, token_slice, head] = (q[batch, token_slice, head].float() * torch.exp2(gate_chunk)).to(
                    dtype
                )
            if kg_ref is not None:
                last_gate = gate_chunk[-1]
                kg_ref[batch, token_slice, head] = (
                    k[batch, token_slice, head].float() * torch.exp2(last_gate[None, :] - gate_chunk)
                ).to(dtype)
    return w_ref, u_ref, qg_ref, kg_ref


def _chunk_gla_reference(
    q: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    state: torch.Tensor,
    coefficients: torch.Tensor,
    layout: SequenceLayout,
    scale: float,
) -> torch.Tensor:
    dtype = q.dtype
    _, _, heads, _ = q.shape
    out = torch.zeros_like(value)
    for batch, token_start, chunk_length, state_chunk in _iter_chunks(layout):
        for head in range(heads):
            token_slice = slice(token_start, token_start + chunk_length)
            scaled_q = (q[batch, token_slice, head].float() * scale).to(dtype).float()
            gated_q = (scaled_q * torch.exp2(gate[batch, token_slice, head].float())).to(dtype).float()
            state_output = gated_q @ state[state_chunk, head].float().transpose(0, 1)

            a_chunk = coefficients[batch, token_slice, head, :chunk_length].float()
            a_chunk = torch.tril(a_chunk).to(dtype).float()
            value_output = a_chunk @ value[batch, token_slice, head].float()
            out[batch, token_slice, head] = (state_output + value_output).to(dtype)
    return out


@pytest.mark.parametrize(
    ("dtype", "tokens", "hidden", "activation", "is_rms_norm", "with_residual", "with_affine"),
    [
        pytest.param(
            torch.float16,
            37,
            60,
            "silu",
            False,
            True,
            True,
            id="fp16-layernorm-residual-affine-silu-tail",
        ),
        pytest.param(
            torch.bfloat16,
            33,
            96,
            "sigmoid",
            True,
            False,
            False,
            id="bf16-rms-no-residual-no-affine-sigmoid-tail",
        ),
    ],
)
@torch.inference_mode()
def test_layer_norm_gated_fwd_kernel(
    dtype: torch.dtype,
    tokens: int,
    hidden: int,
    activation: str,
    is_rms_norm: bool,
    with_residual: bool,
    with_affine: bool,
) -> None:
    torch.manual_seed(11)
    x_cpu = _randn((tokens, hidden), dtype, scale=0.4)
    gate_cpu = _randn((tokens, hidden), dtype, scale=0.7)
    residual_cpu = _randn((tokens, hidden), dtype, scale=0.2) if with_residual else None
    weight_cpu = (1 + _randn((hidden,), dtype, scale=0.1)) if with_affine else None
    bias_cpu = _randn((hidden,), dtype, scale=0.1) if with_affine else None
    y_ref, mean_ref, rstd_ref, residual_ref = _layer_norm_gated_reference(
        x_cpu,
        gate_cpu,
        weight_cpu,
        bias_cpu,
        residual_cpu,
        activation,
        is_rms_norm,
    )

    x = x_cpu.to(DEVICE)
    gate = gate_cpu.to(DEVICE)
    residual = None if residual_cpu is None else residual_cpu.to(DEVICE)
    weight = None if weight_cpu is None else weight_cpu.to(DEVICE)
    bias = None if bias_cpu is None else bias_cpu.to(DEVICE)
    y = torch.empty_like(x)
    mean = None if is_rms_norm else torch.empty(tokens, dtype=torch.float32, device=DEVICE)
    rstd = torch.empty(tokens, dtype=torch.float32, device=DEVICE)
    residual_out = torch.empty_like(x) if with_residual else None
    block_tokens = 32

    layer_norm_gated_fwd_kernel[(_ceil_div(tokens, block_tokens),)](
        x=x,
        g=gate,
        y=y,
        w=weight,
        b=bias,
        residual=residual,
        residual_out=residual_out,
        mean=mean,
        rstd=rstd,
        eps=EPS,
        T=tokens,
        D=hidden,
        BT=block_tokens,
        BD=_next_power_of_2(hidden),
        ACTIVATION=activation,
        IS_RMS_NORM=is_rms_norm,
        num_warps=4,
    )

    _assert_close(y, y_ref, dtype)
    _assert_close(rstd, rstd_ref, torch.float32, rtol=8e-4, atol=8e-4)
    if mean is not None and mean_ref is not None:
        _assert_close(mean, mean_ref, torch.float32, rtol=8e-4, atol=8e-4)
    if residual_out is not None:
        _assert_close(residual_out, residual_ref, dtype)


@pytest.mark.parametrize(
    ("dtype", "tokens", "hidden", "activation", "is_rms_norm", "with_residual", "with_affine"),
    [
        pytest.param(
            torch.float16,
            5,
            768,
            "sigmoid",
            True,
            True,
            True,
            id="fp16-rms-residual-affine-sigmoid-non-power-of-two",
        ),
        pytest.param(
            torch.bfloat16,
            3,
            1000,
            "silu",
            False,
            False,
            False,
            id="bf16-layernorm-no-residual-no-affine-silu-tail",
        ),
    ],
)
@torch.inference_mode()
def test_layer_norm_gated_fwd_kernel1(
    dtype: torch.dtype,
    tokens: int,
    hidden: int,
    activation: str,
    is_rms_norm: bool,
    with_residual: bool,
    with_affine: bool,
) -> None:
    torch.manual_seed(12)
    x_cpu = _randn((tokens, hidden), dtype, scale=0.4)
    gate_cpu = _randn((tokens, hidden), dtype, scale=0.7)
    residual_cpu = _randn((tokens, hidden), dtype, scale=0.2) if with_residual else None
    weight_cpu = (1 + _randn((hidden,), dtype, scale=0.1)) if with_affine else None
    bias_cpu = _randn((hidden,), dtype, scale=0.1) if with_affine else None
    y_ref, mean_ref, rstd_ref, residual_ref = _layer_norm_gated_reference(
        x_cpu,
        gate_cpu,
        weight_cpu,
        bias_cpu,
        residual_cpu,
        activation,
        is_rms_norm,
    )

    x = x_cpu.to(DEVICE)
    gate = gate_cpu.to(DEVICE)
    residual = None if residual_cpu is None else residual_cpu.to(DEVICE)
    weight = None if weight_cpu is None else weight_cpu.to(DEVICE)
    bias = None if bias_cpu is None else bias_cpu.to(DEVICE)
    y = torch.empty_like(x)
    mean = None if is_rms_norm else torch.empty(tokens, dtype=torch.float32, device=DEVICE)
    rstd = torch.empty(tokens, dtype=torch.float32, device=DEVICE)
    residual_out = torch.empty_like(x) if with_residual else None

    layer_norm_gated_fwd_kernel1[(tokens,)](
        x=x,
        g=gate,
        y=y,
        w=weight,
        b=bias,
        residual=residual,
        residual_out=residual_out,
        mean=mean,
        rstd=rstd,
        eps=EPS,
        D=hidden,
        BD=_next_power_of_2(hidden),
        ACTIVATION=activation,
        IS_RMS_NORM=is_rms_norm,
        num_warps=4,
    )

    _assert_close(y, y_ref, dtype)
    _assert_close(rstd, rstd_ref, torch.float32, rtol=8e-4, atol=8e-4)
    if mean is not None and mean_ref is not None:
        _assert_close(mean, mean_ref, torch.float32, rtol=8e-4, atol=8e-4)
    if residual_out is not None:
        _assert_close(residual_out, residual_ref, dtype)


@pytest.mark.parametrize(
    ("is_varlen", "dtype", "key_dim"),
    [
        pytest.param(False, torch.float16, 128, id="fixed-fp16-k128-two-chunks"),
        pytest.param(True, torch.bfloat16, 48, id="varlen-bf16-k48-tail-chunks"),
    ],
)
@torch.inference_mode()
def test_chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter(
    is_varlen: bool,
    dtype: torch.dtype,
    key_dim: int,
) -> None:
    torch.manual_seed(21)
    layout = _make_layout(is_varlen)
    heads = 2
    shape = (layout.batch_size, layout.sequence_width, heads, key_dim)
    q_cpu = _randn(shape, dtype, scale=0.25)
    k_cpu = _randn(shape, dtype, scale=0.25)
    # The production KDA path feeds this kernel the FP32 cumulative gate
    # emitted by chunk_local_cumsum.
    gate_cpu = _randn(shape, torch.float32, scale=0.12)
    beta_cpu = torch.rand(shape[:-1], dtype=torch.float32).mul(0.5).add(0.25).to(dtype)
    scale = key_dim**-0.5
    a_ref, aqk_ref = _scaled_dot_reference(
        q_cpu,
        k_cpu,
        gate_cpu,
        beta_cpu,
        layout,
        scale,
        inter_sub_chunk=True,
    )

    q, k, gate, beta = (tensor.to(DEVICE) for tensor in (q_cpu, k_cpu, gate_cpu, beta_cpu))
    a = torch.zeros(
        layout.batch_size,
        layout.sequence_width,
        heads,
        CHUNK_SIZE,
        dtype=torch.float32,
        device=DEVICE,
    )
    aqk = torch.zeros_like(a)
    cu_seqlens, chunk_indices = _npu_metadata(layout)

    chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter[(layout.grid_chunks, layout.batch_size * heads)](
        q=q,
        k=k,
        g=gate,
        beta=beta,
        A=a,
        Aqk=aqk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=layout.sequence_width,
        H=heads,
        K=key_dim,
        BT=CHUNK_SIZE,
        BC=SUB_CHUNK_SIZE,
        NC=CHUNK_SIZE // SUB_CHUNK_SIZE,
    )

    _assert_close(a, a_ref, dtype, rtol=5e-2, atol=2e-2)
    _assert_close(aqk, aqk_ref, dtype, rtol=5e-2, atol=2e-2)


@pytest.mark.parametrize(
    ("is_varlen", "dtype", "key_dim"),
    [
        pytest.param(False, torch.float16, 128, id="fixed-fp16-k128-two-chunks"),
        pytest.param(True, torch.bfloat16, 24, id="varlen-bf16-k24-tail-chunks"),
    ],
)
@torch.inference_mode()
def test_chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra(
    is_varlen: bool,
    dtype: torch.dtype,
    key_dim: int,
) -> None:
    torch.manual_seed(22)
    layout = _make_layout(is_varlen)
    heads = 2
    shape = (layout.batch_size, layout.sequence_width, heads, key_dim)
    q_cpu = _randn(shape, dtype, scale=0.25)
    k_cpu = _randn(shape, dtype, scale=0.25)
    gate_cpu = _randn(shape, dtype, scale=0.12)
    beta_cpu = torch.rand(shape[:-1], dtype=torch.float32).mul(0.5).add(0.25).to(dtype)
    scale = key_dim**-0.5
    a_ref, aqk_ref = _scaled_dot_reference(
        q_cpu,
        k_cpu,
        gate_cpu,
        beta_cpu,
        layout,
        scale,
        inter_sub_chunk=False,
    )

    q, k, gate, beta = (tensor.to(DEVICE) for tensor in (q_cpu, k_cpu, gate_cpu, beta_cpu))
    a = torch.zeros(
        layout.batch_size,
        layout.sequence_width,
        heads,
        CHUNK_SIZE,
        dtype=torch.float32,
        device=DEVICE,
    )
    aqk = torch.zeros_like(a)
    cu_seqlens, chunk_indices = _npu_metadata(layout)

    chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra[
        (
            layout.grid_chunks,
            CHUNK_SIZE // SUB_CHUNK_SIZE,
            layout.batch_size * heads,
        )
    ](
        q=q,
        k=k,
        g=gate,
        beta=beta,
        A=a,
        Aqk=aqk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=layout.sequence_width,
        H=heads,
        K=key_dim,
        BT=CHUNK_SIZE,
        BC=SUB_CHUNK_SIZE,
        BK=max(_next_power_of_2(key_dim), SUB_CHUNK_SIZE),
    )

    _assert_close(a, a_ref, dtype, rtol=5e-2, atol=2e-2)
    _assert_close(aqk, aqk_ref, dtype, rtol=5e-2, atol=2e-2)


@pytest.mark.parametrize(
    ("is_varlen", "dtype", "key_dim", "value_dim", "store_qg", "store_kg"),
    [
        pytest.param(False, torch.float16, 128, 128, True, False, id="fixed-fp16-k128-v128-store-qg"),
        pytest.param(True, torch.bfloat16, 48, 80, False, True, id="varlen-bf16-k48-v80-store-kg"),
    ],
)
@torch.inference_mode()
def test_recompute_w_u_fwd_kernel(
    is_varlen: bool,
    dtype: torch.dtype,
    key_dim: int,
    value_dim: int,
    store_qg: bool,
    store_kg: bool,
) -> None:
    torch.manual_seed(31)
    layout = _make_layout(is_varlen)
    heads = 2
    q_cpu = _randn(
        (layout.batch_size, layout.sequence_width, heads, key_dim),
        dtype,
        scale=0.25,
    )
    k_cpu = _randn(q_cpu.shape, dtype, scale=0.25)
    value_cpu = _randn(
        (layout.batch_size, layout.sequence_width, heads, value_dim),
        dtype,
        scale=0.25,
    )
    beta_cpu = (
        torch.rand(layout.batch_size, layout.sequence_width, heads, dtype=torch.float32).mul(0.5).add(0.25).to(dtype)
    )
    coefficients_cpu = _randn(
        (layout.batch_size, layout.sequence_width, heads, CHUNK_SIZE),
        dtype,
        scale=0.03,
    )
    gate_cpu = _randn(q_cpu.shape, dtype, scale=0.12)
    w_ref, u_ref, qg_ref, kg_ref = _recompute_reference(
        q_cpu if store_qg else None,
        k_cpu,
        value_cpu,
        beta_cpu,
        coefficients_cpu,
        gate_cpu,
        layout,
        store_qg=store_qg,
        store_kg=store_kg,
    )

    q = q_cpu.to(DEVICE) if store_qg else None
    k = k_cpu.to(DEVICE)
    value = value_cpu.to(DEVICE)
    beta = beta_cpu.to(DEVICE)
    coefficients = coefficients_cpu.to(DEVICE)
    gate = gate_cpu.to(DEVICE)
    w = torch.zeros_like(k)
    u = torch.zeros_like(value)
    qg = torch.zeros_like(q) if q is not None else None
    kg = torch.zeros_like(k) if store_kg else None
    cu_seqlens, chunk_indices = _npu_metadata(layout)

    recompute_w_u_fwd_kernel[(layout.grid_chunks, layout.batch_size * heads)](
        q=q,
        k=k,
        qg=qg,
        kg=kg,
        v=value,
        beta=beta,
        w=w,
        u=u,
        A=coefficients,
        gk=gate,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=layout.sequence_width,
        H=heads,
        K=key_dim,
        V=value_dim,
        BT=CHUNK_SIZE,
        BK=64,
        BV=64,
        DOT_PRECISION="ieee",
    )

    _assert_close(w, w_ref, dtype, rtol=6e-2, atol=2e-2)
    _assert_close(u, u_ref, dtype, rtol=6e-2, atol=2e-2)
    if qg is not None and qg_ref is not None:
        _assert_close(qg, qg_ref, dtype)
    if kg is not None and kg_ref is not None:
        _assert_close(kg, kg_ref, dtype)


@pytest.mark.parametrize(
    ("is_varlen", "dtype", "key_dim", "value_dim"),
    [
        pytest.param(False, torch.float16, 128, 144, id="fixed-fp16-k128-v144-two-chunks"),
        pytest.param(True, torch.bfloat16, 48, 80, id="varlen-bf16-k48-v80-tail-chunks"),
    ],
)
@torch.inference_mode()
def test_chunk_gla_fwd_kernel_o(
    is_varlen: bool,
    dtype: torch.dtype,
    key_dim: int,
    value_dim: int,
) -> None:
    torch.manual_seed(41)
    layout = _make_layout(is_varlen)
    heads = 2
    q_cpu = _randn(
        (layout.batch_size, layout.sequence_width, heads, key_dim),
        dtype,
        scale=0.2,
    )
    value_cpu = _randn(
        (layout.batch_size, layout.sequence_width, heads, value_dim),
        dtype,
        scale=0.2,
    )
    gate_cpu = _randn(q_cpu.shape, dtype, scale=0.12)
    state_cpu = _randn(
        (layout.total_chunks, heads, value_dim, key_dim),
        dtype,
        scale=0.05,
    )
    coefficients_cpu = _randn(
        (layout.batch_size, layout.sequence_width, heads, CHUNK_SIZE),
        torch.float32,
        scale=0.02,
    )
    scale = key_dim**-0.5
    out_ref = _chunk_gla_reference(
        q_cpu,
        value_cpu,
        gate_cpu,
        state_cpu,
        coefficients_cpu,
        layout,
        scale,
    )

    q, value, gate, state, coefficients = (
        tensor.to(DEVICE) for tensor in (q_cpu, value_cpu, gate_cpu, state_cpu, coefficients_cpu)
    )
    out = torch.zeros_like(value)
    cu_seqlens, chunk_indices = _npu_metadata(layout)

    chunk_gla_fwd_kernel_o[
        (
            _ceil_div(value_dim, 128),
            layout.grid_chunks,
            layout.batch_size * heads,
        )
    ](
        q=q,
        v=value,
        g=gate,
        h=state,
        o=out,
        A=coefficients,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=layout.sequence_width,
        H=heads,
        K=key_dim,
        V=value_dim,
        BT=CHUNK_SIZE,
    )

    _assert_close(out, out_ref, dtype, rtol=6e-2, atol=3e-2)


@pytest.mark.parametrize(
    ("dtype", "tokens", "with_bias", "beta", "threshold"),
    [
        pytest.param(torch.float16, 65, True, 1.5, 5.0, id="fp16-bias-linear-threshold-tail"),
        pytest.param(torch.bfloat16, 17, False, 1.5, 5.0, id="bf16-no-bias-tail"),
    ],
)
@torch.inference_mode()
def test_kda_gate_fwd_kernel(
    dtype: torch.dtype,
    tokens: int,
    with_bias: bool,
    beta: float,
    threshold: float,
) -> None:
    torch.manual_seed(51)
    heads, head_dim = 3, 40
    gate_cpu = _randn((tokens, heads, head_dim), dtype, scale=0.8)
    gate_cpu[0, :, 0] = torch.tensor(threshold / beta + 2, dtype=dtype)
    a_cpu = _randn((heads,), torch.float32, scale=0.2) - 0.7
    bias_cpu = _randn((heads, head_dim), dtype, scale=0.2) if with_bias else None

    biased_gate = gate_cpu.float()
    if bias_cpu is not None:
        biased_gate = biased_gate + bias_cpu.float()[None, :, :]
    out_ref = -torch.exp(a_cpu.float())[None, :, None] * F.softplus(
        biased_gate,
        beta=beta,
        threshold=threshold,
    )

    gate = gate_cpu.to(DEVICE)
    a = a_cpu.to(DEVICE)
    bias = None if bias_cpu is None else bias_cpu.to(DEVICE)
    out = torch.empty((tokens, heads, head_dim), dtype=torch.float32, device=DEVICE)

    kda_gate_fwd_kernel[lambda meta: (_ceil_div(tokens, meta["BT"]), heads)](
        g=gate,
        A=a,
        y=out,
        g_bias=bias,
        beta=beta,
        threshold=threshold,
        T=tokens,
        H=heads,
        D=head_dim,
        BD=_next_power_of_2(head_dim),
        HAS_BIAS=with_bias,
    )

    _assert_close(out, out_ref, dtype, rtol=2e-3, atol=2e-3)
