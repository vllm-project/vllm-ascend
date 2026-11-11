from dataclasses import dataclass
from typing import Any

import torch
from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    TritonWarmupTensor,
)
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.ascend_jit_warmup import AscendTritonJitKernel
from vllm_ascend.ops.triton.triton_utils import get_ub_size_bytes


@triton.jit(
    do_not_specialize=[
        "total_batch",
    ]
)
def triton_rms_kernel(
    hidden_state_ptr,
    hidden_state_stride_bs,
    norm_output_ptr,
    variance_epsilon,
    total_batch,
    DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    core_id = tl.program_id(0)
    core_num = tl.num_programs(0)
    batch_per_core = tl.cdiv(total_batch, core_num)
    start_batch = core_id * batch_per_core
    end_batch = tl.minimum(start_batch + batch_per_core, total_batch)
    offset_d = tl.arange(0, DIM)

    for row_start in tl.range(start_batch, end_batch, BLOCK_M):
        offset_row = row_start + tl.arange(0, BLOCK_M)
        mask_r = offset_row < total_batch
        mask_row = mask_r[:, None]
        offset_hidden = offset_row[:, None] * hidden_state_stride_bs + offset_d[None, :]

        x = tl.load(hidden_state_ptr + offset_hidden, mask=mask_row)

        variance = tl.sum(x * x, axis=-1) / DIM
        output = x * tl.rsqrt(variance[:, None] + variance_epsilon)

        tl.store(norm_output_ptr + offset_hidden, output, mask=mask_row)


def _rms_row_block_size(dim: int, dtype: torch.dtype) -> int:
    """UB-derived ceiling on ``BLOCK_M`` before the power-of-two floor."""
    resv_buffer = 6144  # reserve 6 KB to prevent UB overflow
    available_ub_size = get_ub_size_bytes() - resv_buffer
    element_size = torch.empty(1, dtype=dtype).element_size()
    if element_size == 4:
        data_multiplier = 5  # input + offsets + mid_out + output + others
    elif element_size == 2:
        data_multiplier = 7  # input + offsets + mid_out + output + others
    else:
        raise NotImplementedError(f"triton_q_rms: element size ({element_size} B) not supported")
    return int(available_ub_size / (dim * element_size * data_multiplier))


def _rms_block_m(total_batch: int, num_vectorcore: int, dim: int, dtype) -> int:
    """Tile size used by ``triton_q_rms``.

    Derived adaptively from the runtime Unified Buffer (UB) size so small
    hidden dims get larger tiles (fewer loop iterations) while large hidden
    dims shrink the tile to avoid UB overflow. ``BLOCK_M`` is floored to a
    power of two (leftover rows are already masked), which keeps the JIT
    constexpr set small.
    """
    row_block_size = _rms_row_block_size(dim, dtype)
    batch_per_core = triton.cdiv(total_batch, num_vectorcore)
    raw = min(row_block_size, int(batch_per_core))
    return 1 << (max(raw, 1).bit_length() - 1)


class TritonQRmsKernel(AscendTritonJitKernel["TritonQRmsKernel.CompileKey"]):
    kernel = triton_rms_kernel

    @dataclass(frozen=True)
    class CompileKey:
        dim: int
        block_m: int
        dtype: torch.dtype

    def dispatch(
        self,
        *,
        dim: int,
        block_m: int,
        dtype: torch.dtype,
    ) -> "TritonQRmsKernel.CompileKey":
        return self.CompileKey(dim=dim, block_m=block_m, dtype=dtype)

    def get_warmup_keys(self, vllm_config: Any) -> list["TritonQRmsKernel.CompileKey"]:
        model_config = vllm_config.model_config
        dim = model_config.get_head_size()
        if dim > 2048:
            return []
        # Every power of two up to the UB-derived ceiling is reachable:
        # BLOCK_M = pow2floor(min(ceiling, cdiv(total_batch, cores))) and
        # total_batch spans all runtime batch sizes.
        row_block_size = max(_rms_row_block_size(dim, model_config.dtype), 1)
        block_m_values = []
        block_m = 1
        while block_m <= row_block_size:
            block_m_values.append(block_m)
            block_m *= 2
        return self._trace_dispatch(self.dispatch)(
            dim=dim,
            dtype=model_config.dtype,
            block_m=tuple(block_m_values),
        )

    def warmup_inputs(self, compile_key: "TritonQRmsKernel.CompileKey") -> dict[str, Any]:
        return dict(
            hidden_state=TritonWarmupTensor(
                compile_key.dtype,
                shape=(1, compile_key.dim),
                strides=(compile_key.dim, 1),
            ),
            norm_output=TritonWarmupTensor(
                compile_key.dtype,
                shape=(1, compile_key.dim),
                strides=(compile_key.dim, 1),
            ),
            variance_epsilon=1e-6,
            total_batch=1,
            DIM=compile_key.dim,
            BLOCK_M=compile_key.block_m,
            num_vectorcore=1,
        )

    @kernel_launcher
    def __call__(
        self,
        hidden_state: torch.Tensor,
        norm_output: torch.Tensor,
        variance_epsilon: float,
        total_batch: int,
        DIM: int,
        BLOCK_M: int,
        num_vectorcore: int,
    ) -> LaunchSpec:
        return (num_vectorcore,), {
            "hidden_state_stride_bs": hidden_state.stride(0),
        }


_TRITON_Q_RMS_KERNEL = TritonQRmsKernel()


def triton_q_rms(
    q,  # bs, 64, 512
    variance_epsilon,
):
    bs, head_num, dim = q.shape
    total_batch = bs * head_num
    q = q.view(total_batch, dim)

    if dim > 2048:
        raise NotImplementedError(f"triton_q_rms: dim > 2048 not supported, got {dim}")

    device_properties = triton.runtime.driver.active.utils.get_device_properties(q.device)
    num_vectorcore = device_properties.get("num_vectorcore", -1)

    BLOCK_M = _rms_block_m(total_batch, num_vectorcore, dim, q.dtype)

    norm_output = torch.empty_like(q)

    _TRITON_Q_RMS_KERNEL(
        hidden_state=q,
        norm_output=norm_output,
        variance_epsilon=variance_epsilon,
        total_batch=total_batch,
        DIM=dim,
        BLOCK_M=BLOCK_M,
        num_vectorcore=num_vectorcore,
    )
    return norm_output.view(bs, head_num, dim)
