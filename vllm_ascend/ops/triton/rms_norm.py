from dataclasses import dataclass
from typing import Any

import torch
from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    TritonWarmupTensor,
    VllmTritonJitKernel,
)
from vllm.triton_utils import tl, triton


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


class TritonQRmsKernel(VllmTritonJitKernel["TritonQRmsKernel.CompileKey"]):
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
    ) -> CompileKey:
        return self.CompileKey(dim=dim, block_m=block_m, dtype=dtype)

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        model_config = vllm_config.model_config
        return self._trace_dispatch(self.dispatch)(
            dim=model_config.get_head_size(),
            dtype=model_config.dtype,
            block_m=(1, 2, 4, 8, 16),
        )

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
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


def _rms_block_m(total_batch: int, num_vectorcore: int) -> int:
    """Tile size used by ``triton_q_rms``.

    Flooring ``BLOCK_M`` to a power of two is math-equivalent (leftover rows
    are already masked) and cuts the JIT constexpr set from 16 values to
    ``{1, 2, 4, 8, 16}``.
    """
    row_block_size = 16
    batch_per_core = triton.cdiv(total_batch, num_vectorcore)
    raw = min(row_block_size, int(batch_per_core))
    return 1 << (max(raw, 1).bit_length() - 1)


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

    BLOCK_M = _rms_block_m(total_batch, num_vectorcore)

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
