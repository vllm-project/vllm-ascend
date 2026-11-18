# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

import torch
import torch_npu
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton


@triton.jit(do_not_specialize=["M"])
def fast_add_rmsnorm_kernel(
    x_ptr,
    res_ptr,
    w_ptr,
    out_ptr,
    out_res_ptr,
    M,
    H: tl.constexpr,
    eps: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    pid = tl.program_id(0)
    per = tl.cdiv(M, NUM_CORES)
    s = pid * per
    e = tl.minimum(s + per, M)
    cols = tl.arange(0, H)
    w = tl.load(w_ptr + cols).to(tl.float32)
    for r in tl.range(s, e):
        off = r * H + cols
        xv = tl.load(x_ptr + off).to(tl.float32)
        rv = tl.load(res_ptr + off).to(tl.float32)
        nr = xv + rv
        tl.store(out_res_ptr + off, nr.to(tl.bfloat16))
        var = tl.sum(nr * nr, axis=0) / H
        yn = nr * tl.math.rsqrt(var + eps) * w
        tl.store(out_ptr + off, yn.to(tl.bfloat16))


@triton.jit(do_not_specialize=["M"])
def fast_swiglu_kernel(
    x_ptr,
    o_ptr,
    M,
    H: tl.constexpr,
    NUM_CORES: tl.constexpr,
    BH: tl.constexpr,
):
    pid = tl.program_id(0)
    n_h = tl.cdiv(H, BH)
    total = M * n_h
    per = tl.cdiv(total, NUM_CORES)
    s = pid * per
    e = tl.minimum(s + per, total)
    offs = tl.arange(0, BH)
    for t in tl.range(s, e):
        row = t // n_h
        hb = t % n_h
        col = hb * BH + offs
        mask = col < H
        a = tl.load(x_ptr + row * (2 * H) + col, mask=mask).to(tl.float32)
        b = tl.load(x_ptr + row * (2 * H) + H + col, mask=mask).to(tl.float32)
        y = (a * tl.sigmoid(a) * b).to(tl.bfloat16)
        tl.store(o_ptr + row * H + col, y, mask=mask)


_VEC_CORES: int | None = None
_get_raw_stream = getattr(getattr(torch_npu, "_C", None), "_npu_getCurrentRawStream", None)
_ADD_RMSNORM_KERNELS: dict[tuple[int, int, float, int], object] = {}
_SWIGLU_KERNELS: dict[tuple[int, int, int, int], object] = {}


def _get_vec_cores() -> int:
    global _VEC_CORES
    if _VEC_CORES is None:
        init_device_properties_triton()
        _VEC_CORES = int(get_vectorcore_num())
    return _VEC_CORES


def npu_fast_add_rms_norm_impl(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    m = x.shape[0]
    if m == 0:
        return torch.empty_like(x), torch.empty_like(residual)
    h = int(x.shape[-1])
    ncores = _get_vec_cores()
    out = torch.empty_like(x)
    out_res = torch.empty_like(residual)
    feps = float(eps)
    if _get_raw_stream is None or torch.npu.is_current_stream_capturing():
        fast_add_rmsnorm_kernel[(ncores,)](
            x,
            residual,
            weight,
            out,
            out_res,
            m,
            h,
            feps,
            ncores,
        )
    else:
        dev_idx = x.device.index if x.device.index is not None else 0
        key = (dev_idx, h, feps, ncores)
        k = _ADD_RMSNORM_KERNELS.get(key)
        if k is None:
            k = fast_add_rmsnorm_kernel[(ncores,)](
                x,
                residual,
                weight,
                out,
                out_res,
                m,
                h,
                feps,
                ncores,
            )
            _ADD_RMSNORM_KERNELS[key] = k
        else:
            st = _get_raw_stream(dev_idx)
            k.run(  # type: ignore[attr-defined]
                ncores,
                1,
                1,
                st,
                k.function,  # type: ignore[attr-defined]
                k.packed_metadata,  # type: ignore[attr-defined]
                None,
                None,
                None,
                x,
                residual,
                weight,
                out,
                out_res,
                m,
                h,
                feps,
                ncores,
            )
    return out, out_res


def npu_fast_add_rms_norm_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x), torch.empty_like(residual)


direct_register_custom_op(
    op_name="npu_fast_add_rms_norm",
    op_func=npu_fast_add_rms_norm_impl,
    fake_impl=npu_fast_add_rms_norm_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)


def npu_fast_swiglu_impl(x: torch.Tensor) -> torch.Tensor:
    m = x.shape[0]
    h = x.shape[-1] // 2
    if m > 0 and x.is_contiguous() and x.dtype == torch.bfloat16 and x.shape[-1] % 2 == 0 and 2048 <= h <= 32768:
        ncores = _get_vec_cores()
        bh = 6144 if h == 12288 else min(4096, triton.next_power_of_2(h))
        out = torch.empty((m, h), device=x.device, dtype=x.dtype)
        if _get_raw_stream is None or torch.npu.is_current_stream_capturing():
            fast_swiglu_kernel[(ncores,)](x, out, m, h, ncores, bh)
        else:
            dev_idx = x.device.index if x.device.index is not None else 0
            key = (dev_idx, h, bh, ncores)
            k = _SWIGLU_KERNELS.get(key)
            if k is None:
                k = fast_swiglu_kernel[(ncores,)](x, out, m, h, ncores, bh)
                _SWIGLU_KERNELS[key] = k
            else:
                st = _get_raw_stream(dev_idx)
                k.run(  # type: ignore[attr-defined]
                    ncores,
                    1,
                    1,
                    st,
                    k.function,  # type: ignore[attr-defined]
                    k.packed_metadata,  # type: ignore[attr-defined]
                    None,
                    None,
                    None,
                    x,
                    out,
                    m,
                    h,
                    ncores,
                    bh,
                )
        return out
    return torch_npu.npu_swiglu(x)


def npu_fast_swiglu_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty((*x.shape[:-1], x.shape[-1] // 2), device=x.device, dtype=x.dtype)


direct_register_custom_op(
    op_name="npu_fast_swiglu",
    op_func=npu_fast_swiglu_impl,
    fake_impl=npu_fast_swiglu_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
