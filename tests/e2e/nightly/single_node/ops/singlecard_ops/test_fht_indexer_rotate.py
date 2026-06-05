#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""NPU parity and timing for the DeepSeek-V4 DSA indexer Hadamard rotation.

Compares, on V4-Flash lightning-indexer shapes (index_head_dim=128):
  existing: DSA rotate_activation + DeviceOperator.indexer_quantize_query
  fused: PTO fused FHT + int8 dynamic quant kernel

The FHT equals the dense rotation up to a fixed output permutation that
cancels in indexer scores q . k^T; parity is asserted on scores. Timings
are printed, following the singlecard_ops convention.
"""

import numpy as np
import pytest
import torch
from scipy.linalg import hadamard as scipy_hadamard

from vllm_ascend.attention.dsa_v1 import rotate_activation
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.fast_hadamard import fast_hadamard_dynamic_quant_last_dim

torch.manual_seed(45)

N = 128  # index_head_dim
N_HEADS = 64  # index_n_heads
T_Q_PARITY = 1024  # keeps score-parity matmul manageable
T_Q_BENCH = 4096  # realistic DeepSeek-V4-Pro max-num-batched-tokens chunk
COMPRESS_RATIO = 4
T_KV_UPDATE_BENCH = T_Q_BENCH // COMPRESS_RATIO  # compressed kv entries produced from this token batch
T_K_CACHE = 4096  # existing compressed indexer kv cache entries used for score parity
TOPK = 512  # index_topk
WARMUP, ITERS = 50, 100


def benchmark_npu(fn, num_iterations=ITERS, num_warmup_iterations=WARMUP):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    times = np.zeros(num_iterations + num_warmup_iterations)
    for i in range(num_iterations + num_warmup_iterations):
        with torch.no_grad():
            start.record()
            fn()
            end.record()
        torch.npu.synchronize()
        times[i] = start.elapsed_time(end)
    return np.amin(times[num_warmup_iterations:]) * 1000  # us


def existing_indexer_quantize(x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return DeviceOperator.indexer_quantize_query(rotate_activation(x, h))


def fused_indexer_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q, scale = fast_hadamard_dynamic_quant_last_dim(x)
    return q, scale.to(torch.float16)


def dequant(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale.reshape(-1, 1).float()


def topk_overlap(s: torch.Tensor, s_ref: torch.Tensor) -> float:
    top = s.topk(TOPK, dim=-1).indices
    ref = s_ref.topk(TOPK, dim=-1).indices
    return (top.unsqueeze(-1) == ref.unsqueeze(-2)).any(-1).float().mean().item()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_fht_indexer_rotate_parity_and_perf(dtype: torch.dtype):
    dtype_name = str(dtype).split(".")[-1]
    torch.manual_seed(45)
    h = torch.tensor(scipy_hadamard(N, dtype=float), dtype=torch.float32, device="npu").to(dtype)
    q = torch.randn(T_Q_PARITY * N_HEADS, N, dtype=dtype, device="npu")
    k_cache = torch.randn(T_K_CACHE, N, dtype=dtype, device="npu")
    q_bench = torch.randn(T_Q_BENCH * N_HEADS, N, dtype=dtype, device="npu")
    kv_update = torch.randn(T_KV_UPDATE_BENCH, N, dtype=dtype, device="npu")

    # fp32 dense reference scores
    s_ref = rotate_activation(q, h).float() @ rotate_activation(k_cache, h).float().T

    # Existing DSA/indexer component path.
    qq_existing, qs_existing = existing_indexer_quantize(q, h)
    kq_existing, ks_existing = existing_indexer_quantize(k_cache, h)
    s_existing = dequant(qq_existing, qs_existing) @ dequant(kq_existing, ks_existing).T

    # Candidate replacement for rotate_activation + indexer_quantize_query.
    qq, qs = fused_indexer_quantize(q)
    kq, ks = fused_indexer_quantize(k_cache)
    s_fused = dequant(qq, qs) @ dequant(kq, ks).T

    assert ((s_fused - s_existing).norm() / s_existing.norm()).item() < 0.05
    for name, s, min_ov in (
        ("existing", s_existing, 0.98),
        ("fused", s_fused, 0.98),
    ):
        ov = topk_overlap(s, s_ref)
        rel = ((s - s_ref).norm() / s_ref.norm()).item()
        print(f"{dtype_name} {name}: top{TOPK} overlap {ov * 100:.2f}% rel_err {rel:.4f}")
        assert ov >= min_ov

    t_current_q = benchmark_npu(lambda: existing_indexer_quantize(q_bench, h))
    t_fused_q = benchmark_npu(lambda: fused_indexer_quantize(q_bench))
    t_current_kv_update = benchmark_npu(lambda: existing_indexer_quantize(kv_update, h))
    t_fused_kv_update = benchmark_npu(lambda: fused_indexer_quantize(kv_update))
    t_current_q_kv_update = benchmark_npu(
        lambda: (
            existing_indexer_quantize(q_bench, h),
            existing_indexer_quantize(kv_update, h),
        )
    )
    t_fused_q_kv_update = benchmark_npu(
        lambda: (
            fused_indexer_quantize(q_bench),
            fused_indexer_quantize(kv_update),
        )
    )
    t_rotate = benchmark_npu(lambda: rotate_activation(q_bench, h))
    print(f"[{dtype_name} q {T_Q_BENCH * N_HEADS}x{N}] current {t_current_q:.1f} us | fused int8 {t_fused_q:.1f} us")
    print(
        f"[{dtype_name} kv_update {T_KV_UPDATE_BENCH}x{N}] current {t_current_kv_update:.1f} us | "
        f"fused int8 {t_fused_kv_update:.1f} us"
    )
    print(
        f"[{dtype_name} q+kv_update {T_Q_BENCH * N_HEADS}x{N}+{T_KV_UPDATE_BENCH}x{N}] "
        f"current {t_current_q_kv_update:.1f} us | "
        f"fused int8 {t_fused_q_kv_update:.1f} us"
    )
    print(f"[{dtype_name} {T_Q_BENCH * N_HEADS}x{N}] rotate_only {t_rotate:.1f} us")
