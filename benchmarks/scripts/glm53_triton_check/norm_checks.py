# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def run(baseline, candidate):
    torch.manual_seed(941)
    tested = []
    for rows, dim, dtype, activation, residual, prenorm in [
        (3, 128, torch.bfloat16, "swish", True, True),
        (7, 256, torch.float32, "sigmoid", False, True),
        (4, 768, torch.bfloat16, "sigmoid", True, True),
        (33, 128, torch.float16, "silu", False, False),
    ]:
        x = torch.randn(rows, dim, device="npu", dtype=dtype)
        g = torch.randn_like(x)
        w, b = [torch.randn(dim, device="npu", dtype=dtype) for _ in range(2)]
        r = torch.randn_like(x, dtype=torch.float32) if residual else None
        kwargs = dict(activation=activation, residual=r, prenorm=prenorm, residual_in_fp32=True)
        a = baseline.rms_norm_gated(x, g, w, b, **kwargs)
        z = candidate.rms_norm_gated(x, g, w, b, **kwargs)
        torch.testing.assert_close(a, z, rtol=0, atol=0)
        tested.append(
            dict(rows=rows, dim=dim, dtype=str(dtype), activation=activation, residual=residual, prenorm=prenorm)
        )
    for dim in [128, 768]:
        x = torch.randn(5, dim, device="npu", dtype=torch.bfloat16)
        g = torch.randn_like(x)
        w = torch.randn(dim, device="npu", dtype=x.dtype)
        kwargs = dict(out_dtype=x.dtype)
        a = baseline.layer_norm_gated_fwd(x, g, w, None, **kwargs)
        z = candidate.layer_norm_gated_fwd(x, g, w, None, **kwargs)
        torch.testing.assert_close(a, z, rtol=0, atol=0)
        assert z[1] is not None and z[2] is not None
        a = baseline.layer_norm_gated_fwd(x, g, w, None, is_rms_norm=True, **kwargs)
        z = candidate.layer_norm_gated_fwd(x, g, w, None, is_rms_norm=True, return_stats=False, **kwargs)
        assert z[1] is None and z[2] is None
        torch.testing.assert_close(a[0], z[0], rtol=0, atol=0)
        torch.testing.assert_close(a[3], z[3], rtol=0, atol=0)
    return {"bitwise_equal": tested, "default_stats_and_no_stats_dims": [128, 768]}
