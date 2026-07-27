# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Accuracy test for npu_recurrent_kda."""

from __future__ import annotations

import os
import sys

import torch
import torch_npu
from golden import recurrent_kda_golden
from utils import compare_tensors_by_ratio


def _device():
    device_id = int(os.environ.get("TEST_DEVICE_ID", "0"))
    return torch.device(f"npu:{device_id}")


def make_inputs(*, layout="BSND", batch=2, seq_len=2, h=2, hv=4, kdim=128, vdim=128, seed=0, with_initial_state=True):
    torch.manual_seed(seed)
    if layout == "BSND":
        q_shape = (batch, seq_len, h, kdim)
        v_shape = (batch, seq_len, hv, vdim)
        g_shape = (batch, seq_len, hv, kdim)
        beta_shape = (batch, seq_len, hv)
        cu_seqlens = [seq_len * i for i in range(batch + 1)]
        seq_num = batch
    elif layout == "TND":
        total_tokens = batch * seq_len
        q_shape = (total_tokens, h, kdim)
        v_shape = (total_tokens, hv, vdim)
        g_shape = (total_tokens, hv, kdim)
        beta_shape = (total_tokens, hv)
        cu_seqlens = [seq_len * i for i in range(batch + 1)]
        seq_num = batch
    else:
        raise ValueError(layout)

    q = torch.randn(q_shape, dtype=torch.bfloat16)
    k = torch.randn(q_shape, dtype=torch.bfloat16)
    v = torch.randn(v_shape, dtype=torch.bfloat16)
    g = torch.randn(g_shape, dtype=torch.float32) * 0.5
    beta = torch.randn(beta_shape, dtype=torch.float32)
    initial_state = torch.randn((seq_num, hv, vdim, kdim), dtype=torch.float32) * 0.02 if with_initial_state else None
    A_log = torch.randn((hv,), dtype=torch.float32) * 0.1
    dt_bias = torch.randn((hv, kdim), dtype=torch.float32) * 0.1
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "initial_state": initial_state,
        "cu_seqlens": cu_seqlens,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "layout": layout,
    }


def run_case(desc, kwargs, op_kwargs, rtol=0.02, atol=0.01):
    print(f"\n=== {desc} ===")
    inp = make_inputs(**kwargs)
    golden = recurrent_kda_golden(**inp, output_final_state=True, **op_kwargs)

    dev = _device()
    torch_npu.npu.set_device(dev)
    from fla_npu.ops.ascendc import recurrent_kda

    call_kwargs = {**op_kwargs, "output_final_state": True, "layout": inp["layout"]}
    call_kwargs["cu_seqlens"] = torch.tensor(inp["cu_seqlens"], dtype=torch.int64, device=dev)
    initial_state_arg = inp["initial_state"].to(dev) if inp["initial_state"] is not None else None
    out, final_state = recurrent_kda(
        inp["q"].to(dev),
        inp["k"].to(dev),
        inp["v"].to(dev),
        inp["g"].to(dev),
        inp["beta"].to(dev),
        initial_state_arg,
        A_log=inp["A_log"].to(dev) if op_kwargs.get("use_gate_in_kernel", False) else None,
        dt_bias=inp["dt_bias"].to(dev) if op_kwargs.get("use_gate_in_kernel", False) else None,
        **call_kwargs,
    )
    torch_npu.npu.synchronize()

    out_ok = compare_tensors_by_ratio(golden[0], out.cpu(), "out", rtol=rtol, atol=atol)
    state_ok = compare_tensors_by_ratio(golden[1], final_state.cpu(), "final_state", rtol=rtol, atol=atol)
    return out_ok and state_ok


def run_multisequence_beta_visibility_case():
    print("\n=== BSND multi-sequence beta scalar visibility ===")
    generator = torch.Generator().manual_seed(20260746)
    lengths = [4, 7, 2, 5, 3]
    cu_seqlens = [0]
    for length in lengths:
        cu_seqlens.append(cu_seqlens[-1] + length)

    tokens, h, hv, kdim, vdim = cu_seqlens[-1], 8, 32, 128, 256

    def randn(shape, dtype, scale):
        return (
            (torch.randn(shape, generator=generator, dtype=torch.float32) * scale)
            .to(dtype)
            .contiguous()
        )

    q = randn((1, tokens, h, kdim), torch.bfloat16, 0.05)
    k = randn((1, tokens, h, kdim), torch.bfloat16, 0.05)
    v = randn((1, tokens, hv, vdim), torch.bfloat16, 0.05)
    g = (
        -(0.1 + torch.rand((1, tokens, hv, kdim), generator=generator) * 0.4)
    ).to(torch.float16)
    beta = randn((1, tokens, hv), torch.float32, 0.5)
    op_kwargs = {
        "layout": "BSND",
        "scale": None,
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": False,
        "use_gate_in_kernel": False,
        "use_beta_sigmoid_in_kernel": True,
        "allow_neg_eigval": True,
        "safe_gate": False,
        "lower_bound": -5.0,
        "state_v_first": True,
    }
    inputs = {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "initial_state": None,
        "cu_seqlens": cu_seqlens,
        "ssm_state_indices": None,
        "A_log": None,
        "dt_bias": None,
        "num_accepted_tokens": None,
    }
    golden = recurrent_kda_golden(**inputs, **op_kwargs)

    dev = _device()
    torch_npu.npu.set_device(dev)
    from fla_npu.ops.ascendc import recurrent_kda

    device_inputs = {
        key: value.to(dev) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }
    device_inputs["cu_seqlens"] = torch.tensor(
        cu_seqlens, dtype=torch.int64, device=dev
    )
    out, final_state = recurrent_kda(**device_inputs, **op_kwargs)
    torch_npu.npu.synchronize()

    out_ok = compare_tensors_by_ratio(
        golden[0], out.cpu(), "out", rtol=0.02, atol=0.01
    )
    state_ok = compare_tensors_by_ratio(
        golden[1], final_state.cpu(), "final_state", rtol=0.02, atol=0.01
    )
    return out_ok and state_ok


def main():
    results = [
        run_case(
            "BSND raw gate, safe_gate=False, beta sigmoid",
            {"layout": "BSND", "batch": 2, "seq_len": 2, "seed": 1},
            {
                "use_qk_l2norm_in_kernel": True,
                "use_gate_in_kernel": True,
                "use_beta_sigmoid_in_kernel": True,
                "allow_neg_eigval": False,
                "safe_gate": False,
            },
        ),
        run_case(
            "BSND raw gate, safe_gate=True",
            {"layout": "BSND", "batch": 2, "seq_len": 2, "seed": 2},
            {
                "use_qk_l2norm_in_kernel": True,
                "use_gate_in_kernel": True,
                "use_beta_sigmoid_in_kernel": True,
                "allow_neg_eigval": True,
                "safe_gate": True,
                "lower_bound": -4.0,
            },
        ),
        run_case(
            "TND precomputed log gate",
            {"layout": "TND", "batch": 2, "seq_len": 2, "seed": 3, "with_initial_state": False},
            {
                "use_qk_l2norm_in_kernel": False,
                "use_gate_in_kernel": False,
                "use_beta_sigmoid_in_kernel": False,
                "safe_gate": False,
            },
        ),
        run_multisequence_beta_visibility_case(),
    ]
    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
