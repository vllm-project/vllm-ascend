#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the exact torch_npu entry point used by the NoPE SFA adapter.

Run inside the target container after sourcing the custom operator environment:
    python smoke_nope_prolog_npu.py --device 0
Uses small synthetic W8A8 weights and private KV storage. This is an eager ABI
and padding-write check, not model-accuracy, performance, or FULL-graph proof.
"""

import argparse
import json

import torch
import torch_npu

ACL_FORMAT_FRACTAL_NZ = 29


def main():
    """Build synthetic W8A8 inputs, check Dr=0 outputs and isolate KV writes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="Logical NPU index in this process")
    args = parser.parse_args()
    torch_npu.npu.set_device(args.device)
    # Unlike the model runner, this standalone process has not enabled NZ.
    # Otherwise format_cast can silently leave these INT8 weights in ND.
    torch.npu.config.allow_internal_format = True
    torch.manual_seed(42)
    device = torch.device(f"npu:{args.device}")
    dtype = torch.bfloat16
    hidden_size, q_rank, kv_rank, heads, qk_dim = 1024, 1536, 512, 2, 256
    tile_size, block_size, blocks = 128, 16, 2
    packed_width = kv_rank + (kv_rank // tile_size) * 4

    def weight(rows, columns):
        tensor = torch.randint(-4, 5, (rows, columns), dtype=torch.int8, device=device)
        tensor = torch_npu.npu_format_cast(tensor, ACL_FORMAT_FRACTAL_NZ)
        actual_format = torch_npu.get_npu_format(tensor)
        if actual_format != ACL_FORMAT_FRACTAL_NZ:
            raise RuntimeError(
                f"MlaPrologV3 INT8 weights require FRACTAL_NZ (29), got format {actual_format}. "
                "Check torch.npu.config.allow_internal_format and the torch_npu/device combination."
            )
        return tensor

    common = {
        "weight_dq": weight(hidden_size, q_rank),
        "weight_uq_qr": weight(q_rank, heads * qk_dim),
        "weight_dkv_kr": weight(hidden_size, kv_rank),
        "weight_uk": torch.randn(heads, qk_dim, kv_rank, dtype=dtype, device=device) / 16,
        "rmsnorm_gamma_cq": torch.ones(q_rank, dtype=dtype, device=device),
        "rmsnorm_gamma_ckv": torch.ones(kv_rank, dtype=dtype, device=device),
        "rmsnorm_epsilon_cq": 1e-6,
        "rmsnorm_epsilon_ckv": 1e-6,
        "dequant_scale_w_dq": torch.full((1, q_rank), 0.01, device=device),
        "dequant_scale_w_uq_qr": torch.full((1, heads * qk_dim), 0.01, device=device),
        "dequant_scale_w_dkv_kr": torch.full((1, kv_rank), 0.01, device=device),
        "kr_cache": torch.empty(0, dtype=dtype, device=device),
        "k_nope_clip_alpha": torch.ones(1, device=device),
        "weight_quant_mode": 2,
        "kv_cache_quant_mode": 3,
        "query_quant_mode": 0,
        "query_norm_flag": True,
        "ckvkr_repo_mode": 1,
        "quant_scale_repo_mode": 1,
        "tile_size": tile_size,
        "qc_qr_scale": 1.0,
        "kc_scale": 1.0,
        "cache_mode": "PA_BSND",
    }
    op = torch_npu.npu_mla_prolog_v3

    def run(hidden, slot_values):
        rows = hidden.shape[0]
        token_x, scales = torch_npu.npu_dynamic_quant(hidden.contiguous())
        cache = torch.full((blocks, block_size, 1, packed_width), 37, dtype=torch.int8, device=device)
        before = cache.clone()
        slots = torch.tensor(slot_values, dtype=torch.int64, device=device)
        result = op(
            token_x=token_x,
            dequant_scale_x=scales.view(rows, 1),
            rope_sin=torch.empty(rows, 0, dtype=dtype, device=device),
            rope_cos=torch.empty(rows, 0, dtype=dtype, device=device),
            kv_cache=cache,
            cache_index=slots,
            **common,
        )
        torch_npu.npu.synchronize()
        assert len(result) == 5, f"Expected torch_npu's five-result ABI, got {len(result)}"
        query, query_rope, _, qc, qc_scale = result
        assert query.shape[0] == rows and query.shape[-1] == kv_rank
        assert query.numel() == rows * heads * kv_rank
        assert query_rope.shape[0] == rows and query_rope.shape[-1] == 0
        assert qc.shape[0] == rows and qc.numel() == rows * q_rank
        if qc.dtype == torch.int8:
            assert qc_scale is not None and qc_scale.shape[0] == rows and qc_scale.numel() == rows
        assert bool(torch.isfinite(query).all()), "Non-finite query output"
        untouched = torch.ones(blocks * block_size, dtype=torch.bool, device=device)
        valid_slots = [slot for slot in slot_values if slot >= 0]
        untouched[torch.tensor(valid_slots, dtype=torch.int64, device=device)] = False
        flat = cache.view(-1, packed_width)
        assert torch.equal(flat[untouched], before.view(-1, packed_width)[untouched]), "Unmapped KV slot was modified"
        assert not torch.equal(cache, before), "No KV cache writes observed"
        return query, flat

    with torch.inference_mode():
        run(torch.randn(1, hidden_size, dtype=dtype, device=device), [0])
        hidden = torch.randn(4, hidden_size, dtype=dtype, device=device)
        reference_query, reference_cache = run(hidden, [0, 1, 2, 3])
        padded_query, padded_cache = run(hidden, [0, 1, 2, -1])
        torch.testing.assert_close(padded_query[:3], reference_query[:3], rtol=1e-2, atol=1e-2)
        assert torch.equal(padded_cache[:3], reference_cache[:3]), "Valid KV writes changed with a padding slot"

    print(
        json.dumps(
            {
                "status": "passed",
                "entry": "torch_npu.npu_mla_prolog_v3",
                "torch": torch.__version__,
                "torch_npu": torch_npu.__version__,
                "torch_npu_path": torch_npu.__file__,
                "soc": str(torch_npu.npu.get_soc_version()),
                "device": str(device),
                "operator_calls": 3,
                "checked": [
                    "INT8 weights use FRACTAL_NZ (29)",
                    "Dr=0 shapes",
                    "W8A8+C8",
                    "single token",
                    "four-token bucket",
                    "-1 slot leaves unmapped cache unchanged",
                ],
                "not_checked": [
                    "whole model accuracy",
                    "ACL graph capture/replay",
                    "performance",
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
