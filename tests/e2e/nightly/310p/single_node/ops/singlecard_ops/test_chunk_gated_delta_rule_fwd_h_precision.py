"""
Precision test: chunk_gated_delta_rule_fwd_h on Ascend 310P.
Uses the vllm-ascend Python API (chunk_gated_delta_rule_pytorch).
Compares NPU fp16 vs CPU fp32 reference.
"""

import math
import sys

import torch
import torch_npu

from vllm_ascend._310p.ops.fla.chunk_gated_delta_rule import chunk_gated_delta_rule_pytorch

torch.manual_seed(42)
torch_npu.npu.set_device(0)

CHUNK_SIZE = 64


def cosine(a, b, thr=0.99):
    a, b = a.flatten().double(), b.flatten().double()
    if a.norm() == 0 and b.norm() == 0:
        return 1.0, True
    if a.norm() == 0 or b.norm() == 0:
        return 0.0, False
    c = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    return c, not math.isnan(c) and c >= thr


def run_test(label, B, T, Hqk, Hv, K, V, dtype=torch.float16):
    q = torch.randn(B, T, Hqk, K, dtype=dtype)
    k = torch.randn(B, T, Hqk, K, dtype=dtype)
    v = torch.randn(B, T, Hv, V, dtype=dtype)
    g = -torch.rand(B, T, Hv, dtype=torch.float32) * 0.2
    beta = (0.1 + 0.4 * torch.rand(B, T, Hv, dtype=torch.float32)).to(dtype)
    init = torch.randn(B, Hv, V, K, dtype=dtype) * 0.01

    # CPU fp32 reference
    ref_out, ref_state = chunk_gated_delta_rule_pytorch(
        q=q.float(),
        k=k.float(),
        v=v.float(),
        g=g.float(),
        beta=beta.float(),
        initial_state=init.float(),
        output_final_state=True,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )

    # NPU
    npu_out, npu_state = chunk_gated_delta_rule_pytorch(
        q=q.to("npu"),
        k=k.to("npu"),
        v=v.to("npu"),
        g=g.to("npu"),
        beta=beta.to("npu"),
        initial_state=init.to("npu"),
        output_final_state=True,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )
    torch.npu.synchronize()
    npu_out = npu_out.cpu().float()
    npu_state = npu_state.cpu().float() if npu_state is not None else None

    cos_out, ok_out = cosine(npu_out, ref_out)
    cos_st, ok_st = cosine(npu_state, ref_state) if npu_state is not None else (1.0, True)
    mae_out = (npu_out - ref_out).abs().mean().item()

    status = "OK" if (ok_out and ok_st) else "FAIL"
    print(f"  {label}: out_cos={cos_out:.6f} state_cos={cos_st:.6f} mae={mae_out:.6f} [{status}]")
    return ok_out and ok_st


def main():
    print("chunk_gated_delta_rule 310P precision test")
    print(f"Device: {torch_npu.npu.get_device_name(0)}")
    print()

    ok = True
    ok &= run_test("small", B=1, T=64, Hqk=2, Hv=4, K=16, V=16)
    ok &= run_test("Qwen3 1chunk", B=1, T=64, Hqk=4, Hv=8, K=192, V=128)
    ok &= run_test("Qwen3 8chunk", B=1, T=512, Hqk=4, Hv=8, K=192, V=128)
    ok &= run_test("batch=4", B=4, T=128, Hqk=2, Hv=4, K=128, V=128)

    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
