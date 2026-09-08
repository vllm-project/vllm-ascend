import argparse

import torch
import torch_npu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "case",
        choices=(
            "functional",
            "alpha",
            "broadcast",
            "broadcast_negated",
        ),
    )
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    torch_npu.npu.set_device(0)
    lhs = torch.randn(128, 6144, device="npu", dtype=torch.bfloat16)
    rhs = torch.randn_like(lhs)
    bias = torch.randn(lhs.shape[-1], device="npu", dtype=lhs.dtype)
    neg_bias = -bias
    work = lhs.clone()

    if not args.profile_only:
        candidate = torch.add(lhs, neg_bias, alpha=-1.0)
        torch.testing.assert_close(candidate, torch.add(lhs, bias), atol=0, rtol=0)

    def run() -> torch.Tensor:
        if args.case == "functional":
            return torch.add(lhs, rhs)
        if args.case == "broadcast":
            return work.add_(bias)
        if args.case == "broadcast_negated":
            return work.add_(neg_bias, alpha=-1.0)
        return torch.add(lhs, rhs, alpha=0.5)

    for _ in range(5):
        run()
    torch.npu.synchronize()
    for _ in range(20):
        run()
    torch.npu.synchronize()
    print(f"AXPY_LOWERING_PROBE_PASSED case={args.case}")


if __name__ == "__main__":
    main()
