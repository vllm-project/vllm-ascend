import argparse

import torch
import torch_npu  # noqa: F401

import vllm_ascend.vllm_ascend_C  # noqa: F401


def check_meta() -> None:
    x = torch.empty((2, 3), device="meta", dtype=torch.float16)
    y = torch.ops._C_ascend.custom_muls(x, 1.25)
    assert tuple(y.shape) == (2, 3)
    assert y.dtype == torch.float16
    assert y.device.type == "meta"


def check_npu(device: int) -> None:
    torch.npu.set_device(device)
    cases = [
        ((45, 128), torch.bfloat16),
        ((6912, 128), torch.bfloat16),
        ((108, 64), torch.bfloat16),
        ((45, 128), torch.float16),
        ((45, 128), torch.float32),
    ]
    for shape, dtype in cases:
        x = torch.randn(shape, device="npu", dtype=dtype)
        scalar = -0.75
        y = torch.ops._C_ascend.custom_muls(x, scalar)
        ref = x * scalar
        torch.npu.synchronize()
        diff = (y.float().cpu() - ref.float().cpu()).abs()
        print(f"{dtype} {shape}: max={float(diff.max())} mean={float(diff.mean())}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--meta-only", action="store_true")
    args = parser.parse_args()

    assert hasattr(torch.ops._C_ascend, "custom_muls")
    print("has_custom_muls True")
    check_meta()
    print("meta ok")
    if not args.meta_only:
        check_npu(args.device)


if __name__ == "__main__":
    main()
