# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the A3 Cube/Vector KDA verify core with a direct state recurrence."""

import argparse
import time

import torch
import torch.nn.functional as F
from reference import DIM, MAX_TOKENS, oracle, prepare, prepared_reference


def fixture(jobs, tokens, seed):
    rng = torch.Generator().manual_seed(seed)
    shape = (jobs, tokens, DIM)
    q = F.normalize(torch.randn(shape, generator=rng), dim=-1) * DIM**-0.5
    k = F.normalize(torch.randn(shape, generator=rng), dim=-1)
    v = torch.randn(shape, generator=rng)
    decay = torch.sigmoid(torch.randn(shape, generator=rng) + 1.5)
    beta = torch.sigmoid(torch.randn((jobs, tokens), generator=rng))
    state = torch.randn((jobs, DIM, DIM), generator=rng).half()
    return q, k, v, decay, beta, state


def check(result, expected, tokens):
    errors = []
    for actual, ref in zip(result, expected, strict=True):
        actual = actual[:, :tokens].detach().cpu().double()
        torch.testing.assert_close(actual, ref, rtol=2e-3, atol=1e-3)
        errors.append((actual - ref).abs().max().item())
    return errors


def timed_us(kernel, packed, iterations):
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        kernel(*packed)
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1e6 / iterations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "npu"), default="npu")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--tokens", type=int, choices=range(1, MAX_TOKENS + 1), default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=0, help="Calls per A3 timing sample (0 skips timing)")
    args = parser.parse_args()
    if min(args.batch, args.heads) < 1 or args.iterations < 0:
        parser.error("batch and heads must be positive, and iterations must be nonnegative")
    if args.device == "cpu" and args.iterations:
        parser.error("timing requires --device npu")

    jobs = args.batch * args.heads
    inputs = fixture(jobs, args.tokens, args.seed)
    checkpoint_before = inputs[-1].clone()
    expected = oracle(*inputs)
    if args.device == "cpu":
        packed = prepare(*inputs)
        result = prepared_reference(*packed, args.tokens)
    else:
        import torch_npu  # noqa: F401  (register the NPU backend)
        from kernel import make_kernel

        name = torch.npu.get_device_name(torch.npu.current_device())
        if "910_93" not in name and "910C" not in name:
            parser.error(f"expected an Ascend A3 (910_93 or 910C), got {name}")
        packed = prepare(*(tensor.npu() for tensor in inputs))
        kernel = make_kernel(jobs, args.tokens)
        result = kernel(*packed)
    errors = check(result, expected, args.tokens)
    torch.testing.assert_close(packed[1].cpu(), checkpoint_before, rtol=0, atol=0)
    print(
        f"{args.device}: B={args.batch} H={args.heads} L={args.tokens} passed; "
        f"max abs error: correction={errors[0]:.3e}, output={errors[1]:.3e}"
    )

    if args.iterations:
        serial = make_kernel(jobs, args.tokens, prefetch=False)
        check(serial(*packed), expected, args.tokens)
        for _ in range(5):
            serial(*packed)
            kernel(*packed)
        # Alternate the order to reduce sensitivity to linear timing drift.
        a, b, c, d = [timed_us(fn, packed, args.iterations) for fn in (serial, kernel, kernel, serial)]
        serial_us, prefetch_us = (a + d) / 2, (b + c) / 2
        torch.testing.assert_close(packed[1].cpu(), checkpoint_before, rtol=0, atol=0)
        print(
            f"verify core: wait-first={serial_us:.2f} us/call, prefetch={prefetch_us:.2f} us/call, "
            f"wait-first/prefetch={serial_us / prefetch_us:.3f}x ({args.iterations} calls/sample)"
        )


if __name__ == "__main__":
    main()
