"""A/B benchmark: _apply_top_k_top_p_pytorch(NPU) vs npu_top_k_top_p(NPU direct).

Runs on Ascend NPU. Focuses on the non-reduce_sample path
(enable_reduce_sample=False) to avoid TP-group setup.

Usage:
    python benchmarks/ops/bench_topk_topp_sampler.py
    python benchmarks/ops/bench_topk_topp_sampler.py --batch 32 --vocab 152000
"""

import argparse
import logging

import torch
import torch_npu  # noqa: F401  -- must import before vllm_ascend on NPU

from vllm_ascend.sample.sampler import _apply_top_k_top_p_pytorch


# ---------------------------------------------------------------------------
# Stub get_ascend_config so we don't need full vLLM init.
# We force enable_reduce_sample=False to test the non-reduce path only.
# ---------------------------------------------------------------------------
class _StubAscendConfig:
    enable_reduce_sample = False


def _stub_get_ascend_config():
    return _StubAscendConfig()


import vllm_ascend.sample.sampler as _sampler_mod  # noqa: E402

_sampler_mod.get_ascend_config = _stub_get_ascend_config

logging.getLogger("vllm_ascend.sample.sampler").setLevel(logging.WARNING)

# None means the request does not pass that field.
TOP_K_VALUES = [1, None, 128000, 4096]
TOP_P_VALUES = [None, 1, 0.95, 0.5]

CASES = [{"top_k": k, "top_p": p} for k in TOP_K_VALUES for p in TOP_P_VALUES]


def _npu_top_k_top_p_direct(logits, k, p):
    """Direct call to torch_npu.npu_top_k_top_p, no wrapper overhead.

    This is path B: the fused NPU operator without any Python-level
    branching, config checks, or fast-path logic.
    """
    if p is None and k is None:
        return logits
    return torch_npu.npu_top_k_top_p(logits, k=k, p=p)


def make_inputs(batch: int, vocab: int, k_val, p_val, device: str, dtype: torch.dtype):
    """Create logits, k tensor, p tensor for a given scenario."""
    logits = torch.randn(batch, vocab, device=device, dtype=dtype)

    if k_val is None:
        k = None
    else:
        k = torch.full((batch,), k_val, dtype=torch.int32, device=device)

    if p_val is None:
        p = None
    else:
        p = torch.full((batch,), p_val, dtype=torch.float32, device=device)

    return logits, k, p


def case_name(k_val, p_val):
    """Short label for a case, e.g. 'k=1,p=0.95' or 'k=None,p=None'."""
    k_str = "None" if k_val is None else str(k_val)
    p_str = "None" if p_val is None else str(p_val)
    return f"k={k_str},p={p_str}"


def _stats(times):
    """Return min/median/mean/max from a list of ms timings."""
    times_sorted = sorted(times)
    n = len(times_sorted)
    return {
        "min": times_sorted[0],
        "median": times_sorted[n // 2],
        "mean": sum(times_sorted) / n,
        "max": times_sorted[-1],
    }


def bench_npu(fn, logits, k, p, warmup=5, iters=20):
    """Time a sampler function on NPU. Returns stats dict or None on error."""
    try:
        for _ in range(warmup):
            _ = fn(logits.clone(), k, p)
        torch.npu.synchronize()
    except Exception:
        return None

    times = []
    for _ in range(iters):
        torch.npu.synchronize()
        start = torch.Event(enable_timing=True)
        end = torch.Event(enable_timing=True)
        start.record()
        _ = fn(logits.clone(), k, p)
        end.record()
        torch.npu.synchronize()
        times.append(start.elapsed_time(end))

    return _stats(times)


def _fmt_median(stats):
    """Format median for summary table, or 'ERROR' if stats is None."""
    if stats is None:
        return "ERROR"
    return f"{stats['median']:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--vocab", type=int, default=152000)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--no-plot", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    device = "npu"
    dtype = getattr(torch, args.dtype)
    B, V = args.batch, args.vocab
    do_plot = not args.no_plot

    # Collect results for visualization: [(name, s_pt, s_npu)]
    all_results = []

    # Header
    print(f"\n{'=' * 72}")
    print("A/B Benchmark: pytorch(NPU) vs npu_top_k_top_p(NPU direct)")
    print(f"  batch={B}, vocab={V}, dtype={dtype}, device={device}")
    print(f"  warmup={args.warmup}, iters={args.iters}")
    print(f"  cases: {len(CASES)} (cross-product of TOP_K_VALUES x TOP_P_VALUES)")
    print(f"{'=' * 72}")

    # Summary table (median only)
    print(f"\n{'Case':<22} {'A:pt_npu':>11} {'B:npu':>10} {'ratio':>8} {'best':>12}")
    print(f"{'-' * 66}")

    for case in CASES:
        k_val = case["top_k"]
        p_val = case["top_p"]
        name = case_name(k_val, p_val)

        logits, k, p = make_inputs(B, V, k_val, p_val, device, dtype)

        s_pt = bench_npu(_apply_top_k_top_p_pytorch, logits, k, p, args.warmup, args.iters)
        s_npu = bench_npu(_npu_top_k_top_p_direct, logits, k, p, args.warmup, args.iters)

        medians = {}
        if s_pt is not None:
            medians["A:pt_npu"] = s_pt["median"]
        if s_npu is not None:
            medians["B:npu"] = s_npu["median"]
        best = min(medians, key=medians.get) if medians else "N/A"

        pt_str = _fmt_median(s_pt)
        npu_str = _fmt_median(s_npu)
        if s_pt is not None and s_npu is not None and s_npu["median"] > 0:
            ratio = f"{s_pt['median'] / s_npu['median']:.2f}x"
        else:
            ratio = "N/A"

        print(f"{name:<22} {pt_str:>11} {npu_str:>10} {ratio:>8} {best:>12}")

        all_results.append((name, s_pt, s_npu))

    # --- Mixed batch: per-request varying k/p (realistic production scenario) ---
    print(f"\n{'=' * 72}")
    print("Mixed batch (per-request k/p all different, simulating real traffic):")
    print(f"{'=' * 72}")

    import random

    random.seed(42)

    def make_mixed_inputs(batch, vocab, device, dtype, k_list, p_list):
        logits = torch.randn(batch, vocab, device=device, dtype=dtype)
        k = torch.tensor(k_list[:batch], dtype=torch.int32, device=device)
        p = torch.tensor(p_list[:batch], dtype=torch.float32, device=device)
        return logits, k, p

    # Pattern 1: typical production mix
    k_mix_1 = []
    p_mix_1 = []
    for i in range(B):
        choice = i % 5
        if choice == 0:
            k_mix_1.append(1)
            p_mix_1.append(1.0)
        elif choice == 1:
            k_mix_1.append(4096)
            p_mix_1.append(0.95)
        elif choice == 2:
            k_mix_1.append(V)
            p_mix_1.append(1.0)
        elif choice == 3:
            k_mix_1.append(V)
            p_mix_1.append(0.5)
        else:
            k_mix_1.append(128000)
            p_mix_1.append(1.0)

    # Pattern 2: random mix
    k_choices = [1, 10, 50, 4096, 128000, V]
    p_choices = [1.0, 0.95, 0.9, 0.5]
    k_mix_2 = [random.choice(k_choices) for _ in range(B)]
    p_mix_2 = [random.choice(p_choices) for _ in range(B)]

    # Pattern 3: half small-k (k=1), half unrestricted
    k_mix_3 = []
    p_mix_3 = []
    for i in range(B):
        if i < B // 2:
            k_mix_3.append(1)
            p_mix_3.append(1.0)
        else:
            k_mix_3.append(V)
            p_mix_3.append(0.95)

    # Pattern 4: most requests have small k (<4096), one unrestricted, all p=0.95
    k_mix_4 = []
    p_mix_4 = []
    small_k_pool = [1, 10, 50, 100, 500, 1000, 2000, 4096]
    for i in range(B):
        p_mix_4.append(0.95)
        if i == B - 1:
            k_mix_4.append(V)
        else:
            k_mix_4.append(random.choice(small_k_pool))

    mixed_patterns = [
        ("typical_prod", k_mix_1, p_mix_1),
        ("random_mix", k_mix_2, p_mix_2),
        ("half_k1_half_unrestricted", k_mix_3, p_mix_3),
        ("mostly_small_k_one_unrestricted", k_mix_4, p_mix_4),
    ]

    print(f"\n{'Pattern':<30} {'A:pt_npu':>11} {'B:npu':>10} {'ratio':>8} {'best':>12}")
    print(f"{'-' * 74}")

    for name, k_list, p_list in mixed_patterns:
        logits, k, p = make_mixed_inputs(B, V, device, dtype, k_list, p_list)

        k_str = ",".join(str(x) for x in k_list[: min(8, B)])
        p_str = ",".join(str(x) for x in p_list[: min(8, B)])
        if B > 8:
            k_str += ",..."
            p_str += ",..."
        print(f"  k=[{k_str}]")
        print(f"  p=[{p_str}]")

        s_pt = bench_npu(_apply_top_k_top_p_pytorch, logits, k, p, args.warmup, args.iters)
        s_npu = bench_npu(_npu_top_k_top_p_direct, logits, k, p, args.warmup, args.iters)

        medians = {}
        if s_pt is not None:
            medians["A:pt_npu"] = s_pt["median"]
        if s_npu is not None:
            medians["B:npu"] = s_npu["median"]
        best = min(medians, key=medians.get) if medians else "N/A"

        pt_str = _fmt_median(s_pt)
        npu_str = _fmt_median(s_npu)
        if s_pt is not None and s_npu is not None and s_npu["median"] > 0:
            ratio = f"{s_pt['median'] / s_npu['median']:.2f}x"
        else:
            ratio = "N/A"

        print(f"  {name:<28} {pt_str:>11} {npu_str:>10} {ratio:>8} {best:>12}\n")

        all_results.append((name, s_pt, s_npu))

    print(f"{'=' * 72}\n")

    # --- Correctness check ---
    print(f"{'=' * 72}")
    print("Correctness check (top-k only, k=10):")
    print(f"{'=' * 72}")
    logits, k, p = make_inputs(B, V, 10, None, device, dtype)
    out_pt = _apply_top_k_top_p_pytorch(logits.clone(), k, p)
    out_npu = _npu_top_k_top_p_direct(logits.clone(), k, p)
    valid_pt = out_pt != -float("inf")
    valid_npu = out_npu != -float("inf")
    match = (valid_pt == valid_npu).all().item()
    print(f"  pytorch(NPU) vs npu_direct(NPU) valid-position match: {match}")
    if not match:
        mismatch = (valid_pt != valid_npu).sum().item()
        print(f"  mismatched positions: {mismatch}")
    print()

    # --- Visualization ---
    if do_plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("matplotlib not available, skipping visualization.")
            return

        n_uniform = len(CASES)
        uniform_results = all_results[:n_uniform]
        mixed_results = all_results[n_uniform:]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        def plot_group(ax, results, title):
            names = [r[0] for r in results]
            pt_vals = [r[1]["median"] if r[1] else 0 for r in results]
            npu_vals = [r[2]["median"] if r[2] else 0 for r in results]

            x = np.arange(len(names))
            width = 0.35

            ax.bar(x - width / 2, pt_vals, width, label="A: pytorch_npu", color="#4C72B0", alpha=0.85)
            ax.bar(x + width / 2, npu_vals, width, label="B: npu_direct", color="#55A868", alpha=0.85)

            ax.set_ylabel("Median time (ms)")
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
            ax.set_yscale("log")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)

        plot_group(ax1, uniform_results, f"Uniform cases (B={B}, V={V})")
        plot_group(ax2, mixed_results, "Mixed batch patterns")

        fig.suptitle(
            f"A/B Benchmark  (batch={B}, vocab={V}, dtype={dtype}, iters={args.iters})", fontsize=14, fontweight="bold"
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        out_file = "bench_topk_topp_sampler.png"
        fig.savefig(out_file, dpi=150)
        print(f"Visualization saved to {out_file}")
        plt.close(fig)


if __name__ == "__main__":
    main()
