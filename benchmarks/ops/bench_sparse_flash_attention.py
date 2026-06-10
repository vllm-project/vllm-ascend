"""
Benchmark for torch.ops._C_ascend.npu_sparse_flash_attention.

Scans (each varies one shape axis, others fixed):
  decode_T       : batch = T (1 q token per req), kv_len fixed
  decode_kvlen   : batch fixed, kv_len varies
  prefill_T      : batch = 1, T = kv_len (causal prefill of length T)
  prefill_kvlen  : batch = 1, T fixed, kv_len varies (continuation prefill)

Outputs:
  <out_dir>/results.csv
  <out_dir>/plots/<scan>.png   (latency + TFLOPS dual-axis)

Run on an NPU host:
  cd benchmarks/ops
  python bench_sparse_flash_attention.py --out-dir ./sfa_bench_out
"""

import argparse
import csv
import math
import os
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import torch
import torch_npu  # noqa: F401

import vllm_ascend.platform  # noqa: F401


# --- Default model dims (DeepSeek-V3.2 SFA, single TP shard) ---
N_HEADS = 64
D_NOPE = 512
D_ROPE = 64
BLOCK_SIZE = 64
TOPK = 2048
DTYPE = torch.bfloat16
DEVICE = "npu"


@dataclass
class CaseResult:
    scan: str
    scenario: str
    batch: int
    T: int
    kv_len: int
    n_heads: int
    topk: int
    block_size: int
    latency_us: float
    tflops: float


def make_inputs(
    *,
    batch: int,
    T: int,
    kv_len: int,
    cum_query_lens: Sequence[int],
    kv_lens: Sequence[int],
    n_heads: int = N_HEADS,
    topk: int = TOPK,
    d_nope: int = D_NOPE,
    d_rope: int = D_ROPE,
    block_size: int = BLOCK_SIZE,
    dtype: torch.dtype = DTYPE,
    device: str = DEVICE,
):
    """Build a self-consistent kwargs dict for npu_sparse_flash_attention."""
    assert len(cum_query_lens) == batch and cum_query_lens[-1] == T
    assert len(kv_lens) == batch

    max_blocks_per_req = math.ceil(kv_len / block_size)
    # Share one block pool across reqs to keep KV cache memory bounded;
    # block_table indirection still exercises the same kernel path.
    num_blocks = max_blocks_per_req

    query = torch.randn(T, n_heads, d_nope, dtype=dtype, device=device)
    query_rope = torch.randn(T, n_heads, d_rope, dtype=dtype, device=device)
    kv = torch.randn(num_blocks, block_size, 1, d_nope, dtype=dtype, device=device)
    key_rope = torch.randn(num_blocks, block_size, 1, d_rope, dtype=dtype, device=device)

    block_table = (
        torch.arange(num_blocks, dtype=torch.int32, device=device)
        .view(1, num_blocks)
        .expand(batch, -1)
        .contiguous()
    )

    # sparse_indices: [T, kv_heads=1, topk_eff]. Random indices over [0, kv_len)
    # for discrete/scattered KV access. When kv_len < topk we shrink topk_eff to
    # kv_len instead of padding with duplicates — duplicates would hit the same
    # cached row and bias latency downward (perf measurement artifact).
    topk_eff = min(topk, kv_len)
    sparse = torch.randint(0, kv_len, (T, 1, topk_eff), dtype=torch.int32, device=device)

    asl_q = torch.tensor(list(cum_query_lens), dtype=torch.int32, device=device)
    asl_kv = torch.tensor(list(kv_lens), dtype=torch.int32, device=device)

    return dict(
        query=query,
        key=kv,
        value=kv,
        sparse_indices=sparse,
        scale_value=1.0 / math.sqrt(d_nope + d_rope),
        sparse_block_size=1,
        block_table=block_table,
        actual_seq_lengths_query=asl_q,
        actual_seq_lengths_kv=asl_kv,
        query_rope=query_rope,
        key_rope=key_rope,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
    )


def estimate_tflops(T, topk, n_heads, d_nope, d_rope, latency_s):
    # Sparse attention FLOPs per (q_token, head):
    #   QK^T          : 2 * topk * (d_nope + d_rope)
    #   softmax * V   : 2 * topk * d_nope
    flops = T * n_heads * (2 * topk * (d_nope + d_rope) + 2 * topk * d_nope)
    return flops / latency_s / 1e12


def time_op(fn, n_warmup=10, n_iter=50):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    for _ in range(n_warmup):
        fn()
    torch.npu.synchronize()
    times_ms = np.zeros(n_iter, dtype=np.float64)
    for i in range(n_iter):
        start.record()
        fn()
        end.record()
        torch.npu.synchronize()
        times_ms[i] = start.elapsed_time(end)
    return float(np.median(times_ms)) * 1e3  # us


def run_case(*, scan, scenario, batch, T, kv_len, cum_query_lens, kv_lens, n_heads=N_HEADS, topk=TOPK):
    inputs = make_inputs(
        batch=batch,
        T=T,
        kv_len=kv_len,
        cum_query_lens=cum_query_lens,
        kv_lens=kv_lens,
        n_heads=n_heads,
        topk=topk,
    )
    # Effective topk: when kv_len < topk the indexer can't produce topk distinct
    # positions, so the kernel actually processes only kv_len of them.
    topk_eff = min(topk, kv_len)
    op = torch.ops._C_ascend.npu_sparse_flash_attention

    def fn():
        op(**inputs)

    lat_us = time_op(fn)
    tf = estimate_tflops(T, topk_eff, n_heads, D_NOPE, D_ROPE, lat_us / 1e6)
    return CaseResult(
        scan=scan,
        scenario=scenario,
        batch=batch,
        T=T,
        kv_len=kv_len,
        n_heads=n_heads,
        topk=topk_eff,
        block_size=BLOCK_SIZE,
        latency_us=lat_us,
        tflops=tf,
    )


# ---------- Scans ----------

def scan_decode_T(kv_len=4096):
    out = []
    for T in [1, 4, 16, 32, 64, 128, 256]:
        batch = T
        cum_q = list(range(1, batch + 1))
        kv_lens = [kv_len] * batch
        try:
            r = run_case(
                scan="decode_T", scenario="decode",
                batch=batch, T=T, kv_len=kv_len,
                cum_query_lens=cum_q, kv_lens=kv_lens,
            )
            print(f"  decode_T batch={batch:<4} kv={kv_len:<6} -> {r.latency_us:8.1f} us  {r.tflops:5.2f} TFLOPS")
            out.append(r)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] batch={batch} kv={kv_len}: {type(e).__name__}: {e}")
    return out


def scan_decode_kvlen(batch=32):
    out = []
    # Include small kv_len values (<topk=2048) to cover short-context decode.
    # When kv_len < topk, the kernel processes topk_eff = kv_len indices, so
    # these points characterize "dense-style" attention performance.
    for kv_len in [64, 128, 200, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
        T = batch
        cum_q = list(range(1, batch + 1))
        kv_lens = [kv_len] * batch
        try:
            r = run_case(
                scan="decode_kvlen", scenario="decode",
                batch=batch, T=T, kv_len=kv_len,
                cum_query_lens=cum_q, kv_lens=kv_lens,
            )
            print(f"  decode_kvlen batch={batch:<3} kv={kv_len:<6} -> {r.latency_us:8.1f} us  {r.tflops:5.2f} TFLOPS")
            out.append(r)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] batch={batch} kv={kv_len}: {type(e).__name__}: {e}")
    return out


def scan_prefill_T():
    out = []
    for T in [256, 512, 1024, 2048, 4096, 8192]:
        batch = 1
        kv_len = T
        try:
            r = run_case(
                scan="prefill_T", scenario="prefill",
                batch=batch, T=T, kv_len=kv_len,
                cum_query_lens=[T], kv_lens=[kv_len],
            )
            print(f"  prefill_T T=kv={T:<5}                   -> {r.latency_us:8.1f} us  {r.tflops:5.2f} TFLOPS")
            out.append(r)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] T={T}: {type(e).__name__}: {e}")
    return out


def scan_prefill_kvlen(T=1024):
    out = []
    for kv_len in [T, 2048, 4096, 8192, 16384, 32768, 65536]:
        if kv_len < T:
            continue
        batch = 1
        try:
            r = run_case(
                scan="prefill_kvlen", scenario="prefill",
                batch=batch, T=T, kv_len=kv_len,
                cum_query_lens=[T], kv_lens=[kv_len],
            )
            print(f"  prefill_kvlen T={T:<5} kv={kv_len:<6} -> {r.latency_us:8.1f} us  {r.tflops:5.2f} TFLOPS")
            out.append(r)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] T={T} kv={kv_len}: {type(e).__name__}: {e}")
    return out


# ---------- Output ----------

def write_csv(results, path):
    fields = list(asdict(results[0]).keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))


def plot_scan(results, scan_name, x_field, x_label, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rs = [r for r in results if r.scan == scan_name]
    if not rs:
        return
    xs = [getattr(r, x_field) for r in rs]
    lat = [r.latency_us for r in rs]
    tf = [r.tflops for r in rs]

    fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
    ax1.plot(xs, lat, "o-", color="tab:blue", label="Latency")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Latency (us)", color="tab:blue")
    if len(set(xs)) > 1 and min(xs) > 0 and max(xs) / max(min(xs), 1) >= 8:
        ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, which="both", linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(xs, tf, "s--", color="tab:red", label="TFLOPS")
    ax2.set_ylabel("TFLOPS", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    for x, y in zip(xs, lat):
        ax1.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 6),
                     fontsize=8, color="tab:blue", ha="center")

    plt.title(f"npu_sparse_flash_attention — {scan_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="./sfa_bench_out")
    p.add_argument(
        "--scans", nargs="+",
        default=["decode_T", "decode_kvlen", "prefill_T", "prefill_kvlen"],
        choices=["decode_T", "decode_kvlen", "prefill_T", "prefill_kvlen"],
    )
    p.add_argument("--decode-kv-len", type=int, default=4096,
                   help="kv_len used during decode_T scan")
    p.add_argument("--decode-batch", type=int, default=32,
                   help="batch (= T) used during decode_kvlen scan")
    p.add_argument("--prefill-T", type=int, default=1024,
                   help="T used during prefill_kvlen scan")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    plots_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"Device: {DEVICE}  dtype: {DTYPE}  N={N_HEADS} D_nope={D_NOPE} D_rope={D_ROPE} "
          f"block={BLOCK_SIZE} topk={TOPK}\n")

    all_results = []
    if "decode_T" in args.scans:
        print(f"[scan] decode_T  (kv_len={args.decode_kv_len})")
        all_results += scan_decode_T(kv_len=args.decode_kv_len)
    if "decode_kvlen" in args.scans:
        print(f"[scan] decode_kvlen  (batch={args.decode_batch})")
        all_results += scan_decode_kvlen(batch=args.decode_batch)
    if "prefill_T" in args.scans:
        print("[scan] prefill_T")
        all_results += scan_prefill_T()
    if "prefill_kvlen" in args.scans:
        print(f"[scan] prefill_kvlen  (T={args.prefill_T})")
        all_results += scan_prefill_kvlen(T=args.prefill_T)

    if not all_results:
        print("No results collected.")
        return

    csv_path = os.path.join(args.out_dir, "results.csv")
    write_csv(all_results, csv_path)
    print(f"\nWrote {csv_path}")

    plot_scan(all_results, "decode_T",      "batch",  "batch (= T)",  os.path.join(plots_dir, "decode_T.png"))
    plot_scan(all_results, "decode_kvlen",  "kv_len", "kv_len",       os.path.join(plots_dir, "decode_kvlen.png"))
    plot_scan(all_results, "prefill_T",     "T",      "T (= kv_len)", os.path.join(plots_dir, "prefill_T.png"))
    plot_scan(all_results, "prefill_kvlen", "kv_len", "kv_len",       os.path.join(plots_dir, "prefill_kvlen.png"))
    print(f"Wrote plots → {plots_dir}")


if __name__ == "__main__":
    main()
