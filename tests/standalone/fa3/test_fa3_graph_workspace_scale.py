# Copyright (c) 2026. Reproduction-only diagnostic, not a correctness test.
#
# Experiment C6: map (batch, max_seqlen_k) -> FA3 workspace size / crash boundary.
#
# Background: the planned production fix (build scheduler_metadata for
# max_model_len so the baked tile count over-covers any decode length) crashes
# with "MTE accesses an invalid GM address" at production scale but NOT in the
# batch=3 standalone tests.  Leading hypothesis: FA3's internal workspace scales
# with  batch_size x max_seqlen_k  (see flash_attn_npu.patch):
#     lseTasksUpper = num_heads * seqlen_q * kvSegUpper * 2
#     wsSplit       = lseTasksUpper * (4 + head_size_og * 4)
#     kvSegUpper = ceil(max_seqlen_k/128),   seqlen_q = batch_size (decode tokens)
# so the over-cover fix forces a ~2 GB workspace at batch=256 / max_model_len.
#
# This experiment sweeps (batch, maxk) and reports, per cell:
#   - OK        : graph captures + replays without faulting
#   - CRASH(n)  : the subprocess died (MTE page fault / segfault)
#   - ws_measured: measured device-memory delta across the capture (best effort)
#   - ws_pred   : workspace size predicted by the formula above
#
# KEY ISOLATION: the K/V blocks are SHARED across all sequences (every page_table
# row points at the same ceil(maxk/128) blocks), so K/V memory stays ~67 MB even
# at maxk=32768.  The FA3 workspace, however, is sized by (batch, maxk) through
# the metadata plan alone — exactly the quantity we want to isolate.  The
# page_table width ALWAYS MATCHES the plan's tile count (consistent), so a crash
# here cannot be the C4 out-of-bounds-block artifact; it must be the workspace.
#
# Read:
#   crash boundary tracks  batch x maxk   -> workspace-size hypothesis confirmed;
#       gives the max (batch, maxk) FA3 decode can support (informs "bounded
#       maxk" option).
#   no crash even at batch=256 / maxk=32768 -> workspace is NOT the cause; the
#       production crash must be a page_table-width mismatch or something else.
#
# Usage:
#   python test_fa3_graph_workspace_scale.py                  # full sweep (subprocess/cell)
#   python test_fa3_graph_workspace_scale.py --batch 64 --maxk 2048   # single cell
#   python test_fa3_graph_workspace_scale.py --predict         # formula table only, no NPU

import argparse
import subprocess
import sys
from importlib import util as importlib_util

import torch

HEAD_SIZE = 128
NUM_HEADS = 32
NUM_KV_HEADS = 8
BLOCK_SIZE = 128
DTYPE = torch.bfloat16
SCALE = 1.0 / (HEAD_SIZE ** 0.5)

SWEEP_BATCHES = [16, 32, 64, 128, 256]
SWEEP_MAXKS = [2048, 4096, 8192, 16384, 32768]


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def predicted_ws_bytes(batch: int, maxk: int) -> int:
    """FA3 split workspace, from the formula in flash_attn_npu.patch:
        lseTasksUpper = num_heads * seqlen_q * kvSegUpper * 2
        wsSplit       = lseTasksUpper * (4 + head_size_og * 4)
    where seqlen_q = batch (total decode query tokens) and
    kvSegUpper = ceil(max_seqlen_k / BLOCK_SIZE)."""
    kv_seg_upper = _ceil_div(maxk, BLOCK_SIZE)
    lse_tasks = NUM_HEADS * batch * kv_seg_upper * 2
    return lse_tasks * (4 + HEAD_SIZE * 4)


def load_fa3():
    for mod_name in ("flash_attn_npu_3", "flash_attn_npu_3"):
        if importlib_util.find_spec(mod_name) is not None:
            try:
                mod = __import__(
                    mod_name,
                    fromlist=["flash_attn_with_kvcache", "get_scheduler_metadata"],
                )
                return (
                    mod.flash_attn_with_kvcache,
                    mod.get_scheduler_metadata,
                    mod_name,
                )
            except (ImportError, AttributeError) as exc:
                print(f"[import] {mod_name} found but failed: {exc}", flush=True)
    return None, None, None


def run_cell(batch: int, maxk: int) -> int:
    fa3_kvcache, get_scheduler_metadata, mod_name = load_fa3()
    if fa3_kvcache is None:
        print(f"[{batch}x{maxk}] SKIP (FA3 not installed)", flush=True)
        return 3

    blocks_per_seq = _ceil_div(maxk, BLOCK_SIZE)
    print(
        f"[{batch}x{maxk}] setup: blocks/seq={blocks_per_seq} "
        f"ws_pred={predicted_ws_bytes(batch, maxk) / 1e6:.1f} MB ({mod_name})",
        flush=True,
    )

    torch.npu.empty_cache()

    q = torch.randn(batch, NUM_HEADS, HEAD_SIZE, dtype=DTYPE).npu()
    # shared K/V: only blocks_per_seq blocks; every seq points at the same blocks
    k = torch.randn(
        blocks_per_seq, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE
    ).npu()
    v = torch.randn_like(k)

    cu_q = torch.arange(batch + 1, dtype=torch.int32).npu()
    shared_row = list(range(blocks_per_seq))
    page_table = torch.tensor([shared_row] * batch, dtype=torch.int32).npu()
    plan_buf = torch.full((batch,), maxk, dtype=torch.int32).npu()
    cache_seqlens_buf = torch.full((batch,), maxk, dtype=torch.int32).npu()

    print(f"[{batch}x{maxk}] building scheduler_metadata ...", flush=True)
    meta = get_scheduler_metadata(
        batch_size=batch,
        max_seqlen_q=1,
        max_seqlen_k=maxk,
        num_heads_q=NUM_HEADS,
        num_heads_kv=NUM_KV_HEADS,
        headdim=HEAD_SIZE,
        cache_seqlens=plan_buf,
        qkv_dtype=DTYPE,
        cu_seqlens_q=cu_q,
        page_size=BLOCK_SIZE,
        causal=True,
    )

    torch.npu.synchronize()
    base = torch.npu.memory_allocated()

    def run():
        return fa3_kvcache(
            q,
            k,
            v,
            cache_seqlens=cache_seqlens_buf,
            page_table=page_table,
            cu_seqlens_q=cu_q,
            max_seqlen_q=1,
            softmax_scale=SCALE,
            causal=True,
            window_size=(-1, -1),
            scheduler_metadata=meta,
        )

    print(f"[{batch}x{maxk}] capturing graph ...", flush=True)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = run()
    torch.npu.synchronize()

    after = torch.npu.memory_allocated()
    delta = after - base

    print(f"[{batch}x{maxk}] replaying ...", flush=True)
    cache_seqlens_buf.copy_(
        torch.full((batch,), min(BLOCK_SIZE, maxk), dtype=torch.int32).npu()
    )
    graph.replay()
    torch.npu.synchronize()
    _ = captured.clone()

    print(
        f"[{batch}x{maxk}] OK  ws_measured={delta / 1e6:.1f} MB  "
        f"ws_pred={predicted_ws_bytes(batch, maxk) / 1e6:.1f} MB",
        flush=True,
    )
    return 0


def predict():
    print("predicted FA3 workspace (MB) = 33024 * batch * ceil(maxk/128) / 1e6")
    print(f"{'batch':>6} {'maxk':>8} {'blocks/s':>9} {'ws_pred(MB)':>12}")
    for maxk in [1024, 2048, 4096, 8192, 16384, 32768]:
        for batch in SWEEP_BATCHES:
            print(
                f"{batch:>6} {maxk:>8} {_ceil_div(maxk, BLOCK_SIZE):>9} "
                f"{predicted_ws_bytes(batch, maxk) / 1e6:>12.1f}"
            )


def sweep():
    cells = [(b, m) for m in SWEEP_MAXKS for b in SWEEP_BATCHES]
    # order by predicted workspace ascending so the first crash marks the boundary
    # and every small cell completes before the first (possibly device-hanging) fault.
    cells.sort(key=lambda bm: predicted_ws_bytes(*bm))

    print("=" * 86)
    print(
        f"{'batch':>6} {'maxk':>8} {'blocks/s':>9} {'ws_pred(MB)':>12} "
        f"{'result':>10}  {'ws_meas(MB)'}"
    )
    print("=" * 86)
    for batch, maxk in cells:
        cmd = [sys.executable, __file__, "--batch", str(batch), "--maxk", str(maxk)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            print(
                f"{batch:>6} {maxk:>8} {_ceil_div(maxk, BLOCK_SIZE):>9} "
                f"{predicted_ws_bytes(batch, maxk) / 1e6:>12.1f} {'TIMEOUT':>10}"
            )
            continue

        result = "OK" if r.returncode == 0 else f"CRASH({r.returncode})"
        ws_meas = ""
        for line in r.stdout.splitlines():
            if "ws_measured=" in line:
                ws_meas = line.split("ws_measured=")[1].split()[0]
        print(
            f"{batch:>6} {maxk:>8} {_ceil_div(maxk, BLOCK_SIZE):>9} "
            f"{predicted_ws_bytes(batch, maxk) / 1e6:>12.1f} {result:>10}  {ws_meas}"
        )
        if r.returncode != 0:
            tail = "\n".join(r.stderr.splitlines()[-6:])
            if tail.strip():
                print(f"         stderr tail: {tail}")
    print("=" * 86)
    print("If a CRASH leaves the device in a bad state, reset it before re-running.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int)
    ap.add_argument("--maxk", type=int)
    ap.add_argument("--predict", action="store_true")
    args = ap.parse_args()

    if args.predict:
        predict()
        return
    if args.batch is not None and args.maxk is not None:
        sys.exit(run_cell(args.batch, args.maxk))
    if not torch.npu.is_available():
        raise SystemExit("No NPU device available.")
    sweep()


if __name__ == "__main__":
    main()
