#!/usr/bin/env python3
"""Inspect rank*.pt produced by save_profiling_data: shapes, counters, sample trace IDs."""

import argparse
import os
import sys

import torch


def main():
    p = argparse.ArgumentParser(description="Inspect profiling rank*.pt contents")
    p.add_argument("rank_pt", help="path to rank000.pt etc.")
    p.add_argument("--cores", type=int, default=8, help="max cores per type to print detail")
    args = p.parse_args()

    path = args.rank_pt
    if not os.path.isfile(path):
        print(f"not found: {path}", file=sys.stderr)
        sys.exit(1)

    obj = torch.load(path, map_location="cpu")
    print(f"file: {path}")
    print(f"loaded type: {type(obj).__name__}")

    if isinstance(obj, (list, tuple)):
        print(f"num groups (core types): {len(obj)}")
        for gi, t in enumerate(obj):
            if not isinstance(t, torch.Tensor):
                print(f"  [{gi}] {type(t)} (skip)")
                continue
            nz = int((t != 0).sum().item())
            print(
                f"  [{gi}] shape={tuple(t.shape)} dtype={t.dtype} "
                f"nonzero={nz}/{t.numel()} ({100.0 * nz / max(1, t.numel()):.4f}%)"
            )
            # Counters: trace_collector uses record_count = tensor[cid,0] - 1
            n_with_records = 0
            max_rc = 0
            for cid in range(t.shape[0]):
                rc = int(t[cid, 0].item())
                if rc > 1:
                    n_with_records += 1
                max_rc = max(max_rc, rc)
            print(f"      cores with counter>1 (likely has trace records): {n_with_records}/{t.shape[0]}, max(counter)={max_rc}")
            nprint = min(args.cores, t.shape[0])
            for cid in range(nprint):
                rc = int(t[cid, 0].item())
                last = int(t[cid, -1].item()) if t.shape[1] > 1 else 0
                # trace_collector: record_count = raw_count - 1
                nrec = max(0, rc - 1)
                s1 = int(t[cid, 1].item()) if t.shape[1] > 1 and rc > 1 else 0
                s2 = int(t[cid, 2].item()) if t.shape[1] > 2 and rc > 2 else 0
                print(
                    f"      core[{cid}] [0]=counter={rc} => ~{nrec} records, "
                    f"[-1]=ts={last}, [1]={s1}, [2]={s2}"
                )
        return

    if isinstance(obj, torch.Tensor):
        t = obj
        nz = int((t != 0).sum().item())
        print(f"single tensor shape={tuple(t.shape)} nonzero={nz}/{t.numel()}")
        return

    print(f"unexpected payload: {obj!r}"[:500])


if __name__ == "__main__":
    main()
