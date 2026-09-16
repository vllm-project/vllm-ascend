"""Separate persistent-mapping correctness from full-scale capacity gates."""

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from engram_vmm.lookup import Inputs
from engram_vmm.mapping import VmmMapping

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--full", action="store_true")
p.add_argument("--hbm-gib", type=int, default=0)
a = p.parse_args()
assert "expandable_segments:True" in os.environ.get("PYTORCH_NPU_ALLOC_CONF", "")
rank = int(os.environ["LOCAL_RANK"])
torch.npu.set_device(rank)
dist.init_process_group("gloo")
world = dist.get_world_size()
q = SimpleNamespace(rank=rank, cpu_group=dist.group.WORLD)
reservation = [torch.empty(256 << 20, dtype=torch.uint8, device=f"npu:{rank}") for _ in range(a.hbm_gib * 4)]
print("HBM_RESERVED", rank, a.hbm_gib, flush=True)
rows_list = [384006168, 384016682] if a.full else [4097, 8193]
tables = []
for slot, rows in enumerate(rows_list):
    cm = VmmMapping(f"{a.root}/{slot}.codes", rows, 256, torch.int8, q)
    sm = VmmMapping(f"{a.root}/{slot}.scales", rows, 8, torch.float32, q)
    tables.append(SimpleNamespace(codes_map=cm, scales_map=sm))
    # Sparse deterministic rows cover every writer stripe, table ends and
    # 4M-row pointer-table boundaries, without an enormous reference tensor.
    points = sorted(
        set(
            [0, rows - 1]
            + [(rows - 1) * r // world for r in range(world)]
            + ([4194303, 4194304, 4194305] if a.full else [])
        )
    )
    for j, index in enumerate(points):
        if j % world == rank:
            codes = ((torch.arange(256, device="cpu") + index + slot) % 255 - 127).to(torch.int8).view(1, 256)
            scales = torch.full((1, 8), 2.0 ** (-slot - 7), device="cpu")
            # Direct CPU->imported-host memcpy is rejected by this CANN build.
            # Stage through local NPU only during startup/checkpoint loading.
            cm.publish(index, codes)
            sm.publish(index, scales)
    torch.npu.synchronize()
    dist.barrier()
    tables[-1].points = points
inputs = Inputs(tables, [1, 14], 128, 24)
for n in (0, 1, 17, 128, 3, 0):
    hashes = torch.empty(n, 2, 24, dtype=torch.int64, device="cpu")
    for slot, t in enumerate(tables):
        if n:
            hashes[:, slot] = torch.tensor([t.points[i % len(t.points)] for i in range(n * 24)], device="cpu").view(
                n, 24
            )
    mask = torch.arange(n, device="cpu") % 2 == 0
    out = inputs.prepare(hashes, mask)
    torch.npu.synchronize()
    for slot, layer in enumerate([1, 14]):
        index = hashes[:, slot].unsqueeze(-1)
        ref = (
            (((torch.arange(256, device="cpu") + index + slot) % 255 - 127).float() * 2.0 ** (-slot - 7))
            .bfloat16()
            .reshape(n, 6144)
        )
        actual = out["engram_lookups"][layer].cpu()
        assert torch.equal(actual[:n], ref), (rank, n, layer)
        assert torch.count_nonzero(actual[n:]) == 0
    assert torch.equal(out["engram_mask"][:n].cpu(), mask)
    print("VMM_LOOKUP_PASS", rank, n, flush=True)
dist.barrier()
assert tables[0].codes_map.lifecycle_counts() == (4, 0)
inputs.close()
del inputs
for t in reversed(tables):
    t.scales_map.close()
    t.codes_map.close()
dist.barrier()
assert tables[0].codes_map.lib.host_vmm_live_mappings() == 0
assert not list(Path(a.root).iterdir()), "Descriptor files leaked"
dist.destroy_process_group()
print("VMM_PROBE_PASS", rank, "full", a.full, "hbm_gib", a.hbm_gib, flush=True)
