"""Standalone test for FillLocalCopy kernel: correctness, dynamic refresh, graph capture."""

import ctypes

import torch
import torch_npu

SO = "/mnt/sdb/cta/vllm-ascend-main/fill_kernel/libfill_local_copy.so"
lib = ctypes.CDLL(SO)
lib.FillLocalCopy.argtypes = [ctypes.c_uint64] * 8 + [ctypes.c_void_p]

NMAX, K_BYTES, V_BYTES = 8, 1152, 128
N_SLOTS = 4096


def make_buffers():
    torch.npu.set_device(0)
    src_k = torch.randint(0, 255, (16, K_BYTES), dtype=torch.uint8, device="npu")
    src_v = torch.randint(0, 255, (16, V_BYTES), dtype=torch.uint8, device="npu")
    dst_k = torch.full((N_SLOTS, K_BYTES), 0xAA, dtype=torch.uint8, device="npu")
    dst_v = torch.full((N_SLOTS, V_BYTES), 0xAA, dtype=torch.uint8, device="npu")
    rows = torch.zeros(NMAX, dtype=torch.int32, device="npu")
    slots = torch.zeros(NMAX, dtype=torch.int32, device="npu")
    valid = torch.zeros(NMAX, dtype=torch.int32, device="npu")
    params = torch.tensor([NMAX, K_BYTES, V_BYTES], dtype=torch.int32, device="npu")
    return src_k, src_v, dst_k, dst_v, rows, slots, valid, params


def launch(src_k, src_v, dst_k, dst_v, rows, slots, valid, params):
    stream = torch_npu.npu.current_stream().npu_stream
    lib.FillLocalCopy(
        rows.data_ptr(),
        slots.data_ptr(),
        valid.data_ptr(),
        params.data_ptr(),
        src_k.data_ptr(),
        dst_k.data_ptr(),
        src_v.data_ptr(),
        dst_v.data_ptr(),
        ctypes.c_void_p(stream),
    )


def set_entries(rows, slots, valid, entries):
    rows_p = torch.zeros(NMAX, dtype=torch.int32)
    slots_p = torch.zeros(NMAX, dtype=torch.int32)
    valid_p = torch.zeros(NMAX, dtype=torch.int32)
    for i, (r, s) in enumerate(entries):
        rows_p[i], slots_p[i], valid_p[i] = r, s, 1
    rows.copy_(rows_p, non_blocking=True)
    slots.copy_(slots_p, non_blocking=True)
    valid.copy_(valid_p, non_blocking=True)


def check(tag, src_k, src_v, dst_k, dst_v, entries, written=()):
    ok = True
    for r, s in entries:
        if not torch.equal(dst_k[s], src_k[r]) or not torch.equal(dst_v[s], src_v[r]):
            ok = False
            print(f"[{tag}] MISMATCH row={r} slot={s}")
    mask = torch.ones(N_SLOTS, dtype=torch.bool)
    for _, s in entries:
        mask[s] = False
    for s in written:
        mask[s] = False
    if dst_k[mask].cpu().min().item() != 0xAA or dst_k[mask].cpu().max().item() != 0xAA:
        ok = False
        print(f"[{tag}] UNTOUCHED SLOT CORRUPTED (K)")
    if dst_v[mask].cpu().min().item() != 0xAA or dst_v[mask].cpu().max().item() != 0xAA:
        ok = False
        print(f"[{tag}] UNTOUCHED SLOT CORRUPTED (V)")
    print(f"[{tag}] {'PASS' if ok else 'FAIL'} ({len(entries)} entries)")
    return ok


src_k, src_v, dst_k, dst_v, rows, slots, valid, params = make_buffers()

# --- test 1: eager, 3 valid entries + 5 empty (must be strict no-op) ---
e1 = [(5, 37), (11, 1024), (2, 399)]
set_entries(rows, slots, valid, e1)
launch(src_k, src_v, dst_k, dst_v, rows, slots, valid, params)
torch.npu.synchronize()
ok1 = check("eager#1", src_k, src_v, dst_k, dst_v, e1)

# --- test 2: refresh indices in-place (dynamic descriptor test), overlapping slots ---
e2 = [(9, 1024), (3, 37)]
set_entries(rows, slots, valid, e2)
launch(src_k, src_v, dst_k, dst_v, rows, slots, valid, params)
torch.npu.synchronize()
# slot 1024 now holds row9 (overwritten), slot 37 holds row3, slot 399 still holds old row2
written1 = {37, 1024, 399}
ok2 = check("eager#2-refresh", src_k, src_v, dst_k, dst_v, e2, written1)
ok2 &= torch.equal(dst_k[399], src_k[2])

# --- test 3: graph capture + replay with refreshed descriptors ---
g = torch_npu.npu.NPUGraph()
e3 = [(7, 2048)]
set_entries(rows, slots, valid, e3)
torch.npu.synchronize()
with torch_npu.npu.graph(g):
    launch(src_k, src_v, dst_k, dst_v, rows, slots, valid, params)
g.replay()
torch.npu.synchronize()
ok3 = check("graph#capture-replay1", src_k, src_v, dst_k, dst_v, e3, written1)

# replay with different indices -> kernel must re-read descriptors from memory
e4 = [(13, 3000), (1, 500)]
set_entries(rows, slots, valid, e4)
torch.npu.synchronize()
g.replay()
torch.npu.synchronize()
ok4 = check("graph#replay2-dynamic", src_k, src_v, dst_k, dst_v, e4, written1 | {2048})

print("ALL PASS" if (ok1 and ok2 and ok3 and ok4) else "SOME FAILED")
