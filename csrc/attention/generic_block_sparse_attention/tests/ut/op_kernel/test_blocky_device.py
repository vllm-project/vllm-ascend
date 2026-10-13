"""Direct kernel regression (no system OPP installation).

source /usr/local/Ascend/cann/set_env.sh
python3 test_blocky_device.py --device 2
Builds the production entry with CANN's inline-wrapper convention, loads the
binary through ACL, and checks O/LSE against a token-mask Torch oracle on CPU.
Host tiling and metadata API checks are covered by their separate unit tests.
"""
import argparse
import ctypes as C
from itertools import product
import os
from pathlib import Path
import re
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent
OP = HERE.parents[2]
KEYS = {
    0: {
        (False, False): 9250000130001002,
        (False, True): 9250000130023222,
        (True, False): 9250000171001002,
        (True, True): 9250000171023222,
    },
    1: {
        (False, False): 9250000130101002,
        (False, True): 9250000130123222,
        (True, False): 9250000171101002,
        (True, True): 9250000171123222,
    },
}


def build(cann, target, key):
    cmd = [str(cann / "tools/bisheng_compiler/bin/bisheng"), "-xcce", "--cce-aicore-only",
           "--cce-aicore-arch=dav-c310", "-std=c++17", "-O2", "-c", f"-DTILING_KEY_VAR={key}ULL"]
    for inc in ("asc/include", "asc/include/basic_api", "asc/include/adv_api",
                "compiler/tikcpp/tikcfw", "include"):
        cmd += ["-I" + str(cann / inc)]
    subprocess.run(cmd + [str(HERE / "blocky_device_entry.cpp"), "-o", str(target)], check=True)


def tiling_type():
    types = {"uint32_t": C.c_uint32, "uint64_t": C.c_uint64, "float": C.c_float}
    source = (OP / "op_kernel/kernel_common.hpp").read_text()

    def fields_of(struct_name):
        body = re.search(rf"struct {struct_name} \{{(.*?)\n\}};", source, re.S).group(1)
        return [(name, types[t]) for t, name in re.findall(
            r"^    (uint32_t|uint64_t|float|GbsaBaseTiling|GbsaMmPhaseL1Tiling) (\w+);", body, re.M)]

    base = type("GbsaBaseTiling", (C.Structure,), {"_fields_": fields_of("GbsaBaseTiling")})
    mm = type("GbsaMmPhaseL1Tiling", (C.Structure,), {"_fields_": fields_of("GbsaMmPhaseL1Tiling")})
    types.update(GbsaBaseTiling=base, GbsaMmPhaseL1Tiling=mm)
    fields = fields_of("GenericBlockSparseAttentionTilingData")
    return type("Tiling", (C.Structure,), {
        "_anonymous_": ("baseTileInfo", "mmPhaseL1TileInfo"),
        "_fields_": fields,
    })


def signature(values):
    h = 2166136261
    for v in values:
        for byte in int(v).to_bytes(4, "little"):
            h = ((h ^ byte) * 16777619) & 0xffffffff
    return C.c_int32(h).value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--ys", default="1,3,16,17,63,64,65,95,127,128")
    parser.add_argument("--cann", type=Path, default=Path("/usr/local/Ascend/cann"))
    parser.add_argument("--artifact-dir", type=Path,
                        help="reuse precompiled direct-kernel binaries from this directory")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--fd", action="store_true", help="split/combine regression with eight partials")
    parser.add_argument("--fd-case", choices=("full", "clipped", "empty"), default="full",
                        help="FD visible range: full, causal tail with empty partials, or all partials empty")
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--head-dims", help="comma-separated D sweep (overrides --head-dim)")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--softmax-precision", type=int, choices=(0, 1), default=0)
    parser.add_argument("--variant", choices=("tnd-fp16", "tnd-bf16", "pa-fp16", "pa-bf16"),
                        help="compile and run only this layout/dtype variant")
    parser.add_argument("--kv-base-tile", type=int, choices=(128, 256, 384, 512),
                        help="override the base KV tile for focused pipeline debugging")
    parser.add_argument("--stage-tail-sizes",
                        help="comma-separated 1..512 valid-KV sizes for exact high-precision stage coverage")
    args = parser.parse_args()
    if args.fd_case != "full" and not args.fd:
        parser.error("--fd-case requires --fd")
    if not 1 <= args.topk <= 256 or (args.fd and args.topk < 12):
        parser.error("topk must be 1..256 (at least 12 for the FD regression)")
    head_dims = list(map(int, args.head_dims.split(","))) if args.head_dims else [args.head_dim]
    if not head_dims or any(not 1 <= d <= 512 for d in head_dims):
        parser.error("head-dim must be 1..512")
    stage_tail_sizes = (list(map(int, args.stage_tail_sizes.split(",")))
                        if args.stage_tail_sizes else None)
    if stage_tail_sizes and any(not 1 <= n <= 512 for n in stage_tail_sizes):
        parser.error("stage-tail-sizes must contain values in 1..512")
    if stage_tail_sizes and (args.softmax_precision != 0 or args.fd):
        parser.error("stage-tail-sizes is only supported by non-FD softmax-precision=0 runs")
    if stage_tail_sizes and max(stage_tail_sizes) > 256 and args.topk < 2:
        parser.error("stage-tail-sizes above 256 require topk>=2")
    if not 1 <= args.group_size <= 128:
        parser.error("group-size must be 1..128")
    q_base_tile = next(tile for tile in (16, 32, 64, 128) if args.group_size <= tile)
    if args.kv_base_tile is not None and args.kv_base_tile > 256 and args.softmax_precision != 0:
        parser.error("kv-base-tile above 256 is only supported by softmax-precision=0")
    if args.kv_base_tile == 512 and q_base_tile > 32:
        parser.error("kv-base-tile=512 requires group-size<=32")
    if stage_tail_sizes and (any(d > 256 for d in head_dims) or q_base_tile > 32 or
                             args.kv_base_tile not in (None, 512)):
        parser.error("stage-tail-sizes requires an effective 512 KV base tile")
    work = args.artifact_dir or Path(tempfile.mkdtemp(prefix="gbsa-blocky-"))
    work.mkdir(parents=True, exist_ok=True)
    print("artifacts:", work, flush=True)
    binaries = {}
    selected_keys = KEYS[args.softmax_precision]
    if args.variant:
        variant = (args.variant.startswith("pa-"), args.variant.endswith("bf16"))
        selected_keys = {variant: selected_keys[variant]}
    for variant, key in selected_keys.items():
        binary = work / f"{key}.o"
        if not binary.exists():
            build(args.cann, binary, key)
        binaries[variant] = binary
        print("compiled", key, flush=True)
    if args.build_only:
        all_lse_keys = {key for precision_keys in KEYS.values() for key in precision_keys.values()}
        all_keys = all_lse_keys | {key - 100000000 for key in all_lse_keys}
        all_keys |= {9250000071101012, 9250000071101022}
        for key in sorted(all_keys):
            build(args.cann, work / f"{key}.o", key)
            print("compiled", key, flush=True)
        return

    import torch
    import torch_npu
    torch.set_num_threads(1)
    torch_npu.npu.set_device(args.device)
    device = f"npu:{args.device}"
    acl = C.CDLL(str(args.cann / "lib64/libascendcl.so"))
    acl.aclrtBinaryLoadFromFile.argtypes = [C.c_char_p, C.c_void_p, C.POINTER(C.c_void_p)]
    acl.aclrtBinaryGetFunction.argtypes = [C.c_void_p, C.c_char_p, C.POINTER(C.c_void_p)]
    acl.aclrtLaunchKernelWithHostArgs.argtypes = [C.c_void_p, C.c_uint32, C.c_void_p, C.c_void_p,
                                                 C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t]
    acl.aclrtBinaryUnLoad.argtypes = [C.c_void_p]

    def check(rc):
        if rc:
            raise RuntimeError(f"ACL error {rc}")

    Tiling = tiling_type()
    torch.manual_seed(19)
    passed = 0
    for (paged, bf16), binary in binaries.items():
        bh, fn = C.c_void_p(), C.c_void_p()
        check(acl.aclrtBinaryLoadFromFile(os.fsencode(binary), None, C.byref(bh)))
        check(acl.aclrtBinaryGetFunction(bh, b"gbsa_blocky_test", C.byref(fn)))
        dtype = torch.bfloat16 if bf16 else torch.float16
        geometries = ([(256, n) for n in stage_tail_sizes] if stage_tail_sizes else
                      [(y, None) for y in map(int, args.ys.split(","))])
        for (y, stage_tail_size), d in product(geometries, head_dims):
            # Two padded batches and two KV heads, including empty attention rows.
            group = args.group_size
            b, nq, nk, topk = 2, 2 * group, 2, args.topk
            qlens, qstorage = [2, 3], [3, 5]
            kvlens, kvstorage = [923, 1379], [1024, 1536]
            cuq = [0, 3, 8]
            cukv = [0, 1024, 2560]
            if stage_tail_size is not None:
                b, nq, nk = 1, group, 1
                qlens, qstorage = [1], [1]
                kvlens, kvstorage = [stage_tail_size], [512]
                cuq, cukv = [0, 1], [0, 512]
            elif args.fd:
                b, nq, nk = 1, group, 1
                qlens, qstorage = [1], [1]
                kvlens, kvstorage = [4093], [4096]
                if args.fd_case != "full":
                    kvlens = [1533 if args.fd_case == "clipped" else 1]
                cuq, cukv = [0, 1], [0, 4096]
            q = torch.randn(cuq[-1], nq, d, dtype=dtype) * 0.2
            densek = torch.randn(cukv[-1], nk, d, dtype=dtype) * 0.2
            densev = torch.randn_like(densek)
            idx = torch.full((nk, cuq[-1], topk), -1, dtype=torch.int32)
            counts = torch.zeros(nk, cuq[-1], dtype=torch.int32)
            ref = torch.zeros_like(q, dtype=torch.float32)
            reflse = torch.full((cuq[-1], nq, 1), torch.finfo(torch.float32).min)
            for batch in range(b):
                for t in range(qlens[batch]):
                    qt = cuq[batch] + t
                    for h in range(nk):
                        # Include the causal boundary and candidates beyond it. Indices remain sorted.
                        candidate_end = kvstorage[batch] if args.fd else kvlens[batch]
                        if stage_tail_size is not None:
                            ids = list(range((stage_tail_size + y - 1) // y))
                        else:
                            ids = [i for i in range((candidate_end + y - 1) // y) if (i + h) % 3 != 1]
                        if args.fd and args.fd_case == "empty":
                            ids = [i for i in ids if i > 0]
                        ids = ids[-topk:]
                        if t == 0 and h == 0 and not args.fd and stage_tail_size is None:
                            ids = []  # Empty attention output.
                        counts[h, qt] = len(ids)
                        idx[h, qt, :len(ids)] = torch.tensor(ids, dtype=torch.int32)
                        visible = kvlens[batch] - qlens[batch] + t + 1
                        tokens = [s for i in ids for s in range(i * y, min((i + 1) * y, visible))]
                        if tokens:
                            heads = slice(h * group, (h + 1) * group)
                            storage_tokens = torch.tensor(tokens) + cukv[batch]
                            scores = q[qt, heads].float() @ densek[storage_tokens, h].float().T / d**0.5
                            ref[qt, heads] = scores.softmax(-1) @ densev[storage_tokens, h].float()
                            reflse[qt, heads, 0] = scores.logsumexp(-1)
            pages = None
            if paged:
                # Independent, non-contiguous first-axis K/V strides; reversed physical pages.
                npages = sum(kvstorage) // 128
                maxpages = max(kvstorage) // 128
                kback = torch.full((npages * 2, 128, nk, d), 13, dtype=dtype)
                vback = torch.full((npages * 3, 128, nk, d), 17, dtype=dtype)
                k, v = kback[::2], vback[::3]
                pagecpu = torch.full((b, maxpages), -1, dtype=torch.int32)
                cursor = 0
                for batch in range(b):
                    for p in range(kvstorage[batch] // 128):
                        physical = npages - 1 - cursor
                        pagecpu[batch, p] = physical
                        start = cukv[batch] + p * 128
                        k[physical].copy_(densek[start:start + 128])
                        v[physical].copy_(densev[start:start + 128])
                        cursor += 1
                # Transfer backing tensors to retain the strided views.
                k, v = kback.to(device)[::2], vback.to(device)[::3]
                pages = pagecpu.to(device)
            else:
                k, v = densek.to(device), densev.to(device)
                maxpages = 0
            q, idx, counts = q.to(device), idx.to(device), counts.to(device)
            cuqt = torch.tensor(cuq, dtype=torch.int64, device=device)
            cukvt = None if paged else torch.tensor(cukv, dtype=torch.int64, device=device)
            uq = torch.tensor(qlens, dtype=torch.int32, device=device)
            ukv = torch.tensor(kvlens, dtype=torch.int32, device=device)
            out = torch.full_like(q, float("nan"))
            lse = torch.full((cuq[-1], nq, 1), float("nan"), device=device)
            metadata = torch.zeros(1024, dtype=torch.int32)
            metadata[:12] = torch.tensor([
                0x5341534D, 8, 272, 8, sum(qlens) * nk, 0, 0, 0, 2, 0, 0,
                signature([b, nq, nk, d, 1, y, topk, cuq[-1], 1])], dtype=torch.int32)
            if args.fd:
                candidate_count = int(counts[0, 0].cpu())
                metadata[5:11] = torch.tensor([8, (candidate_count + 7) // 8, 1, 3, candidate_count, 8])
                for core in range(8):
                    metadata[16 + core * 4:20 + core * 4] = torch.tensor([
                        0, 1, core * candidate_count // 8, (core + 1) * candidate_count // 8])
                metadata[144:148] = torch.tensor([0, 0, 0, 8])
            metadata = metadata.to(device)
            td = Tiling()
            if d <= 256:
                auto_kv_base_tile = 512 if q_base_tile <= 32 else 256 if q_base_tile <= 64 else 128
            else:
                auto_kv_base_tile = 384 if q_base_tile <= 16 else 256 if q_base_tile <= 32 else 128
            if args.softmax_precision == 1 and auto_kv_base_tile > 256:
                auto_kv_base_tile = 256
            kv_base_tile = auto_kv_base_tile
            if args.kv_base_tile is not None:
                kv_base_tile = args.kv_base_tile
            embed_aligned = 128 if d <= 128 else 256 if d <= 256 else 512
            mm1_l1_tile_n = 256 if d <= 128 and q_base_tile <= 64 else 128
            mm2_l1_tile_n = embed_aligned
            v_l1_buf_num = 1 if d > 128 and q_base_tile <= 32 else 2
            if d > 256:
                v_l1_buf_num = 2 if q_base_tile == 64 else 1
            lo_workspace_bytes = 8 * 2 * q_base_tile * ((d + 15) // 16 * 16) * 4 if d > 256 and q_base_tile > 64 else 0
            values = dict(batch=b, numHeads=nq, kvHeads=nk, embeddingSize=d, blockShapeX=1,
                blockShapeY=y, blockSize=128 if paged else 0, topK=topk, qBlockNum=cuq[-1],
                maxBlocksPerBatch=maxpages, scaleValue=d**-0.5, groupSize=group, qBaseTile=q_base_tile,
                kvBaseTile=kv_base_tile,
                mm1L1TileM=q_base_tile, mm1L1TileN=mm1_l1_tile_n, mm1L1TileKLeft=embed_aligned,
                mm1L1TileKRight=128 if d > 256 else embed_aligned,
                mm2L1TileM=q_base_tile, mm2L1TileN=mm2_l1_tile_n, mm2L1TileKLeft=kv_base_tile,
                mm2L1TileKRight=kv_base_tile,
                qL1BufNum=1, kL1BufNum=2, vL1BufNum=v_l1_buf_num, pL1BufNum=3,
                kStride0=k.stride(0) if paged else 0, vStride0=v.stride(0) if paged else 0,
                totalQTokens=cuq[-1], totalKvTokens=npages * 128 if paged else cukv[-1],
                mm2OutSize=lo_workspace_bytes)
            max_blocks = (mm1_l1_tile_n + 2 * y - 2) // y
            if paged:
                max_blocks = min(mm1_l1_tile_n, max_blocks * 2)
            group_words = (8 + max_blocks * (3 if paged else 4) + 7) // 8 * 8
            sparse_tile_words = (kv_base_tile + mm1_l1_tile_n - 1) // mm1_l1_tile_n * group_words
            sparse_offset = (max(lo_workspace_bytes, topk * 4) + 511) // 512 * 512
            sparse_end = sparse_offset + 8 * 4 * sparse_tile_words * 8
            values.update(kvOffsetInfoOffset=sparse_offset, kvOffsetTileWords=sparse_tile_words)
            if args.fd:
                values.update(fdStaticEnabled=1, fdLseSubStride=((group + 1) // 2 + 7) // 8 * 8,
                              fdPartialCapacity=8,
                              fdPartialLseOffset=sparse_end + 4096,
                              fdPartialOOffset=sparse_end + 8192)
            for name, value in values.items():
                setattr(td, name, value)
            td_tensor = torch.tensor(list(bytes(td)), dtype=torch.uint8, device=device)
            # The direct entry reserves 16 MiB for AscendC before GetUserWorkspace().
            # D>256, qBaseTile=128 adds full LO double buffers; FD partials follow
            # them. A fixed 20-MiB allocation leaves only 4 MiB for these regions.
            user_workspace_bytes = sparse_end
            if args.fd:
                user_workspace_bytes = max(
                    user_workspace_bytes,
                    td.fdPartialLseOffset + td.fdPartialCapacity * 2 * td.fdLseSubStride * 4,
                    td.fdPartialOOffset + td.fdPartialCapacity * group * ((d + 15) // 16 * 16) * 4,
                )
            workspace_bytes = 16 * 1024 * 1024 + ((user_workspace_bytes + 511) // 512 * 512)
            workspace = torch.zeros(workspace_bytes, dtype=torch.uint8, device=device)
            tensors = [q, k, v, idx, counts, metadata, cuqt, cukvt, uq, ukv, pages, out, lse, workspace, td_tensor]
            argv = (C.c_void_p * len(tensors))(*[t.data_ptr() if t is not None else 0 for t in tensors])
            stream = torch_npu.npu.current_stream().npu_stream
            print(f"launch Y={y} D={d} G={group} KV_tile={kv_base_tile} tail={stage_tail_size} "
                  f"PA={paged} dtype={dtype} FD={args.fd} fd_case={args.fd_case}",
                  flush=True)
            check(acl.aclrtLaunchKernelWithHostArgs(fn, 8, C.c_void_p(stream), None,
                                                   argv, C.sizeof(argv), None, 0))
            torch_npu.npu.synchronize()
            actual, actual_lse = out.cpu().float(), lse.cpu()
            diff = (actual - ref).abs()
            print("O max error by 64-D slice:",
                  [diff[..., start:start + 64].max().item() for start in range(0, d, 64)], flush=True)
            torch.testing.assert_close(actual, ref, atol=0.01 if bf16 else 0.003, rtol=0.04)
            torch.testing.assert_close(actual_lse, reflse, atol=0.04 if bf16 else 0.008, rtol=0.005)
            print(f"PASS max_O_error={(actual - ref).abs().max().item():.6f}", flush=True)
            passed += 1
        check(acl.aclrtBinaryUnLoad(bh))
    print(f"PASS: {passed} direct kernel O/LSE cases", flush=True)


if __name__ == "__main__":
    main()
