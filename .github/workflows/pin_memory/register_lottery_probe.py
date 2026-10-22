# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
Registered-host-memory lottery probe (NPU, single card).

为什么有这个探针：干净机上微基准的 registered buffer 呈逐 buffer/方向抽奖
（有的 26 GB/s、有的 9.8–12.4 GB/s），训练环 8B 的 avg 被慢尾拖垮
（avg/min = 3.5）。本探针在同一进程内把三种宿主内存放一起对照，并把每个
buffer 的 NUMA 节点与大页落点打印出来，一次跑完即可判定根因方向。

三组对照（每组 N 个 64MiB buffer，逐个 H2D/D2H 计时）：
  A  register-on-malloc：posix_memalign + mlock + npuHostRegister(MAPPED)
     —— 逐字节复刻 DeepSpeed native registered 路径（pin_memory.py 的 C++
     分配器 + npu_accelerator.register_host_memory 的 ACL_HOST_REG_MAPPED）。
  B  driver 分配：aclrtMallocHost(HUGE_FIRST) —— torch pin 分配器的底层路径，
     微基准里 torch 档零抽奖的候选解释。
  C  torch pinned：tensor.pin_memory() —— 零抽奖参照系。
  E  THP-then-lock：posix_memalign + fault-in + MADV_HUGEPAGE，等 khugepaged
     合并成 2MB 页后再 mlock+注册 —— 判 A 比 B/C 的 20% 恒定差是否来自页粒度。

判别矩阵（命令见 run_commands.txt）：
  default            基线（无 mbind，与训练环同路径）
  --mode interleave  注册前 mbind(MPOL_INTERLEAVE) —— 判 NUMA 落点假说
  --mode bind0       注册前 mbind(MPOL_BIND, node0)
  --huge             注册前 MADV_HUGEPAGE —— 判 IOMMU 页表粒度假说

判读：
  - B/C 全快、A 抽奖 → 根因在 register-on-malloc 页，修复方向 = 换分配路径
  - interleave/bind0 后 A 抽奖消失 → NUMA 放置问题，修复 = mbind
  - --huge 改变抽奖率 → 页表粒度问题（THP 2MB 页把 IOMMU 遍历省 512 倍）
  - 全部无关、B 也抽奖 → 内核 4.19 + IORT 固件层，软件改不动

用法（先 source set_env.sh 使 libascendcl.so 可被加载）：
  source ~/Ascend/cann-8.5.1/set_env.sh
  ASCEND_RT_VISIBLE_DEVICES=0 python register_lottery_probe.py
"""

import argparse
import ctypes
import json
import os
import time

import regex as re
import torch

# CANN 常量，与 deepspeed/accelerator/npu_accelerator.py 保持一致。
ACL_SUCCESS = 0
ACL_HOST_REG_MAPPED = 0x2
ACL_MEM_MALLOC_HUGE_FIRST = 0

# Linux 内存策略常量（man mbind / man madvise）。
MPOL_INTERLEAVE = 3
MPOL_BIND = 2
MADV_HUGEPAGE = 14


def _libc():
    """glibc 包装：posix_memalign / mbind / madvise / mlock / free。"""
    libc = ctypes.CDLL(None)
    libc.posix_memalign.restype = ctypes.c_int
    libc.posix_memalign.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_size_t]
    # 本机 glibc 未导出 mbind 符号（libc.so.6 亦无），改走原始 syscall，见 _mbind()。
    libc.syscall.restype = ctypes.c_long
    libc.madvise.restype = ctypes.c_int
    libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.mlock.restype = ctypes.c_int
    libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.munlock.restype = ctypes.c_int
    libc.munlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.free.restype = None
    libc.free.argtypes = [ctypes.c_void_p]
    libc.memset.restype = ctypes.c_void_p
    libc.memset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    return libc


# aarch64 通用 syscall 表（/usr/include/asm-generic/unistd.h）：__NR_mbind = 235。
_NR_MBIND = 235


def _mbind(libc, ptr, nbytes, policy, nodemask, maxnode):
    """mbind syscall：变参调用必须逐参显式转型，否则 64 位值被 c_int 截断。"""
    mask = ctypes.c_ulong(nodemask)
    return libc.syscall(
        ctypes.c_long(_NR_MBIND),
        ctypes.c_void_p(ptr.value),
        ctypes.c_ulong(nbytes),
        ctypes.c_int(policy),
        ctypes.byref(mask),
        ctypes.c_ulong(maxnode),
        ctypes.c_uint(0),
    )


def _load_ascendcl():
    """加载 libascendcl.so（需已 source CANN set_env.sh 把 lib64 加进 LD_LIBRARY_PATH）。"""
    try:
        lib = ctypes.CDLL("libascendcl.so")
        lib.aclrtMallocHost.restype = ctypes.c_int32
        lib.aclrtMallocHost.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int32]
        lib.aclrtFreeHost.restype = ctypes.c_int32
        lib.aclrtFreeHost.argtypes = [ctypes.c_void_p]
        return lib
    except OSError:
        return None


def _npurt():
    """torch_npu 的运行时 API 模块（含 npuHostRegister/npuHostUnregister）。"""
    return torch.npu.npurt()


def discover_nodes():
    nodes = []
    base = "/sys/devices/system/node"
    for name in os.listdir(base):
        m = re.match(r"node(\d+)$", name)
        if m:
            nodes.append(int(m.group(1)))
    return sorted(nodes)


def alloc_group_a(nbytes, mode, use_huge, bind_node, libc, npurt):
    """posix_memalign + (mbind) + (MADV_HUGEPAGE) + mlock + npuHostRegister(MAPPED)。"""
    ptr = ctypes.c_void_p()
    rc = libc.posix_memalign(ctypes.byref(ptr), os.sysconf("SC_PAGESIZE"), nbytes)
    if rc != 0:
        return None, f"posix_memalign rc={rc}"
    # mbind 必须在页被 fault-in 之前设置；mlock 会触发 first touch，所以这里先于 mlock。
    if mode in ("interleave", "bind0"):
        nodes = discover_nodes()
        if mode == "interleave":
            policy, nodemask = MPOL_INTERLEAVE, (1 << len(nodes)) - 1
        else:
            policy, nodemask = MPOL_BIND, 1 << bind_node
        if _mbind(libc, ptr, nbytes, policy, nodemask, max(nodes) + 1) != 0:
            libc.free(ptr)
            return None, "mbind failed"
    if use_huge:
        libc.madvise(ptr, nbytes, MADV_HUGEPAGE)
    # mlock 自身把页 fault-in（与 DeepSpeed C++ 分配器同序，无额外 memset）。
    if libc.mlock(ptr, nbytes) != 0:
        libc.free(ptr)
        return None, "mlock failed"
    rc = npurt.npuHostRegister(ptr.value, nbytes, ACL_HOST_REG_MAPPED)
    if rc != ACL_SUCCESS:
        libc.munlock(ptr, nbytes)
        libc.free(ptr)
        return None, f"npuHostRegister rc={rc}"
    return ptr.value, None


def alloc_group_b(nbytes, ascendcl):
    if ascendcl is None:
        return None, "libascendcl.so not found (source set_env.sh?)"
    ptr = ctypes.c_void_p()
    rc = ascendcl.aclrtMallocHost(ctypes.byref(ptr), nbytes, ACL_MEM_MALLOC_HUGE_FIRST)
    if rc != ACL_SUCCESS:
        return None, f"aclrtMallocHost rc={rc}"
    return ptr.value, None


def alloc_group_e(nbytes, libc):
    """posix_memalign + MADV_HUGEPAGE + fault-in：madvise 必须先于 fault，fault-time
    THP 才会直接给 2MB 页（本机 khugepaged 全量扫描太慢，等合并不现实）。"""
    ptr = ctypes.c_void_p()
    rc = libc.posix_memalign(ctypes.byref(ptr), os.sysconf("SC_PAGESIZE"), nbytes)
    if rc != 0:
        return None, f"posix_memalign rc={rc}"
    libc.madvise(ptr, nbytes, MADV_HUGEPAGE)
    libc.memset(ptr, 0, nbytes)  # fault-in：THP=always + madvise → 直接分配 2MB 页
    return ptr.value, None


def alloc_group_c(nbytes):
    # torch_npu 的 pin 分配器（微基准 torch 档同路径，零抽奖参照）。
    return torch.empty(nbytes, dtype=torch.uint8).pin_memory(), None


def host_view(buf):
    """把 A/B 的裸指针包成 CPU uint8 tensor（frombuffer 共享内存、零拷贝）。"""
    if buf.get("tensor") is not None:
        return buf["tensor"]
    array = (ctypes.c_char * buf["nbytes"]).from_address(buf["ptr"])
    return torch.frombuffer(array, dtype=torch.uint8)


def measure_buffer(dev, host, nbytes, warmup, iters):
    """与上游 bench 相同的拷贝操作（copy_ + non_blocking），逐步独立 Event 计时。"""
    stream = torch.npu.current_stream()
    ev_start = torch.npu.Event(enable_timing=True)
    ev_end = torch.npu.Event(enable_timing=True)
    times = {"h2d_s": [], "d2h_s": []}
    for key, copy_fn in (
        ("h2d_s", lambda: dev.copy_(host, non_blocking=True)),
        ("d2h_s", lambda: host.copy_(dev, non_blocking=True)),
    ):
        for _ in range(warmup):
            copy_fn()
            stream.synchronize()
        for _ in range(iters):
            ev_start.record(stream)
            copy_fn()
            ev_end.record(stream)
            stream.synchronize()
            times[key].append(ev_start.elapsed_time(ev_end) / 1000.0)
    return times


def read_numa_entries():
    # 本机 4.19 内核 numa_maps 行只有起始地址（无 "-end"），区间延伸到下一行起始地址。
    entries = []
    with open("/proc/self/numa_maps") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        m = re.match(r"([0-9a-f]+)\s", line)
        if not m:
            continue
        # 找下一行起始地址作为本行区间结束；末行给一个极大值。
        end = int(m.group(1), 16) + (1 << 48)
        for nxt in lines[i + 1 :]:
            n = re.match(r"([0-9a-f]+)\s", nxt)
            if n:
                end = int(n.group(1), 16)
                break
        counts = {int(g.group(1)): int(g.group(2)) for g in re.finditer(r"N(\d+)=(\d+)", line)}
        entries.append((int(m.group(1), 16), end, counts))
    return entries


def read_huge_entries():
    entries = []
    current = None
    with open("/proc/self/smaps") as f:
        for line in f:
            m = re.match(r"([0-9a-f]+)-([0-9a-f]+)\s", line)
            if m:
                current = [int(m.group(1), 16), int(m.group(2), 16), 0]
                entries.append(current)
                continue
            m = re.match(r"AnonHugePages:\s+(\d+) kB", line)
            if m and current is not None:
                current[2] = int(m.group(1)) * 1024
    return entries


def placement_for(ptr, nbytes, numa_entries, huge_entries):
    """汇总 [ptr, ptr+nbytes) 的 NUMA 节点页数占比与 AnonHugePages 覆盖字节。"""
    begin, end = ptr, ptr + nbytes
    node_pages = {}
    huge_bytes = 0
    for s, e, counts in numa_entries:
        if min(e, end) > max(s, begin):
            for node, pages in counts.items():
                node_pages[node] = node_pages.get(node, 0) + pages
    for s, e, huge in huge_entries:
        if min(e, end) > max(s, begin):
            huge_bytes += huge
    total = sum(node_pages.values())
    node_pct = {}
    if total:
        node_pct = {node: round(100.0 * pages / total, 1) for node, pages in node_pages.items()}
    return node_pct, huge_bytes


def machine_info():
    info = {
        "host": os.uname().nodename,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "nodes": discover_nodes(),
    }
    for key in ("enabled", "defrag"):
        path = f"/sys/kernel/mm/transparent_hugepage/{key}"
        if os.path.exists(path):
            with open(path) as f:
                info[f"thp_{key}"] = f.read().strip()
        else:
            info[f"thp_{key}"] = "?"
    with open("/proc/cmdline") as f:
        info["cmdline_has_iommu"] = "iommu" in f.read()
    return info


def gbps(nbytes, seconds_list):
    """字节数 / 耗时 → GB/s；取平均与最小（最小≈快路径上界）。"""
    avg = nbytes / (sum(seconds_list) / len(seconds_list)) / 1e9
    fastest = nbytes / min(seconds_list) / 1e9
    return round(avg, 2), round(fastest, 2)


def fmt_nodes(node_pct):
    if not node_pct:
        return "-"
    parts = [f"N{n}:{p}%" for n, p in sorted(node_pct.items())]
    return ",".join(parts[:3])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--buf-mib", type=int, default=64, help="per-buffer size in MiB")
    parser.add_argument("--n-buf", type=int, default=12, help="buffers per group")
    parser.add_argument("--iters", type=int, default=20, help="timed copies per buffer per direction")
    parser.add_argument("--warmup", type=int, default=3, help="untimed warmup copies")
    parser.add_argument(
        "--mode",
        choices=("default", "interleave", "bind0", "sweep"),
        default="default",
        help="group A mbind policy, applied before fault-in; sweep = per-node scan",
    )
    parser.add_argument("--bind-node", type=int, default=0, help="MPOL_BIND target node for --mode bind0")
    parser.add_argument("--huge", action="store_true", help="MADV_HUGEPAGE on group A before mlock")
    parser.add_argument("--groups", default="abc", help="subset of groups, e.g. 'ac' or 'aeb'")
    parser.add_argument(
        "--thp-wait", type=int, default=45, help="seconds to wait for khugepaged before locking group E buffers"
    )
    parser.add_argument("--fast-gbps", type=float, default=18.0, help="fast/slow divider")
    parser.add_argument("--out", default=None, help="JSON output path (default: timestamped in cwd)")
    return parser.parse_args()


def main():
    args = parse_args()
    npurt = _npurt()  # 初始化 NPU 运行时（register / aclrtMallocHost 之前）
    libc = _libc()
    ascendcl = _load_ascendcl() if "b" in args.groups else None
    nbytes = args.buf_mib * 1024 * 1024

    dev = torch.empty(nbytes, dtype=torch.uint8, device="npu")

    allocators = {
        "a": lambda: alloc_group_a(nbytes, args.mode, args.huge, args.bind_node, libc, npurt),
        "b": lambda: alloc_group_b(nbytes, ascendcl),
        "c": lambda: alloc_group_c(nbytes),
        "e": lambda: alloc_group_e(nbytes, libc),
    }

    buffers = []
    if args.mode == "sweep":
        # 单进程节点扫描：所有节点轮转分配，避免时间漂移伪装成节点差异。
        sweep_nodes = discover_nodes()
        for _ in range(args.n_buf):
            for node in sweep_nodes:
                ptr_or_tensor, error = alloc_group_a(nbytes, "bind0", args.huge, node, libc, npurt)
                buffer = {"group": "a", "nbytes": nbytes, "error": error}
                if isinstance(ptr_or_tensor, int):
                    buffer["ptr"] = ptr_or_tensor
                else:
                    buffer["tensor"] = ptr_or_tensor
                buffers.append(buffer)
    else:
        # 分配顺序也交错：A/B/C 轮转，避免某一组吃满同一段地址/状态。
        for _ in range(args.n_buf):
            for group in args.groups:
                ptr_or_tensor, error = allocators[group]()
                buffer = {"group": group, "nbytes": nbytes, "error": error}
                if isinstance(ptr_or_tensor, int):
                    buffer["ptr"] = ptr_or_tensor
                else:
                    buffer["tensor"] = ptr_or_tensor
                if group == "e":
                    buffer["pending_lock"] = True
                buffers.append(buffer)

    # E 组：mlock 会阻止 khugepaged 合并，所以先等合并、再上锁+注册。
    if any(buffer.get("pending_lock") for buffer in buffers):
        time.sleep(args.thp_wait)
        for buffer in buffers:
            if not buffer.get("pending_lock"):
                continue
            ptr = buffer["ptr"]
            if libc.mlock(ptr, nbytes) != 0:
                buffer["error"] = "mlock failed"
                continue
            rc = npurt.npuHostRegister(ptr, nbytes, ACL_HOST_REG_MAPPED)
            if rc != ACL_SUCCESS:
                libc.munlock(ptr, nbytes)
                buffer["error"] = f"npuHostRegister rc={rc}"
                continue
            buffer["pending_lock"] = False

    # numa_maps / smaps 反映当前映射，必须在分配之后、释放之前读取。
    numa_entries = read_numa_entries()
    huge_entries = read_huge_entries()

    try:
        rows = []
        for buffer in buffers:
            row = {
                "group": buffer["group"],
                "nbytes": nbytes,
                "ptr": buffer.get("ptr"),
                "error": buffer.get("error"),
            }
            if buffer.get("error") is None:
                host = host_view(buffer)
                times = measure_buffer(dev, host, nbytes, args.warmup, args.iters)
                h2d_avg, h2d_min = gbps(nbytes, times["h2d_s"])
                d2h_avg, d2h_min = gbps(nbytes, times["d2h_s"])
                node_pct, huge_bytes = placement_for(
                    buffer.get("ptr") or host.data_ptr(), nbytes, numa_entries, huge_entries
                )
                row.update(
                    {
                        "h2d_avg_gbps": h2d_avg,
                        "h2d_min_gbps": h2d_min,
                        "d2h_avg_gbps": d2h_avg,
                        "d2h_min_gbps": d2h_min,
                        "node_pct": node_pct,
                        "huge_mib": round(huge_bytes / 1024 / 1024, 1),
                    }
                )
            rows.append(row)

        verdict = {}
        print(
            f"\n== register lottery probe ==  mode={args.mode} huge={args.huge} "
            f"buf={args.buf_mib}MiB x {args.n_buf}  fast>={args.fast_gbps} GB/s"
        )
        for group in args.groups:
            group_rows = [r for r in rows if r["group"] == group and r.get("error") is None]
            errors = [r for r in rows if r["group"] == group and r.get("error")]
            print(f"\n[{group}]  #buf H2Davg H2Dmin D2Havg D2Hmin  nodes(huge-MiB)")
            for r in group_rows:
                marker = "S" if r["h2d_avg_gbps"] < args.fast_gbps else " "
                print(
                    f"  {marker} {r['h2d_avg_gbps']:7.2f} {r['h2d_min_gbps']:6.2f} "
                    f"{r['d2h_avg_gbps']:7.2f} {r['d2h_min_gbps']:6.2f}  "
                    f"{fmt_nodes(r['node_pct'])} ({r['huge_mib']})"
                )
            for r in errors:
                print(f"  alloc failed: {r['error']}")
            h2d_slow = sum(1 for r in group_rows if r["h2d_avg_gbps"] < args.fast_gbps)
            d2h_slow = sum(1 for r in group_rows if r["d2h_avg_gbps"] < args.fast_gbps)
            verdict[group] = {"h2d_slow": h2d_slow, "d2h_slow": d2h_slow, "n": len(group_rows)}
            print(f"  -> H2D slow {h2d_slow}/{len(group_rows)}, D2H slow {d2h_slow}/{len(group_rows)}")

        out_path = args.out or f"register_lottery_probe_{time.strftime('%Y-%m-%d_%H%M%S')}.json"
        with open(out_path, "w") as f:
            json.dump({"info": machine_info(), "args": vars(args), "rows": rows, "verdict": verdict}, f, indent=2)
        print(f"\nJSON written: {out_path}")
    finally:
        for buffer in buffers:
            if buffer.get("ptr") is not None and buffer["group"] in ("a", "e"):
                npurt.npuHostUnregister(buffer["ptr"])
                libc.munlock(buffer["ptr"], buffer["nbytes"])
                libc.free(buffer["ptr"])
            elif buffer.get("ptr") is not None and buffer["group"] == "b" and ascendcl is not None:
                ascendcl.aclrtFreeHost(buffer["ptr"])


if __name__ == "__main__":
    main()
