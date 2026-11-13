"""NUMA free memory scanner.

Scans /proc/kpageflags per NUMA node to locate free (buddy) memory segments,
optionally intersected with target address ranges. Scans run in parallel
(per-NUMA) threads and batch kpageflags reads to keep syscall count low.
"""

import argparse
import array
import os
import platform
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import regex as re

PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
KPF_BUDDY = 10
ASCEND_910B = "ASCEND_910B"
ASCEND_910C = "ASCEND_910C"
ASCEND_950 = "ASCEND_950"
ASCEND_UNKNOWN = "ASCEND_UNKNOWN"

# ===== Address range configuration =====
RANGES = [
    0x29580000000,
    0xA9580000000,
    0x129580000000,
    0x1A9580000000,
]

RANGE_SIZE = 682 * 1024 * 1024 * 1024  # 682GB

_print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    """Thread-safe print with immediate flush."""
    with _print_lock:
        print(*args, **kwargs)
        sys.stdout.flush()


def fmt(addr):
    return f"0x{addr:016x}"


def addr_to_pfn(addr):
    return addr // PAGE_SIZE


def pfn_to_addr(pfn):
    return pfn * PAGE_SIZE


def is_buddy(flags):
    return (flags >> KPF_BUDDY) & 1


def intersect(a_start, a_end, b_start, b_end):
    s = max(a_start, b_start)
    e = min(a_end, b_end)
    if s <= e:
        return s, e
    return None


def round_up(a, b):
    return (a + b - 1) // b * b


def round_down(a, b):
    return a // b * b


def get_all_nodes():
    base = "/sys/devices/system/node/"
    return sorted([int(d.replace("node", "")) for d in os.listdir(base) if d.startswith("node")])


@lru_cache(maxsize=1)
def get_memory_block_size():
    path = "/sys/devices/system/memory/block_size_bytes"
    try:
        with open(path) as f:
            block_size = int(f.read().strip(), 16)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read memory block size from {path}: {exc}") from exc
    if block_size <= 0:
        raise RuntimeError(f"Invalid memory block size in {path}: {block_size}")
    return block_size


@lru_cache(maxsize=None)
def get_node_blocks(node):
    path = f"/sys/devices/system/node/node{node}/"
    block_size = get_memory_block_size()
    blocks = []
    for name in os.listdir(path):
        if name.startswith("memory"):
            idx_str = name.replace("memory", "")
            if idx_str.isdigit():
                idx = int(idx_str)
                start = idx * block_size // PAGE_SIZE
                end = (idx * block_size + block_size) // PAGE_SIZE - 1
                blocks.append((start, end))
    return tuple(sorted(blocks))


@lru_cache(maxsize=1)
def get_force_max_zoneorder():
    try:
        kernel_ver = platform.release()
    except OSError:
        kernel_ver = ""
    if kernel_ver:
        path = f"/boot/config-{kernel_ver}"
        try:
            with open(path, 'r', errors='ignore') as f:
                content = f.read()
            match = re.search(r'CONFIG_FORCE_MAX_ZONEORDER=(\d+)', content)
            if match:
                return int(match.group(1))
        except OSError:
            pass
    try:
        with open("/etc/os-release", encoding="utf-8", errors="ignore") as f:
            if re.search(r"(?im)^ID=(?:\"ubuntu\"|ubuntu)$", f.read()):
                return 13
    except OSError:
        pass
    return 11


def get_zone_page_size():
    return PAGE_SIZE * (1 << (get_force_max_zoneorder() - 1))


def build_target_ranges():
    target_ranges = []
    for base in RANGES:
        start = addr_to_pfn(base)
        end = addr_to_pfn(base + RANGE_SIZE - 1)
        target_ranges.append((start, end))
    return target_ranges


def get_ascend_soc_type(verbose=True):
    """Detect Ascend SoC type. Degrades to UNKNOWN when acl is unavailable."""
    try:
        import acl  # type: ignore[import-not-found,import-untyped]
    except ImportError:
        if verbose:
            safe_print("[mem_scan] acl not available, treat as UNKNOWN")
        return ASCEND_UNKNOWN

    name = acl.get_soc_name()
    if name is None:
        if verbose:
            safe_print("[mem_scan] acl.get_soc_name() failed.")
        return ASCEND_UNKNOWN
    if verbose:
        safe_print(f"[mem_scan] soc name: {name}")
    if "Ascend910B" in name:
        return ASCEND_910B
    if "Ascend910_93" in name:
        return ASCEND_910C
    if "Ascend910_95" in name or "Ascend950" in name:
        return ASCEND_950
    return ASCEND_UNKNOWN


def scan_node_free_segments(node, on_block=None):
    """Yield (start_pfn, end_pfn) free segments of a NUMA node.

    Reads /proc/kpageflags in one batched read per memory block instead of
    per-PFN reads, cutting syscall count by ~1000x on large nodes. Calls
    ``on_block(done, total)`` after each block for progress reporting.
    """
    blocks = get_node_blocks(node)
    total_blocks = len(blocks)
    with open("/proc/kpageflags", "rb") as f:
        merged_start = None
        prev_state = None
        prev_end = None

        for bi, (start_pfn, end_pfn) in enumerate(blocks):
            if prev_end is not None and start_pfn > prev_end + 1:
                if prev_state:
                    yield (merged_start, prev_end)
                merged_start = None
                prev_state = None

            count = end_pfn - start_pfn + 1
            f.seek(start_pfn * 8)
            data = f.read(count * 8)
            got = len(data) // 8
            if got == 0:
                if on_block is not None:
                    on_block(bi + 1, total_blocks)
                continue

            flags = array.array('Q')
            flags.frombytes(data[: got * 8])

            for i in range(got):
                pfn = start_pfn + i
                free = is_buddy(flags[i])
                if prev_state is None:
                    prev_state = free
                    merged_start = pfn
                elif free != prev_state:
                    if prev_state:
                        yield (merged_start, pfn - 1)
                    merged_start = pfn
                    prev_state = free

            prev_end = start_pfn + got - 1
            if on_block is not None:
                on_block(bi + 1, total_blocks)

        if prev_state:
            yield (merged_start, prev_end)


def get_aligned_free_size_mb(free_start, free_end, min_size, zone_page):
    start = round_up(free_start * PAGE_SIZE, zone_page)
    end = round_down((free_end + 1) * PAGE_SIZE, zone_page)
    size = end - start
    if min_size < zone_page:
        size = size // zone_page * min_size
    else:
        size = size // min_size * min_size
    if size < min_size:
        return None
    return size >> 20


# ===== Compatibility functions (preserve signatures for external imports; single-threaded) =====


def stat_node_free_mb(node, min_mb, zone_page):
    total_free_mb = 0
    min_size = min_mb << 20
    for free_start, free_end in scan_node_free_segments(node):
        size_mb = get_aligned_free_size_mb(free_start, free_end, min_size, zone_page)
        if size_mb is not None:
            total_free_mb += size_mb
    return total_free_mb


def stat_range_node_free_mb(node, min_mb, target_ranges, zone_page):
    total_free_mb = 0
    min_size = min_mb << 20
    for free_start, free_end in scan_node_free_segments(node):
        for r_start, r_end in target_ranges:
            inter = intersect(free_start, free_end, r_start, r_end)
            if not inter:
                continue
            size_mb = get_aligned_free_size_mb(inter[0], inter[1], min_size, zone_page)
            if size_mb is not None:
                total_free_mb += size_mb
    return total_free_mb


def classic_scan_for_node(node, min_mb, zone_page):
    print(f"\n===== NUMA node {node} =====")
    total_free_mb = stat_node_free_mb(node, min_mb, zone_page)
    print(f"Total FREE: {total_free_mb / 1024:.2f} GB")
    return total_free_mb


def run_for_node(node, min_mb, target_ranges, zone_page):
    print(f"\n===== NUMA node {node} =====")
    total_intersect_mb = 0
    min_mb_bytes = min_mb << 20
    for free_start, free_end in scan_node_free_segments(node):
        for r_start, r_end in target_ranges:
            inter = intersect(free_start, free_end, r_start, r_end)
            if not inter:
                continue
            s, e = inter
            s = round_up(s * PAGE_SIZE, zone_page)
            e = round_down((e + 1) * PAGE_SIZE, zone_page)
            this_size = e - s
            if min_mb_bytes < zone_page:
                this_size = this_size // zone_page * min_mb_bytes
            else:
                this_size = this_size // min_mb_bytes * min_mb_bytes
            if this_size < min_mb_bytes:
                continue
            size_mb = this_size >> 20
            total_intersect_mb += size_mb
            print(f"FREE∩RANGE: {fmt(pfn_to_addr(s))} - {fmt(pfn_to_addr(e))}  size={size_mb:.2f} MB")
    print(f"Total intersect FREE: {total_intersect_mb:.2f} MB")
    return total_intersect_mb


# ===== Parallel scanning =====


def _scan_node(node, min_mb, target_ranges, zone_page, use_ranges, progress=None, on_block=None):
    """Scan one NUMA node. Returns (node, total_mb, lines, elapsed).

    When ``progress`` is given, updates ``progress[node][2]`` with the
    running aligned free MB so the UI can show live available memory.
    """
    t0 = time.time()
    lines = []
    total = 0
    min_size = min_mb << 20

    for free_start, free_end in scan_node_free_segments(node, on_block):
        if use_ranges:
            for r_start, r_end in target_ranges:
                inter = intersect(free_start, free_end, r_start, r_end)
                if not inter:
                    continue
                s, e = inter
                size_mb = get_aligned_free_size_mb(s, e, min_size, zone_page)
                if size_mb is not None:
                    total += size_mb
                    lines.append(f"FREE∩RANGE: {fmt(pfn_to_addr(s))} - {fmt(pfn_to_addr(e))}  size={size_mb:.2f} MB")
        else:
            size_mb = get_aligned_free_size_mb(free_start, free_end, min_size, zone_page)
            if size_mb is not None:
                total += size_mb
        if progress is not None:
            progress[node][2] = total

    elapsed = time.time() - t0
    return node, total, lines, elapsed


def _draw_node_bar(node, done, total, free_mb, width=24):
    pct = done * 100 // total if total else 100
    filled = width * pct // 100
    bar = "#" * filled + "-" * (width - filled)
    return f"node {node} [{bar}] {pct:3d}% free={free_mb / 1024:.1f}GB"


def _make_progress_cb(node, progress, tty):
    """Build on_block callback: update shared progress (tty) or print lines."""
    if tty:

        def tty_cb(done, _total):
            progress[node][0] = done

        return tty_cb
    last = [0]

    def line_cb(done, n_total):
        pct = done * 100 // n_total if n_total else 100
        if done == n_total or pct - last[0] >= 10:
            safe_print(f"[mem_scan] node {node}: progress {pct}% free={progress[node][2] / 1024:.1f}GB")
            last[0] = pct

    return line_cb


def _progress_drawer(progress, nodes, stop):
    """Redraw one progress-bar line per node in place until stop is set."""
    n_lines = len(nodes)
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()
    try:
        for n in nodes:
            sys.stdout.write(_draw_node_bar(n, progress[n][0], progress[n][1], progress[n][2]) + "\n")
        sys.stdout.flush()
        while not stop.wait(0.2):
            sys.stdout.write(f"\033[{n_lines}A")
            for n in nodes:
                sys.stdout.write("\r\033[K" + _draw_node_bar(n, progress[n][0], progress[n][1], progress[n][2]) + "\n")
            sys.stdout.flush()
        sys.stdout.write(f"\033[{n_lines}A")
        for n in nodes:
            sys.stdout.write("\r\033[K" + _draw_node_bar(n, progress[n][0], progress[n][1], progress[n][2]) + "\n")
        sys.stdout.flush()
    finally:
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()


def _resolve_workers(nodes, workers):
    if workers is not None and workers > 0:
        return min(workers, len(nodes)) if nodes else 1
    return min(len(nodes), 8) if nodes else 1


def _resolve_nodes(node):
    """Return node list to scan. Validates a user-specified node id."""
    all_nodes = get_all_nodes()
    if node is None:
        return all_nodes
    if node in all_nodes:
        return [node]
    safe_print(f"[mem_scan] error: NUMA node {node} does not exist (available: {all_nodes})")
    return []


def _scan_all(nodes, min_mb, zone_page, use_ranges, target_ranges, workers, show_progress=True):
    """Scan all NUMA nodes in parallel, return {node: (total_mb, lines, elapsed)}."""
    results = {}
    tty = show_progress and sys.stdout.isatty()
    progress = {n: [0, len(get_node_blocks(n)), 0] for n in nodes}
    drawer = None
    stop = None
    if tty:
        stop = threading.Event()
        drawer = threading.Thread(target=_progress_drawer, args=(progress, nodes, stop), daemon=True)
        drawer.start()

    def cb_for(n):
        if not show_progress:
            return None
        return _make_progress_cb(n, progress, tty)

    prog_for = progress if show_progress else None
    with ThreadPoolExecutor(max_workers=_resolve_workers(nodes, workers)) as ex:
        futs = {
            ex.submit(_scan_node, n, min_mb, target_ranges, zone_page, use_ranges, prog_for, cb_for(n)): n
            for n in nodes
        }
        for fut in as_completed(futs):
            node, total, lines, elapsed = fut.result()
            results[node] = (total, lines, elapsed)
    if stop is not None:
        stop.set()
    if drawer is not None:
        drawer.join()
    return results


def _detect_soc(verbose):
    try:
        return get_ascend_soc_type(verbose=verbose)
    except Exception as exc:
        if verbose:
            safe_print(f"[mem_scan] get soc type failed: {exc}, fallback to UNKNOWN")
        return ASCEND_UNKNOWN


def show(node=None, min_mb=1024, workers=None):
    nodes = _resolve_nodes(node)
    if not nodes:
        return 0
    ascend_soc_type = _detect_soc(True)
    zone_page = get_zone_page_size()
    use_ranges = ascend_soc_type == ASCEND_910C or ascend_soc_type == ASCEND_UNKNOWN
    target_ranges = build_target_ranges() if use_ranges else None
    workers_eff = _resolve_workers(nodes, workers)

    mode = "range" if use_ranges else "classic"
    safe_print(
        f"[mem_scan] start: nodes={nodes}, min_mb={min_mb}, workers={workers_eff}, "
        f"page={hex(PAGE_SIZE)}, zone_page={hex(zone_page)}, soc={ascend_soc_type}, mode={mode}"
    )

    t0 = time.time()
    results = _scan_all(nodes, min_mb, zone_page, use_ranges, target_ranges, workers)

    if use_ranges:
        for n in nodes:
            _, lines, _ = results[n]
            if not lines:
                continue
            safe_print(f"\n----- NUMA node {n} -----")
            for line in lines:
                safe_print(line)

    total_all = sum(results[n][0] for n in nodes)
    elapsed = time.time() - t0
    safe_print("\n===== Summary =====")
    for n in nodes:
        node_total, _, node_elapsed = results[n]
        safe_print(f"NUMA node {n:>2}: {node_total / 1024:>8.2f} GB  ({node_elapsed:>5.2f}s)")
    safe_print("-" * 40)
    safe_print(f"Total free : {total_all / 1024:>8.2f} GB  ({elapsed:>5.2f}s, workers={workers_eff})")
    safe_print(f"PageSize={hex(PAGE_SIZE)}  ZonePageSize={hex(zone_page)}")
    return total_all


def stat(node=None, min_mb=1024, workers=None):
    nodes = _resolve_nodes(node)
    if not nodes:
        return 0.0
    ascend_soc_type = _detect_soc(False)
    zone_page = get_zone_page_size()
    use_ranges = ascend_soc_type == ASCEND_910C
    target_ranges = build_target_ranges() if use_ranges else None
    results = _scan_all(nodes, min_mb, zone_page, use_ranges, target_ranges, workers, show_progress=False)
    return sum(results[n][0] for n in nodes) / 1024


def add_arguments(parser):
    """Add memory scan arguments to an argument parser."""
    parser.add_argument("-n", "--node", type=int, help="NUMA node ID (default: all)")
    parser.add_argument("-m", "--min-mb", type=int, default=1024, help="min segment size in MB")
    parser.add_argument("-w", "--workers", type=int, default=None, help="thread pool size (default: min(nodes, 8))")


def main(argv=None):
    parser = argparse.ArgumentParser(description="NUMA free memory intersect tool")
    add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        show(args.node, args.min_mb, args.workers)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"mem_scan error: {exc}\n")


if __name__ == "__main__":
    main()
