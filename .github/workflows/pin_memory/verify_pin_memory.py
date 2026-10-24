# SPDX-License-Identifier: Apache-2.0
# 驱动重装后 pin_memory 一键验收脚本（Ascend 910B4，CANN 9.0.0）
#
# 用法:
#   python verify_pin_memory.py          完整验收（含映射 D2D 决定性测试）
#   python verify_pin_memory.py --safe   跳过映射 D2D（在坏驱动上重复跑会烧 SDMA 通道）
#
# 判据（2026-09-29 实测校准）:
#   [1] 驱动承认 SDMA/PCIEDMA 支持 host 映射        （坏驱动 25.3.rc1.2 全报 207000）
#   [2] 从映射 VA 做 64MiB D2D 拷贝不崩             （坏驱动烧 SDMA 通道: sdmaBlkFsmState=0x7）
#   [3] aclrtMemcpyAsync 层 registered > 15 GB/s    （9 月正常 23.4，坏驱动 ~12.6 封顶）
#   [4] DeepSpeed 钩子全链路 registered > 15 GB/s  （同上）
import ctypes
import glob
import os
import statistics
import subprocess
import sys
import time

import regex as re

SAFE_MODE = "--safe" in sys.argv

# 阈值（GB/s），9 月正常水平为 ~23.4
REG_BW_THRESHOLD = 15.0


ENV_KEYS = ("LD_LIBRARY_PATH", "ASCEND_HOME_PATH", "ASCEND_OPP_PATH", "PATH")


def ensure_ascend_env():
    """确保 set_env.sh 的环境变量在**进程启动前**就位。

    glibc 动态加载器在进程启动时缓存 LD_LIBRARY_PATH，启动后 putenv 对 dlopen 无效，
    所以发现环境不对不能原地改，只能用正确的环境重新 exec 自己。
    """
    setup = os.path.expanduser("~/miniconda3/envs/cann/Ascend/ascend-toolkit/set_env.sh")
    if not os.path.exists(setup):
        return
    out = subprocess.run(["bash", "-c", f"source {setup} && env"], capture_output=True, text=True)
    env_map = {}
    for line in out.stdout.splitlines():
        if "=" not in line or line.startswith(("BASH_FUNC", "_=")):
            continue
        key, _, value = line.partition("=")
        env_map[key] = value

    def cann_root(ld_path):
        # LD_LIBRARY_PATH 首段形如 .../Ascend/cann-9.0.0/lib64；取 CANN 版本目录名
        if not ld_path:
            return None
        match = re.search(r"Ascend/([^/]+)", ld_path.split(":")[0])
        return match.group(1) if match else ld_path.split(":")[0]

    mismatched = [k for k in ENV_KEYS if k != "LD_LIBRARY_PATH" and os.environ.get(k) != env_map.get(k)]
    # LD_LIBRARY_PATH 单独比首段 CANN 根：set_env.sh 每次 source 都会重复追加前缀，精确比较恒不等
    if cann_root(os.environ.get("LD_LIBRARY_PATH", "")) != cann_root(env_map.get("LD_LIBRARY_PATH", "")):
        mismatched.append("LD_LIBRARY_PATH")
    if not mismatched:
        return
    if "--ascend-env-relaunched" in sys.argv:
        print(f"  警告: 环境变量仍不一致 {mismatched}（可能 set_env.sh 指向了旧 CANN），继续执行")
        return
    print(f"  环境变量与 set_env.sh 不一致（{mismatched}），用正确环境重新启动 ...")
    sys.stdout.flush()
    new_env = dict(os.environ)
    new_env.update(env_map)
    os.execve(sys.executable, [sys.executable, *sys.argv, "--ascend-env-relaunched"], new_env)


def find_ascendcl():
    """按优先级找 libascendcl.so：ASCEND_HOME_PATH -> cann-9.0.0 -> cann-8.5.0 -> 全盘 glob。"""
    cands = []
    home = os.environ.get("ASCEND_HOME_PATH", "")
    if home:
        cands += [f"{home}/lib64/libascendcl.so", f"{home}/aarch64-linux/lib64/libascendcl.so"]
    asc = os.path.expanduser("~/miniconda3/envs/cann/Ascend")
    for ver in ("cann-9.0.0", "cann-8.5.0"):
        cands.append(f"{asc}/{ver}/aarch64-linux/lib64/libascendcl.so")
    cands += glob.glob(f"{asc}/**/lib64/libascendcl.so", recursive=True)
    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("找不到 libascendcl.so，请确认 CANN 安装路径")


def driver_version():
    try:
        out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=30)
        for line in out.stdout.splitlines():
            if "Version" in line:
                return line.strip()
    except Exception:
        pass
    return "unknown (npu-smi 不可用)"


def raw_alloc(nbytes):
    ptr = ctypes.c_void_p()
    assert libc.posix_memalign(ctypes.byref(ptr), 4096, nbytes) == 0
    return ptr


def main():
    global libc, acl
    ensure_ascend_env()

    import torch
    import torch_npu  # noqa: F401
    import torch_npu._C as npu_C

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.posix_memalign.restype = ctypes.c_int
    libc.posix_memalign.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_size_t]
    libc.free.argtypes = [ctypes.c_void_p]

    acl = ctypes.CDLL(find_ascendcl())
    acl.aclrtSetDevice.restype = ctypes.c_int
    acl.aclrtSetDevice.argtypes = [ctypes.c_int32]
    acl.aclrtHostRegisterV2.restype = ctypes.c_int
    acl.aclrtHostRegisterV2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32]
    acl.aclrtHostUnregister.restype = ctypes.c_int
    acl.aclrtHostUnregister.argtypes = [ctypes.c_void_p]
    acl.aclrtHostGetDevicePointer.restype = ctypes.c_int
    acl.aclrtHostGetDevicePointer.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32]
    acl.aclrtHostMemMapCapabilities.restype = ctypes.c_int
    acl.aclrtHostMemMapCapabilities.argtypes = [ctypes.c_uint32, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
    acl.aclrtMemcpyAsync.restype = ctypes.c_int
    acl.aclrtMemcpyAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int32,
        ctypes.c_void_p,
    ]
    acl.aclrtSynchronizeStream.restype = ctypes.c_int
    acl.aclrtSynchronizeStream.argtypes = [ctypes.c_void_p]

    passed, failed = [], []

    def check(name, ok, detail):
        (passed if ok else failed).append(name)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    print("=== [0] 环境 ===")
    print(f"  torch {torch.__version__} | torch_npu {torch_npu.__version__}")
    print(f"  CANN runtime {torch_npu._C._get_cann_version('runtime')}")
    print(f"  ascendcl: {acl._name}")
    print(f"  {driver_version()}")
    print(f"  safe 模式: {SAFE_MODE}")

    torch.npu.set_device(0)
    assert acl.aclrtSetDevice(0) == 0

    print("\n=== [1] host 映射能力查询（坏驱动下全引擎 207000） ===")
    cap_ok = False
    for hac, name in [(4, "PCIEDMA"), (6, "SDMA")]:
        cap = ctypes.c_int32(-1)
        rc = acl.aclrtHostMemMapCapabilities(0, hac, ctypes.byref(cap))
        print(f"  {name}: rc={rc} capability={cap.value}")
        if rc == 0 and cap.value == 1:
            cap_ok = True
    check("capabilities", cap_ok, "SDMA/PCIEDMA 至少一个报告支持 host 映射")

    N = 64 * 1024 * 1024
    STREAM = npu_C._npu_getCurrentRawStream(0)
    dev = torch.empty(N // 4, dtype=torch.float32, device="npu")
    dptr = dev.data_ptr()

    if SAFE_MODE:
        print("\n=== [2] 映射 D2D 决定性测试：--safe 模式跳过 ===")
    else:
        print("\n=== [2] 决定性测试：从映射 VA 的 D2D 拷贝 64MiB ===")
        print("  !! 注意：如果驱动没修好，这一步会烧掉一个 SDMA 通道（可重启恢复）")
        h = raw_alloc(N)
        try:
            rc_reg = acl.aclrtHostRegisterV2(h, N, 0x2)
            m = ctypes.c_void_p()
            rc_g = acl.aclrtHostGetDevicePointer(h, ctypes.byref(m), 0)
            rc_c = acl.aclrtMemcpyAsync(dptr, N, m, N, 3, STREAM)
            rc_s = acl.aclrtSynchronizeStream(STREAM)
            check(
                "mapped-d2d",
                rc_reg == 0 and rc_g == 0 and rc_c == 0 and rc_s == 0,
                f"register rc={rc_reg} GetDevicePointer rc={rc_g} memcpy rc={rc_c} sync rc={rc_s}",
            )
            acl.aclrtHostUnregister(h)
        except Exception as e:
            check("mapped-d2d", False, f"异常: {type(e).__name__}: {e}")
        libc.free(h)

    def bench_memcpy(kind, src, dst, iters=11, warmup=3):
        for _ in range(warmup):
            assert acl.aclrtMemcpyAsync(dst, N, src, N, kind, STREAM) == 0
            acl.aclrtSynchronizeStream(STREAM)
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            acl.aclrtMemcpyAsync(dst, N, src, N, kind, STREAM)
            acl.aclrtSynchronizeStream(STREAM)
            ts.append(time.perf_counter() - t0)
        return N / statistics.median(ts) / 1e9

    print("\n=== [3] 驱动层带宽：plain vs registered（64MiB, median of 11） ===")
    h = raw_alloc(N)
    h2 = raw_alloc(N)
    try:
        assert acl.aclrtHostRegisterV2(h, N, 0x2) == 0
        bw_plain = bench_memcpy(1, h2, dptr)
        bw_reg = bench_memcpy(1, h, dptr)
        print(f"  H2D plain {bw_plain:6.2f} GB/s | registered {bw_reg:6.2f} GB/s | {bw_reg / bw_plain:.2f}x")
        check("driver-bandwidth", bw_reg > REG_BW_THRESHOLD, f"registered 超过 {REG_BW_THRESHOLD} GB/s")
        acl.aclrtHostUnregister(h)
    except Exception as e:
        check("driver-bandwidth", False, f"异常: {type(e).__name__}: {e}")
    libc.free(h)
    libc.free(h2)

    print("\n=== [4] DeepSpeed 钩子全链路（NativePinnedMemory: mlock + npurt 注册） ===")
    try:
        from deepspeed.utils.pin_memory import NativePinnedMemory

        mgr = NativePinnedMemory()
        src = torch.ones(N // 4, dtype=torch.float32)
        locked = mgr.pin(src, make_copy=True)
        print(f"  ds_pinned={mgr.is_pinned(locked)} device_registered={bool(mgr._device_registered)}")

        plain_t = torch.empty(N // 4, dtype=torch.float32)
        plain_t.copy_(src)

        def bench_torch(src_t):
            for _ in range(3):
                dev.copy_(src_t, non_blocking=True)
            torch.npu.synchronize()
            ts = []
            for _ in range(11):
                t0 = time.perf_counter()
                dev.copy_(src_t, non_blocking=True)
                torch.npu.synchronize()
                ts.append(time.perf_counter() - t0)
            return N / statistics.median(ts) / 1e9

        bw_plain_t = bench_torch(plain_t)
        bw_reg_t = bench_torch(locked)
        print(
            f"  H2D plain {bw_plain_t:6.2f} GB/s | hook registered {bw_reg_t:6.2f} GB/s | {bw_reg_t / bw_plain_t:.2f}x"
        )
        check("hook-bandwidth", bw_reg_t > REG_BW_THRESHOLD, f"钩子路径 registered 超过 {REG_BW_THRESHOLD} GB/s")
        mgr.unpin(locked)
    except Exception as e:
        check("hook-bandwidth", False, f"异常: {type(e).__name__}: {e}")

    print("\n=== 汇总 ===")
    print(f"  PASS: {len(passed)}  FAIL: {len(failed)}")
    if failed:
        print(f"  失败项: {failed} -> 驱动还没修好")
        print("  参考: 2026-09-29 坏驱动 25.3.rc1.2 的症状是 [1] 全 207000、[2] 烧 SDMA 通道、")
        print("        9 月正常驱动 25.5.1 下 registered 23.4 GB/s")
    else:
        print("  全部通过 -> 驱动修好了，pin_memory 完整生效")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
