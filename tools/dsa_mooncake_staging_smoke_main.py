"""DSA PD Mooncake staging smoke test, adapted for the main-tree port.

Adapted from tools/dsa_mooncake_staging_smoke.py (upstream glm52 branch,
commit 1e9ebb64) for branch port/d2d-d2h-staging:
- global_te.get_transfer_engine(hostname, device_name): main tree hardcodes
  the "ascend" protocol inside mooncake_transfer_engine.py, no 3rd arg.
- fake receiving thread must also provide _dsa_pd_mooncake_staging_lock
  (object.__new__ skips __init__ where the lock is normally created).

Verified run recipe (101, linjinghan container, 2026-10-14):
- server and client MUST sit on different NPU devices (mooncake ADXL
  channels use HCCL rank-table semantics: one rank per device, self-device
  connect is rejected with "not support connect with self device").
- export HCCL_INTRA_ROCE_ENABLE=1 on both roles: default HCCS channel
  setup fails inside the container (PrepareHcclComm fails), RoCE works.
- use a real host IP (e.g. 172.27.4.101), not 127.0.0.1.
- run the script from a clean directory: the repo tools/ dir contains a
  bisect/ package that shadows the stdlib bisect module.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import threading
import time
from types import SimpleNamespace
from typing import Any

import torch
import torch_npu  # noqa: F401  (must precede mooncake import on Ascend)

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
    KVCacheRecvingThread,
)
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import (
    global_te,
)
from vllm_ascend.kv_offload.dsa_pd_mooncake import (
    register_dsa_pd_mooncake_offloaded_kv_tensor,
)


ACL_MEMCPY_DEVICE_TO_DEVICE = 3
DEFAULT_STAGING_BYTES = 2 * 1024 * 1024


def _set_npu_device(device_id: int) -> None:
    torch.npu.set_device(device_id)


def _load_acl() -> ctypes.CDLL:
    for path in ("libascendcl.so", "libacl.so"):
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue
    raise RuntimeError("failed to load Ascend ACL runtime library")


def _init_engine(hostname: str, protocol: str, device_name: str | None) -> Any:
    # main tree: protocol is hardcoded to "ascend" inside get_transfer_engine
    return global_te.get_transfer_engine(hostname, device_name)


def _get_rpc_port(transfer_engine: Any) -> int:
    for attr_name in ("get_rpc_port", "get_rpc_port_name"):
        getter = getattr(transfer_engine, attr_name, None)
        if getter is not None:
            return int(getter())
    for attr_name in ("rpc_port", "local_rpc_port"):
        value = getattr(transfer_engine, attr_name, None)
        if value is not None:
            return int(value)
    raise RuntimeError("cannot determine Mooncake TransferEngine RPC port")


def _fill_source_buffer(nbytes: int, fill_value: int) -> torch.Tensor:
    source = torch.empty((nbytes,), dtype=torch.int8, device="npu")
    source.fill_(int(fill_value))
    torch.npu.synchronize()
    return source


def run_server(args: argparse.Namespace) -> None:
    _set_npu_device(args.device_id)
    transfer_engine = _init_engine(args.hostname, args.protocol, args.device_name)
    source = _fill_source_buffer(args.nbytes, args.fill_value)
    global_te.register_buffer([int(source.data_ptr())], [int(args.nbytes)])
    session = f"{args.hostname}:{_get_rpc_port(transfer_engine)}"
    print(
        json.dumps(
            {
                "role": "server",
                "session": session,
                "ptr": int(source.data_ptr()),
                "nbytes": int(args.nbytes),
                "fill_value": int(args.fill_value),
                "device_id": int(args.device_id),
            }
        ),
        flush=True,
    )
    deadline = time.time() + args.keepalive_seconds
    while time.time() < deadline:
        time.sleep(1)
        torch.npu.synchronize()
    os._exit(0)


def _make_fake_recv_thread(transfer_engine: Any, staging_bytes: int) -> Any:
    thread = object.__new__(KVCacheRecvingThread)
    thread.engine = transfer_engine
    thread.kv_caches = {}
    thread.block_len_per_addr = [[int(staging_bytes)]]
    thread._dsa_pd_mooncake_staging_storage = None
    thread._dsa_pd_mooncake_staging_tensor = None
    thread._dsa_pd_mooncake_staging_bytes = 0
    thread._dsa_pd_mooncake_staging_lock = threading.Lock()
    # Mirror the async-landing attributes normally created in __init__
    # (object.__new__ skips it); read the knobs the same way envs.py does.
    thread._landing_stream = torch.npu.Stream()
    thread._pipeline_banks = max(1, int(os.getenv("VLLM_ASCEND_DSA_PD_MOONCAKE_BANKS", "4")))
    thread._landing_streams = [
        torch.npu.Stream() for _ in range(max(1, thread._pipeline_banks))
    ]
    thread._landing_stream = thread._landing_streams[0]
    thread._pipeline_landing_enabled = os.getenv("VLLM_ASCEND_DSA_PD_MOONCAKE_PIPELINE", "0") == "1"
    thread.vllm_config = SimpleNamespace()
    return thread


def _allocate_swapped_target(nbytes: int) -> torch.Tensor:
    empty_with_swapped_memory = getattr(torch_npu, "empty_with_swapped_memory", None)
    if empty_with_swapped_memory is None:
        raise RuntimeError("torch_npu.empty_with_swapped_memory is unavailable")
    target = empty_with_swapped_memory((nbytes,), dtype=torch.int8, device="npu")
    register_dsa_pd_mooncake_offloaded_kv_tensor(target)
    return target


def _copy_swapped_sample_to_scratch(
    target: torch.Tensor,
    sample_bytes: int,
) -> tuple[torch.Tensor, str]:
    sample_bytes = min(int(sample_bytes), int(target.numel()))
    scratch = torch.empty((sample_bytes,), dtype=torch.int8, device="npu")
    try:
        scratch.copy_(target[:sample_bytes], non_blocking=False)
        torch.npu.synchronize()
        return scratch, "torch_copy"
    except BaseException as copy_exc:
        try:
            import vllm_ascend.vllm_ascend_C  # noqa: F401

            src_ptrs = torch.tensor([int(target.data_ptr())], dtype=torch.int64)
            dst_ptrs = torch.tensor([int(scratch.data_ptr())], dtype=torch.int64)
            sizes = torch.tensor([sample_bytes], dtype=torch.int64)
            torch.ops._C_ascend.swap_blocks_batch(
                src_ptrs,
                dst_ptrs,
                sizes,
                ACL_MEMCPY_DEVICE_TO_DEVICE - 1,
            )
            torch.npu.synchronize()
            return scratch, f"swap_blocks_batch_after={copy_exc!r}"
        except BaseException as swap_exc:
            acl = _load_acl()
            acl.aclrtMemcpy.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            acl.aclrtMemcpy.restype = ctypes.c_int
            ret = int(
                acl.aclrtMemcpy(
                    ctypes.c_void_p(int(scratch.data_ptr())),
                    ctypes.c_size_t(sample_bytes),
                    ctypes.c_void_p(int(target.data_ptr())),
                    ctypes.c_size_t(sample_bytes),
                    ctypes.c_int(ACL_MEMCPY_DEVICE_TO_DEVICE),
                )
            )
            if ret != 0:
                raise RuntimeError(
                    "failed to copy swapped target to scratch: "
                    f"torch_copy={copy_exc!r}, swap_blocks_batch={swap_exc!r}, aclrtMemcpy={ret}"
                )
            torch.npu.synchronize()
            return scratch, f"aclrtMemcpy_after={copy_exc!r}; swap={swap_exc!r}"


def _verify_swapped_target(target: torch.Tensor, fill_value: int, verify_bytes: int) -> tuple[bool, str]:
    scratch, copy_mode = _copy_swapped_sample_to_scratch(target, verify_bytes)
    sample = scratch.cpu()
    expected = torch.full_like(sample, int(fill_value))
    return bool(torch.equal(sample, expected)), copy_mode


def run_client(args: argparse.Namespace) -> None:
    _set_npu_device(args.device_id)
    transfer_engine = _init_engine(args.hostname, args.protocol, args.device_name)
    target = _allocate_swapped_target(args.nbytes)
    thread = _make_fake_recv_thread(transfer_engine, args.staging_bytes)

    ret = thread._batch_transfer_sync_read_with_optional_staging(
        args.remote_session,
        [int(target.data_ptr())],
        [int(args.remote_ptr)],
        [int(args.nbytes)],
        "staging-smoke",
    )
    if ret != 0:
        raise RuntimeError(f"staged Mooncake transfer failed: ret={ret}")

    ok, copy_mode = _verify_swapped_target(target, args.fill_value, args.verify_bytes)
    print(
        json.dumps(
            {
                "role": "client",
                "ok": ok,
                "copy_mode": copy_mode,
                "target_ptr": int(target.data_ptr()),
                "target_nbytes": int(target.numel()),
                "staging_bytes": int(thread._dsa_pd_mooncake_staging_bytes),
                "verify_bytes": int(min(args.verify_bytes, args.nbytes)),
                "device_id": int(args.device_id),
            }
        ),
        flush=True,
    )
    if not ok:
        raise RuntimeError("swapped target verification failed")
    os._exit(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Mooncake -> fixed NPU staging chunk -> swapped KV with safe readback."
    )
    parser.add_argument("--role", choices=("server", "client"), required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--hostname", default="127.0.0.1")
    parser.add_argument("--protocol", default="ascend")
    parser.add_argument("--device-name", default=None)
    parser.add_argument("--nbytes", type=int, default=3 * 1024 * 1024)
    parser.add_argument("--fill-value", type=int, default=77)
    parser.add_argument("--keepalive-seconds", type=int, default=300)
    parser.add_argument("--remote-session", default="")
    parser.add_argument("--remote-ptr", type=int, default=0)
    parser.add_argument("--staging-bytes", type=int, default=DEFAULT_STAGING_BYTES)
    parser.add_argument("--verify-bytes", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.role == "server":
        run_server(args)
    else:
        if not args.remote_session or args.remote_ptr <= 0:
            raise ValueError("--remote-session and --remote-ptr are required for client role")
        run_client(args)


if __name__ == "__main__":
    main()
