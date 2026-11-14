# SPDX-License-Identifier: Apache-2.0
"""Byte check for Mooncake READ -> HBM staging -> Manager sparse-copy landing.

Run the roles on different idle NPU devices, with a real host IP and
HCCL_INTRA_ROCE_ENABLE=1. Use python -P with the repo on PYTHONPATH to avoid
tools/bisect shadowing the standard library. No model weights are loaded.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from types import SimpleNamespace
from typing import Any

import torch
import torch_npu  # noqa: F401  (must precede mooncake import on Ascend)
from memfabric_hybrid import offload

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.adapter import MooncakeSFAHostKV
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.staging import MooncakeSFAStagingReader
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    SparseKVOffloadManager,
    empty_aligned_int8_cpu_tensors,
)
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import (
    global_te,
)

DEFAULT_STAGING_BYTES = 2 * 1024 * 1024


def _set_npu_device(device_id: int) -> None:
    torch.npu.set_device(device_id)


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


def make_smoke_host_kv(nbytes: int, *, layer_name: str = "host", store_port: int = 8500):
    """Build a one-rank cache fixture with real offload storage and Manager copy.

    Only model/distributed initialization is omitted. Allocation, cache view
    export, Host adaptation and sparse_copy use the serving implementations.
    """
    if nbytes <= 0:
        raise ValueError("nbytes must be positive")
    alignment = 2 * 1024 * 1024
    block_size = 128
    blocks = (nbytes + block_size * 2 - 1) // (block_size * 2)
    tensor_bytes = blocks * block_size * 2
    pool_bytes = (2 * ((tensor_bytes + alignment - 1) // alignment) + 1) * alignment
    config = offload.OffloadConfig()
    config.device_id = torch.npu.current_device()
    config.reserve_size = config.alloc_size = pool_bytes
    config.world_size = 1
    config.rank_id = 0
    config.scene = offload.Scene.SHARED
    config.store_url = f"tcp://127.0.0.1:{store_port}"
    if offload.initialize(config) != 0:
        raise RuntimeError("Smoke fixture offload.initialize failed")
    raw_k, raw_v = empty_aligned_int8_cpu_tensors([tensor_bytes, tensor_bytes], alignment)
    host_k, host_v = [tensor.view(torch.bfloat16).view(blocks, block_size, 1, 1) for tensor in (raw_k, raw_v)]
    hot_tokens = 128
    hot_k, hot_v = [
        torch.zeros((1, hot_tokens + 2 * block_size, 1, 1), dtype=torch.bfloat16, device="npu") for _ in range(2)
    ]
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.tp_rank = 0
    manager.tp_group = SimpleNamespace(world_size=1)
    manager.use_fused_copy_sfa = True
    manager.offload_layer_names = [layer_name]
    manager.layer_name_to_offload_id = {layer_name: 0}
    manager.k_caches_cpu, manager.v_caches_cpu = [host_k], [host_v]
    manager.topk_buffers_k, manager.topk_buffers_v = [hot_k], [hot_v]
    manager.topk_buffer_size = hot_tokens
    manager.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=[layer_name], kv_cache_spec=SimpleNamespace(block_size=block_size))
        ]
    )
    caches = {layer_name: (None, None, host_k, host_v, hot_k, hot_v)}
    adapter = MooncakeSFAHostKV(manager, caches)
    return adapter, raw_k[:nbytes]


def _verify_host_target(target: torch.Tensor, fill_value: int, verify_bytes: int) -> tuple[bool, str]:
    sample = target[: min(verify_bytes, target.numel())]
    expected = torch.full_like(sample, int(fill_value))
    return bool(torch.equal(sample, expected)), "manager_sparse_copy"


def run_client(args: argparse.Namespace) -> None:
    _set_npu_device(args.device_id)
    transfer_engine = _init_engine(args.hostname, args.protocol, args.device_name)
    host_kv, target = make_smoke_host_kv(args.nbytes, store_port=args.offload_port)
    thread = MooncakeSFAStagingReader(transfer_engine, host_kv, [[args.staging_bytes]])

    ret = thread.batch_transfer_sync_read(
        args.remote_session,
        [int(target.data_ptr())],
        [int(args.remote_ptr)],
        [int(args.nbytes)],
        "staging-smoke",
    )
    if ret != 0:
        raise RuntimeError(f"staged Mooncake transfer failed: ret={ret}")

    ok, copy_mode = _verify_host_target(target, args.fill_value, args.verify_bytes)
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
        raise RuntimeError("Host fixture verification failed")
    os._exit(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Mooncake -> fixed NPU staging -> Manager Host byte fixture.")
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
    parser.add_argument("--offload-port", type=int, default=8500)
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
