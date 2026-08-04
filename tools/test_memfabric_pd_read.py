# SPDX-License-Identifier: Apache-2.0
"""Minimal two-node MemFabric pull-read diagnostic.

This script starts no vLLM service. The Prefill process owns a deterministic
NPU buffer and publishes only its MemFabric session and address over a tiny TCP
control channel. The Decode process creates a local destination, pulls the bytes
with ``batch_transfer_sync_read``, and verifies them exactly. Only the remote
Prefill source is registered, matching the production pull-read path.

Run Prefill first::

    python tools/test_memfabric_pd_read.py \
        --role prefill --local-ip 10.0.0.1 --device-id 0

Then run Decode on the peer machine::

    python tools/test_memfabric_pd_read.py \
        --role decode --local-ip 10.0.0.2 --prefill-ip 10.0.0.1 \
        --device-id 0

The default destination is an aligned ``memfabric_hybrid.offload`` pinned CPU
buffer, matching KVOffloadDecodeManager. Use ``--dst-memory npu`` as an HBM to
HBM control test.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import ipaddress
import json
import os
import socket
import sys
import time
from typing import Any

CONTROL_PORT = 29599
DEFAULT_NUM_BYTES = 4 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 120.0
CPU_BUFFER_ALIGNMENT = 2 * 1024 * 1024
GIB = 1024 * 1024 * 1024
PATTERN_PERIOD = 251
MAX_CONTROL_MESSAGE_BYTES = 64 * 1024


def _ipv4(value: str) -> str:
    try:
        address = ipaddress.ip_address(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid IP address: {value}") from error
    if address.version != 4:
        raise argparse.ArgumentTypeError("MemFabric test requires a numeric IPv4 address")
    return str(address)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=("prefill", "decode"))
    parser.add_argument(
        "--local-ip",
        required=True,
        type=_ipv4,
        help="Numeric IPv4 address advertised by this machine to MemFabric.",
    )
    parser.add_argument(
        "--prefill-ip",
        type=_ipv4,
        help="Prefill machine IPv4 address; required on Decode.",
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--control-port", type=int, default=CONTROL_PORT)
    parser.add_argument(
        "--num-bytes",
        type=_positive_int,
        default=DEFAULT_NUM_BYTES,
        help="Number of bytes in the Prefill source buffer.",
    )
    parser.add_argument(
        "--dst-memory",
        choices=("offload", "npu"),
        default="offload",
        help="Decode destination type; offload matches KVOffloadDecodeManager.",
    )
    parser.add_argument(
        "--dram-pool-gb",
        type=_positive_int,
        default=1,
        help="Decode offload pool size in GiB (used only with --dst-memory offload).",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Control-channel connect/accept/result timeout in seconds.",
    )
    args = parser.parse_args()
    if args.device_id < 0:
        parser.error("--device-id must be non-negative")
    if not 1 <= args.control_port <= 65535:
        parser.error("--control-port must be in [1, 65535]")
    if args.role == "decode" and args.prefill_ip is None:
        parser.error("--prefill-ip is required for Decode")
    return args


def _load_torch():
    try:
        import torch
        import torch_npu  # noqa: F401
    except ImportError as error:
        raise RuntimeError("torch and torch_npu must be installed") from error
    return torch


def _make_pattern(torch, num_bytes: int):
    base = torch.arange(PATTERN_PERIOD, dtype=torch.uint8, device="cpu")
    repeats = (num_bytes + PATTERN_PERIOD - 1) // PATTERN_PERIOD
    return base.repeat(repeats)[:num_bytes].contiguous()


def _sha256(tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def _new_transfer_engine(role: str, local_ip: str, device_id: int):
    from vllm_ascend.distributed.kv_transfer.utils.memfabric_transfer_engine import (
        MEMFABRIC_ROLE_DECODE,
        MEMFABRIC_ROLE_PREFILL,
        GlobalMemfabricTE,
    )

    memfabric_role = MEMFABRIC_ROLE_PREFILL if role == "prefill" else MEMFABRIC_ROLE_DECODE
    manager = GlobalMemfabricTE()
    manager.configure(role=memfabric_role, device_id=device_id)
    engine = manager.get_transfer_engine(local_ip)
    return manager, engine


def _shutdown_transfer_engine(engine) -> None:
    """Release MemFabric connections before its process-global runtime."""
    raw_engine = getattr(engine, "_engine", None)
    if raw_engine is None:
        return
    errors = []
    for method_name in ("destroy", "unInitialize"):
        method = getattr(raw_engine, method_name, None)
        if method is None:
            continue
        try:
            method()
        except Exception as error:
            errors.append(f"{method_name}: {type(error).__name__}: {error}")
    if errors:
        raise RuntimeError("MemFabric shutdown failed: " + "; ".join(errors))


def _send_json(sock: socket.socket, message: dict[str, Any]) -> None:
    sock.sendall(json.dumps(message, separators=(",", ":")).encode() + b"\n")


def _recv_json(sock: socket.socket) -> dict[str, Any]:
    payload = bytearray()
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("control connection closed before a complete message")
        payload.extend(chunk)
        if len(payload) > MAX_CONTROL_MESSAGE_BYTES:
            raise ValueError("control message is too large")
        newline = payload.find(b"\n")
        if newline >= 0:
            return json.loads(payload[:newline])


def _connect_with_retry(host: str, port: int, timeout: float) -> socket.socket:
    deadline = time.monotonic() + timeout
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            remaining = max(deadline - time.monotonic(), 0.1)
            return socket.create_connection((host, port), timeout=min(5.0, remaining))
        except OSError as error:
            last_error = error
            time.sleep(0.5)
    raise TimeoutError(f"timed out connecting to {host}:{port}: {last_error}")


def _allocate_offload_destination(torch, device_id: int, num_bytes: int, pool_gb: int):
    try:
        from memfabric_hybrid import offload
    except ImportError as error:
        raise RuntimeError("memfabric_hybrid.offload is required for CPU destination") from error

    config = offload.OffloadConfig()
    config.device_id = device_id
    config.size = pool_gb * GIB
    config.world_size = 1
    config.rank_id = 0
    offload.initialize(config)

    backing = offload.empty(
        [num_bytes + CPU_BUFFER_ALIGNMENT],
        dtype=torch.uint8,
        pin_memory=True,
    )
    offset = (-backing.data_ptr()) % CPU_BUFFER_ALIGNMENT
    destination = backing[offset : offset + num_bytes]
    destination.fill_(0xFF)
    if destination.data_ptr() % CPU_BUFFER_ALIGNMENT != 0:
        raise RuntimeError("failed to create a 2 MiB-aligned offload buffer")
    return offload, backing, destination


def _compare(torch, actual, expected) -> tuple[bool, str]:
    actual_cpu = actual.detach().cpu().contiguous()
    if torch.equal(actual_cpu, expected):
        return True, ""
    mismatch = torch.nonzero(actual_cpu != expected, as_tuple=False).flatten()
    first = int(mismatch[0]) if mismatch.numel() else -1
    detail = (
        f"mismatches={mismatch.numel()}, first_offset={first}, "
        f"actual={int(actual_cpu[first]) if first >= 0 else 'n/a'}, "
        f"expected={int(expected[first]) if first >= 0 else 'n/a'}"
    )
    return False, detail


def _run_prefill(args: argparse.Namespace) -> None:
    torch = _load_torch()
    torch.npu.set_device(torch.device(f"npu:{args.device_id}"))

    expected = _make_pattern(torch, args.num_bytes)
    source = expected.to(f"npu:{args.device_id}")
    torch.npu.synchronize()

    manager, engine = _new_transfer_engine("prefill", args.local_ip, args.device_id)
    manager.register_buffer([source.data_ptr()], [args.num_bytes])
    metadata = {
        "session": manager.unique_id,
        "source_ptr": source.data_ptr(),
        "num_bytes": args.num_bytes,
        "pattern_period": PATTERN_PERIOD,
        "sha256": _sha256(expected),
    }

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((args.local_ip, args.control_port))
            server.listen(1)
            server.settimeout(args.timeout)
            print(
                f"[P] ready: session={manager.unique_id}, source_hbm={hex(source.data_ptr())}, "
                f"bytes={args.num_bytes}, control={args.local_ip}:{args.control_port}",
                flush=True,
            )
            conn, peer = server.accept()
            with conn:
                conn.settimeout(args.timeout)
                print(
                    f"[P] Decode control connection from {peer[0]}:{peer[1]}",
                    flush=True,
                )
                _send_json(conn, metadata)
                result = _recv_json(conn)
    finally:
        _shutdown_transfer_engine(engine)

    if not result.get("ok"):
        raise RuntimeError(f"Decode read verification failed: {result.get('error', result)}")
    print(
        f"[P] PASS: Decode pulled and verified {result['num_bytes']} bytes, sha256={result['sha256']}",
        flush=True,
    )


def _run_decode(args: argparse.Namespace) -> None:
    torch = _load_torch()
    torch.npu.set_device(torch.device(f"npu:{args.device_id}"))

    with _connect_with_retry(args.prefill_ip, args.control_port, args.timeout) as conn:
        conn.settimeout(args.timeout)
        engine = None
        offload_module = None
        backing = None
        destination = None
        result = None
        operation_error = None
        try:
            metadata = _recv_json(conn)
            num_bytes = int(metadata["num_bytes"])
            if num_bytes <= 0:
                raise ValueError(f"invalid source size: {num_bytes}")
            if int(metadata["pattern_period"]) != PATTERN_PERIOD:
                raise ValueError("Prefill and Decode pattern versions differ")

            expected = _make_pattern(torch, num_bytes)
            expected_hash = _sha256(expected)
            if metadata["sha256"] != expected_hash:
                raise ValueError("Prefill metadata checksum does not match the test pattern")

            if args.dst_memory == "offload":
                offload_module, backing, destination = _allocate_offload_destination(
                    torch,
                    args.device_id,
                    num_bytes,
                    args.dram_pool_gb,
                )
            else:
                destination = torch.full(
                    (num_bytes,),
                    0xFF,
                    dtype=torch.uint8,
                    device=f"npu:{args.device_id}",
                )
                backing = destination
                torch.npu.synchronize()

            # Match the production D-side order: KVOffloadDecodeManager owns
            # the CPU pool before the connector initializes its transfer
            # engine. A pull destination is local and is not registered.
            _, engine = _new_transfer_engine("decode", args.local_ip, args.device_id)
            print(
                f"[D] read: peer_session={metadata['session']}, "
                f"local_dst_{args.dst_memory}={hex(destination.data_ptr())}, "
                f"peer_src_hbm={hex(int(metadata['source_ptr']))}, bytes={num_bytes}",
                flush=True,
            )
            ret = engine.batch_transfer_sync_read(
                metadata["session"],
                [destination.data_ptr()],
                [int(metadata["source_ptr"])],
                [num_bytes],
            )
            if ret != 0:
                raise RuntimeError(f"batch_transfer_sync_read returned {ret}")
            torch.npu.synchronize()

            ok, detail = _compare(torch, destination, expected)
            if not ok:
                raise RuntimeError(f"data mismatch after successful read: {detail}")
            actual_hash = _sha256(destination.detach().cpu())
            result = {
                "ok": True,
                "num_bytes": num_bytes,
                "sha256": actual_hash,
                "dst_memory": args.dst_memory,
            }
        except Exception as error:
            operation_error = error

        cleanup_errors = []
        if engine is not None:
            try:
                _shutdown_transfer_engine(engine)
            except Exception as error:
                cleanup_errors.append(str(error))
        del destination
        del backing
        if offload_module is not None:
            try:
                offload_module.uninitialize()
            except Exception as error:
                cleanup_errors.append(f"offload.uninitialize: {type(error).__name__}: {error}")
        if cleanup_errors and operation_error is None:
            operation_error = RuntimeError("; ".join(cleanup_errors))

        if operation_error is not None:
            with contextlib.suppress(OSError):
                _send_json(
                    conn,
                    {
                        "ok": False,
                        "error": (f"{type(operation_error).__name__}: {operation_error}"),
                    },
                )
            raise operation_error

        assert result is not None
        _send_json(conn, result)
        print(
            f"[D] PASS: ret=0, bytes={result['num_bytes']}, "
            f"sha256={result['sha256']}, destination={result['dst_memory']}",
            flush=True,
        )


def main() -> int:
    args = _parse_args()
    os.environ.setdefault("ASCEND_TRANSFER_TIMEOUT", str(int(args.timeout)))
    try:
        if args.role == "prefill":
            _run_prefill(args)
        else:
            _run_decode(args)
    except Exception as error:
        print(f"[{args.role[0].upper()}] FAIL: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
