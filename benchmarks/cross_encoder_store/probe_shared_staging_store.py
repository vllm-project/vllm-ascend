#!/usr/bin/env python3
"""Probe one long-lived NPU staging slab with Store.

The probe starts two short-lived processes on two NPUs and (unless an existing
master address is supplied) a private ``mooncake_master``.  Each process owns
one MooncakeDistributedStore. The default topology mirrors vLLM-Ascend's
shared ``global_te`` path. The optional two-engine topology uses a second
externally created engine. The store-owned topology mirrors KV Pool: Store
creates its own engine and registers the same slab via ``store.register_buffer``.

In one invocation it verifies all properties needed by the proposed design:

* Store setup accepts both an externally shared engine and a Store-owned one;
* an EC-style direct engine write can read the engine-registered staging slab;
* Store PUT can read that same slab into the peer's Host-DRAM segment;
* Store GET can write multiple objects directly into subranges of the slab;
* both the existing KV Store's contiguous Device-buffer shape and the EC
  Store's 24-byte-header plus Device-payload shape work;
* EC write and Store PUT can be invoked concurrently against the exact same
  source without corrupting either result (API-call overlap is reported; this
  is not a transport-throughput benchmark);
* PUT completion is safe for immediate source reuse (the source is overwritten
  before the later GET and the original bytes must still be returned);
* registrations can be explicitly checked at shutdown or retained until
  normal interpreter exit, matching KV Pool ownership.

This is intentionally not a model/vLLM E2E and does not modify Store data
format or production code.

Run the production-shaped Store-owned lifetime probe over RoCE with::

    python3 probe_shared_staging_store.py --host <business-ip> \
        --npu-a 0 --npu-b 1 --link-mode roce \
        --engine-topology store-owned --cleanup-mode process-lifetime \
        --global-resource-config \
        '{"store":{"comm_resource_config":{"protocol_desc":["roce:device"]}}}' \
        --iterations 7 --timeout 600 \
        --output-dir <empty-output-directory>
"""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

ALIGNMENT = 2 * 1024**2
HEADER_BYTES = 24
OBJECT_BYTES = 4 * 1024**2
OBJECT_STRIDE = ((OBJECT_BYTES + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
STAGING_BYTES = 64 * 1024**2
GLOBAL_SEGMENT_BYTES = 256 * 1024**2
LOCAL_BUFFER_BYTES = 64 * 1024**2
DEFAULT_TIMEOUT_SECONDS = 180.0
STORE_ROCE_RESOURCE_CONFIG = {
    "store": {
        "comm_resource_config": {
            "protocol_desc": ["roce:device"],
        },
    },
}
DEFAULT_STORE_ROCE_RESOURCE_CONFIG = json.dumps(
    STORE_ROCE_RESOURCE_CONFIG,
    separators=(",", ":"),
)
_PROCESS_LIFETIME_OWNERS: list[tuple[Any, ...]] = []


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _wait_json(path: Path, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.05)
    raise TimeoutError(f"timed out waiting for {path}")


def _wait_port(host: str, port: int, process: subprocess.Popen[Any], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"mooncake_master exited with {process.returncode}")
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"mooncake_master did not listen on {host}:{port}")


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _default_host() -> str:
    host = socket.gethostbyname(socket.gethostname())
    if host.startswith("127."):
        raise RuntimeError("could not infer a non-loopback host; pass --host")
    return host


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in (
        "mooncake-transfer-engine-npu",
        "mooncake-transfer-engine",
        "torch",
        "torch-npu",
    ):
        with suppress(importlib.metadata.PackageNotFoundError):
            versions[name] = importlib.metadata.version(name)
    return versions


def _aligned_npu_tensor(torch: Any, nbytes: int) -> tuple[Any, Any]:
    raw = torch.empty(nbytes + ALIGNMENT - 1, dtype=torch.uint8, device="npu")
    offset = (-raw.data_ptr()) % ALIGNMENT
    tensor = raw.narrow(0, offset, nbytes)
    if tensor.data_ptr() % ALIGNMENT:
        raise RuntimeError("failed to allocate a 2 MiB-aligned NPU slab")
    return raw, tensor


def _fill_object(tensor: Any, offset: int, header: int, payload: int) -> None:
    tensor.narrow(0, offset, HEADER_BYTES).fill_(header)
    tensor.narrow(0, offset + HEADER_BYTES, OBJECT_BYTES - HEADER_BYTES).fill_(payload)


def _object_matches(tensor: Any, offset: int, header: int, payload: int) -> bool:
    return bool(
        (tensor.narrow(0, offset, HEADER_BYTES) == header).all().item()
        and (tensor.narrow(0, offset + HEADER_BYTES, OBJECT_BYTES - HEADER_BYTES) == payload)
        .all()
        .item()
    )


def _npu_memory(torch: Any) -> dict[str, int] | dict[str, str]:
    try:
        free_bytes, total_bytes = torch.npu.mem_get_info()
        return {"free_bytes": int(free_bytes), "total_bytes": int(total_bytes)}
    except BaseException as error:
        return {"error": repr(error)}


def _require_result(name: str, result: Any, expected: int | None = None) -> list[int]:
    values = [int(value) for value in result]
    if not values or any(value < 0 for value in values):
        raise RuntimeError(f"{name} failed: {values!r}")
    if expected is not None and any(value != expected for value in values):
        raise RuntimeError(f"{name} returned unexpected byte counts: {values!r}")
    return values


def _worker(args: argparse.Namespace) -> int:
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    before = _npu_memory(torch)

    # Mooncake Store currently treats mere presence of this variable as true,
    # including the text value "0".  Non-Fabric mode therefore requires it to
    # be absent before Store creates its TransferEngine.
    os.environ.pop("ASCEND_ENABLE_USE_FABRIC_MEM", None)
    os.environ.pop("ASCEND_BUFFER_POOL", None)
    if args.engine_topology == "store-owned":
        os.environ["ASCEND_GLOBAL_RESOURCE_CONFIG"] = args.global_resource_config
    else:
        os.environ.pop("ASCEND_GLOBAL_RESOURCE_CONFIG", None)
    os.environ.pop("ASCEND_USE_ASYNC_TRANSFER", None)
    if args.link_mode == "roce":
        os.environ["HCCL_INTRA_PCIE_ENABLE"] = "0"
        os.environ["HCCL_INTRA_ROCE_ENABLE"] = "1"
    else:
        os.environ.pop("HCCL_INTRA_PCIE_ENABLE", None)
        os.environ.pop("HCCL_INTRA_ROCE_ENABLE", None)

    # Match MooncakeBackend._setup_store(): load Store bindings before the
    # process-wide TransferEngine bindings are used.
    from mooncake.store import (
        MooncakeDistributedStore,
        ObjectDataType,
        ReplicateConfig,
    )
    from mooncake.engine import TransferEngine

    ec_engine = TransferEngine()
    init_result = int(ec_engine.initialize(args.host, "P2PHANDSHAKE", "ascend", ""))
    if init_result != 0:
        raise RuntimeError(f"EC TransferEngine.initialize failed: {init_result}")
    ec_endpoint = f"{args.host}:{ec_engine.get_rpc_port()}"
    if args.engine_topology == "two":
        if args.store_buffer_pool:
            os.environ["ASCEND_BUFFER_POOL"] = args.store_buffer_pool
        try:
            engine = TransferEngine()
            init_result = int(
                engine.initialize(args.host, "P2PHANDSHAKE", "ascend", "")
            )
        finally:
            os.environ.pop("ASCEND_BUFFER_POOL", None)
        if init_result != 0:
            raise RuntimeError(
                f"Store TransferEngine.initialize failed: {init_result}"
            )
    elif args.engine_topology == "single":
        engine = ec_engine
    else:
        engine = None
    store_endpoint = (
        args.host
        if args.engine_topology == "store-owned"
        else f"{args.host}:{engine.get_rpc_port()}"
    )

    store = MooncakeDistributedStore()
    setup_signature = (
        str(inspect.signature(store.setup))
        if hasattr(store.setup, "__signature__")
        else None
    )
    try:
        setup_kwargs = {}
        if engine is not None:
            setup_kwargs["engine"] = engine.get_engine()
        setup_result = int(
            store.setup(
                local_hostname=store_endpoint,
                metadata_server="P2PHANDSHAKE",
                global_segment_size=(
                    GLOBAL_SEGMENT_BYTES
                    if args.engine_topology != "store-owned"
                    or args.role == "target"
                    else 0
                ),
                local_buffer_size=(
                    0
                    if args.engine_topology == "store-owned"
                    else LOCAL_BUFFER_BYTES
                ),
                protocol="ascend",
                rdma_devices="",
                master_server_addr=args.master_address,
                **setup_kwargs,
            )
        )
    except TypeError as error:
        raise RuntimeError(
            "installed Mooncake binding does not accept the requested Store "
            f"setup; setup_signature={setup_signature!r}"
        ) from error
    if setup_result != 0:
        raise RuntimeError(f"MooncakeDistributedStore.setup failed: {setup_result}")

    raw, staging = _aligned_npu_tensor(torch, STAGING_BYTES)
    register_calls = {"ec": 0, "store": 0}
    registered = {"ec": False, "store": False}
    keys: list[str] = []
    try:
        register_calls["ec"] += 1
        register_result = int(
            ec_engine.register_memory(staging.data_ptr(), staging.nbytes)
        )
        if register_result != 0:
            raise RuntimeError(
                f"EC engine.register_memory failed: {register_result}"
            )
        registered["ec"] = True
        if args.engine_topology in ("two", "store-owned"):
            register_calls["store"] += 1
            if args.engine_topology == "store-owned":
                register_result = int(
                    store.register_buffer(staging.data_ptr(), staging.nbytes)
                )
            else:
                register_result = int(
                    engine.register_memory(staging.data_ptr(), staging.nbytes)
                )
            if register_result != 0:
                raise RuntimeError(
                    f"Store staging registration failed: {register_result}"
                )
            registered["store"] = True
        after = _npu_memory(torch)

        _atomic_json(
            args.exchange_dir / f"{args.role}-ready.json",
            {
                "ec_endpoint": ec_endpoint,
                "store_endpoint": store_endpoint,
                "ascend_buffer_pool": os.environ.get("ASCEND_BUFFER_POOL"),
                "engine_topology": args.engine_topology,
                "packages": _package_versions(),
                "register_calls": register_calls,
                "registration_api": (
                    "store.register_buffer"
                    if args.engine_topology == "store-owned"
                    else "TransferEngine.register_memory"
                ),
                "setup_signature": setup_signature,
                "staging_address": staging.data_ptr(),
                "staging_bytes": staging.nbytes,
                "npu_memory_before": before,
                "npu_memory_after": after,
            },
        )

        if args.role == "target":
            done = _wait_json(args.exchange_dir / "initiator-done.json", args.timeout)
            torch.npu.synchronize()
            ec_checks = {
                str(item["name"]): _object_matches(
                    staging,
                    int(item["target_offset"]),
                    int(item["header_pattern"]),
                    int(item["payload_pattern"]),
                )
                for item in done["ec_transfers"]
            }
            result = {
                "ec_checks": ec_checks,
                "register_calls": register_calls,
                "success": all(ec_checks.values()),
            }
            _atomic_json(args.exchange_dir / "target-result.json", result)
            print("TARGET_RESULT " + json.dumps(result, sort_keys=True), flush=True)
            initiator_exit = _wait_json(
                args.exchange_dir / "initiator-exited.json",
                args.timeout,
            )
            if int(initiator_exit["returncode"]) != 0:
                raise RuntimeError(
                    "initiator exited unsuccessfully before target cleanup: "
                    f"{initiator_exit!r}"
                )
            return 0 if result["success"] else 1

        target = _wait_json(args.exchange_dir / "target-ready.json", args.timeout)
        replicate = ReplicateConfig()
        replicate.data_type = ObjectDataType.TENSOR
        replicate.replica_num = 1
        if args.engine_topology != "store-owned":
            replicate.preferred_segment = str(target["store_endpoint"])

        transfer_records: list[dict[str, Any]] = []
        expected_patterns: list[tuple[int, int]] = []
        put_results: list[dict[str, Any]] = []
        concurrency_results: list[dict[str, int]] = []
        source_offset = 0
        run_id = uuid.uuid4().hex

        for iteration in range(args.iterations):
            sequential_pattern = 0x10 + iteration
            concurrent_pattern = 0x80 + iteration
            for kind, pattern in (
                ("sequential", sequential_pattern),
                ("concurrent", concurrent_pattern),
            ):
                name = f"{kind}-{iteration}"
                key = f"shared-staging-probe/{run_id}/{name}"
                store_layout = "contiguous-device" if kind == "sequential" else "ec-header-payload-device"
                target_offset = len(transfer_records) * OBJECT_STRIDE
                header_pattern = pattern
                payload_pattern = pattern + 1
                _fill_object(staging, source_offset, header_pattern, payload_pattern)
                torch.npu.synchronize()

                ec_result: list[int] = []
                store_result: list[list[int]] = []
                operation_errors: list[dict[str, str]] = []
                timings: dict[str, int] = {}
                barrier = threading.Barrier(3) if kind == "concurrent" else None

                def ec_write() -> None:
                    try:
                        if barrier is not None:
                            barrier.wait()
                        timings["ec_start_ns"] = time.perf_counter_ns()
                        ec_result.append(
                            int(
                                ec_engine.batch_transfer_sync_write(
                                    str(target["ec_endpoint"]),
                                    [staging.data_ptr() + source_offset],
                                    [int(target["staging_address"]) + target_offset],
                                    [OBJECT_BYTES],
                                )
                            )
                        )
                    except BaseException as error:
                        operation_errors.append({"operation": "ec_write", "error": repr(error)})
                    finally:
                        timings["ec_end_ns"] = time.perf_counter_ns()

                def store_put() -> None:
                    try:
                        if barrier is not None:
                            barrier.wait()
                        timings["store_start_ns"] = time.perf_counter_ns()
                        if store_layout == "contiguous-device":
                            put_addresses = [[staging.data_ptr() + source_offset]]
                            put_sizes = [[OBJECT_BYTES]]
                        else:
                            put_addresses = [[
                                staging.data_ptr() + source_offset,
                                staging.data_ptr() + source_offset + HEADER_BYTES,
                            ]]
                            put_sizes = [[HEADER_BYTES, OBJECT_BYTES - HEADER_BYTES]]
                        store_result.append(
                            _require_result(
                                f"Store PUT {name}",
                                store.batch_put_from_multi_buffers(
                                    [key],
                                    put_addresses,
                                    put_sizes,
                                    replicate,
                                ),
                            )
                        )
                    except BaseException as error:
                        operation_errors.append({"operation": "store_put", "error": repr(error)})
                    finally:
                        timings["store_end_ns"] = time.perf_counter_ns()

                if kind == "concurrent":
                    ec_thread = threading.Thread(target=ec_write, name=f"ec-{iteration}")
                    store_thread = threading.Thread(target=store_put, name=f"store-{iteration}")
                    ec_thread.start()
                    store_thread.start()
                    barrier.wait()
                    ec_thread.join(args.timeout)
                    store_thread.join(args.timeout)
                    if ec_thread.is_alive() or store_thread.is_alive():
                        raise TimeoutError(f"concurrent operations did not finish for iteration {iteration}")
                    overlap_ns = min(timings["ec_end_ns"], timings["store_end_ns"]) - max(
                        timings["ec_start_ns"], timings["store_start_ns"]
                    )
                    concurrency_results.append(
                        {"api_call_overlap_ns": overlap_ns, "iteration": iteration}
                    )
                else:
                    ec_write()
                    store_put()

                if operation_errors:
                    raise RuntimeError(f"{name} operation errors: {operation_errors!r}")
                if ec_result != [0]:
                    raise RuntimeError(f"EC write {name} failed: {ec_result!r}")
                if len(store_result) != 1:
                    raise RuntimeError(f"Store PUT {name} did not return exactly once")

                # Deliberately destroy the source immediately after both APIs
                # return.  A later GET must still reproduce the old pattern.
                _fill_object(staging, source_offset, 0xEE, 0xEE)
                torch.npu.synchronize()
                keys.append(key)
                expected_patterns.append((header_pattern, payload_pattern))
                transfer_records.append(
                    {
                        "header_pattern": header_pattern,
                        "name": name,
                        "payload_pattern": payload_pattern,
                        "store_layout": store_layout,
                        "target_offset": target_offset,
                    }
                )
                put_results.append({"name": name, "result": store_result[0]})

        if not any(item["api_call_overlap_ns"] > 0 for item in concurrency_results):
            raise RuntimeError(
                "no EC/Store API-call overlap was observed in any concurrent iteration: "
                f"{concurrency_results!r}"
            )

        existence = [int(value) for value in store.batch_is_exist(keys)]
        if existence != [1] * len(keys):
            raise RuntimeError(f"Store existence mismatch: {existence!r}")

        replica_endpoints: dict[str, list[str]] = {}
        for key in keys:
            descriptors = store.get_replica_desc(key)
            endpoints = [
                str(descriptor.get_memory_descriptor().buffer_descriptor.transport_endpoint)
                for descriptor in descriptors
                if descriptor.is_memory_replica()
            ]
            replica_endpoints[key] = endpoints
            target_store_endpoint = str(target["store_endpoint"])
            if args.engine_topology == "store-owned":
                # Store chooses its native TransferEngine port after setup, so
                # the descriptor is host:port while local_hostname is only the
                # host.  Only the target contributes a global segment here.
                placed_on_target = len(endpoints) == 1 and (
                    endpoints[0] == target_store_endpoint
                    or endpoints[0].startswith(f"{target_store_endpoint}:")
                )
            else:
                placed_on_target = endpoints == [target_store_endpoint]
            if not placed_on_target:
                raise RuntimeError(
                    f"Store object {key!r} was not allocated in the requested "
                    f"remote Host-DRAM segment: {endpoints!r}"
                )

        destination_base = OBJECT_STRIDE
        destinations = [destination_base + index * OBJECT_STRIDE for index in range(len(keys))]
        if destinations[-1] + OBJECT_BYTES > staging.nbytes:
            raise RuntimeError("configured iterations exceed the staging slab")
        for offset in destinations:
            _fill_object(staging, offset, 0, 0)
        torch.npu.synchronize()

        get_results = _require_result(
            "batched Store GET",
            store.batch_get_into_multi_buffers(
                keys,
                [
                    [staging.data_ptr() + offset]
                    if record["store_layout"] == "contiguous-device"
                    else [staging.data_ptr() + offset, staging.data_ptr() + offset + HEADER_BYTES]
                    for offset, record in zip(destinations, transfer_records, strict=True)
                ],
                [
                    [OBJECT_BYTES]
                    if record["store_layout"] == "contiguous-device"
                    else [HEADER_BYTES, OBJECT_BYTES - HEADER_BYTES]
                    for record in transfer_records
                ],
            ),
            expected=OBJECT_BYTES,
        )
        torch.npu.synchronize()
        get_checks = {
            key: _object_matches(staging, offset, patterns[0], patterns[1])
            for key, offset, patterns in zip(keys, destinations, expected_patterns, strict=True)
        }

        initiator = {
            "batched_get_checks": get_checks,
            "batched_get_results": get_results,
            "concurrency_results": concurrency_results,
            "ec_transfers": transfer_records,
            "existence": existence,
            "put_results": put_results,
            "register_calls": register_calls,
            "replica_endpoints": replica_endpoints,
            "remote_store_segment": target["store_endpoint"],
            "success": all(get_checks.values()),
        }
        _atomic_json(args.exchange_dir / "initiator-done.json", initiator)
        target_result = _wait_json(args.exchange_dir / "target-result.json", args.timeout)
        initiator["target_result"] = target_result
        initiator["success"] = bool(initiator["success"] and target_result["success"])
        _atomic_json(args.exchange_dir / "initiator-result.json", initiator)
        print("INITIATOR_RESULT " + json.dumps(initiator, sort_keys=True), flush=True)
        return 0 if initiator["success"] else 1
    finally:
        if args.role == "initiator":
            for key in keys:
                with suppress(BaseException):
                    store.remove(key, True)
        if args.cleanup_mode == "process-lifetime":
            _PROCESS_LIFETIME_OWNERS.append(
                (store, ec_engine, engine, raw, staging)
            )
            print(
                "RETAIN_PROCESS_LIFETIME "
                + json.dumps(
                    {
                        "address": f"0x{staging.data_ptr():x}",
                        "engine_topology": args.engine_topology,
                        "register_calls": register_calls,
                        "role": args.role,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        else:
            try:
                cleanup = (
                    (("store", engine), ("ec", ec_engine))
                    if args.cleanup_order == "store-first"
                    else (("ec", ec_engine), ("store", engine))
                )
                for owner, owner_engine in cleanup:
                    if not registered[owner]:
                        continue
                    try:
                        if (
                            owner == "store"
                            and args.engine_topology == "store-owned"
                        ):
                            unregister_result = int(
                                store.unregister_buffer(staging.data_ptr())
                            )
                        else:
                            unregister_result = int(
                                owner_engine.unregister_memory(staging.data_ptr())
                            )
                        unregister_error = None
                    except BaseException as error:
                        unregister_result = None
                        unregister_error = repr(error)
                    print(
                        "UNREGISTER "
                        + json.dumps(
                            {
                                "address": f"0x{staging.data_ptr():x}",
                                "cleanup_order": args.cleanup_order,
                                "exception": unregister_error,
                                "owner": owner,
                                "register_calls": register_calls[owner],
                                "result": unregister_result,
                                "role": args.role,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            finally:
                store.close()
        _ = raw


def _start_worker(
    args: argparse.Namespace,
    role: str,
    npu: str,
) -> tuple[subprocess.Popen[Any], Any, Path]:
    log_path = args.output_dir / f"{role}.log"
    log = log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = npu
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("ASCEND_ENABLE_USE_FABRIC_MEM", None)
    env.pop("ASCEND_BUFFER_POOL", None)
    env.pop("ASCEND_GLOBAL_RESOURCE_CONFIG", None)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--role",
        role,
        "--host",
        args.host,
        "--master-address",
        args.master_address,
        "--exchange-dir",
        str(args.output_dir),
        "--timeout",
        str(args.timeout),
        "--iterations",
        str(args.iterations),
        "--link-mode",
        args.link_mode,
        "--engine-topology",
        args.engine_topology,
        "--cleanup-order",
        args.cleanup_order,
        "--cleanup-mode",
        args.cleanup_mode,
        "--global-resource-config",
        args.global_resource_config,
    ]
    if args.store_buffer_pool:
        command.extend(("--store-buffer-pool", args.store_buffer_pool))
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    return process, log, log_path


def _terminate(processes: list[subprocess.Popen[Any]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def _run(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise RuntimeError(f"output directory must be empty: {args.output_dir}")
    master_process: subprocess.Popen[Any] | None = None
    master_log = None
    processes: list[subprocess.Popen[Any]] = []
    worker_logs: list[Any] = []
    try:
        if args.master_address is None:
            executable = shutil.which("mooncake_master")
            if executable is None:
                raise RuntimeError("mooncake_master is not installed or not on PATH")
            port = _free_port()
            args.master_address = f"127.0.0.1:{port}"
            master_log = (args.output_dir / "mooncake_master.log").open("w", encoding="utf-8")
            master_process = subprocess.Popen(
                [executable, "--port", str(port)],
                stdout=master_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            _wait_port("127.0.0.1", port, master_process, min(args.timeout, 30.0))

        print(
            "CONFIG "
            + json.dumps(
                {
                    "cleanup_mode": args.cleanup_mode,
                    "cleanup_order": args.cleanup_order,
                    "engine_topology": args.engine_topology,
                    "fabric_mem": False,
                    "global_segment_bytes_per_process": GLOBAL_SEGMENT_BYTES,
                    "host": args.host,
                    "iterations": args.iterations,
                    "link_mode": args.link_mode,
                    "local_buffer_bytes_per_process": LOCAL_BUFFER_BYTES,
                    "master_address": args.master_address,
                    "npu_a": args.npu_a,
                    "npu_b": args.npu_b,
                    "object_header_bytes": HEADER_BYTES,
                    "object_bytes": OBJECT_BYTES,
                    "object_stride": OBJECT_STRIDE,
                    "registration_owner": {
                        "single": "TransferEngine-global_te-style",
                        "two": "same-slab-two-explicit-engines",
                        "store-owned": "EC-engine-plus-store.register_buffer",
                    }[args.engine_topology],
                    "staging_bytes_per_process": STAGING_BYTES,
                    "store_buffer_pool": args.store_buffer_pool,
                },
                sort_keys=True,
            ),
            flush=True,
        )

        target_process, target_log, target_path = _start_worker(args, "target", args.npu_b)
        processes.append(target_process)
        worker_logs.append(target_log)
        deadline = time.monotonic() + args.timeout
        while not (args.output_dir / "target-ready.json").exists():
            if target_process.poll() is not None:
                raise RuntimeError(
                    f"target exited before readiness with {target_process.returncode}:\n"
                    + target_path.read_text(errors="replace")
                )
            if time.monotonic() >= deadline:
                raise TimeoutError("target did not become ready")
            time.sleep(0.05)

        initiator_process, initiator_log, initiator_path = _start_worker(args, "initiator", args.npu_a)
        processes.append(initiator_process)
        worker_logs.append(initiator_log)
        initiator_exit_announced = False
        while any(process.poll() is None for process in processes):
            failed = [process for process in processes if process.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(
                    "worker failed:\n"
                    + "\n".join(
                        f"[{path.name}]\n{path.read_text(errors='replace')}"
                        for path in (target_path, initiator_path)
                        if path.exists()
                    )
                )
            if initiator_process.poll() == 0 and not initiator_exit_announced:
                _atomic_json(
                    args.output_dir / "initiator-exited.json",
                    {"returncode": 0},
                )
                initiator_exit_announced = True
            if time.monotonic() >= deadline:
                raise TimeoutError("workers exceeded the probe timeout")
            time.sleep(0.1)

        initiator_result = _wait_json(args.output_dir / "initiator-result.json", 2.0)
        worker_readiness = {
            role: _wait_json(args.output_dir / f"{role}-ready.json", 2.0)
            for role in ("target", "initiator")
        }
        unregister_results: dict[str, dict[str, dict[str, Any]]] = {}
        lifetime_retained: dict[str, bool] = {}
        unsafe_exit_markers: dict[str, list[str]] = {}
        for role in ("target", "initiator"):
            log_text = (args.output_dir / f"{role}.log").read_text(errors="replace")
            records = [
                json.loads(line.removeprefix("UNREGISTER "))
                for line in log_text.splitlines()
                if line.startswith("UNREGISTER ")
            ]
            unregister_results[role] = {
                str(record["owner"]): record for record in records
            }
            lifetime_retained[role] = any(
                line.startswith("RETAIN_PROCESS_LIFETIME ")
                for line in log_text.splitlines()
            )
            lowered = log_text.lower()
            unsafe_exit_markers[role] = [
                marker
                for marker in (
                    "103900",
                    "invalid memhandle",
                    "segmentation fault",
                )
                if marker in lowered
            ]

        buffer_pool_evidence = {}
        for role in ("target", "initiator"):
            log_text = (args.output_dir / f"{role}.log").read_text(errors="replace")
            buffer_pool_evidence[role] = any(
                marker in log_text
                for marker in ("Set adxl.BufferPool", "when buffer pool is enabled")
            )

        if args.cleanup_mode == "process-lifetime":
            cleanup_success = (
                all(lifetime_retained.values())
                and all(not records for records in unregister_results.values())
                and all(not markers for markers in unsafe_exit_markers.values())
            )
        else:
            cleanup_success = all(
                owners.get("ec", {}).get("result") == 0
                and owners.get("ec", {}).get("exception") is None
                and (
                    args.engine_topology == "single"
                    or (
                        owners.get("store", {}).get("result") == 0
                        and owners.get("store", {}).get("exception") is None
                    )
                )
                for owners in unregister_results.values()
            )

        summary = {
            "buffer_pool_evidence": buffer_pool_evidence,
            "cleanup_mode": args.cleanup_mode,
            "cleanup_success": cleanup_success,
            "initiator_result": initiator_result,
            "lifetime_retained": lifetime_retained,
            "output_dir": str(args.output_dir),
            "process_returncodes": {
                "target": target_process.returncode,
                "initiator": initiator_process.returncode,
            },
            "success": (
                initiator_result["success"]
                and all(process.returncode == 0 for process in processes)
                and cleanup_success
                and (
                    bool(args.store_buffer_pool)
                    or not any(buffer_pool_evidence.values())
                )
            ),
            "unregister_results": unregister_results,
            "unsafe_exit_markers": unsafe_exit_markers,
            "worker_readiness": worker_readiness,
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
        return 0 if summary["success"] else 1
    except BaseException as error:
        failure = {"error": repr(error), "output_dir": str(args.output_dir), "success": False}
        (args.output_dir / "summary.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("SUMMARY " + json.dumps(failure, sort_keys=True), flush=True)
        return 1
    finally:
        _terminate(processes)
        for log in worker_logs:
            log.close()
        if master_process is not None:
            with suppress(ProcessLookupError):
                os.killpg(master_process.pid, signal.SIGTERM)
            try:
                master_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(master_process.pid, signal.SIGKILL)
                master_process.wait()
        if master_log is not None:
            master_log.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=None)
    parser.add_argument("--npu-a", default="0")
    parser.add_argument("--npu-b", default="1")
    parser.add_argument("--master-address")
    parser.add_argument("--link-mode", choices=("hccs", "roce"), default="hccs")
    parser.add_argument(
        "--engine-topology",
        choices=("single", "two", "store-owned"),
        default="single",
    )
    parser.add_argument(
        "--cleanup-mode",
        choices=("explicit", "process-lifetime"),
        default="explicit",
    )
    parser.add_argument(
        "--cleanup-order",
        choices=("store-first", "ec-first"),
        default="store-first",
    )
    parser.add_argument("--store-buffer-pool")
    parser.add_argument(
        "--global-resource-config",
        default=DEFAULT_STORE_ROCE_RESOURCE_CONFIG,
    )
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--role", choices=("initiator", "target"), help=argparse.SUPPRESS)
    parser.add_argument("--exchange-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.iterations <= 7:
        parser.error("--iterations must be between 1 and 7 for the 64 MiB slab")
    try:
        global_resource_config = json.loads(args.global_resource_config)
    except json.JSONDecodeError as error:
        parser.error(f"--global-resource-config must be valid JSON: {error}")
    if not isinstance(global_resource_config, dict):
        parser.error("--global-resource-config must be a JSON object")
    if args.engine_topology == "store-owned" and args.link_mode == "roce":
        store_protocols = (
            global_resource_config.get("store", {})
            .get("comm_resource_config", {})
            .get("protocol_desc")
        )
        if not isinstance(store_protocols, list) or "roce:device" not in store_protocols:
            parser.error(
                "store-owned RoCE requires "
                "store.comm_resource_config.protocol_desc to include roce:device; "
                "an empty config lets the native Store engine select FabricMem"
            )
    if args.engine_topology == "store-owned" and args.store_buffer_pool:
        parser.error("store-owned topology does not support --store-buffer-pool")
    if (
        args.cleanup_mode == "process-lifetime"
        and args.engine_topology != "store-owned"
    ):
        parser.error(
            "process-lifetime cleanup requires --engine-topology store-owned"
        )
    args.host = args.host or _default_host()
    if args.worker:
        if args.role is None or args.exchange_dir is None or args.master_address is None:
            parser.error("worker mode requires --role, --exchange-dir, and --master-address")
        args.output_dir = args.exchange_dir
    elif args.output_dir is None:
        args.output_dir = Path(tempfile.mkdtemp(prefix="shared-staging-store-probe-", dir="/tmp"))
    return args


def main() -> int:
    args = _parse_args()
    return _worker(args) if args.worker else _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
