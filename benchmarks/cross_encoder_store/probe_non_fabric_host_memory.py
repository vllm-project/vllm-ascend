#!/usr/bin/env python3
"""Probe non-FabricMem Ascend writes into a remote pinned Host range.

This is deliberately Store-free: two short-lived processes create ordinary
Mooncake TransferEngines on separate NPUs, reproduce the registration order
seen in the failing cross-encoder run, and write into the peer's 1 GiB pinned
Host range. It does not start vLLM, a model, Mooncake Store, or an E2E test.

Run in the same 910C container and source overlay as the failed case:

    python3 benchmarks/cross_encoder_store/probe_non_fabric_host_memory.py \
        --host 172.61.8.214 --npu-a 0 --npu-b 1 --link-mode roce

The default cases use a fresh pair of engines each time:

* header-only: one 24-byte pinned-Host source fragment;
* store-shape: header plus a 2 MiB device prefix and the observed
  1,475,072-byte aligned device suffix.
* device-to-host: one aligned 2 MiB device source into remote pinned Host;
* host-to-device-read: one remote pinned-Host source read into local device.
* device-store-shape: header plus prefix coalesced in an aligned device
  bounce, followed by an aligned device suffix.
* dual-engine-registration: try to register an EC-owned device subrange in
  the Store engine while the EC registration is still live.
* registration-handoff: unregister the EC owner first, then register the same
  device subrange in the Store engine.
* dual-engine-long-lived-staging: keep the same complete aligned producer slab
  registered in both engines, then repeatedly issue concurrent EC-to-device
  and Store-shaped-to-Host writes from the same immutable source region.

Run only the long-lived dual-engine case over RoCE:

    python3 benchmarks/cross_encoder_store/probe_non_fabric_host_memory.py \
        --host 172.61.8.214 --npu-a 0 --npu-b 1 --link-mode roce \
        --engine-topology two --cases dual-engine-long-lived-staging \
        --iterations 100 --cleanup-order store-first

Both target the beginning of the peer's registered 1 GiB pinned Host range.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

ALIGNMENT = 2 * 1024**2
PRODUCER_BYTES = 136_314_880
LOCAL_HOST_BYTES = 512 * 1024**2
GLOBAL_HOST_BYTES = 1024**3
HEADER_BYTES = 24
PREFIX_BYTES = ALIGNMENT
SUFFIX_BYTES = 1_475_072
DEFAULT_TIMEOUT_SECONDS = 120.0
LONG_LIVED_TRANSFER_BYTES = 4 * 1024**2
DEFAULT_LONG_LIVED_ITERATIONS = 32


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


def _aligned_npu_tensor(torch: Any, nbytes: int) -> tuple[Any, Any]:
    raw = torch.empty(
        nbytes + ALIGNMENT - 1,
        dtype=torch.uint8,
        device="npu",
    )
    offset = (-raw.data_ptr()) % ALIGNMENT
    tensor = raw.narrow(0, offset, nbytes)
    if tensor.data_ptr() % ALIGNMENT:
        raise RuntimeError("failed to allocate a 2 MiB-aligned NPU range")
    return raw, tensor


def _register(engine: Any, name: str, tensor: Any) -> None:
    result = engine.register_memory(tensor.data_ptr(), tensor.nbytes)
    print(
        f"REGISTER name={name} address=0x{tensor.data_ptr():x} "
        f"bytes={tensor.nbytes} result={result}",
        flush=True,
    )
    if result != 0:
        raise RuntimeError(f"registration failed for {name}: {result}")


def _unregister(engine: Any, name: str, tensor: Any) -> int:
    result = int(engine.unregister_memory(tensor.data_ptr()))
    print(
        f"UNREGISTER name={name} address=0x{tensor.data_ptr():x} "
        f"result={result}",
        flush=True,
    )
    return result


def _try_register(engine: Any, name: str, tensor: Any) -> dict[str, Any]:
    try:
        result = engine.register_memory(tensor.data_ptr(), tensor.nbytes)
    except BaseException as error:
        observation = {
            "exception": repr(error),
            "result": None,
            "succeeded": False,
        }
    else:
        observation = {
            "exception": None,
            "result": int(result),
            "succeeded": result == 0,
        }
    print(
        "REGISTER_PROBE "
        + json.dumps(
            {
                "address": f"0x{tensor.data_ptr():x}",
                "bytes": tensor.nbytes,
                "name": name,
                **observation,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return observation


def _verify_bytes(tensor: Any, offset: int, length: int, expected: int) -> bool:
    view = tensor.narrow(0, offset, length)
    return bool((view == expected).all().item())


def _print_npu_memory(torch: Any, stage: str) -> None:
    try:
        free_bytes, total_bytes = torch.npu.mem_get_info()
    except BaseException as error:
        print(f"NPU_MEMORY stage={stage} unavailable={error!r}", flush=True)
        return
    print(
        f"NPU_MEMORY stage={stage} free_bytes={free_bytes} "
        f"total_bytes={total_bytes}",
        flush=True,
    )


def _worker(args: argparse.Namespace) -> int:
    import torch
    import torch_npu  # noqa: F401
    from mooncake.engine import TransferEngine

    torch.npu.set_device(0)
    _print_npu_memory(torch, "before-engines")

    # Engine initialization captures ASCEND_BUFFER_POOL. Keep it absent from
    # the EC engine; when requested, expose it only while constructing the
    # second engine. Two direct engines can also be tested without it.
    if args.engine_topology == "single" and args.store_buffer_pool:
        os.environ["ASCEND_BUFFER_POOL"] = args.store_buffer_pool
    else:
        os.environ.pop("ASCEND_BUFFER_POOL", None)
    ec_engine = TransferEngine()
    result = ec_engine.initialize(args.host, "P2PHANDSHAKE", "ascend", "")
    if result != 0:
        raise RuntimeError(f"EC TransferEngine initialization failed: {result}")
    _print_npu_memory(torch, "after-ec-engine")

    if args.engine_topology == "two":
        if args.store_buffer_pool:
            os.environ["ASCEND_BUFFER_POOL"] = args.store_buffer_pool
        else:
            os.environ.pop("ASCEND_BUFFER_POOL", None)
        try:
            engine = TransferEngine()
            result = engine.initialize(args.host, "P2PHANDSHAKE", "ascend", "")
        finally:
            os.environ.pop("ASCEND_BUFFER_POOL", None)
        if result != 0:
            raise RuntimeError(f"Store TransferEngine initialization failed: {result}")
        _print_npu_memory(torch, "after-store-engine")
    else:
        engine = ec_engine

    ec_endpoint = f"{args.host}:{ec_engine.get_rpc_port()}"
    endpoint = f"{args.host}:{engine.get_rpc_port()}"
    print(
        f"ENGINE role={args.role} topology={args.engine_topology} "
        f"ec_endpoint={ec_endpoint} store_endpoint={endpoint} "
        f"store_buffer_pool={args.store_buffer_pool or '0:0'}",
        flush=True,
    )

    producer_raw, producer = _aligned_npu_tensor(torch, PRODUCER_BYTES)
    local_host = torch.empty(
        LOCAL_HOST_BYTES,
        dtype=torch.uint8,
        device="cpu",
        pin_memory=True,
    )
    global_host = torch.empty(
        GLOBAL_HOST_BYTES,
        dtype=torch.uint8,
        device="cpu",
        pin_memory=True,
    )
    header = None
    store_prefix_raw = None
    store_prefix = None
    suffix_raw = None
    suffix = None
    registered: list[tuple[str, Any]] = []
    ec_registered: list[tuple[str, Any]] = []

    try:
        # Keep the EC slab owned only by the EC engine. Store Host ranges and
        # its temporary device range belong only to the Store engine.
        _register(ec_engine, "producer", producer)
        ec_registered.append(("producer", producer))
        for name, tensor in (
            ("store-local", local_host),
            ("store-global", global_host),
        ):
            _register(engine, name, tensor)
            registered.append((name, tensor))

        global_host[: HEADER_BYTES + PREFIX_BYTES + SUFFIX_BYTES].zero_()
        if args.case in ("dual-engine-registration", "registration-handoff"):
            if args.engine_topology != "two":
                raise RuntimeError(f"{args.case} requires --engine-topology two")
            _atomic_json(
                args.exchange_dir / f"{args.case}-{args.role}-ready.json",
                {
                    "endpoint": endpoint,
                    "global_address": global_host.data_ptr(),
                    "global_bytes": global_host.nbytes,
                },
            )
            overlap = producer.narrow(0, 0, SUFFIX_BYTES)
            if args.role == "initiator":
                if args.case == "registration-handoff":
                    _unregister(ec_engine, "producer-before-handoff", producer)
                    ec_registered.clear()
                observation = _try_register(engine, args.case, overlap)
                if observation["succeeded"]:
                    _unregister(engine, args.case, overlap)
                _atomic_json(
                    args.exchange_dir / f"{args.case}-done.json",
                    {"registration": observation},
                )
            else:
                done = _wait_json(
                    args.exchange_dir / f"{args.case}-done.json",
                    args.timeout,
                )
                observation = done["registration"]
                result = {
                    "case": args.case,
                    "registration": observation,
                    "success": (
                        observation["succeeded"]
                        if args.case == "registration-handoff"
                        else True
                    ),
                }
                _atomic_json(
                    args.exchange_dir / f"{args.case}-result.json",
                    result,
                )
                print("RESULT " + json.dumps(result, sort_keys=True), flush=True)
            if args.role == "initiator":
                result = _wait_json(
                    args.exchange_dir / f"{args.case}-result.json",
                    args.timeout,
                )
                return 0 if result["success"] else 1
            return 0 if result["success"] else 1

        if args.case == "dual-engine-long-lived-staging":
            if args.engine_topology != "two":
                raise RuntimeError(
                    "dual-engine-long-lived-staging requires --engine-topology two"
                )

            # The exact same complete, aligned slab stays registered in both
            # engines for every iteration. Inner transfer regions are reused;
            # neither registration is released between waves.
            _register(engine, "producer-store-long-lived", producer)
            registered.append(("producer-store-long-lived", producer))
            producer[:LONG_LIVED_TRANSFER_BYTES].zero_()
            global_host[:LONG_LIVED_TRANSFER_BYTES].zero_()
            torch.npu.synchronize()
            _atomic_json(
                args.exchange_dir / f"{args.case}-{args.role}-ready.json",
                {
                    "ec_endpoint": ec_endpoint,
                    "store_endpoint": endpoint,
                    "global_address": global_host.data_ptr(),
                    "producer_address": producer.data_ptr(),
                    "producer_bytes": producer.nbytes,
                },
            )

            if args.role == "target":
                iteration_checks: list[dict[str, Any]] = []
                for iteration in range(args.iterations):
                    submitted = _wait_json(
                        args.exchange_dir
                        / f"{args.case}-iteration-{iteration}-submitted.json",
                        args.timeout,
                    )
                    torch.npu.synchronize()
                    pattern = int(submitted["pattern"])
                    checks = {
                        "ec_device": _verify_bytes(
                            producer,
                            0,
                            LONG_LIVED_TRANSFER_BYTES,
                            pattern,
                        ),
                        "store_host": _verify_bytes(
                            global_host,
                            0,
                            LONG_LIVED_TRANSFER_BYTES,
                            pattern,
                        ),
                    }
                    record = {
                        "checks": checks,
                        "iteration": iteration,
                        "pattern": pattern,
                        "success": all(checks.values()),
                    }
                    iteration_checks.append(record)
                    _atomic_json(
                        args.exchange_dir
                        / f"{args.case}-iteration-{iteration}-checked.json",
                        record,
                    )

                result = {
                    "case": args.case,
                    "iterations": iteration_checks,
                    "success": all(item["success"] for item in iteration_checks),
                }
                _atomic_json(
                    args.exchange_dir / f"{args.case}-result.json",
                    result,
                )
                print("RESULT " + json.dumps(result, sort_keys=True), flush=True)
                return 0 if result["success"] else 1

            target = _wait_json(
                args.exchange_dir / f"{args.case}-target-ready.json",
                args.timeout,
            )
            submissions: list[dict[str, Any]] = []
            for iteration in range(args.iterations):
                pattern = 1 + iteration % 254
                producer[:LONG_LIVED_TRANSFER_BYTES].fill_(pattern)
                torch.npu.synchronize()

                barrier = threading.Barrier(3)
                operation_results: dict[str, int] = {}
                operation_errors: dict[str, str] = {}
                timings: dict[str, int] = {}

                def transfer(
                    name: str,
                    transfer_engine: Any,
                    target_endpoint: str,
                    target_address: int,
                ) -> None:
                    try:
                        barrier.wait()
                        timings[f"{name}_start_ns"] = time.perf_counter_ns()
                        operation_results[name] = int(
                            transfer_engine.batch_transfer_sync_write(
                                target_endpoint,
                                [producer.data_ptr()],
                                [target_address],
                                [LONG_LIVED_TRANSFER_BYTES],
                            )
                        )
                    except BaseException as error:
                        operation_errors[name] = repr(error)
                    finally:
                        timings[f"{name}_end_ns"] = time.perf_counter_ns()

                ec_thread = threading.Thread(
                    target=transfer,
                    args=(
                        "ec",
                        ec_engine,
                        str(target["ec_endpoint"]),
                        int(target["producer_address"]),
                    ),
                    name=f"dual-engine-ec-{iteration}",
                )
                store_thread = threading.Thread(
                    target=transfer,
                    args=(
                        "store",
                        engine,
                        str(target["store_endpoint"]),
                        int(target["global_address"]),
                    ),
                    name=f"dual-engine-store-{iteration}",
                )
                ec_thread.start()
                store_thread.start()
                barrier.wait()
                ec_thread.join(args.timeout)
                store_thread.join(args.timeout)
                if ec_thread.is_alive() or store_thread.is_alive():
                    raise TimeoutError(
                        f"dual-engine transfers did not settle in iteration {iteration}"
                    )
                if operation_errors:
                    raise RuntimeError(
                        f"dual-engine transfer errors in iteration {iteration}: "
                        f"{operation_errors!r}"
                    )
                if operation_results != {"ec": 0, "store": 0}:
                    raise RuntimeError(
                        f"dual-engine transfer results in iteration {iteration}: "
                        f"{operation_results!r}"
                    )

                overlap_ns = min(
                    timings["ec_end_ns"], timings["store_end_ns"]
                ) - max(timings["ec_start_ns"], timings["store_start_ns"])
                submitted = {
                    "api_call_overlap_ns": overlap_ns,
                    "iteration": iteration,
                    "pattern": pattern,
                    "results": operation_results,
                }
                submissions.append(submitted)
                _atomic_json(
                    args.exchange_dir
                    / f"{args.case}-iteration-{iteration}-submitted.json",
                    submitted,
                )
                checked = _wait_json(
                    args.exchange_dir
                    / f"{args.case}-iteration-{iteration}-checked.json",
                    args.timeout,
                )
                if not checked["success"]:
                    raise RuntimeError(
                        f"target verification failed in iteration {iteration}: "
                        f"{checked!r}"
                    )

            result = _wait_json(
                args.exchange_dir / f"{args.case}-result.json",
                args.timeout,
            )
            result["submissions"] = submissions
            result["all_api_calls_overlapped"] = all(
                item["api_call_overlap_ns"] > 0 for item in submissions
            )
            _atomic_json(
                args.exchange_dir / f"{args.case}-initiator-observations.json",
                {
                    "all_api_calls_overlapped": result[
                        "all_api_calls_overlapped"
                    ],
                    "submissions": submissions,
                },
            )
            print("INITIATOR_RESULT " + json.dumps(result, sort_keys=True), flush=True)
            return 0 if result["success"] else 1

        if args.role == "target" and args.case == "host-to-device-read":
            global_host[:SUFFIX_BYTES].fill_(0x55)
        if args.role == "target" and args.case == "device-to-device":
            producer[:SUFFIX_BYTES].zero_()
            torch.npu.synchronize()

        if args.role == "initiator":
            header = torch.empty(
                HEADER_BYTES,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            )
            suffix_raw, suffix = _aligned_npu_tensor(torch, SUFFIX_BYTES)
            store_prefix_raw, store_prefix = _aligned_npu_tensor(
                torch,
                PREFIX_BYTES + HEADER_BYTES,
            )
            _register(engine, "header", header)
            registered.append(("header", header))
            _register(engine, "suffix", suffix)
            registered.append(("suffix", suffix))
            _register(engine, "store-prefix", store_prefix)
            registered.append(("store-prefix", store_prefix))

            header.fill_(0x11)
            producer.narrow(0, PRODUCER_BYTES - PREFIX_BYTES, PREFIX_BYTES).fill_(0x22)
            producer.narrow(0, 0, PREFIX_BYTES).fill_(0x44)
            if args.case == "device-to-device":
                producer.narrow(0, 0, SUFFIX_BYTES).fill_(0x66)
            if args.case == "device-store-shape":
                store_prefix.narrow(0, 0, HEADER_BYTES).fill_(0x11)
                store_prefix.narrow(0, HEADER_BYTES, PREFIX_BYTES).fill_(0x22)
            else:
                store_prefix.fill_(0x22)
            suffix.fill_(0x44 if args.case == "device-to-host" else 0x33)
            torch.npu.synchronize()

        ready_path = args.exchange_dir / f"{args.case}-{args.role}-ready.json"
        _atomic_json(
            ready_path,
            {
                "endpoint": endpoint,
                "global_address": global_host.data_ptr(),
                "global_bytes": global_host.nbytes,
                "producer_address": producer.data_ptr(),
            },
        )

        done_path = args.exchange_dir / f"{args.case}-done.json"
        if args.role == "target":
            done = _wait_json(done_path, args.timeout)
            transfer_result = int(done["transfer_result"])
            if args.case == "host-to-device-read":
                checks = {"payload": bool(done.get("payload_check", False))}
            elif args.case == "device-to-host":
                checks = {
                    "payload": _verify_bytes(
                        global_host,
                        0,
                        SUFFIX_BYTES,
                        0x44,
                    )
                }
            elif args.case == "device-to-device":
                checks = {
                    "payload": _verify_bytes(
                        producer,
                        0,
                        SUFFIX_BYTES,
                        0x66,
                    )
                }
            else:
                checks = {"header": _verify_bytes(global_host, 0, HEADER_BYTES, 0x11)}
            if args.case in ("store-shape", "device-store-shape"):
                checks["prefix"] = _verify_bytes(
                    global_host,
                    HEADER_BYTES,
                    PREFIX_BYTES,
                    0x22,
                )
                checks["suffix"] = _verify_bytes(
                    global_host,
                    HEADER_BYTES + PREFIX_BYTES,
                    SUFFIX_BYTES,
                    0x33,
                )
            success = transfer_result == 0 and all(checks.values())
            result_path = args.exchange_dir / f"{args.case}-result.json"
            _atomic_json(
                result_path,
                {
                    "case": args.case,
                    "checks": checks,
                    "success": success,
                    "transfer_result": transfer_result,
                },
            )
            print(
                "RESULT "
                + json.dumps(
                    {
                        "case": args.case,
                        "checks": checks,
                        "success": success,
                        "transfer_result": transfer_result,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            _print_npu_memory(torch, "after-transfer")
            return 0 if success else 1

        target = _wait_json(
            args.exchange_dir / f"{args.case}-target-ready.json",
            args.timeout,
        )
        target_endpoint = str(target["endpoint"])
        target_global = int(target["global_address"])
        if args.case == "header-only":
            source_addresses = [header.data_ptr()]
            target_addresses = [target_global]
            lengths = [HEADER_BYTES]
        elif args.case == "device-to-host":
            assert suffix is not None
            source_addresses = [suffix.data_ptr()]
            target_addresses = [target_global]
            lengths = [SUFFIX_BYTES]
        elif args.case == "device-to-device":
            source_addresses = [producer.data_ptr()]
            target_addresses = [int(target["producer_address"])]
            lengths = [SUFFIX_BYTES]
        elif args.case == "host-to-device-read":
            assert suffix is not None
            # The binding names the first list `buffers` for both opcodes.
            # For READ it is the local destination; peer addresses stay in
            # the second list as TransferRequest.target_offset.
            source_addresses = [suffix.data_ptr()]
            target_addresses = [target_global]
            lengths = [SUFFIX_BYTES]
        elif args.case == "device-store-shape":
            assert suffix is not None
            assert store_prefix is not None
            source_addresses = [store_prefix.data_ptr(), suffix.data_ptr()]
            target_addresses = [
                target_global,
                target_global + HEADER_BYTES + PREFIX_BYTES,
            ]
            lengths = [HEADER_BYTES + PREFIX_BYTES, SUFFIX_BYTES]
        else:
            assert suffix is not None
            assert store_prefix is not None
            source_addresses = [
                header.data_ptr(),
                store_prefix.data_ptr(),
                suffix.data_ptr(),
            ]
            target_addresses = [
                target_global,
                target_global + HEADER_BYTES,
                target_global + HEADER_BYTES + PREFIX_BYTES,
            ]
            lengths = [HEADER_BYTES, PREFIX_BYTES, SUFFIX_BYTES]

        print(
            "TRANSFER "
            + json.dumps(
                {
                    "case": args.case,
                    "lengths": lengths,
                    "source_addresses": [f"0x{value:x}" for value in source_addresses],
                    "target_addresses": [f"0x{value:x}" for value in target_addresses],
                    "target_endpoint": target_endpoint,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if args.case == "host-to-device-read":
            transfer_result = engine.batch_transfer_sync_read(
                target_endpoint,
                source_addresses,
                target_addresses,
                lengths,
            )
        else:
            transfer_result = engine.batch_transfer_sync_write(
                target_endpoint,
                source_addresses,
                target_addresses,
                lengths,
            )
        print(f"TRANSFER_RESULT case={args.case} result={transfer_result}", flush=True)
        _print_npu_memory(torch, "after-transfer")
        done = {"transfer_result": transfer_result}
        if args.case == "host-to-device-read":
            torch.npu.synchronize()
            done["payload_check"] = _verify_bytes(suffix, 0, SUFFIX_BYTES, 0x55)
        _atomic_json(done_path, done)
        result_data = _wait_json(
            args.exchange_dir / f"{args.case}-result.json",
            args.timeout,
        )
        return 0 if result_data["success"] else 1
    finally:
        cleanup_groups = (
            ((engine, registered), (ec_engine, ec_registered))
            if args.cleanup_order == "store-first"
            else ((ec_engine, ec_registered), (engine, registered))
        )
        print(f"CLEANUP_ORDER value={args.cleanup_order}", flush=True)
        cleanup_results: list[dict[str, Any]] = []
        for cleanup_engine, registrations in cleanup_groups:
            for name, tensor in reversed(registrations):
                try:
                    unregister_result = _unregister(cleanup_engine, name, tensor)
                except BaseException as error:
                    cleanup_results.append(
                        {
                            "exception": repr(error),
                            "name": name,
                            "result": None,
                        }
                    )
                else:
                    cleanup_results.append(
                        {
                            "exception": None,
                            "name": name,
                            "result": unregister_result,
                        }
                    )
        _atomic_json(
            args.exchange_dir / f"{args.case}-{args.role}-cleanup.json",
            {
                "cleanup_order": args.cleanup_order,
                "registrations": cleanup_results,
                "role": args.role,
            },
        )
        # Keep allocation owners visibly live until after unregister completes.
        _ = (producer_raw, store_prefix_raw, suffix_raw)


def _default_host() -> str:
    host = socket.gethostbyname(socket.gethostname())
    if host.startswith("127."):
        raise RuntimeError("could not infer a non-loopback host; pass --host")
    return host


def _run_case(args: argparse.Namespace, case: str) -> dict[str, Any]:
    case_dir = args.output_dir / case
    case_dir.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve()
    processes: list[tuple[str, subprocess.Popen[Any], Any]] = []

    def start(role: str, npu: str) -> None:
        env = os.environ.copy()
        env["ASCEND_ENABLE_USE_FABRIC_MEM"] = "0"
        env["ASCEND_RT_VISIBLE_DEVICES"] = npu
        env["PYTHONUNBUFFERED"] = "1"
        env.pop("ASCEND_BUFFER_POOL", None)
        env.pop("ASCEND_USE_ASYNC_TRANSFER", None)
        if args.link_mode == "roce":
            env["HCCL_INTRA_ROCE_ENABLE"] = "1"
        else:
            env.pop("HCCL_INTRA_ROCE_ENABLE", None)
        log = (case_dir / f"{role}.log").open("w", encoding="utf-8")
        command = [
            sys.executable,
            str(script),
            "--worker",
            "--role",
            role,
            "--case",
            case,
            "--host",
            args.host,
            "--exchange-dir",
            str(case_dir),
            "--timeout",
            str(args.timeout),
            "--link-mode",
            args.link_mode,
            "--engine-topology",
            args.engine_topology,
            "--cleanup-order",
            args.cleanup_order,
            "--iterations",
            str(args.iterations),
        ]
        if args.store_buffer_pool:
            command.extend(("--store-buffer-pool", args.store_buffer_pool))
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        processes.append((role, process, log))

    deadline = time.monotonic() + args.timeout
    try:
        start("target", args.npu_b)
        target_process = processes[0][1]
        ready_path = case_dir / f"{case}-target-ready.json"
        while not ready_path.exists():
            if target_process.poll() is not None:
                target_log = (case_dir / "target.log").read_text(
                    encoding="utf-8",
                    errors="replace",
                )
                raise RuntimeError(
                    f"{case} target exited before readiness "
                    f"with {target_process.returncode}:\n{target_log}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{case} timed out waiting for target readiness")
            time.sleep(0.05)
        _wait_json(ready_path, 1.0)
        start("initiator", args.npu_a)

        for role, process, _ in processes:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"{case} timed out while waiting for {role}")
            process.wait(timeout=remaining)
    except BaseException:
        for _, process, _ in processes:
            if process.poll() is None:
                process.terminate()
        for _, process, _ in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        raise
    finally:
        for _, _, log in processes:
            log.close()

    result_path = case_dir / f"{case}-result.json"
    result = _wait_json(result_path, 1.0)
    result["process_returncodes"] = {
        role: process.returncode for role, process, _ in processes
    }
    result["logs"] = {
        role: str(case_dir / f"{role}.log") for role, _, _ in processes
    }
    if case == "dual-engine-long-lived-staging":
        result["transfer_success"] = bool(result["success"])
        result["initiator_observations"] = _wait_json(
            case_dir / f"{case}-initiator-observations.json",
            1.0,
        )
        cleanup = {}
        for role, _, _ in processes:
            cleanup_path = case_dir / f"{case}-{role}-cleanup.json"
            cleanup[role] = (
                _wait_json(cleanup_path, 1.0) if cleanup_path.exists() else None
            )
        result["cleanup"] = cleanup
        cleanup_succeeded = all(
            record is not None
            and all(
                item["exception"] is None and item["result"] == 0
                for item in record["registrations"]
            )
            for record in cleanup.values()
        )
        result["cleanup_success"] = cleanup_succeeded
        result["success"] = bool(
            result["transfer_success"]
            and cleanup_succeeded
            and all(code == 0 for code in result["process_returncodes"].values())
        )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=None)
    parser.add_argument("--npu-a", default="0")
    parser.add_argument("--npu-b", default="1")
    parser.add_argument(
        "--link-mode",
        choices=("hccs", "roce"),
        default="hccs",
    )
    parser.add_argument(
        "--engine-topology",
        choices=("single", "two"),
        default="single",
    )
    parser.add_argument("--store-buffer-pool")
    parser.add_argument(
        "--cleanup-order",
        choices=("store-first", "ec-first"),
        default="store-first",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_LONG_LIVED_ITERATIONS,
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=(
            "header-only",
            "store-shape",
            "device-to-host",
            "device-to-device",
            "host-to-device-read",
            "device-store-shape",
            "dual-engine-registration",
            "registration-handoff",
            "dual-engine-long-lived-staging",
        ),
        default=(
            "header-only",
            "store-shape",
            "device-to-host",
            "device-to-device",
            "host-to-device-read",
            "device-store-shape",
            "dual-engine-registration",
            "registration-handoff",
        ),
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--role",
        choices=("initiator", "target"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--case",
        choices=(
            "header-only",
            "store-shape",
            "device-to-host",
            "host-to-device-read",
            "device-store-shape",
            "device-to-device",
            "dual-engine-registration",
            "registration-handoff",
            "dual-engine-long-lived-staging",
        ),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--exchange-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    args.host = args.host or _default_host()
    if args.worker:
        if args.role is None or args.case is None or args.exchange_dir is None:
            parser.error("worker mode requires --role, --case, and --exchange-dir")
        args.output_dir = None
    elif args.output_dir is None:
        args.output_dir = Path(
            tempfile.mkdtemp(prefix="non-fabric-host-probe-", dir="/tmp")
        )
    return args


def main() -> int:
    args = _parse_args()
    if args.worker:
        return _worker(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"OUTPUT_DIR={args.output_dir}")
    print(
        "CONFIG "
        + json.dumps(
            {
                "engine_topology": args.engine_topology,
                "cleanup_order": args.cleanup_order,
                "fabric_mem": False,
                "global_host_bytes": GLOBAL_HOST_BYTES,
                "host": args.host,
                "link_mode": args.link_mode,
                "local_host_bytes": LOCAL_HOST_BYTES,
                "npu_a": args.npu_a,
                "npu_b": args.npu_b,
                "producer_bytes": PRODUCER_BYTES,
                "store_buffer_pool": args.store_buffer_pool,
                "iterations": args.iterations,
            },
            sort_keys=True,
        )
    )

    results = []
    for case in args.cases:
        try:
            result = _run_case(args, case)
        except BaseException as error:
            result = {"case": case, "error": repr(error), "success": False}
        results.append(result)
        print("CASE_RESULT " + json.dumps(result, sort_keys=True))

    summary = {
        "engine_topology": args.engine_topology,
        "cleanup_order": args.cleanup_order,
        "link_mode": args.link_mode,
        "output_dir": str(args.output_dir),
        "results": results,
        "store_buffer_pool": args.store_buffer_pool,
        "success": all(result.get("success", False) for result in results),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("SUMMARY " + json.dumps(summary, sort_keys=True))
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
