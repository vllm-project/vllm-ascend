# SPDX-License-Identifier: Apache-2.0
"""Launch an Engram owner or consumer before an existing service command."""

import argparse
import base64
import contextlib
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def model_identity(model):
    digest = hashlib.sha256()
    for name in ("config.json", "quant_model_weights.safetensors.index.json"):
        digest.update(name.encode())
        digest.update((Path(model) / name).read_bytes())
    return digest.hexdigest()


def wait_ready(pool_dir, identity, timeout, owner=None, stopped=lambda: False):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if stopped():
            raise InterruptedError("Engram startup was interrupted")
        if owner is not None and owner.poll() is not None:
            raise RuntimeError(f"Engram owner exited with status {owner.returncode}")
        try:
            root = Path(pool_dir)
            stat = root.stat()
            if stat.st_uid != os.getuid() or stat.st_mode & 0o077:
                raise ValueError("Shared Engram directory must be uid-owned and mode0700")
            manifest = json.loads((root / "manifest.json").read_text())
            if manifest["identity"] != identity:
                raise ValueError("Shared Engram checkpoint identity mismatch")
            if manifest["schema"] != 1:
                raise ValueError("Unsupported shared Engram manifest schema")
            if manifest["state"] == "ready":
                with socket.create_connection(tuple(manifest["endpoint"]), timeout=5) as conn:
                    conn.settimeout(5)
                    conn.sendall((json.dumps({"op": "ping", "token": manifest["token"]}) + "\n").encode())
                    with conn.makefile("rb") as wire:
                        response = json.loads(wire.readline(4096))
                    if response.get("alive") is True:
                        return
        except (OSError, json.JSONDecodeError, KeyError):
            pass
        time.sleep(1)
    raise TimeoutError(f"Engram owner was not ready within {timeout} seconds")


def bootstrap_local(pool_dir, identity, host, port, timeout, stopped=lambda: False):
    """Copy VMM handle descriptors over a trusted network into a private local directory."""
    deadline = time.monotonic() + timeout
    response = None
    while time.monotonic() < deadline:
        if stopped():
            raise InterruptedError("Engram startup was interrupted")
        try:
            with socket.create_connection((host, port), timeout=5) as conn:
                conn.settimeout(5)
                request = {"op": "bootstrap", "identity": identity}
                conn.sendall((json.dumps(request) + "\n").encode())
                with conn.makefile("rb") as wire:
                    response = json.loads(wire.readline((1 << 20) + 1))
            if "error" not in response:
                break
            if response["error"] != "Engram pool is not ready":
                raise ValueError("Engram owner rejected bootstrap")
        except (OSError, json.JSONDecodeError):
            pass
        time.sleep(1)
    else:
        raise TimeoutError(f"Engram owner was not ready within {timeout} seconds")

    manifest = response["manifest"]
    if manifest.get("schema") != 1 or manifest.get("identity") != identity or manifest.get("state") != "ready":
        raise ValueError("Incompatible shared Engram manifest")
    descriptors = response["descriptors"]
    names = {entry["descriptor"] for entry in manifest["tensors"].values()}
    if not names or set(descriptors) != names or any(len(n) != 64 or set(n) - set("0123456789abcdef") for n in names):
        raise ValueError("Invalid Engram descriptors")
    decoded = {name: base64.b64decode(descriptors[name], validate=True) for name in names}
    if any(not value or len(value) > 4096 for value in decoded.values()):
        raise ValueError("Invalid Engram descriptor size")
    root = Path(pool_dir)
    root.mkdir(mode=0o700)
    try:
        for name, value in decoded.items():
            for filename, content in ((name, value), (name + ".ready", b"")):
                fd = os.open(root / filename, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                with os.fdopen(fd, "wb") as file:
                    file.write(content)
        fd = os.open(root / "manifest.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w") as file:
            json.dump(manifest, file)
    except BaseException:
        shutil.rmtree(root)
        raise


def begin_shard_loading(pool_dir, identity, membership, timeout):
    from vllm_ascend.engram_shard_layout import layout_identity, validate_owner_set

    membership = validate_owner_set(membership)
    root = Path(pool_dir)
    root.mkdir(mode=0o700, exist_ok=True)
    if root.stat().st_uid != os.getuid() or root.stat().st_mode & 0o077:
        raise ValueError("Sharded Engram requires a private bundle")
    manifest = dict(
        schema=1,
        kind="sharded",
        identity=identity,
        state="loading",
        membership=membership,
        layout_id=layout_identity(membership, identity),
        expires_at=time.time() + timeout,
    )
    fd = os.open(root / "manifest.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as file:
        json.dump(manifest, file)
    return manifest


def bootstrap_shards(pool_dir, identity, membership, timeout, stopped=lambda: False, *, pending=False):
    from concurrent.futures import ThreadPoolExecutor

    from vllm_ascend.engram_shard_layout import layout_identity, validate_owner_set

    membership = validate_owner_set(membership)
    root = Path(pool_dir)
    # The owner may already have created root/backing. Never overwrite a previous bundle.
    root.mkdir(mode=0o700, exist_ok=True)
    if pending:
        plan = json.loads((root / "manifest.json").read_text())
        if (
            plan.get("schema") != 1
            or plan.get("kind") != "sharded"
            or plan.get("identity") != identity
            or plan.get("membership") != membership
            or plan.get("state") != "loading"
            or plan.get("layout_id") != layout_identity(membership, identity)
            or plan.get("expires_at", 0) <= time.time()
        ):
            raise ValueError("Invalid pending Engram load plan")
    if (
        root.stat().st_uid != os.getuid()
        or root.stat().st_mode & 0o077
        or (not pending and (root / "manifest.json").exists())
    ):
        raise ValueError("Sharded Engram requires a fresh private bundle")
    with ThreadPoolExecutor(max_workers=len(membership["owners"])) as executor:
        futures = [
            executor.submit(
                bootstrap_local, root / f"owner-{rank}", identity, peer["host"], peer["port"], timeout, stopped
            )
            for rank, peer in enumerate(membership["owners"])
        ]
        for future in futures:
            future.result()
    generations = []
    for rank in range(len(membership["owners"])):
        manifest = json.loads((root / f"owner-{rank}" / "manifest.json").read_text())
        if manifest.get("sharding") != dict(
            layout_id=layout_identity(membership, identity), owner_rank=rank, membership=membership
        ):
            raise ValueError("Engram bootstrap shard layout mismatch")
        generations.append(manifest["generation"])
    manifest = dict(
        schema=1,
        kind="sharded",
        identity=identity,
        state="ready",
        membership=membership,
        layout_id=layout_identity(membership, identity),
        generations=generations,
    )
    temporary = root / "manifest.tmp"
    temporary.write_text(json.dumps(manifest))
    temporary.replace(root / "manifest.json")


def shard_owner_checks(pool_dir, identity):
    root = Path(pool_dir)
    bundle = json.loads((root / "manifest.json").read_text())
    if not isinstance(bundle, dict) or bundle.get("state") != "ready" or bundle.get("identity") != identity:
        raise ValueError("Shared Engram bundle is not ready for supervision")
    checks = []
    generations = bundle.get("generations")
    if not isinstance(generations, list) or len(generations) not in (2, 4):
        raise ValueError("Invalid Engram owner generations")
    for rank, generation in enumerate(generations):
        owner = json.loads((root / f"owner-{rank}" / "manifest.json").read_text())
        if not isinstance(owner, dict) or owner.get("generation") != generation or owner.get("identity") != identity:
            raise ValueError("Shared Engram supervision generation mismatch")
        checks.append(dict(endpoint=owner["endpoint"], token=owner["token"], generation=generation))
    if not checks:
        raise ValueError("Empty shared Engram owner set")
    return checks


def check_shard_owners(checks, identity):
    """Check the generation captured at bootstrap, never adopt a replacement."""
    for rank, owner in enumerate(checks):
        try:
            with socket.create_connection(tuple(owner["endpoint"]), timeout=1) as conn:
                conn.settimeout(1)
                conn.sendall((json.dumps(dict(op="ping", token=owner["token"])) + "\n").encode())
                with conn.makefile("rb") as wire:
                    response = json.loads(wire.readline(4096))
            if not isinstance(response, dict):
                raise ValueError("Invalid Engram owner health response")
            if not (
                response.get("alive") is True
                and response.get("state") == "ready"
                and response.get("identity") == identity
                and response.get("generation") == owner["generation"]
            ):
                raise ValueError("Owner no longer matches the active ready generation")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise RuntimeError(f"Engram owner {rank} is unavailable, draining or replaced") from exc


def discover_owners(url, identity, timeout, *, host=None, port=29699, node_id=None, stopped=lambda: False):
    from vllm_ascend.engram_shard_layout import validate_owner_set

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    body = None
    if host is not None:
        body = json.dumps(
            dict(host=host, port=port, node_id=node_id or socket.gethostname(), identity=identity)
        ).encode()
    while time.monotonic() < deadline:
        if stopped():
            raise InterruptedError("Engram registry discovery interrupted")
        path = "/register" if body is not None else "/members"
        request = urllib.request.Request(
            url.rstrip("/") + path, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with opener.open(request, timeout=5) as response:
                raw = response.read((1 << 20) + 1)
                if len(raw) > 1 << 20:
                    raise ValueError("Oversized Engram registry response")
                result = json.loads(raw)
            if result.get("identity") not in (None, identity):
                raise ValueError("Engram registry checkpoint mismatch")
            if result.get("state") == "ready":
                spec = validate_owner_set(result["membership"])
                rank = None
                if host is not None:
                    peer = dict(host=host, port=port)
                    if peer not in spec["owners"]:
                        raise ValueError("Local owner is absent from registry membership")
                    rank = spec["owners"].index(peer)
                return spec, rank
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                raise ValueError(f"Engram registry rejected request: HTTP {exc.code}") from exc
        except (OSError, json.JSONDecodeError):
            pass
        time.sleep(1)
    raise TimeoutError("Engram owner registry did not publish membership")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("owner", "consumer"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pool-dir", required=True)
    parser.add_argument("--listen-host", help="Reachable owner address; required for the owner role")
    parser.add_argument("--owner-host", help="Owner address for network bootstrap")
    parser.add_argument("--owner-port", type=int, help="Fixed owner port for network bootstrap")
    parser.add_argument("--owner-registry", help="Early registry URL on the conductor/kvmaster pod")
    parser.add_argument("--registry-owner-port", type=int, default=29699)
    parser.add_argument("--owner-set", help="Static ordered owner membership JSON")
    parser.add_argument("--owner-rank", type=int)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--overlap-load", action="store_true", help="Overlap sharded table and model loading")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Service command after --")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.timeout <= 0 or (args.role == "owner" and not args.listen_host):
        parser.error("A service command, positive timeout, and owner listen address are required")
    network = args.owner_host is not None or args.owner_port is not None
    if network and (not args.owner_port or (args.role == "consumer" and not args.owner_host)):
        parser.error("Network bootstrap requires an owner port, plus owner host on consumers")
    if args.owner_port is not None and not 1 <= args.owner_port <= 65535:
        parser.error("Owner port must be between 1 and 65535")

    if args.owner_registry and (args.owner_set or network or args.owner_rank is not None):
        parser.error("Registry discovery cannot be combined with static/single-owner arguments")
    if not 1 <= args.registry_owner_port <= 65535:
        parser.error("Registry owner port must be between 1 and 65535")
    membership = None
    if args.owner_set:
        from vllm_ascend.engram_shard_layout import owner_set

        membership = owner_set(args.owner_set)
        if args.owner_host is not None or args.owner_port is not None:
            parser.error("--owner-set cannot be combined with single-owner endpoint flags")
        if args.role == "owner" and (args.owner_rank is None or not 0 <= args.owner_rank < len(membership["owners"])):
            parser.error("Sharded owner requires a valid --owner-rank")
    elif args.owner_rank is not None:
        parser.error("--owner-rank requires --owner-set")

    if args.overlap_load and not (membership or args.owner_registry):
        parser.error("--overlap-load requires sharded owners")

    owner = None
    service = None
    bootstrap_thread = None
    bootstrap_errors = []
    owner_checks = None
    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True
        if service is not None and service.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(service.pid, signum)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        identity = model_identity(args.model)
        if args.owner_registry:
            membership, args.owner_rank = discover_owners(
                args.owner_registry,
                identity,
                args.timeout,
                host=args.listen_host if args.role == "owner" else None,
                port=args.registry_owner_port,
                stopped=lambda: stop,
            )
            root = Path(args.pool_dir)
            root.mkdir(mode=0o700)
            spec_path = root / "owners.json"
            spec_path.write_text(json.dumps(membership))
            args.owner_set = str(spec_path)
        if args.role == "owner":
            backing_dir = str(Path(args.pool_dir) / "backing") if membership else args.pool_dir
            if membership and not args.owner_registry:
                Path(args.pool_dir).mkdir(mode=0o700)
            if Path(backing_dir).exists():
                raise ValueError("Owner requires a new, nonexistent pool directory")
            owner_command = [
                sys.executable,
                "-m",
                "vllm_ascend.shared_engram",
                "--model",
                args.model,
                "--pool-dir",
                backing_dir,
                "--device",
                str(args.device),
                "--listen-host",
                args.listen_host,
            ]
            if membership:
                peer = membership["owners"][args.owner_rank]
                if args.listen_host != peer["host"]:
                    raise ValueError("Owner listen host differs from static membership")
                owner_command += [
                    "--owner-set",
                    args.owner_set,
                    "--owner-rank",
                    str(args.owner_rank),
                    "--port",
                    str(peer["port"]),
                    "--network-bootstrap",
                ]
            if network:
                owner_command += ["--port", str(args.owner_port), "--network-bootstrap"]
            owner = subprocess.Popen(owner_command)
            print(json.dumps(dict(stage="OWNER_STARTED", time=time.time(), pid=owner.pid)), file=sys.stderr, flush=True)
        if membership:

            def bootstrap():
                bootstrap_shards(
                    args.pool_dir,
                    identity,
                    membership,
                    args.timeout,
                    lambda: stop or (owner is not None and owner.poll() is not None),
                    pending=args.overlap_load,
                )

            if args.overlap_load:
                plan = begin_shard_loading(args.pool_dir, identity, membership, args.timeout)

                def background_bootstrap():
                    try:
                        bootstrap()
                        print(json.dumps(dict(stage="SHARDS_READY", time=time.time())), file=sys.stderr, flush=True)
                    except BaseException as exc:
                        bootstrap_errors.append(exc)
                        plan["state"] = "failed"
                        temporary = Path(args.pool_dir) / "manifest.failed.tmp"
                        temporary.write_text(json.dumps(plan))
                        temporary.replace(Path(args.pool_dir) / "manifest.json")

                bootstrap_thread = threading.Thread(target=background_bootstrap, daemon=True)
                bootstrap_thread.start()
            else:
                bootstrap()
                print(json.dumps(dict(stage="SHARDS_READY", time=time.time())), file=sys.stderr, flush=True)
        elif network and args.role == "consumer":
            bootstrap_local(args.pool_dir, identity, args.owner_host, args.owner_port, args.timeout, lambda: stop)
        else:
            wait_ready(args.pool_dir, identity, args.timeout, owner, lambda: stop)
        if stop:
            return 1
        service = subprocess.Popen(command, start_new_session=True)
        print(json.dumps(dict(stage="MODEL_STARTED", time=time.time(), pid=service.pid)), file=sys.stderr, flush=True)
        if stop:
            request_stop(signal.SIGTERM, None)
        while service.poll() is None:
            if bootstrap_errors:
                os.killpg(service.pid, signal.SIGTERM)
                service.wait()
                raise RuntimeError("Engram owner bootstrap failed") from bootstrap_errors[0]
            if membership and (bootstrap_thread is None or not bootstrap_thread.is_alive()):
                try:
                    if owner_checks is None:
                        owner_checks = shard_owner_checks(args.pool_dir, identity)
                    check_shard_owners(owner_checks, identity)
                except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                    print(
                        json.dumps(dict(stage="SHARD_UNAVAILABLE", time=time.time(), action="stopping_service")),
                        file=sys.stderr,
                        flush=True,
                    )
                    request_stop(signal.SIGTERM, None)
                    service.wait()
                    raise RuntimeError("Shared Engram became unavailable; service stopped") from exc
            if owner is not None and owner.poll() is not None:
                os.killpg(service.pid, signal.SIGTERM)
                service.wait()
                raise RuntimeError("Engram owner exited while the service was running")
            time.sleep(1)
        return service.returncode
    finally:
        stop = True
        if bootstrap_thread is not None:
            bootstrap_thread.join(timeout=10)
        if owner is not None and owner.poll() is None:
            owner.terminate()
            # The owner intentionally waits for all remote leases to drain.
            owner.wait()


if __name__ == "__main__":
    sys.exit(main())
