# SPDX-License-Identifier: Apache-2.0
"""One real Mooncake initialization per fresh process, without weights or peers."""

import argparse
import ctypes
import hashlib
import json
import os
import shutil
from pathlib import Path


def event(stage, **values):
    print(json.dumps({"event": "transport", "stage": stage, **values}), flush=True)


def library_fingerprints():
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            path = fields[5]
            if any(name in path.lower() for name in ("hixl", "mooncake", "libascendcl", "libruntime", "libascend_hal")):
                paths.add(path)
    for name in sorted(paths):
        try:
            digest = hashlib.sha256()
            with open(name, "rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            event("library_digest", path=name, sha256=digest.hexdigest())
        except OSError as exc:
            event("library_digest_unavailable", path=name, error=str(exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--user", type=int, required=True, choices=(0, 1))
    args = parser.parse_args()
    event("process", pid=os.getpid(), user=args.user)
    # Observe HIXL's documented tool locations without substituting a tool,
    # guessing a bond/device ID, or changing any network configuration.
    for name in ("/usr/local/Ascend/driver/tools/hccn_tool", shutil.which("hccn_tool")):
        if name:
            path = Path(name)
            event(
                "tool_path",
                path=name,
                resolved=str(path.resolve()),
                exists=path.is_file(),
                executable=os.access(path, os.X_OK),
            )
    # Import and select only inside this isolated child, as a spawned worker does.
    event("torch_import_start")
    import torch
    import torch_npu  # noqa: F401

    event("set_device_start", user=args.user)
    torch.npu.set_device(args.user)
    event("set_device_return", current=torch.npu.current_device())
    # This is our own exported ABI. The SDK declarations live in the C++ helper.
    library = ctypes.CDLL(args.library)
    snapshot = library.a5_mapping_snapshot
    snapshot.argtypes = [ctypes.c_int]
    snapshot.restype = ctypes.c_int

    def observe(stage):
        event(stage + "_start")
        rc = snapshot(args.user)
        event(stage + "_return", rc=rc)
        assert rc == 0, f"SDK mapping failed at {stage}: {rc}"

    observe("after_torch")
    event("mooncake_import_start")
    from mooncake.engine import TransferEngine  # type: ignore[import-not-found]
    from vllm.utils.network_utils import get_ip

    observe("after_import")
    library_fingerprints()
    # Match PP=1 GlobalTE exactly: no device-name override or protocol override.
    hostname = get_ip()
    event("construct_start")
    engine = TransferEngine()
    event("initialize_start", hostname=hostname, metadata="P2PHANDSHAKE", protocol="ascend", device_name="")
    rc = engine.initialize(hostname, "P2PHANDSHAKE", "ascend", "")
    event("initialize_return", rc=rc)
    # Preserve post-call mapping even on a nonzero return; never retry initialize.
    observe("after_initialize")
    library_fingerprints()
    assert rc == 0, f"TransferEngine initialization failed: {rc}"
    event("destroy_start")
    del engine
    event("complete")


if __name__ == "__main__":
    main()
