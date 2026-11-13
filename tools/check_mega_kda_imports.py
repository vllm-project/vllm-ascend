# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Locate Mega-KDA operator import hangs without loading model weights.

Run inside the inference environment:

    timeout 90s python tools/check_mega_kda_imports.py

Imports can trigger the packages' own library loading or extension
compilation.  This script does not change sys.path, environment variables,
or compiler caches; it prints all Python thread stacks periodically while
each import is in flight, mirroring the diagnostic shipped with
cann-recipes-infer_kimi_k3 PR #50.
"""

import argparse
import faulthandler
import importlib
import inspect
import os
import socket
import sys
import time

OPERATORS = {
    "torch": (),
    "torch_npu": (),
    "ops.mega_recurrent_kda": ("mega_recurrent_kda",),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump-after",
        type=float,
        default=15,
        help="Print all Python thread stacks every N seconds while importing (default: 15).",
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        default=list(OPERATORS),
        help="Import only these modules, in the given order.",
    )
    args = parser.parse_args()
    if args.dump_after <= 0:
        parser.error("--dump-after must be positive")

    print(f"Host: {socket.gethostname()}  PID: {os.getpid()}", flush=True)
    print(f"Python: {sys.executable} ({sys.version.split()[0]})", flush=True)
    print(f"Working directory: {os.getcwd()}", flush=True)
    faulthandler.enable()
    for name in args.modules:
        print(f"BEGIN {name}", flush=True)
        started = time.monotonic()
        faulthandler.dump_traceback_later(args.dump_after, repeat=True)
        try:
            module = importlib.import_module(name)
            print(f"  file: {getattr(module, '__file__', None)}", flush=True)
            for symbol in OPERATORS.get(name, ()):
                value = getattr(module, symbol)
                if not callable(value):
                    raise TypeError(f"{name}.{symbol} is not callable")
                print(f"  export OK: {symbol}", flush=True)
                try:
                    print(f"  signature: {inspect.signature(value)}", flush=True)
                except (TypeError, ValueError):
                    print("  signature: unavailable (binary export)", flush=True)
        except Exception:
            print(f"FAILED {name} after {time.monotonic() - started:.2f}s", flush=True)
            raise
        finally:
            faulthandler.cancel_dump_traceback_later()
        print(f"DONE {name} in {time.monotonic() - started:.2f}s", flush=True)
    print(
        "Imports passed. Required: PR #6 20-argument BF16/NZ ABI with explicit RMS epsilon. "
        "An import/signature check does not prove weight layout, cache semantics or numerical correctness.",
        flush=True,
    )


if __name__ == "__main__":
    main()
