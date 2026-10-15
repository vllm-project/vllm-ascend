#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wait for and verify cross-Encoder Store path markers in Encoder logs."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

PUT_MARKER = "Stored encoder output in Mooncake Store"
GET_PATTERN = re.compile(r"Loaded (\d+) encoder output\(s\) from Mooncake Store")
FAILURE_MARKERS = (
    "failed to put embedding tensor",
    "TENANT_QUOTA_EXCEEDED",
    "Mooncake I/O completion is unconfirmed",
    "Traceback (most recent call last)",
)


def read_logs(paths: list[Path]) -> str:
    return "\n".join(path.read_text(errors="replace") for path in paths)


def counts(text: str) -> tuple[int, int]:
    puts = text.count(PUT_MARKER)
    gets = sum(int(match.group(1)) for match in GET_PATTERN.finditer(text))
    return puts, gets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, nargs="+", required=True)
    parser.add_argument("--expected-puts", type=int)
    parser.add_argument("--expected-gets", type=int)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.expected_puts is None and args.expected_gets is None:
        parser.error("set --expected-puts and/or --expected-gets")

    deadline = time.monotonic() + args.timeout
    while True:
        text = read_logs(args.logs)
        failure = next((marker for marker in FAILURE_MARKERS if marker in text), None)
        if failure:
            raise RuntimeError(f"Encoder log contains failure marker: {failure}")
        puts, gets = counts(text)
        puts_ready = args.expected_puts is None or puts >= args.expected_puts
        gets_ready = args.expected_gets is None or gets >= args.expected_gets
        if puts_ready and gets_ready:
            if args.expected_puts is not None and puts != args.expected_puts:
                raise RuntimeError(f"expected exactly {args.expected_puts} PUTs, got {puts}")
            if args.expected_gets is not None and gets != args.expected_gets:
                raise RuntimeError(f"expected exactly {args.expected_gets} GETs, got {gets}")
            print(f"store_path_ok puts={puts} gets={gets}")
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for Store markers: puts={puts}, gets={gets}")
        time.sleep(1)


if __name__ == "__main__":
    main()
