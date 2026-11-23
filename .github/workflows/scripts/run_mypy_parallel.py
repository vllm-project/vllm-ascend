# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the CI mypy versions concurrently, with progress and grouped logs."""

import os
import signal
import subprocess
import time
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

PYTHON_VERSIONS = ("3.10", "3.11", "3.12")
HEARTBEAT_SECONDS = 20.0
POLL_SECONDS = 0.1


def report_result(version: str, log_dir: Path, returncode: int, *, cancelled: bool = False) -> None:
    status = "cancelled" if cancelled else "passed" if returncode == 0 else "failed"
    print(f"[mypy] Python {version}: {status}, exit={returncode}", flush=True)
    print(f"::group::mypy Python {version} ({status})", flush=True)
    # Preserve diagnostic lines verbatim so the mypy problem matcher works.
    print((log_dir / f"mypy-{version}.log").read_text(encoding="utf-8", errors="replace"), end="")
    print("\n::endgroup::", flush=True)
    if status == "failed":
        print(f"::error title=mypy Python {version}::mypy exited with status {returncode}", flush=True)


def main(heartbeat_seconds: float = HEARTBEAT_SECONDS) -> int:
    cancelled = Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda _signum, _frame: cancelled.set())
    root = Path(__file__).resolve().parents[3]
    pending: dict[str, subprocess.Popen] = {}
    failed = False
    started_at = time.monotonic()
    with TemporaryDirectory(prefix="vllm-mypy-") as directory:
        log_dir = Path(directory)
        try:
            for version in PYTHON_VERSIONS:
                if cancelled.is_set():
                    break
                with (log_dir / f"mypy-{version}.log").open("wb") as log:
                    pending[version] = subprocess.Popen(
                        ["bash", str(root / "tools/mypy.sh"), "1", version],
                        cwd=root,
                        env={**os.environ, "MYPY_CACHE_DIR": str(log_dir / f"cache-{version}")},
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                print(f"[mypy] Python {version}: started", flush=True)

            next_heartbeat = time.monotonic() + heartbeat_seconds
            while pending and not cancelled.is_set():
                for version, process in list(pending.items()):
                    returncode = process.poll()
                    if returncode is not None:
                        report_result(version, log_dir, returncode)
                        failed |= returncode != 0
                        del pending[version]
                if pending and time.monotonic() >= next_heartbeat:
                    elapsed = time.monotonic() - started_at
                    print(f"[mypy] Running Python {', '.join(pending)}; {elapsed:.0f}s elapsed", flush=True)
                    next_heartbeat = time.monotonic() + heartbeat_seconds
                cancelled.wait(POLL_SECONDS)
            return 130 if cancelled.is_set() else int(failed)
        finally:
            # These checks only write disposable caches. On cancellation, stop
            # entire process groups, including children that outlive Bash.
            for process in pending.values():
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            for version, process in pending.items():
                report_result(version, log_dir, process.wait(), cancelled=True)


if __name__ == "__main__":
    raise SystemExit(main())
