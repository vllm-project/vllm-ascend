# SPDX-License-Identifier: Apache-2.0
"""Gated diagnostics for the OSCAR INT2 path.

Replaces the historical hard-coded probe paths (absolute home-directory
appends and unconditional /tmp writes) that the frozen RC53 overlay carried.
Everything here is OFF unless ``OSCAR_DIAG=1``; outputs land under
``OSCAR_DIAG_DIR`` (default ``/tmp/oscar_diag``). Compute paths never touch
the filesystem through this module in the default configuration.
"""

from __future__ import annotations

import contextlib
import os
from contextlib import contextmanager


def diag_enabled() -> bool:
    return os.environ.get("OSCAR_DIAG", "0") == "1"


def diag_dir() -> str:
    return os.environ.get("OSCAR_DIAG_DIR", "/tmp/oscar_diag")


def write(relname: str, text: str) -> None:
    """Append one diagnostic record (no-op unless OSCAR_DIAG=1)."""
    if not diag_enabled():
        return
    try:
        os.makedirs(diag_dir(), exist_ok=True)
        with open(os.path.join(diag_dir(), relname), "a") as f:
            f.write(text if text.endswith("\n") else text + "\n")
    except Exception:
        pass


def dump_path(relname: str) -> str:
    """Path for binary/json diagnostic dumps (always under OSCAR_DIAG_DIR)."""
    d = diag_dir()
    with contextlib.suppress(Exception):
        os.makedirs(d, exist_ok=True)
    return os.path.join(d, relname)


class _Sink:
    """Discard-only stand-in for a file object."""

    def write(self, *a, **k):  # noqa: D102
        pass

    def flush(self):  # noqa: D102
        pass


@contextmanager
def open_diag(relname: str):
    """``with open_diag("x.log") as f:`` — ``f`` is a sink unless OSCAR_DIAG=1."""
    if not diag_enabled():
        yield _Sink()
        return
    try:
        os.makedirs(diag_dir(), exist_ok=True)
    except Exception:
        yield _Sink()
        return
    try:
        f = open(os.path.join(diag_dir(), relname), "a")  # noqa: SIM115
    except Exception:
        # Setup failures degrade to the sink; failures while the caller
        # holds the file must propagate, so the yield is NOT in this try.
        yield _Sink()
        return
    with contextlib.closing(f):
        yield f
