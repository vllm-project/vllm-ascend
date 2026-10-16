# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backport vllm-project/vllm#59984 for the supported dependency pins."""

import contextlib
import os
import signal
import sys

import psutil
from vllm.utils import system_utils


def _kill_process_tree(pid: int) -> None:
    """Keep killing the tree when a listed child exits before its name is read."""
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    children = parent.children(recursive=True)
    for child in children:
        with contextlib.suppress(ProcessLookupError, psutil.NoSuchProcess):
            system_utils.logger.warning(
                "[shutdown] force killing sub process %s pid %d",
                child.name(),
                child.pid,
            )
            os.kill(child.pid, signal.SIGKILL)

    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)


def _patch_kill_process_tree() -> None:
    original = system_utils.kill_process_tree
    # shutdown() may import its alias before platform registration runs.
    for name in (
        "vllm.utils.system_utils",
        "vllm.utils",
        "vllm.v1.utils",
        "vllm.entrypoints.launchers.dp_supervisor",
    ):
        module = sys.modules.get(name)
        if module is not None and module.__dict__.get("kill_process_tree") is original:
            module.__dict__["kill_process_tree"] = _kill_process_tree


_patch_kill_process_tree()
