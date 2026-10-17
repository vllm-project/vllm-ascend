# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
from unittest.mock import patch

import psutil


def _kill_process_tree(pid: int) -> None:
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        return

    for process in [*children, parent]:
        with contextlib.suppress(psutil.NoSuchProcess):
            process.kill()


def shutdown_engine(engine_core) -> None:
    """Finish E2E teardown when a child exits during forced cleanup."""
    # vLLM c4df37d reads child.name() outside its NoSuchProcess guard.
    # Keep this workaround scoped to teardown until the upstream fix is available.
    with patch("vllm.v1.utils.kill_process_tree", _kill_process_tree):
        engine_core.shutdown()
