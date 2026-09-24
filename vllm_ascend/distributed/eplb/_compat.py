# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Compatibility helpers for vLLM EPLB stream APIs."""

from contextlib import contextmanager
from typing import Any

import torch
from vllm.distributed.eplb import eplb_utils


@contextmanager
def device_stream(stream: Any):
    """Activate a device stream across supported vLLM revisions."""
    upstream_context = getattr(eplb_utils, "device_stream", None)
    if upstream_context is not None:
        with upstream_context(stream):
            yield
        return
    if stream is None:
        yield
        return
    previous_stream = torch.accelerator.current_stream()
    torch.accelerator.set_stream(stream)
    try:
        yield
    finally:
        torch.accelerator.set_stream(previous_stream)


def communicator_stream(communicator: Any):
    """Read the stream stored by either vLLM communicator contract."""
    return getattr(communicator, "_stream", getattr(communicator, "_cuda_stream", None))
