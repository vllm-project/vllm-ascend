# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from contextlib import contextmanager

import torch


class PrefillLayerWindow:
    """Limit how far the host runs ahead of the prefill attention stream."""

    def __init__(self, max_inflight_layers: int):
        self.max_inflight_layers = max_inflight_layers
        self._events = deque()

    @contextmanager
    def layer(self):
        event = None
        if len(self._events) == self.max_inflight_layers:
            event = self._events[0]
            if not event.query():
                event.synchronize()
            self._events.popleft()

        yield

        # Record on the compute stream after the layer's inter-stream waits.
        self._events.append(torch.npu.current_stream().record_event(event))
