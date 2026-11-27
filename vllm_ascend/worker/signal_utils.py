# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Keep engine-owned termination handlers across lazy backend initialization."""

import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def preserve_termination_handlers() -> Iterator[None]:
    # Some compiler backend imports install process-wide handlers. The engine
    # must retain its shutdown protocol; backend atexit cleanup is unaffected.
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield
    finally:
        for signum, handler in handlers.items():
            # None denotes a native handler not representable by signal.signal.
            if handler is not None and signal.getsignal(signum) is not handler:
                signal.signal(signum, handler)
