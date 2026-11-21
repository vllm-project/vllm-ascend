# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Keep shutdown-signal ownership with the worker's caller.

Some torch-npu graph SDKs install process-wide handlers on their first import.
Lazy imports during profiling/capture must not replace EngineCore/WorkerProc
shutdown handlers. SDK atexit finalizers remain registered and run normally.
"""

import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from vllm.logger import logger


@contextmanager
def preserve_worker_signal_handlers() -> Iterator[None]:
    if threading.current_thread() is not threading.main_thread():
        # The affected SDK imports also install handlers only on the main
        # thread; Python does not allow another thread to restore them.
        yield
        return
    handlers = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield
    finally:
        for number, handler in handlers.items():
            # None denotes a native-installed handler which Python cannot
            # reinstall. Do not pass it to signal.signal or guess its value.
            if handler is not None and signal.getsignal(number) != handler:
                signal.signal(number, handler)
                logger.info("Restored caller's %s handler after graph SDK initialization.", number.name)
