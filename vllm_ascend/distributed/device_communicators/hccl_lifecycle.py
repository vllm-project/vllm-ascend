# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from collections.abc import Callable
from weakref import WeakMethod


class HcclSleepAdapter:
    """Keep a PG owner's sleep entry point alive across native PG teardown.

    Callbacks reuse the coordinator's PG registry and topology. A failed native
    operation may have made partial progress: only final teardown is safe after
    that, rather than an automatic retry or a raw Suspend-to-Destroy fallback.
    """

    def __init__(self, release: Callable[[], None], restore: Callable[[], None]):
        # The upstream group registry uses weak references. Bound callbacks
        # must not keep retired coordinators alive through a reference cycle.
        self._release = WeakMethod(release)
        self._restore = WeakMethod(restore)
        self._suspended = False
        self._failed = False
        self._closed = False

    def validate(self) -> None:
        if self._closed or self._failed:
            raise RuntimeError("HCCL sleep lifecycle is closed or failed; restart the worker.")

    def suspend(self) -> None:
        self.validate()
        if self._suspended:
            return
        try:
            release = self._release()
            if release is None:
                raise RuntimeError("The HCCL process-group owner no longer exists.")
            release()
        except Exception:
            self._failed = True
            raise
        self._suspended = True

    def resume(self) -> None:
        self.validate()
        if not self._suspended:
            return
        try:
            restore = self._restore()
            if restore is None:
                raise RuntimeError("The HCCL process-group owner no longer exists.")
            restore()
        except Exception:
            self._failed = True
            raise
        self._suspended = False

    def close(self) -> None:
        """Prevent subsequent recovery; the coordinator performs final release."""
        self._closed = True
