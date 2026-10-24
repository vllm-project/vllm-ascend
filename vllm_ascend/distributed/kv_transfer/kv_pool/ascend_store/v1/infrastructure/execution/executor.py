"""Provide serial execution for caller-supplied tasks."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import ParamSpec, TypeVar

Parameters = ParamSpec("Parameters")
ResultT = TypeVar("ResultT")


class Executor:
    """Own one initialized worker thread; task results and exceptions belong to callers."""

    def __init__(self, name: str, initializer: Callable[[], None]) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name, initializer=initializer)
        self._stopped = False

    def start(self) -> None:
        # The pool starts lazily; this barrier confirms its thread initializer has finished.
        self._pool.submit(lambda: None).result()

    def submit(
        self, fn: Callable[Parameters, ResultT], /, *args: Parameters.args, **kwargs: Parameters.kwargs
    ) -> Future[ResultT]:
        return self._pool.submit(fn, *args, **kwargs)

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        self._stopped = True

    @property
    def stopped(self) -> bool:
        """Whether close has finished draining tasks and joining the worker thread."""
        return self._stopped
