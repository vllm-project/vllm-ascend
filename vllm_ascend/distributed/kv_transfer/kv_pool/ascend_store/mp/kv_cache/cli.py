"""Process entry point for the AscendStore multiprocess KV cache server."""

import argparse
import logging
import signal
import threading
from collections.abc import Callable, Sequence
from types import FrameType
from typing import Any

from vllm_ascend import envs

from .server import DEFAULT_SCHEDULER_THREADS, DEFAULT_WORKER_THREADS, KVCacheServer

logger = logging.getLogger(__name__)

_AUTOMATIC_ABORT_EXIT_CODE = 1
_SIGINT_EXIT_CODE = 128 + signal.SIGINT

# ==============================
# Command-line entry point
# ==============================

# The CLI turns process arguments into existing service configuration and keeps
# process supervision outside the RPC and KV cache layers. It intentionally
# exposes only settings that a server operator owns today.


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected vLLM Ascend command."""
    logging.basicConfig(level=logging.INFO)
    args = _build_parser().parse_args(argv)
    if args.command == "kv-cache-server":
        return _run_kv_cache_server(args.bind_url, args.scheduler_threads, args.worker_threads)
    raise RuntimeError(f"Unsupported command: {args.command}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vllm-ascend")
    commands = parser.add_subparsers(dest="command", required=True)

    server_parser = commands.add_parser("kv-cache-server", help="Run the AscendStore multiprocess KV cache server.")
    default_bind_url = envs.VLLM_ASCEND_STORE_SERVER_URL
    if not default_bind_url:
        raise ValueError("VLLM_ASCEND_STORE_SERVER_URL must be a non-empty string")
    server_parser.add_argument(
        "--bind-url",
        type=_non_empty_url,
        default=default_bind_url,
        help=f"ZMQ URL to bind (default: {default_bind_url}).",
    )
    server_parser.add_argument(
        "--scheduler-threads",
        type=_positive_int,
        default=DEFAULT_SCHEDULER_THREADS,
        help=f"Scheduler execution threads (default: {DEFAULT_SCHEDULER_THREADS}).",
    )
    server_parser.add_argument(
        "--worker-threads",
        type=_positive_int,
        default=DEFAULT_WORKER_THREADS,
        help=f"Worker execution threads (default: {DEFAULT_WORKER_THREADS}).",
    )
    return parser


def _non_empty_url(value: str) -> str:
    if not value:
        raise argparse.ArgumentTypeError("expected a non-empty ZMQ URL")
    return value


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a value greater than 0, got {parsed}")
    return parsed


def _run_kv_cache_server(bind_url: str, scheduler_threads: int, worker_threads: int) -> int:
    shutdown = _ServerShutdown()
    shutdown.install()
    control_thread: threading.Thread | None = None

    try:
        server = KVCacheServer(bind_url, scheduler_threads=scheduler_threads, worker_threads=worker_threads)
        control_thread = threading.Thread(
            target=_coordinate_server_shutdown,
            args=(server, shutdown),
            daemon=True,
            name="ascend-store-kv-shutdown",
        )
        control_thread.start()
        logger.info("AscendStore KV cache server listening on %s", server.endpoint)
        server.run()
    finally:
        shutdown.notify_server_stopped()
        if control_thread is not None:
            control_thread.join()
        shutdown.restore()

    return shutdown.exit_code


# ==============================
# Server process shutdown
# ==============================

# Signal handlers only record intent because they run in the main thread and
# may interrupt code that already owns a server lock. A control thread performs
# the lifecycle calls, while the recorded outcome lets the process distinguish
# a completed drain, an automatic abort, and a later Ctrl-C that cancels draining.


class _ServerShutdown:
    """Coordinate process signals with one server run and its exit status."""

    def __init__(self) -> None:
        self.shutdown_requested = threading.Event()
        self.force_abort_requested = threading.Event()
        self.graceful_shutdown_unavailable = threading.Event()
        self.server_stopped = threading.Event()
        self._abort_waiter = threading.Event()
        self._previous_handlers: dict[int, Callable[[int, FrameType | None], Any] | int | None] = {}

    @property
    def exit_code(self) -> int:
        if self.force_abort_requested.is_set():
            return _SIGINT_EXIT_CODE
        if self.graceful_shutdown_unavailable.is_set():
            return _AUTOMATIC_ABORT_EXIT_CODE
        return 0

    def install(self) -> None:
        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                self._previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle)
        except BaseException:
            self.restore()
            raise

    def restore(self) -> None:
        for signum, handler in self._previous_handlers.items():
            signal.signal(signum, handler)
        self._previous_handlers.clear()

    def notify_server_stopped(self) -> None:
        """Release the control thread after the server stops on its own."""
        self.server_stopped.set()
        self.shutdown_requested.set()
        self._abort_waiter.set()

    def wait_for_abort_or_server_stop(self) -> None:
        """Wait without treating a completed server run as a forced abort."""
        self._abort_waiter.wait()

    def _handle(self, signum: int, _frame: FrameType | None) -> None:
        if self.server_stopped.is_set():
            return
        if not self.shutdown_requested.is_set():
            self.shutdown_requested.set()
        elif signum == signal.SIGINT:
            self.force_abort_requested.set()
            self._abort_waiter.set()


def _coordinate_server_shutdown(server: KVCacheServer, shutdown: _ServerShutdown) -> None:
    shutdown.shutdown_requested.wait()
    if shutdown.server_stopped.is_set():
        return

    logger.info("Graceful shutdown started; press Ctrl-C again to abort outstanding requests")
    if not server.request_stop():
        shutdown.graceful_shutdown_unavailable.set()
        logger.warning("Graceful shutdown is unavailable; aborting the KV cache server")
        server.abort()
        return

    shutdown.wait_for_abort_or_server_stop()
    if shutdown.server_stopped.is_set():
        return

    logger.warning("Forced KV cache server abort requested")
    server.abort()


if __name__ == "__main__":
    raise SystemExit(main())
