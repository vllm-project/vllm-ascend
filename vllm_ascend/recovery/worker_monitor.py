import time
import threading

import pickle
import zmq
from vllm.logger import logger
from vllm.utils.network_utils import make_zmq_socket


class RecoveryMonitor:
    def __init__(
        self,
        worker_rank: int,
        local_rank: int,
        main_monitor_addr: str,
        recover_step_sub_addr: str,
        recover_report_push_addr: str,
    ):
        self._worker_rank = worker_rank
        self._local_rank = local_rank
        self._in_recovery = False

        ctx = zmq.Context()

        self._pull_sock = make_zmq_socket(
            ctx, main_monitor_addr, zmq.PULL, bind=True,
        )

        self._recover_step_sock = make_zmq_socket(
            ctx, recover_step_sub_addr, zmq.SUB, bind=False,
        )
        self._recover_step_sock.setsockopt_string(zmq.SUBSCRIBE, "")

        self._recover_report_sock = make_zmq_socket(
            ctx, recover_report_push_addr, zmq.PUSH, bind=False,
        )

        self._poller = zmq.Poller()
        self._poller.register(self._pull_sock, zmq.POLLIN)
        self._poller.register(self._recover_step_sock, zmq.POLLIN)

        self._thread: threading.Thread = threading.Thread(
            target=self._run, name=f"RecoveryMonitor-{self._worker_rank}", daemon=True
        )

    @property
    def in_recovery(self) -> bool:
        return self._in_recovery

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while True:
            try:
                events = dict(self._poller.poll(timeout=1000))

                if self._pull_sock in events:
                    self._handle_exception_msg()

                if self._recover_step_sock in events:
                    self._handle_recover_step_msg()

            except Exception:
                logger.exception("[RecoveryMonitor] Unexpected error, restarting loop")
                time.sleep(1)

    def _handle_exception_msg(self) -> None:
        buffer = self._pull_sock.recv()

        try:
            msg = pickle.loads(buffer)
        except Exception:
            logger.exception("[RecoveryMonitor] Failed to deserialize exception msg")
            return

        logger.info("[RecoveryMonitor] Received exception msg: %s", msg)
        # TODO: handle exception

    def _handle_recover_step_msg(self) -> None:
        buffer = self._recover_step_sock.recv()

        try:
            msg = pickle.loads(buffer)
        except Exception:
            logger.exception("[RecoveryMonitor] Failed to deserialize recover step msg")
            return

        logger.info("[RecoveryMonitor] Received recover step msg: %s", msg)
        # TODO: handle recover step msg
