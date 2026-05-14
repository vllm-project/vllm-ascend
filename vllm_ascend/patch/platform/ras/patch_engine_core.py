import pickle
import zmq
from contextlib import ExitStack
import threading

from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import EngineCoreProc
from vllm.utils.network_utils import make_zmq_socket
from vllm_ascend.recovery.engine_core_handler import RecoveryHandler


_RECOVERY_MSG_PREFIX = b"\x00REC"

class RecoveryHandler:
    def __init__(self, engine_core: Any, worker_count: int, engine_index: int = 0, expect_coordinator: bool = False):
        self._engine_core = engine_core
        self._worker_count = worker_count
        self._engine_index = engine_index
        self._expect_coordinator = expect_coordinator

        self._ctx = zmq.Context()
        self._coord_push_sock: zmq.Socket | None = None
        self._coord_sub_sock: zmq.Socket | None = None

        self._coord_ready_event = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="RecoveryHandler", daemon=True
        )
        self._thread.start()

    def setup_worker_sockets(
        self,
        cmd_pub_addr: str,
        worker_msg_pull_addr: str,
    ) -> None:
        # TODO: setup worker sockets
        return

    def wait_for_worker_subscriptions(self) -> None:
        # TODO: wait for worker subscriptions
        return

    def connect_coordinator(self, coord_sub_addr: str, coord_push_addr: str) -> None:
        self._coord_sub_sock = make_zmq_socket(
            self._ctx, coord_sub_addr, zmq.SUB, bind=False,
        )
        self._coord_sub_sock.setsockopt_string(zmq.SUBSCRIBE, "")

        self._coord_push_sock = make_zmq_socket(
            self._ctx, coord_push_addr, zmq.PUSH, bind=False,
        )

        ping_msg = _RECOVERY_MSG_PREFIX + pickle.dumps(
            {"type": "PING", "engine_index": self._engine_index}
        )
        self._coord_push_sock.send(ping_msg)

        self._coord_ready_event.set()

        logger.info(
            "=========[RecoveryHandler] Connected to coordinator: sub=%s, push=%s",
            coord_sub_addr, coord_push_addr,
        )

    def notify_recovery_needed(self, plan: RecoveryPlan) -> None:
        # TODO: notify recovery needed
        return

    def _run(self) -> None:
        if self._expect_coordinator:
            logger.info("======[RecoveryHandler] Waiting for coordinator connection...======")
            self._coord_ready_event.wait()
            logger.info("======[RecoveryHandler] Coordinator connected, entering main loop======")

            if self._coord_sub_sock is not None:
                buffer = self._coord_sub_sock.recv()
                if buffer == b"RECOVERY_READY":
                    logger.info("======[RecoveryHandler] Received RECOVERY_READY message======")

        poller = zmq.Poller()
        if self._coord_sub_sock is not None:
            poller.register(self._coord_sub_sock, zmq.POLLIN)

        while True:

            events = dict(poller.poll(timeout=1000))

            if self._coord_sub_sock is not None and self._coord_sub_sock in events:
                self._handle_coord_msg()

    def _handle_worker_msg(self) -> None:
        #TODO: handle worker message
        return

    def _handle_fault_report(self, msg: dict) -> None:
        #TODO: handle fault report
        return

    def _handle_step_result(self, msg: dict) -> None:
        #TODO: handle step result
        return

    def _handle_coord_msg(self) -> None:
        #TODO: handle coordinator message
        return

    def _notify_coordinator_start_recovery(self) -> None:
        #TODO: notify coordinator recovery start
        return

    def _notify_coordinator_recover_done(self, success: bool) -> None:
        #TODO: notify coordinator recovery done
        return

    def _execute_recovery(self) -> None:
        #TODO: execute recovery
        return

    def _execute_ec_step(self, step: Any) -> bool:
        #TODO: execute ec step
        return True

    def _execute_worker_step(self, step: Any) -> bool:
        #TODO: execute worker step
        return True

    def _broadcast_recovery_done(self) -> None:
        #TODO: broadcast recovery_done to all workers
        return


_original_engine_core_proc_init = EngineCoreProc.__init__

def _patched_engine_core_proc_init(
    self,
    vllm_config,
    local_client,
    handshake_address,
    executor_class,
    log_stats,
    client_handshake_address=None,
    *,
    engine_index=0,
):
    _original_engine_core_proc_init(
        self,
        vllm_config,
        local_client,
        handshake_address,
        executor_class,
        log_stats,
        client_handshake_address,
        engine_index=engine_index,
    )

    worker_count = vllm_config.parallel_config.local_world_size

    self._recovery_handler = RecoveryHandler(
        engine_core=self,
        worker_count=worker_count,
        engine_index=engine_index,
    )

    self._recovery_handler.start()
    self._recovery_handler.wait_for_worker_subscriptions()

EngineCoreProc.__init__ = _patched_engine_core_proc_init


_original_process_input_sockets = EngineCoreProc.process_input_sockets

def _patched_process_input_sockets(
    self,
    input_addresses,
    coord_input_address,
    identity,
    ready_event,
):
    add_request_decoder = MsgpackDecoder(EngineCoreRequest)
    generic_decoder = MsgpackDecoder()

    with ExitStack() as stack, zmq.Context() as ctx:
        input_sockets = [
            stack.enter_context(
                make_zmq_socket(
                    ctx, input_address, zmq.DEALER, identity=identity, bind=False
                )
            )
            for input_address in input_addresses
        ]
        if coord_input_address is None:
            coord_socket = None
        else:
            coord_socket = stack.enter_context(
                make_zmq_socket(
                    ctx,
                    coord_input_address,
                    zmq.XSUB,
                    identity=identity,
                    bind=False,
                )
            )
            coord_socket.send(b"\x01")

        poller = zmq.Poller()
        for input_socket in input_sockets:
            input_socket.send(b"")
            poller.register(input_socket, zmq.POLLIN)

        if coord_socket is not None:
            assert coord_socket.recv() == b"READY"
            poller.register(coord_socket, zmq.POLLIN)

            if self._recovery_handler is not None:
                buffer = coord_socket.recv()
                if buffer.startswith(_RECOVERY_MSG_PREFIX):
                    payload = buffer[len(_RECOVERY_MSG_PREFIX):]
                    try:
                        msg = pickle.loads(payload)
                        if msg[0] == "RECOVERY_ADDRESSES":
                            _, sub_addr, push_addr = msg
                            logger.info(
                                "=========================Received recovery addresses: sub=%s, push=%s=========================",
                                sub_addr, push_addr,
                            )
                            self._recovery_handler.connect_coordinator(sub_addr, push_addr)
                    except Exception as e:
                        logger.exception(f"Failed to deserialize recovery addr msg: {e}")

        ready_event.set()
        del ready_event

        while True:
            for input_socket, _ in poller.poll():
                type_frame, *data_frames = input_socket.recv_multipart(copy=False)

                if input_socket == coord_socket and type_frame.buffer == b"READY":
                    continue

                if type_frame.buffer.startswith(_RECOVERY_MSG_PREFIX):
                    continue

                request_type = EngineCoreRequestType(bytes(type_frame.buffer))

                request = None
                if request_type == EngineCoreRequestType.ADD:
                    req = add_request_decoder.decode(data_frames)
                    try:
                        request = self.preprocess_add_request(req)
                    except Exception:
                        self._handle_request_preproc_error(req)
                        continue
                else:
                    request = generic_decoder.decode(data_frames)

                    if request_type == EngineCoreRequestType.ABORT:
                        self.aborts_queue.put_nowait(request)

                self.input_queue.put_nowait((request_type, request))


EngineCoreProc.process_input_sockets = _patched_process_input_sockets